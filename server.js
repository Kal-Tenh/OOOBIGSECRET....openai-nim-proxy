// server.js - OpenAI to NVIDIA NIM API Proxy
// Features: mandatory proxy key, per-model thinking control, keep-alive, timing logs.

const crypto = require('crypto');
const express = require('express');
const cors = require('cors');
const axios = require('axios');

// =============================================================================
// 1. ENVIRONMENT (set these in your host's environment variables)
// =============================================================================

const PORT = process.env.PORT || 3000;
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;     // your NVIDIA key
const PROXY_API_KEY = process.env.PROXY_API_KEY; // password clients must send

// Refuse to start without both keys, so the proxy can never run open to the public.
if (!PROXY_API_KEY) {
  console.error('FATAL: PROXY_API_KEY is not set. Refusing to start without authentication.');
  process.exit(1);
}
if (!NIM_API_KEY) {
  console.error('FATAL: NIM_API_KEY is not set. Requests to NVIDIA NIM would fail.');
  process.exit(1);
}

// =============================================================================
// 2. SETTINGS
// =============================================================================

// Show the model's reasoning inside <think> tags (false = hide it)
const SHOW_REASONING = false;

// false = ask each model to think as little as possible (see THINKING_OFF)
// true  = turn thinking on
const ENABLE_THINKING_MODE = false;

// Keep-alive: if NVIDIA is slow, open the response and send tiny pings
// so the host's gateway doesn't cut the connection.
const KEEPALIVE_ENABLED = true;
const KEEPALIVE_DELAY_MS = 10000;
const KEEPALIVE_INTERVAL_MS = 10000;

// Max time to wait for NVIDIA before giving up
const UPSTREAM_TIMEOUT_MS = 240000;

// =============================================================================
// 3. MODEL TABLES
// =============================================================================

// Names clients can use -> real NIM model ids
const MODEL_MAPPING = {
  'glm-5.3': 'z-ai/glm-5.3',
  'glm-5.3-flash': 'z-ai/glm-5.3-flash',
  'gpt-3.5-turbo': 'deepseek-ai/deepseek-v4-flash-0731',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking',
  'claude-3-sonnet': 'deepseek-ai/deepseek-v4-pro-0813',
  'kimi-k3': 'moonshotai/kimi-k3',
  'deepseek': 'deepseek-ai/deepseek-v4.1-flash'
};

// Params that specific NIM models reject (they return 400 "immutable")
const STRIP_PARAMS = {
  'moonshotai/kimi-k3': ['frequency_penalty', 'presence_penalty']
};

// Per-model "thinking off" settings, sent as chat_template_kwargs.
//  - GLM-5.3 has no real on/off switch (its template ignores enable_thinking),
//    so the best we can do is the lowest reasoning effort.
//  - Kimi K3 accepts a plain off switch.
//  - DeepSeek V4.1 Flash: thinking:false is the best known flag.
// Models not listed here get no chat_template_kwargs at all.
const THINKING_OFF = {
  'z-ai/glm-5.3': { reasoning_effort: 'low' },
  'z-ai/glm-5.3-flash': { reasoning_effort: 'low' },
  'moonshotai/kimi-k3': { thinking: false, enable_thinking: false },
  'deepseek-ai/deepseek-v4.1-flash': { thinking: false }
};

// Remembers results for unmapped model names so each is probed only once
const modelProbeCache = new Map();

// =============================================================================
// 4. HELPERS
// =============================================================================

// Constant-time check of the "Authorization: Bearer <key>" header
function isAuthorized(header = '') {
  const expected = Buffer.from(`Bearer ${PROXY_API_KEY}`);
  const given = Buffer.from(header);
  return given.length === expected.length && crypto.timingSafeEqual(given, expected);
}

// Pull the most useful error message out of an axios error
function extractErrorMessage(error) {
  return (
    error.response?.data?.error?.message ||
    error.response?.data?.message ||
    (typeof error.response?.data === 'string' ? error.response.data : null) ||
    error.message ||
    'Internal server error'
  );
}

// With responseType 'stream', error bodies arrive as streams. Read them fully
// so the real error message can be shown.
async function readErrorBody(error) {
  const d = error.response?.data;
  if (d && typeof d.on === 'function') {
    const body = await new Promise((resolve) => {
      let b = '';
      d.on('data', (c) => (b += c));
      d.on('end', () => resolve(b));
      d.on('error', () => resolve(b));
    });
    try {
      error.response.data = JSON.parse(body);
    } catch {
      error.response.data = body;
    }
  }
}

function errorBody(message, code) {
  return { error: { message, type: 'invalid_request_error', code } };
}

function setSSEHeaders(res) {
  res.status(200);
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no'); // stop proxies from buffering the stream
}

// Decide which NIM model to use for the incoming model name
async function resolveModel(model) {
  const mapped = MODEL_MAPPING[model];
  if (mapped) return mapped;

  if (modelProbeCache.has(model)) return modelProbeCache.get(model);

  let resolved = null;

  // Probe: maybe the client sent a real NIM model id
  try {
    const probe = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      { model, messages: [{ role: 'user', content: 'test' }], max_tokens: 1 },
      {
        headers: { Authorization: `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
        validateStatus: (status) => status < 500,
        timeout: 30000
      }
    );
    if (probe.status >= 200 && probe.status < 300) resolved = model;
  } catch (e) {
    console.error('Model probe failed:', extractErrorMessage(e));
  }

  // Fallback by name
  if (!resolved) {
    const m = model.toLowerCase();
    if (m.includes('gpt-4') || m.includes('claude-opus') || m.includes('405b')) {
      resolved = 'meta/llama-3.1-405b-instruct';
    } else if (m.includes('claude') || m.includes('gemini') || m.includes('70b')) {
      resolved = 'meta/llama-3.1-70b-instruct';
    } else {
      resolved = 'meta/llama-3.1-8b-instruct';
    }
  }

  modelProbeCache.set(model, resolved);
  return resolved;
}

// Build the request we send to NIM from the client's OpenAI-style request
function buildNimRequest(body, nimModel) {
  const { messages, temperature, max_tokens, stream, top_p, stop, presence_penalty, frequency_penalty } = body;

  const nimRequest = {
    model: nimModel,
    messages,
    temperature: temperature ?? 0.6,
    max_tokens: max_tokens ?? 9024,
    stream: Boolean(stream)
  };

  if (top_p !== undefined) nimRequest.top_p = top_p;
  if (stop !== undefined) nimRequest.stop = stop;
  if (presence_penalty !== undefined) nimRequest.presence_penalty = presence_penalty;
  if (frequency_penalty !== undefined) nimRequest.frequency_penalty = frequency_penalty;

  // Thinking control (per model)
  if (ENABLE_THINKING_MODE) {
    nimRequest.chat_template_kwargs = { thinking: true };
  } else if (THINKING_OFF[nimModel]) {
    nimRequest.chat_template_kwargs = THINKING_OFF[nimModel];
  }

  (STRIP_PARAMS[nimModel] || []).forEach((p) => delete nimRequest[p]);

  return nimRequest;
}

// Keep-alive: starts a timer; if NVIDIA is still silent after KEEPALIVE_DELAY_MS,
// it opens the response and sends a ping every KEEPALIVE_INTERVAL_MS.
function createKeepAlive(res, isStream) {
  let timer = null;
  let interval = null;

  const stop = () => {
    if (timer) { clearTimeout(timer); timer = null; }
    if (interval) { clearInterval(interval); interval = null; }
  };

  const start = () => {
    if (!KEEPALIVE_ENABLED) return;

    timer = setTimeout(() => {
      timer = null;
      if (res.headersSent || res.writableEnded) return;

      let ping;
      if (isStream) {
        setSSEHeaders(res);
        ping = ': keep-alive\n\n'; // SSE comment, ignored by clients
      } else {
        res.writeHead(200, {
          'Content-Type': 'application/json',
          'Cache-Control': 'no-cache',
          'X-Accel-Buffering': 'no'
        });
        ping = ' '; // leading whitespace is valid before JSON
      }
      res.flushHeaders();
      res.write(ping);
      console.log(`Keep-alive started (${isStream ? 'stream' : 'non-stream'})`);

      interval = setInterval(() => {
        try { res.write(ping); } catch (e) {}
      }, KEEPALIVE_INTERVAL_MS);
    }, KEEPALIVE_DELAY_MS);
  };

  return { start, stop };
}

// Pipe NIM's streaming reply to the client, cleaning up reasoning chunks
function streamToClient(upstream, res) {
  if (!res.headersSent) {
    setSSEHeaders(res);
    res.flushHeaders();
  }

  let buffer = '';
  let reasoningStarted = false;
  let lastWrite = Date.now();

  const write = (text) => {
    lastWrite = Date.now();
    res.write(text);
  };
  const sendData = (obj) => write(`data: ${JSON.stringify(obj)}\n\n`);

  const processLine = (rawLine) => {
    const line = rawLine.trim();
    if (!line.startsWith('data:')) return;

    const payload = line.slice(5).trim();

    if (payload === '[DONE]') {
      write('data: [DONE]\n\n');
      return;
    }

    try {
      const data = JSON.parse(payload);

      // Surface upstream errors sent mid-stream
      if (data.error) {
        console.error('Upstream stream error:', data.error);
        sendData({ error: data.error });
        return;
      }

      const choice = data.choices?.[0];

      // Chunk with no delta (e.g. usage chunk): pass through
      if (!choice || !choice.delta) {
        sendData(data);
        return;
      }

      const reasoning = choice.delta.reasoning_content;
      const content = choice.delta.content;
      delete choice.delta.reasoning_content;

      if (SHOW_REASONING) {
        let combined = '';

        if (reasoning) {
          combined += reasoningStarted ? reasoning : '<think>\n' + reasoning;
          reasoningStarted = true;
        }

        if (content) {
          if (reasoningStarted) {
            combined += '\n</think>\n\n';
            reasoningStarted = false;
          }
          combined += content;
        }

        // Close the think tag if the stream ends while still reasoning
        if (choice.finish_reason && reasoningStarted) {
          combined += '\n</think>\n\n';
          reasoningStarted = false;
        }

        if (combined) choice.delta.content = combined;

        if (combined || choice.finish_reason || choice.delta.role) sendData(data);
      } else {
        // Skip pure reasoning chunks, but never drop role or finish chunks
        if (content || choice.finish_reason || choice.delta.role) sendData(data);
      }
    } catch (e) {
      // Not valid JSON: pass through raw rather than dropping it silently
      write(line + '\n\n');
    }
  };

  // Ping during long silent phases (e.g. hidden reasoning)
  const streamPing = setInterval(() => {
    if (Date.now() - lastWrite > KEEPALIVE_INTERVAL_MS) {
      try { write(': keep-alive\n\n'); } catch (e) {}
    }
  }, KEEPALIVE_INTERVAL_MS);

  res.on('close', () => {
    clearInterval(streamPing);
    try { upstream.destroy(); } catch (e) {}
  });

  upstream.on('data', (chunk) => {
    buffer += chunk.toString();
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    lines.forEach(processLine);
  });

  upstream.on('end', () => {
    clearInterval(streamPing);
    if (buffer.trim()) processLine(buffer);
    res.end();
  });

  upstream.on('error', (err) => {
    clearInterval(streamPing);
    console.error('Stream error:', err.message);
    try {
      sendData({ error: { message: err.message || 'Stream error' } });
    } catch (e) {}
    res.end();
  });
}

// Send a non-streaming reply in OpenAI format
function sendJsonReply(res, model, nimData) {
  const choices = nimData.choices || [];

  const reply = {
    id: `chatcmpl-${Date.now()}`,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: choices.map((choice) => {
      let fullContent = choice.message?.content || '';

      if (SHOW_REASONING && choice.message?.reasoning_content) {
        fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
      }

      return {
        index: choice.index,
        message: { role: choice.message?.role || 'assistant', content: fullContent },
        finish_reason: choice.finish_reason
      };
    }),
    usage: nimData.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
  };

  if (res.headersSent) {
    // Keep-alive already opened the response, so finish it manually
    res.end(JSON.stringify(reply));
  } else {
    res.json(reply);
  }
}

// =============================================================================
// 5. APP SETUP
// =============================================================================

const app = express();
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Public health check (shows no secrets)
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    keepalive: KEEPALIVE_ENABLED
  });
});

// Everything under /v1 requires the proxy key
app.use('/v1', (req, res, next) => {
  if (!isAuthorized(req.headers.authorization)) {
    console.log('401: wrong or missing proxy key');
    return res.status(401).json(errorBody('Unauthorized', 401));
  }
  next();
});

// =============================================================================
// 6. ROUTES
// =============================================================================

// List models (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map((model) => ({
    id: model,
    object: 'model',
    created: Math.floor(Date.now() / 1000),
    owned_by: 'nvidia-nim-proxy'
  }));

  res.json({ object: 'list', data: models });
});

// Chat completions (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  const t0 = Date.now();
  const elapsed = () => ((Date.now() - t0) / 1000).toFixed(0) + 's';

  const controller = new AbortController();
  let keepAlive = null;

  // Client left before we finished: stop timers and cancel the NVIDIA request
  res.on('close', () => {
    if (keepAlive) keepAlive.stop();
    if (!res.writableFinished) controller.abort();
  });

  try {
    const { model, messages, stream } = req.body;

    if (!model || typeof model !== 'string') {
      return res.status(400).json(errorBody('"model" is required and must be a string', 400));
    }
    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json(errorBody('"messages" is required and must be an array', 400));
    }

    keepAlive = createKeepAlive(res, Boolean(stream));
    keepAlive.start();

    const nimModel = await resolveModel(model);
    console.log(`Incoming model: "${model}" | Mapped to: "${nimModel}" | stream: ${Boolean(stream)}`);

    const nimRequest = buildNimRequest(req.body, nimModel);

    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      timeout: UPSTREAM_TIMEOUT_MS,
      signal: controller.signal,
      validateStatus: (status) => status < 400 // 4xx/5xx go to catch with the real body
    });

    keepAlive.stop();
    console.log(`Request to NIM (${nimModel}) completed with status ${response.status} after ${elapsed()}`);

    if (stream) {
      streamToClient(response.data, res);
    } else {
      sendJsonReply(res, model, response.data);
    }
  } catch (error) {
    if (keepAlive) keepAlive.stop();

    // Client already left, nothing to send
    if (error.code === 'ERR_CANCELED' || controller.signal.aborted) {
      console.log(`Client disconnected after ${elapsed()}`);
      return;
    }

    await readErrorBody(error);

    const status = error.response?.status || 500;
    const message = extractErrorMessage(error);

    console.error('Proxy error:', { status, message, data: error.response?.data });

    const body = errorBody(message, status);

    // Keep-alive already sent a 200 header, so the error has to go in the body
    if (res.headersSent) {
      try {
        const isSSE = String(res.getHeader('Content-Type') || '').includes('text/event-stream');
        if (isSSE) {
          res.write(`data: ${JSON.stringify(body)}\n\n`);
          res.write('data: [DONE]\n\n');
          res.end();
        } else {
          res.end(JSON.stringify(body));
        }
      } catch (e) {}
      return;
    }

    res.status(status).json(body);
  }
});

// Catch-all for unsupported endpoints (works on Express 4 and 5)
app.use((req, res) => {
  console.log(`404: ${req.method} ${req.originalUrl}`);
  res.status(404).json(errorBody(`Endpoint ${req.method} ${req.path} not found`, 404));
});

// =============================================================================
// 7. START
// =============================================================================

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Keep-alive: ${KEEPALIVE_ENABLED ? 'ENABLED' : 'DISABLED'}`);
  console.log('Proxy auth: ENABLED');
});
