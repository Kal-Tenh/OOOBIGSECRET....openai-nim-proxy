// server.js - OpenAI to NVIDIA NIM API Proxy (with keep-alive for slow models)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Optional: protect your proxy so strangers can't burn your NIM key.
// Set PROXY_API_KEY in your env vars, then put the same value in JanitorAI's API key field.
const PROXY_API_KEY = process.env.PROXY_API_KEY;

if (!NIM_API_KEY) {
  console.error('⚠️  WARNING: NIM_API_KEY is not set! Requests to NVIDIA NIM will fail with an auth error.');
}

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE
// false = ask each model to think as little as possible (see THINKING_OFF below)
// true  = turn thinking on
const ENABLE_THINKING_MODE = false;

// 🔥 KEEP-ALIVE - Stops the host's gateway from cutting the connection while NVIDIA is slow.
// If NVIDIA hasn't answered after KEEPALIVE_DELAY_MS, we open the response and send a tiny
// ping every KEEPALIVE_INTERVAL_MS until the real reply is ready.
const KEEPALIVE_ENABLED = true;
const KEEPALIVE_DELAY_MS = 10000;
const KEEPALIVE_INTERVAL_MS = 10000;

// Upstream timeout (keep this below JanitorAI's 300000 ms limit)
const UPSTREAM_TIMEOUT_MS = 240000;

// Model mapping (adjust based on available NIM models)
const MODEL_MAPPING = {
  'glm-5.3': 'z-ai/glm-5.3',
  'glm-5.3-flash': 'z-ai/glm-5.3-flash',
  'gpt-3.5-turbo': 'deepseek-ai/deepseek-v4-flash-0731',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking',
  'claude-3-sonnet': 'deepseek-v4-pro-0813',
  'kimi-k3': 'moonshotai/kimi-k3',
  'deepseek': 'deepseek-ai/deepseek-v4.1-flash'
};

// Params that specific NIM models reject (they return 400 "immutable")
const STRIP_PARAMS = {
  'moonshotai/kimi-k3': ['frequency_penalty', 'presence_penalty']
};

// Per-model "thinking off" settings, sent as chat_template_kwargs.
// Each model family uses different flags, so they are not shared:
//  - GLM-5.3 has no real on/off switch (its template ignores enable_thinking),
//    so the best we can do is lowest reasoning effort.
//  - Kimi K3 accepts a plain off switch.
//  - DeepSeek V4.1 Flash: thinking:false is the best known flag.
// Models not listed here get no chat_template_kwargs at all.
const THINKING_OFF = {
  'z-ai/glm-5.3': { reasoning_effort: 'low' },
  'z-ai/glm-5.3-flash': { reasoning_effort: 'low' },
  'moonshotai/kimi-k3': { thinking: false, enable_thinking: false },
  'deepseek-ai/deepseek-v4.1-flash': { thinking: false }
};

// Cache for unmapped model probes so we only probe each name once
const modelProbeCache = new Map();

// Helper: pull the most useful error message out of an axios error
function extractErrorMessage(error) {
  return (
    error.response?.data?.error?.message ||
    error.response?.data?.message ||
    (typeof error.response?.data === 'string' ? error.response.data : null) ||
    error.message ||
    'Internal server error'
  );
}

// Helper: when responseType is 'stream', error bodies arrive as streams.
// Read them fully so we can show the real error message.
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

// Optional auth for /v1 routes
app.use('/v1', (req, res, next) => {
  if (!PROXY_API_KEY) return next();
  if (req.headers.authorization !== `Bearer ${PROXY_API_KEY}`) {
    return res.status(401).json({
      error: { message: 'Unauthorized', type: 'invalid_request_error', code: 401 }
    });
  }
  next();
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    keepalive: KEEPALIVE_ENABLED,
    api_key_configured: Boolean(NIM_API_KEY),
    proxy_auth_enabled: Boolean(PROXY_API_KEY)
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map((model) => ({
    id: model,
    object: 'model',
    created: Math.floor(Date.now() / 1000),
    owned_by: 'nvidia-nim-proxy'
  }));

  res.json({ object: 'list', data: models });
});

// Resolve which NIM model to use for the incoming model name
async function resolveModel(model) {
  const mapped = MODEL_MAPPING[model];
  if (mapped) return mapped;

  if (modelProbeCache.has(model)) return modelProbeCache.get(model);

  let resolved = null;

  // Probe: maybe the client sent a real NIM model id
  try {
    const testResponse = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      { model, messages: [{ role: 'user', content: 'test' }], max_tokens: 1 },
      {
        headers: { Authorization: `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
        validateStatus: (status) => status < 500,
        timeout: 30000
      }
    );
    if (testResponse.status >= 200 && testResponse.status < 300) {
      resolved = model;
    }
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

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  let keepAliveTimer = null;
  let keepAliveInterval = null;
  const controller = new AbortController();

  const stopKeepAlive = () => {
    if (keepAliveTimer) { clearTimeout(keepAliveTimer); keepAliveTimer = null; }
    if (keepAliveInterval) { clearInterval(keepAliveInterval); keepAliveInterval = null; }
  };

  // Client left before we finished: stop timers and cancel the upstream request
  res.on('close', () => {
    stopKeepAlive();
    if (!res.writableFinished) controller.abort();
  });

  try {
    const {
      model, messages, temperature, max_tokens, stream,
      top_p, stop, presence_penalty, frequency_penalty
    } = req.body;

    if (!model || typeof model !== 'string') {
      return res.status(400).json({
        error: { message: '"model" is required and must be a string', type: 'invalid_request_error', code: 400 }
      });
    }

    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({
        error: { message: '"messages" is required and must be an array', type: 'invalid_request_error', code: 400 }
      });
    }

    const setSSEHeaders = () => {
      res.status(200);
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no'); // stop proxies from buffering the stream
    };

    // Keep-alive: if NVIDIA is slow, open the response and send tiny pings so the
    // host's gateway doesn't kill the connection. Fast requests are unaffected.
    if (KEEPALIVE_ENABLED) {
      keepAliveTimer = setTimeout(() => {
        keepAliveTimer = null;
        if (res.headersSent || res.writableEnded) return;

        let ping;
        if (stream) {
          setSSEHeaders();
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
        console.log(`Keep-alive started (${stream ? 'stream' : 'non-stream'})`);

        keepAliveInterval = setInterval(() => {
          try { res.write(ping); } catch (e) {}
        }, KEEPALIVE_INTERVAL_MS);
      }, KEEPALIVE_DELAY_MS);
    }

    const nimModel = await resolveModel(model);
    console.log(`Incoming model: "${model}" | Mapped to: "${nimModel}" | stream: ${Boolean(stream)}`);

    // Transform OpenAI request to NIM format
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

    // Thinking control (per model, see THINKING_OFF above)
    if (ENABLE_THINKING_MODE) {
      nimRequest.chat_template_kwargs = { thinking: true };
    } else if (THINKING_OFF[nimModel]) {
      nimRequest.chat_template_kwargs = THINKING_OFF[nimModel];
    }

    (STRIP_PARAMS[nimModel] || []).forEach((p) => delete nimRequest[p]);

    // Make request to NVIDIA NIM API
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

    stopKeepAlive();
    console.log(`Request to NIM (${nimModel}) completed with status ${response.status}`);

    if (stream) {
      // Streaming response
      if (!res.headersSent) {
        setSSEHeaders();
        res.flushHeaders();
      }

      let buffer = '';
      let reasoningStarted = false;

      const processLine = (line) => {
        line = line.trim();
        if (!line.startsWith('data:')) return;

        const payload = line.slice(5).trim();

        if (payload === '[DONE]') {
          res.write('data: [DONE]\n\n');
          return;
        }

        try {
          const data = JSON.parse(payload);

          // Surface upstream errors sent mid-stream
          if (data.error) {
            console.error('Upstream stream error:', data.error);
            res.write(`data: ${JSON.stringify({ error: data.error })}\n\n`);
            return;
          }

          const choice = data.choices?.[0];

          if (!choice || !choice.delta) {
            // Chunk with no delta (e.g. usage chunk): pass through
            res.write(`data: ${JSON.stringify(data)}\n\n`);
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

            if (combined || choice.finish_reason || choice.delta.role) {
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            }
          } else {
            // Skip pure reasoning chunks, but never drop role or finish chunks
            if (content || choice.finish_reason || choice.delta.role) {
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            }
          }
        } catch (e) {
          // Not valid JSON: pass through raw rather than dropping it silently
          res.write(line + '\n\n');
        }
      };

      // Keep the connection alive during long reasoning phases where we hide reasoning
      let lastWrite = Date.now();
      const streamPing = setInterval(() => {
        if (Date.now() - lastWrite > KEEPALIVE_INTERVAL_MS) {
          try { res.write(': keep-alive\n\n'); lastWrite = Date.now(); } catch (e) {}
        }
      }, KEEPALIVE_INTERVAL_MS);

      res.on('close', () => {
        clearInterval(streamPing);
        try { response.data.destroy(); } catch (e) {}
      });

      const origWrite = res.write.bind(res);
      res.write = (...args) => { lastWrite = Date.now(); return origWrite(...args); };

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        lines.forEach(processLine);
      });

      response.data.on('end', () => {
        clearInterval(streamPing);
        if (buffer.trim()) processLine(buffer);
        res.end();
      });

      response.data.on('error', (err) => {
        clearInterval(streamPing);
        console.error('Stream error:', err.message);
        try {
          res.write(`data: ${JSON.stringify({ error: { message: err.message || 'Stream error' } })}\n\n`);
        } catch (writeErr) {
          // ignore if we can't write anymore
        }
        res.end();
      });
    } else {
      // Non-streaming: transform NIM response to OpenAI format
      const choices = response.data.choices || [];

      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: choices.map((choice) => {
          let fullContent = choice.message?.content || '';

          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }

          return {
            index: choice.index,
            message: {
              role: choice.message?.role || 'assistant',
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };

      if (res.headersSent) {
        // Keep-alive already opened the response, so finish it manually
        res.end(JSON.stringify(openaiResponse));
      } else {
        res.json(openaiResponse);
      }
    }
  } catch (error) {
    stopKeepAlive();

    // Client already left, nothing to send
    if (error.code === 'ERR_CANCELED' || controller.signal.aborted) {
      console.log('Client disconnected before the reply was ready');
      return;
    }

    await readErrorBody(error);

    const status = error.response?.status || 500;
    const message = extractErrorMessage(error);

    console.error('Proxy error:', {
      status,
      message,
      data: error.response?.data
    });

    const errorBody = {
      error: {
        message: message,
        type: 'invalid_request_error',
        code: status
      }
    };

    // Keep-alive already sent a 200 header, so the error has to go in the body
    if (res.headersSent) {
      try {
        const isSSE = String(res.getHeader('Content-Type') || '').includes('text/event-stream');
        if (isSSE) {
          res.write(`data: ${JSON.stringify(errorBody)}\n\n`);
          res.write('data: [DONE]\n\n');
          res.end();
        } else {
          res.end(JSON.stringify(errorBody));
        }
      } catch (e) {}
      return;
    }

    res.status(status).json(errorBody);
  }
});

// Catch-all for unsupported endpoints (works on Express 4 and 5)
app.use((req, res) => {
  console.log(`404: ${req.method} ${req.originalUrl}`);
  res.status(404).json({
    error: {
      message: `Endpoint ${req.method} ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Keep-alive: ${KEEPALIVE_ENABLED ? 'ENABLED' : 'DISABLED'}`);
  console.log(`NIM_API_KEY configured: ${NIM_API_KEY ? 'YES' : 'NO - set this env var!'}`);
  console.log(`Proxy auth: ${PROXY_API_KEY ? 'ENABLED' : 'DISABLED (anyone with the URL can use it)'}`);
});
