// server.js - OpenAI to NVIDIA NIM API Proxy
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

if (!NIM_API_KEY) {
  console.error('⚠️  WARNING: NIM_API_KEY is not set! Requests to NVIDIA NIM will fail with an auth error.');
}

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = false; // Set to true to enable chat_template_kwargs thinking parameter

// Model mapping (adjust based on available NIM models)
const MODEL_MAPPING = {
  'glm-5.3': 'z-ai/glm-5.3',
  'glm-5.3-flash': 'z-ai/glm-5-3-flash',
  'gpt-3.5-turbo': 'deepseek-ai/deepseek-v4-flash-0731',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking',
  'claude-3-sonnet': 'deepseek-v4-pro-0813',
  'kimi-k3': 'moonshotai/kimi-k3',
  'deepseek': 'deepseek-ai/deepseek-v4.1-flash'
};

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

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    api_key_configured: Boolean(NIM_API_KEY)
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));

  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({
        error: {
          message: '"messages" is required and must be an array',
          type: 'invalid_request_error',
          code: 400
        }
      });
    }

    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[model];
    console.log(`Incoming model: "${model}" | Mapped to: "${nimModel}"`);

    if (!nimModel) {
      try {
        const testResponse = await axios.post(`${NIM_API_BASE}/chat/completions`, {
          model: model,
          messages: [{ role: 'user', content: 'test' }],
          max_tokens: 1
        }, {
          headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
          validateStatus: (status) => status < 500
        });

        if (testResponse.status >= 200 && testResponse.status < 300) {
          nimModel = model;
        }
      } catch (e) {
        console.error('Model probe failed:', extractErrorMessage(e));
      }

      if (!nimModel) {
        const modelLower = model.toLowerCase();
        if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
          nimModel = 'meta/llama-3.1-405b-instruct';
        } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
          nimModel = 'meta/llama-3.1-70b-instruct';
        } else {
          nimModel = 'meta/llama-3.1-8b-instruct';
        }
      }
    }

    // Transform OpenAI request to NIM format
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 9024,
      stream: stream || false
    };

    if (ENABLE_THINKING_MODE) {
      nimRequest.chat_template_kwargs = { thinking: true };
    }

    // Make request to NVIDIA NIM API
const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
  headers: {
    'Authorization': `Bearer ${NIM_API_KEY}`,
    'Content-Type': 'application/json'
  },
  responseType: stream ? 'stream' : 'json',
  validateStatus: (status) => status < 400, // let 4xx/5xx fall into catch with real body
});

console.log(`Request to NIM (${nimModel}) completed with status ${response.status}`);

    if (stream) {
      // Handle streaming response with reasoning
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (!line.startsWith('data: ')) return;

          if (line.includes('[DONE]')) {
            res.write(line + '\n\n');
            return;
          }

          try {
            const data = JSON.parse(line.slice(6));

            // Surface upstream errors sent mid-stream instead of passing them through silently
            if (data.error) {
              console.error('Upstream stream error:', data.error);
              res.write(`data: ${JSON.stringify({ error: data.error })}\n\n`);
              return;
            }

            if (data.choices?.[0]?.delta) {
              const reasoning = data.choices[0].delta.reasoning_content;
              const content = data.choices[0].delta.content;

              if (SHOW_REASONING) {
                let combinedContent = '';

                if (reasoning && !reasoningStarted) {
                  combinedContent = '<think>\n' + reasoning;
                  reasoningStarted = true;
                } else if (reasoning) {
                  combinedContent = reasoning;
                }

                if (content && reasoningStarted) {
                  combinedContent += '</think>\n\n' + content;
                  reasoningStarted = false;
                } else if (content) {
                  combinedContent += content;
                }

                if (combinedContent) {
                  data.choices[0].delta.content = combinedContent;
                  delete data.choices[0].delta.reasoning_content;
                  res.write(`data: ${JSON.stringify(data)}\n\n`);
                }
                // if no combinedContent (pure reasoning chunk with SHOW_REASONING off path not hit here), skip
              } else {
                delete data.choices[0].delta.reasoning_content;
                if (content) {
                  data.choices[0].delta.content = content;
                  res.write(`data: ${JSON.stringify(data)}\n\n`);
                }
                // reasoning-only chunk with SHOW_REASONING false -> skip, don't send empty content
              }
            } else {
              // No delta field (e.g. role-only or finish_reason chunk) - pass through as-is
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            }
          } catch (e) {
            // Not valid JSON - pass through raw rather than dropping it silently
            res.write(line + '\n');
          }
        });
      });

      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err.message);
        // Try to notify the client before closing, if headers are already sent
        try {
          res.write(`data: ${JSON.stringify({ error: { message: err.message || 'Stream error' } })}\n\n`);
        } catch (writeErr) {
          // ignore if we can't write anymore
        }
        res.end();
      });
    } else {
      // Transform NIM response to OpenAI format with reasoning
      const choices = response.data.choices || [];

      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: choices.map(choice => {
          let fullContent = choice.message?.content || '';

          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }

          return {
            index: choice.index,
            message: {
              role: choice.message.role,
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

      res.json(openaiResponse);
    }

  } catch (error) {
    const status = error.response?.status || 500;
    const message = extractErrorMessage(error);

    console.error('Proxy error:', {
      status,
      message,
      data: error.response?.data
    });

    // If headers were already sent (mid-stream failure), just end the connection
    if (res.headersSent) {
      try { res.end(); } catch (e) {}
      return;
    }

    res.status(status).json({
      error: {
        message: message,
        type: 'invalid_request_error',
        code: status
      }
    });
  }
});

// Catch-all for unsupported endpointsapp.all('*', (req, res) => {
  console.log(`404: ${req.method} ${req.originalUrl}`);
  res.status(404).json({
    error: {
      message: `Endpoint ${req.method} ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });


app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
  console.log(`NIM_API_KEY configured: ${NIM_API_KEY ? 'YES' : 'NO — set this env var!'}`);
});
