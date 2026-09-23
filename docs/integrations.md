# Using Cachalot from agent harnesses and clients

Start the server:

```bash
cachalot serve --model /Volumes/FastSSD/DeepSeek-V4.1-Flash --port 8000
```

The model id reported by `GET /v1/models` is `deepseek-v4.1-flash` (change with `--model-id`).
Timeouts matter: prefill of a long, cold prompt can take minutes on USB storage, so raise
client timeouts generously. Cachalot streams as soon as the first token exists.

## OpenCode

`~/.config/opencode/opencode.jsonc`:

```jsonc
{
  "provider": {
    "cachalot": {
      "name": "Cachalot (DeepSeek V4.1 Flash, local)",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "headerTimeout": 1800000,
        "chunkTimeout": 1800000,
        "timeout": false
      },
      "models": {
        "deepseek-v4.1-flash": {
          "name": "DeepSeek V4.1 Flash",
          "reasoning": true,
          "tool_call": true,
          "temperature": true,
          "limit": { "context": 32768, "output": 8192 },
          "options": { "temperature": 0.6 }
        }
      }
    }
  },
  "model": "cachalot/deepseek-v4.1-flash"
}
```

## Hermes Agent

Start the server with `./serve.sh` (port 8011, 65,536-token context). Then, in `~/.hermes/config.yaml`
(or a profile's `config.yaml`):

```yaml
model:
  default: deepseek-v4.1-flash
  provider: custom:cachalot
  base_url: http://127.0.0.1:8011/v1
custom_providers:
  - name: cachalot
    base_url: http://127.0.0.1:8011/v1
    model: deepseek-v4.1-flash
    api_mode: chat_completions
    api_key: cachalot          # any non-empty string unless the server was started with --api-key
```

To try it without touching your own Hermes setup, put that file in an empty directory and run
`HERMES_HOME=/that/dir hermes chat`. This is how it was tested (HANDOFF section 15.1).

What to expect, measured 2026-09-23:

- **Hermes refuses any endpoint whose context window is below 64,000 tokens** and never sends a request.
  `serve.sh` serves 65,536 for that reason; `chat.sh` stays at 32,768.
- **The first request of a session is slow.** Hermes's system prompt plus its 24 tool schemas is about
  13,500 tokens, which is roughly 4 minutes of prefill on this machine. Later turns reuse that prefix from
  the prefix cache. Hermes raises its stream read timeout to 1800 s for a local endpoint, and the server sends
  SSE keep-alive comments while it prefills, so the wait does not time out.
- **One request at a time.** Hermes's side calls (session titles, compression) queue behind the main
  request. A request whose client has disconnected is dropped instead of being prefilled.
- **Images work**, both `hermes chat --image` and OpenAI `image_url` content parts (URL or base64 data URI).
- `reasoning_effort` accepts OpenAI-style values: `none`/`minimal` turn thinking off; `low`, `medium`,
  `high`, `xhigh`/`max` map onto DeepSeek's effort levels.

## aider

```bash
export OPENAI_API_BASE=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=cachalot
aider --model openai/deepseek-v4.1-flash --timeout 1800
```

## Continue (VS Code / JetBrains)

`~/.continue/config.yaml`:

```yaml
models:
  - name: DeepSeek V4.1 Flash (Cachalot)
    provider: openai
    model: deepseek-v4.1-flash
    apiBase: http://127.0.0.1:8000/v1
    apiKey: cachalot
    roles: [chat, edit, apply]
```

## OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="cachalot", timeout=1800)

stream = client.chat.completions.create(
    model="deepseek-v4.1-flash",
    messages=[{"role": "user", "content": "Summarize the CSA2 attention design."}],
    stream=True,
    stream_options={"include_usage": True},
    extra_body={"thinking": True, "reasoning_effort": 50},
)
for chunk in stream:
    delta = chunk.choices[0].delta if chunk.choices else None
    if delta and getattr(delta, "reasoning_content", None):
        print(delta.reasoning_content, end="", flush=True)   # thinking stream
    if delta and delta.content:
        print(delta.content, end="", flush=True)
    if chunk.usage:
        print("\n", chunk.usage.model_dump())
```

## Request extensions

| Field | Values | Effect |
|---|---|---|
| `thinking` | `true` / `false` | DeepSeek thinking mode; reasoning arrives as `reasoning_content` |
| `reasoning_effort` | `1`–`100`, `"low"`, `"high"`, `"max"` | Official V4.1 reasoning budget; implies `thinking: true` |
| `top_k` | int | Additional tail truncation before sampling |
| `seed` | int | Deterministic sampling |
| `stream_options.include_usage` | `true` | Final SSE chunk carries `usage`, including `usage.cachalot.reused_prefix_tokens` and timings |

Tool calls: pass OpenAI `tools`; the server renders them through the checkpoint's own protocol
and returns `tool_calls` with `finish_reason: "tool_calls"`. Tool results go back as
`{"role": "tool", ...}` messages as usual.

## Operational notes

- **One request at a time.** The runtime is stateful and the machine is I/O-bound; concurrent
  requests queue in arrival order. `GET /v1/stats` shows `busy`, hit rates and the prefix cache.
- **Keep the conversation growing.** The prefix cache keys on the exact token sequence. Harnesses
  that append messages (the normal case) pay only for new tokens each turn. Editing earlier
  messages or changing the system prompt invalidates the cached prefix.
- **Context length** defaults to 32,768 tokens (`--max-seq-len`). Cold prefill of a long prompt
  is bounded by SSD bandwidth: expect roughly 0.3 GB of expert reads per prompt token until the
  cache warms.
