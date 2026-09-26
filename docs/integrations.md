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
- **Only the server's very first Hermes request is slow.** Hermes's system prompt plus its 24 tool schemas is
  about 13,500 tokens, roughly 3 minutes of cold prefill on this machine. The server snapshots where that
  system block ends, so every later session reuses it (about 1 s of prefill), and `serve.sh` keeps the
  snapshot in `~/.cache/cachalot/prefix-snapshots` so a restarted server reuses it too (about 3 s).
  Changing the Hermes toolset or system prompt, the expert bank or the Cachalot version pays the cold
  prefill once more. Hermes raises its stream read timeout to 1800 s for a local endpoint, and the server
  sends SSE keep-alive comments while it prefills, so the wait does not time out.
- **One request at a time.** Hermes's side calls (session titles, compression) queue behind the main
  request. A request whose client has disconnected is dropped instead of being prefilled.
- **Images work**, both `hermes chat --image` and OpenAI `image_url` content parts (URL or base64 data URI).
- `reasoning_effort` accepts OpenAI-style values: `none`/`minimal` turn thinking off; `low`, `medium`,
  `high`, `xhigh`/`max` map onto DeepSeek's effort levels. Hermes (September 2026 builds) sends `medium`, so
  its sessions run in thinking mode at effort 50.
- **Each working directory is its own system prompt.** Hermes writes `Current working directory: …` into its
  system prompt, about 3,900 tokens in, ahead of the ~9,700 tokens of tool schemas. The first request from a
  new project (or after a Hermes update, which also changes the prompt's wording) pays the cold prefill once;
  the server keeps the 32 most recently used system blocks on disk (loading the ones it did not preload when
  a request needs them), so a batch of `delegate_task` subagents does not push the main agent's block out.
- **Tool calls come back re-serialized.** Hermes returns the model's tool-call arguments with the keys in
  another order, and an empty thinking block as a single space. The server recognizes its own reply and
  reuses it (`spliced=N` on the `[request]` line), so each turn prefills only what is new.
- **Compression** starts at ≥ 75 % of the window for any model under 512k tokens (85 % when Hermes's 64k
  floor binds), so ~49-56k tokens here, regardless of `compression.threshold`. `compression.threshold_tokens`
  lowers it. The summary request shares no prefix with the conversation, so it is a full prefill, and a
  summary of a long session takes far longer than Hermes allows: a 6,258-token summary prompt took 850 s here
  (a 3,386-token summary), against `auxiliary.compression.timeout` 120 s. **Point `auxiliary.compression` at a
  hosted model** (or give it a hosted `fallback_chain` entry) and keep Cachalot for the main agent.
- **Subagents** (`delegate_task`) each put their task's context into their system prompt, in front of the tool
  schemas, so every subagent's first turn is a ~15k-token prefill (~3 min). `docs/hermes-subagent-context.md`
  has the numbers and the Hermes change that would share one block among them.

**MiniMax-M3 instead.** Stop the server and start `./serve-minimax.sh`: same URL, model id `minimax-m3`,
131,072-token context. Tool calls and thinking work (send a `reasoning_effort` to turn thinking on; in one test the
model declined a tool call with thinking off and made it with thinking on). Its attention is full, so the cache
grows ~120 KB per token (~2.4 GB at Hermes's 20k); the server keeps 8 system-block snapshots on disk in
`~/.cache/cachalot/prefix-snapshots-minimax`. Since 0.20.0 a cold prefill runs ~170-240 tok/s (a 17k-token system
block in ~100 s, once) and decode 3.1-3.6 tok/s (HANDOFF sections 18, 18.1). Since 0.21.0: 64k tokens measured and fitting (wired 78.5
of 80 GiB, prefill 148 tok/s, decode 2.65 tok/s; section 18.2); past that is untested. The server also keeps its
expert cache across a restart (`resident-set.json` beside the snapshots, ~9 s to read back at startup). Since 0.22.0
the script also reads ~10 % of every expert from the X10Pro's copy when it is mounted (decode -5 %, cold prefill
-9 %), and decode attention past 4k tokens runs through its own kernel (a token -5 % at 32k, -8 % at 64k); short
chat turns decoded at 4.1-4.7 tok/s through the server (HANDOFF section 18.3). With the X10Pro unplugged it runs
from the internal SSD alone, as before. Since 0.23.0 the experts come from a bias-free bank
(`~/MiniMax-M3-coded-bank`, 6 % fewer bytes per read, byte-identical output; read wait -9 %, short tool turns
4.7-4.9 tok/s; HANDOFF section 18.4), and the internal checkpoint keeps only the non-expert weights. The first
request after the upgrade prefills its system block once more (the snapshot identity includes the shard sizes).

A step-by-step manual test for Hermes Agent Desktop is in `docs/manual-tests/hermes-desktop.md`.

**GLM-5.3-Flash instead of DeepSeek.** (Since 0.19.0 GLM reads from the X10Pro over USB, which is much slower.)
Stop the server and start `./serve-glm.sh`: same URL
(`http://127.0.0.1:8011/v1`), model id `glm-5.3-flash`, 131,072-token context. Tool calls and thinking work;
images and the reply splice do not yet. A cold prefill runs ~90 tok/s (Hermes's ~20k-token system prompt: ~4
minutes, once); the system block is kept on disk in `~/.cache/cachalot/prefix-snapshots-glm`, so a restarted
server reuses it (HANDOFF sections 17, 17.1). Decode is 3.3-3.6 tok/s, slower than DeepSeek's.

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
