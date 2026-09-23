# Manual test: Hermes Agent Desktop against Cachalot

What this checks: the Desktop app's own UI flows (which no automated run has exercised), a long session
growing toward Hermes's compression threshold, and the three server-side speedups that make agent turns
cheap: the system-prompt snapshot (HANDOFF §15.4), the reply splice (§15.5) and the image-row cache (§15.5).

Budget about an hour. The first request of a new Hermes version costs ~3 minutes (cold prefill); after that a
turn should cost seconds of prefill plus decode time.

## 0. Before you start

- Nothing else may run a model: no `chat.sh`, no benchmark, no second `serve.sh`.
- Close heavy apps if you can. Swap growth is one of the things being watched, and it starts at whatever
  is already in use. Note the starting value:

```bash
sysctl vm.swapusage
```

## 1. Start the server with the request dump on

In a terminal of its own:

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && CACHALOT_SERVER_DUMP=/tmp/cachalot-requests.jsonl ./serve.sh 2>&1 | tee /tmp/cachalot-serve.log
```

Wait for `Uvicorn running on http://127.0.0.1:8011` (loading takes 1-2 minutes). Before sending anything
from Desktop, confirm the server answers; this must print a JSON line with `deepseek-v4.1-flash`:

```bash
curl -s http://127.0.0.1:8011/v1/models
```

If the log says `a runtime is already running; not starting a second one`, something else is holding the
model (a `chat.sh`, a benchmark, an old server); stop it first. Up to 0.12.0 the guard also matched *any*
command line containing "cachalot", including this very `tee /tmp/cachalot-serve.log`; 0.12.1 matches only
`cachalot.cli`. The line `prefix snapshots: N loaded` tells you whether a
system-prompt snapshot from an earlier run matches this version (N ≥ 1 means the first request is fast).

In a second terminal, follow the per-request lines while you test:

```bash
tail -f /tmp/cachalot-serve.log | grep --line-buffered "^\[request\]"
```

Each line reads `prompt=… reused=… prefilled=… prefill=…s completion=… decode=…s (… tok/s) images=… spliced=… miss/tok=… hit=… finish=…`.
`reused` is how much of the prompt came from the prefix cache, `spliced` how many of the model's own earlier
replies the server recognized in Hermes's re-rendered history, `miss/tok` how many experts per generated token
had to be read from disk.

A new Hermes version, a new working directory, or a new Cachalot version (0.12.0 invalidates earlier disk
snapshots) each cost one cold prefill of the ~13.7k-token system block, ~3 minutes. After that, every new
chat in the same project starts in about a second.

## 2. Point Hermes Desktop at the server

Add Cachalot as a custom provider. In `~/.hermes/config.yaml` your existing `custom_providers:` list has a
`vllm-metal` entry; add a second entry under it (keep the indentation):

```yaml
  - name: cachalot
    base_url: http://127.0.0.1:8011/v1
    model: deepseek-v4.1-flash
    api_mode: chat_completions
    api_key: cachalot
```

**Profiles have their own config.** Hermes Desktop runs whichever profile is active (the log folder tells
you: `~/.hermes/profiles/<name>/logs/`). The entry above goes into *that* profile's config:
`~/.hermes/config.yaml` for `default`, `~/.hermes/profiles/<name>/config.yaml` for any other. For this test,
switch Desktop to the `default` profile, which already has the entry, or add the same entry to the active
profile's `custom_providers`.

Then, in Hermes Desktop, pick the model `deepseek-v4.1-flash` from provider `cachalot` in the model
selector (or set it as the default model in Settings). Nothing else in your config needs to change, but read
section 5 about compression before starting the long session.

## 3. Short checks (10 minutes)

Start a **new chat** in Desktop and run these one at a time. Expected results are in italics.

1. `What files are in my home directory's Desktop folder? Use a tool.`
   *A tool call (terminal or search), then a list. First request of this Hermes version: `reused=0`,
   ~3 minutes of prefill. Every later new chat: `reused` ≈ 13,400-13,600 and prefill ~1-3 s.*
2. `Create a file /tmp/cachalot-test/hello.py that prints the sum of 1..100, run it, tell me the output.`
   *`write_file` then `terminal`; answer 5050. On the turn after the `write_file` call, the request line
   should show `spliced=1` (or more) and `prefilled` close to the size of the tool result only, not the size
   of the reply plus the tool result.*
3. `Now change it to print the sum of 1..1000 and run it again.`
   *`patch` or `write_file`, then 500500.*
4. Start another **new chat**: `Say hi in five words.`
   *`reused` ≈ the system block again, prefill ~1 s.*

## 4. Images (5 minutes)

5. In a new chat, attach any screenshot with readable text (drag it into the input) and ask
   `Read me every piece of text in this image.`
   *`images=1` on the request line (or Hermes's own `vision_analyze` pre-pass, which is a separate short
   request with `images=1`). The text should match the image.*
6. In the same chat, a follow-up that does not attach anything: `What colour is the largest element in that image?`
   *If Desktop re-sends the image in history, the request line again shows `images=1`, but prefill should be
   short (`reused` close to `prompt`); the ViT is not re-run for the same picture.*

## 5. The long session (30-45 minutes): what to watch

Keep one chat going for 20+ turns of real work: reading files, editing, running commands. Ask for things
that pull large tool results into the context (read a long file, list a big directory), so the context
grows past 30k tokens.

**Compression.** Hermes compresses at ≥ 75 % of any window under 512k tokens, so here at ~49-56k tokens
whatever `compression.threshold` says. Getting there takes a long session; to see it sooner, add
`threshold_tokens: 30000` under `compression:` in your config for this test (remove it afterwards). The summary
is written by the auxiliary model; with `provider: auto` that is Cachalot itself. Measured in the CLI run
(HANDOFF §15.5), the summary request prefilled 2,145 tokens and then **decoded a 1,595-token summary: ~4
minutes**, just inside Hermes's 300 s budget (a second run took 343 s, past it). Compression also adds a
`skill_manage` tool, which changes the system block, so the next main request is one more cold prefill
(~4 minutes); after that `reused` climbs again. If the summary times out, Hermes falls back to its deterministic
compression (`abort_on_summary_failure: false`), so the session continues either way. Note which one
happened, and how long the turn took. If 4-minute compressions are a problem, point Hermes's auxiliary
`compression` provider at a hosted model; that is a config choice, not a server fix.

Record, per turn if you can:

| what | where | healthy | report if |
|---|---|---|---|
| `reused` | `[request]` line | ≈ previous turn's `prompt` or more | drops to 0 or to a 4,096 multiple mid-session |
| `spliced` | `[request]` line | ≥ 1 on turns after a tool call whose arguments Hermes reordered | always 0 while `prefilled` ≫ tool result size |
| `prefill` | `[request]` line | seconds (tool result size × ~25 ms) | minutes on a turn that only added a short message |
| decode tok/s | `[request]` line | ~6-9 tok/s at 13-30k context | below 5 tok/s: note the time and the `miss/tok`; if `miss/tok` is normal (15-35) this is the unexplained slow window of §15.5, and the time stamps are exactly what Job 1 needs |
| swap | `sysctl vm.swapusage` every ~10 turns | flat | growing by >1 GB |
| compression | Desktop UI / `[request]` line with a new, short `prompt` and `reused=0` | happens once, then `reused` climbs again | happens every turn |

## Troubleshooting

| what Desktop shows | cause | fix |
|---|---|---|
| "The reply timed out … Connection error." after ~300 s, and no `[request]` line on the server | the server is not running or not reachable | check the `curl` above; restart `serve.sh` and wait for `Uvicorn running` |
| a long wait, then an answer, first request only | the cold prefill of your system prompt (~3-6 min) | expected once per Hermes version, profile and working directory |
| "timed out" while the server log shows a request still prefilling | Hermes gave up first | should not happen: Hermes allows 1800 s for a local endpoint; send the log lines |

## 6. After the session

Stop the server (Ctrl-C in its terminal). Send me:

```bash
grep "^\[request\]" /tmp/cachalot-serve.log > /tmp/cachalot-requests-summary.txt; wc -l /tmp/cachalot-requests-summary.txt /tmp/cachalot-requests.jsonl
```

plus a note of anything the Desktop UI did wrong (tool call shown as raw text, an error banner, a stalled
spinner). The request dump (`/tmp/cachalot-requests.jsonl`) holds every request body and every reply's
token ids, so any miss can be traced to the exact token after the fact.

To go back to your usual model, pick it again in the Desktop model selector. The `cachalot` provider entry
can stay in the config.
