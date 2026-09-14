#!/bin/bash
# End-to-end smoke test of the OpenAI-compatible server with the real model.
# Starts the server, waits for /health, runs a streaming and a non-streaming
# chat request plus a multi-turn follow-up (prefix cache), prints usage.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT=${PORT:-8011}
LOG=benchmarks/results/server_smoke.log
mkdir -p benchmarks/results
PYTHONPATH=src ~/venvs/deepseek-v41/bin/python -m cachalot.cli serve --port "$PORT" --max-seq-len 8192 > "$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  if ! kill -0 $PID 2>/dev/null; then echo "server died"; tail -20 "$LOG"; exit 1; fi
  sleep 2
done
echo "--- /v1/models"; curl -s "http://127.0.0.1:$PORT/v1/models" | head -c 400; echo
echo "--- streaming"
curl -sN "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Give me one sentence about cachalots."}],
  "max_tokens": 24, "temperature": 0, "stream": true, "stream_options": {"include_usage": true}}' \
  | ~/venvs/deepseek-v41/bin/python -c '
import sys, json
text = ""
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data: ") or line == "data: [DONE]":
        continue
    chunk = json.loads(line[6:])
    for c in chunk.get("choices", []):
        text += c["delta"].get("content", "")
    if chunk.get("usage"):
        print("usage:", json.dumps(chunk["usage"]))
print("text:", repr(text))'
echo "--- non-streaming follow-up (prefix cache)"
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Give me one sentence about cachalots."},
               {"role": "assistant", "content": "Cachalots are deep-diving whales."},
               {"role": "user", "content": "And one about their teeth."}],
  "max_tokens": 16, "temperature": 0}' | ~/venvs/deepseek-v41/bin/python -c 'import sys,json; d=json.load(sys.stdin); print(json.dumps(d["choices"][0]["message"])); print("usage:", json.dumps(d["usage"]))'
echo "--- /v1/stats"; curl -s "http://127.0.0.1:$PORT/v1/stats" | head -c 600; echo
