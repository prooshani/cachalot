#!/bin/bash
# The shipped interactive configuration, as an OpenAI-compatible HTTP server.
#
# Same env vars as chat.sh (see that file's header for why the pgrep guard
# and the exported-not-piped form matter), pointed at `cachalot serve`
# instead of `cachalot chat`. Point an agent harness (Hermes, OpenCode,
# Continue, aider, ...) at http://127.0.0.1:8011/v1 with model id
# `deepseek-v4.1-flash` and any placeholder API key.
#
# Any argument given here is passed through to `cachalot.cli serve`, so
#   ./serve.sh --port 8080 --api-key secret
# overrides the defaults below.
#
# --max-seq-len is 65536, not chat.sh's 32768: Hermes Agent refuses any
# endpoint whose /v1/models max_context_length is below 64,000 and never
# sends a request (HANDOFF section 15.1). The extra compressed-KV cache
# costs about 84 MB.
set -euo pipefail
cd "$(dirname "$0")"

if pgrep -fl "deepseek-v41/bin/python|cachalot" >/dev/null 2>&1; then
    echo "a runtime is already running; not starting a second one:" >&2
    pgrep -fl "deepseek-v41/bin/python|cachalot" >&2
    exit 1
fi

export CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash
export CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128
export CACHALOT_PAGE_CACHE=1
export CACHALOT_MLX_WIRED_LIMIT_GIB=${CACHALOT_MLX_WIRED_LIMIT_GIB:-80}
export CACHALOT_HOTLIST=/Users/hamedprooshani/cachalot-hotlist.json
export CACHALOT_HOTLIST_GIB=8
export PYTHONPATH=src
# The snapshot where an agent's system prompt ends survives a restart, so the
# first request after one reuses it instead of re-prefilling ~13.5k tokens
# (HANDOFF section 15.4). Empty disables it.
export CACHALOT_SNAPSHOT_DIR=${CACHALOT_SNAPSHOT_DIR-$HOME/.cache/cachalot/prefix-snapshots}

exec ~/venvs/deepseek-v41/bin/python -m cachalot.cli serve \
    --expert-budget-gib 52 \
    --max-seq-len 65536 \
    --port 8011 \
    --default-max-tokens 2000 \
    --default-temperature 0.6 \
    "$@"
