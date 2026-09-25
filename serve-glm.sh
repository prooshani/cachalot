#!/bin/bash
# GLM-5.3-Flash (MLX 4-bit, Vontra/GLM-5.3-Flash-MLX-4bit-MTP) as an OpenAI-compatible HTTP server.
#
# Same shape and port as serve.sh, so an agent harness switches models by restarting the server:
# ./serve.sh serves DeepSeek V4.1 Flash, ./serve-glm.sh serves GLM-5.3-Flash, both on
# http://127.0.0.1:8011/v1. Model id here: glm-5.3-flash. Only one runtime runs at a time (the
# guard below), because both use the same expert-cache memory.
#
# The routed experts (12,096 x 13.5 MiB) stream from the internal SSD into a 52 GiB wired cache;
# the rest of the model (5.5 GiB) stays resident. Text only (no vision, no MTP yet), and the
# prefix cache lives in memory only (no disk snapshots across restarts yet). HANDOFF section 17.
#
# Any argument is passed through to `cachalot.cli serve`, e.g. ./serve-glm.sh --expert-budget-gib 44
set -euo pipefail
cd "$(dirname "$0")"

if pgrep -fl "deepseek-v41/bin/python|cachalot\.cli" >/dev/null 2>&1; then
    echo "a runtime is already running; not starting a second one:" >&2
    pgrep -fl "deepseek-v41/bin/python|cachalot\.cli" >&2
    exit 1
fi

export CACHALOT_MODEL_PATH=/Users/hamedprooshani/GLM-5.3-Flash-MLX-4bit-MTP
export CACHALOT_MODEL_FAMILY=glm
export CACHALOT_PAGE_CACHE=1
export CACHALOT_MLX_WIRED_LIMIT_GIB=${CACHALOT_MLX_WIRED_LIMIT_GIB:-80}
export PYTHONPATH=src
export MLX_METAL_FAST_SYNCH=${MLX_METAL_FAST_SYNCH:-1}

exec ~/venvs/deepseek-v41/bin/python -m cachalot.cli serve \
    --model "$CACHALOT_MODEL_PATH" \
    --expert-budget-gib 52 \
    --max-seq-len 131072 \
    --port 8011 \
    --model-id glm-5.3-flash \
    --default-max-tokens 8192 \
    --default-temperature 0.6 \
    "$@"
