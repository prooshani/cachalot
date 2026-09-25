#!/bin/bash
# GLM-5.3-Flash in the terminal: the GLM counterpart of chat.sh (DeepSeek V4.1 Flash).
#   ./chat-glm.sh                 interactive
#   ./chat-glm.sh --thinking      with GLM's reasoning shown
#   ./chat-glm.sh "a question"    one turn
# Only one runtime runs at a time. HANDOFF section 17.
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

exec ~/venvs/deepseek-v41/bin/python -m cachalot.cli chat \
    --model "$CACHALOT_MODEL_PATH" \
    --expert-budget-gib 52 \
    --max-seq-len 131072 \
    --max-new-tokens 2000 \
    --temperature 0.6 \
    "$@"
