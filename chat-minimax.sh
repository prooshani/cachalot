#!/bin/bash
# MiniMax-M3 in the terminal: the MiniMax counterpart of chat.sh (DeepSeek) and chat-glm.sh (GLM).
#   ./chat-minimax.sh                 interactive
#   ./chat-minimax.sh --thinking      with M3's reasoning shown
#   ./chat-minimax.sh "a question"    one turn
# Only one runtime runs at a time. HANDOFF section 18.
set -euo pipefail
cd "$(dirname "$0")"

if pgrep -fl "deepseek-v41/bin/python|cachalot\.cli" >/dev/null 2>&1; then
    echo "a runtime is already running; not starting a second one:" >&2
    pgrep -fl "deepseek-v41/bin/python|cachalot\.cli" >&2
    exit 1
fi

export CACHALOT_MODEL_PATH=${CACHALOT_MINIMAX_PATH:-/Users/hamedprooshani/MiniMax-M3-MLX-3bit}
export CACHALOT_MODEL_FAMILY=minimax
export CACHALOT_PAGE_CACHE=1
export CACHALOT_MLX_WIRED_LIMIT_GIB=${CACHALOT_MLX_WIRED_LIMIT_GIB:-80}
export PYTHONPATH=src
export MLX_METAL_FAST_SYNCH=${MLX_METAL_FAST_SYNCH:-1}

exec ~/venvs/deepseek-v41/bin/python -m cachalot.cli chat \
    --model "$CACHALOT_MODEL_PATH" \
    --expert-budget-gib 52 \
    --max-seq-len 131072 \
    --max-new-tokens 2000 \
    --temperature 1.0 \
    "$@"
