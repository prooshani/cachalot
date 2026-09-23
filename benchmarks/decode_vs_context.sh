#!/bin/bash
# One arm of decode_vs_context.py under serve.sh's environment. Usage: decode_vs_context.sh FILLER_TOKENS [OUT]
set -euo pipefail
cd "$(dirname "$0")/.."
if pgrep -fl "deepseek-v41/bin/python" >/dev/null 2>&1; then
    echo "a runtime is already running; not starting a second one" >&2; exit 1
fi
export CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash
export CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128
export CACHALOT_PAGE_CACHE=1
export CACHALOT_MLX_WIRED_LIMIT_GIB=${CACHALOT_MLX_WIRED_LIMIT_GIB:-80}
export CACHALOT_HOTLIST=/Users/hamedprooshani/cachalot-hotlist.json
export CACHALOT_HOTLIST_GIB=8
export PYTHONPATH=src
exec ~/venvs/deepseek-v41/bin/python benchmarks/decode_vs_context.py --context "$1" --out "${2:-benchmarks/results/decode_vs_context.jsonl}"
