#!/bin/bash
# MiniMax-M3 (MLX 3-bit, pipenetwork/MiniMax-M3-MLX-3bit) as an OpenAI-compatible HTTP server.
#
# Same port and API as serve.sh (DeepSeek V4.1 Flash) and serve-glm.sh (GLM-5.3-Flash): an agent harness
# switches models by restarting the server. Model id here: minimax-m3. Only one runtime runs at a time.
#
# The routed experts (7,296 x 23.6 MiB, 3-bit; 22.2 MiB each read from the bias-free bank) stream from the internal SSD into a 52 GiB wired cache; the
# rest of the model (6.0 GiB) stays resident. Text only (the conversion has no vision tower or MTP).
# MiniMax Sparse Attention runs as full causal attention (exact to 2,048 tokens). HANDOFF section 18.
#
# Any argument is passed through to `cachalot.cli serve`, e.g. ./serve-minimax.sh --expert-budget-gib 44
set -euo pipefail
cd "$(dirname "$0")"

if pgrep -fl "deepseek-v41/bin/python|cachalot\.cli" >/dev/null 2>&1; then
    echo "a runtime is already running; not starting a second one:" >&2
    pgrep -fl "deepseek-v41/bin/python|cachalot\.cli" >&2
    exit 1
fi

# internal copy (non-expert weights only since 0.23.0); the full download stays on the X10Pro
# (/Volumes/X10Pro/models/MiniMax-M3-MLX-3bit)
export CACHALOT_MODEL_PATH=${CACHALOT_MINIMAX_PATH:-/Users/hamedprooshani/MiniMax-M3-MLX-3bit}
export CACHALOT_MODEL_FAMILY=minimax
export CACHALOT_PAGE_CACHE=1
export CACHALOT_MLX_WIRED_LIMIT_GIB=${CACHALOT_MLX_WIRED_LIMIT_GIB:-80}
export PYTHONPATH=src
export MLX_METAL_FAST_SYNCH=${MLX_METAL_FAST_SYNCH:-1}
# The routed experts come from the bias-free bank (HANDOFF 18.4): one contiguous 22.2 MiB record per expert, the
# biases rebuilt exactly from 2-bit codes; the internal checkpoint holds only the non-expert weights since 0.23.0.
# Same outputs, 6 % fewer bytes, ~9 % less read wait per token. The download on the X10Pro is the full original.
export CACHALOT_MINIMAX_BANK=${CACHALOT_MINIMAX_BANK-$HOME/MiniMax-M3-coded-bank}
# Mirror striping (HANDOFF 18.3, 18.4): the tail 10 % of each expert's weight pieces comes from the bank's copy on
# the X10Pro at the same time as the rest from the internal SSD. Off when the X10Pro is not mounted;
# CACHALOT_MINIMAX_MIRROR= (empty) turns it off.
MIRROR=${CACHALOT_MINIMAX_MIRROR-/Volumes/X10Pro/models/MiniMax-M3-coded-bank}
if [ -n "$MIRROR" ] && [ -f "$MIRROR/bank.json" ]; then
    export CACHALOT_MINIMAX_BANK_MIRROR=$MIRROR
    export CACHALOT_MIRROR_FRACTION=${CACHALOT_MIRROR_FRACTION:-0.10}
fi
# The snapshot where an agent's system prompt ends survives a restart. Empty disables it.
export CACHALOT_SNAPSHOT_DIR=${CACHALOT_MINIMAX_SNAPSHOT_DIR-$HOME/.cache/cachalot/prefix-snapshots-minimax}
# M3's attention is full (60 layers x 4 KV heads x 128), ~120 KB of cache per token: a 20k-token agent block
# is ~2.4 GB. Eight files on disk (~20 GB at most), 5 GiB of snapshots in memory.
export CACHALOT_SNAPSHOT_KEEP=${CACHALOT_SNAPSHOT_KEEP:-8}
export CACHALOT_GLM_PREFIX_GIB=${CACHALOT_GLM_PREFIX_GIB:-5}

# MiniMax's generation_config: temperature 1.0, top_p 0.95
exec ~/venvs/deepseek-v41/bin/python -m cachalot.cli serve \
    --model "$CACHALOT_MODEL_PATH" \
    --expert-budget-gib 52 \
    --max-seq-len 131072 \
    --port 8011 \
    --model-id minimax-m3 \
    --default-max-tokens 8192 \
    --default-temperature 1.0 \
    "$@"
