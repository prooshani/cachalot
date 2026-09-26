#!/bin/bash
# MiniMax-M3 decode-path quality gate for the GQA decode kernel (HANDOFF 18.3): three processes on the same text,
# teacher-forced decode log-probs after an N-token prefill, then KL / top-1 of each arm against the old path.
#   stock  MLX's SDPA (CACHALOT_MINIMAX_GQA_DECODE_MIN=0)
#   gqa    the kernel from 4,096 cached tokens
#   noise  MLX's SDPA with prefill chunk 4,096: the model's own rounding noise, the bar the kernel must stay under
# Speed is not read here (separate processes drift; use TF_ALTERNATE, see glm_prefill_timeline.py).
#   N=32768 TF=400 OFF=2000000 OUT=/some/dir benchmarks/minimax_decode_gate.sh
cd "$(dirname "$0")/.."
N=${N:-32768}; OFF=${OFF:-1000000}; TF=${TF:-400}
OUT=${OUT:-/tmp}
FILLER=${FILLER_FILE:?set FILLER_FILE (e.g. git ls-files '*.md' '*.py' | sort | xargs cat > filler.txt)}
for arm in ${ARMS:-stock gqa noise}; do
  extra=()
  case $arm in
    stock) extra=(CACHALOT_MINIMAX_GQA_DECODE_MIN=0);;
    gqa) extra=(CACHALOT_MINIMAX_GQA_DECODE_MIN=4096);;
    noise) extra=(CACHALOT_MINIMAX_GQA_DECODE_MIN=0 CACHALOT_MINIMAX_PREFILL_CHUNK=4096);;
  esac
  echo "=== $arm N=$N $(date +%T)"
  env CACHALOT_MODEL_PATH=${CACHALOT_MINIMAX_PATH:-/Users/hamedprooshani/MiniMax-M3-MLX-3bit} CACHALOT_PAGE_CACHE=1 \
      CACHALOT_MLX_WIRED_LIMIT_GIB=80 MLX_METAL_FAST_SYNCH=1 PYTHONPATH=src \
      FILLER_FILE=$FILLER FILLER_OFFSET=$OFF TF_DECODE=$TF TF_OUT=$OUT/tf.$arm.$N.$OFF.npy "${extra[@]}" \
      ~/venvs/deepseek-v41/bin/python benchmarks/glm_prefill_timeline.py $N 2>&1 | grep -E "RESULT|TF_DECODE|Error|Traceback"
done
for pair in "stock gqa" "stock noise" "gqa noise"; do
  set -- $pair
  echo "$1 vs $2: $(~/venvs/deepseek-v41/bin/python benchmarks/glm_prefill_timeline.py --compare $OUT/tf.$1.$N.$OFF.npy $OUT/tf.$2.$N.$OFF.npy)"
done
