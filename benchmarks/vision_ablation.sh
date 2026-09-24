#!/bin/bash
# One arm of the vision ablation (HANDOFF section 15.7): start serve.sh with CACHALOT_VISION_ABLATE=$1
# ("none" for the shipped model), score benchmarks/vision_ablation.py's cases, stop the server.
#   for a in none delims engram_mask bias_vl; do benchmarks/vision_ablation.sh $a; done
set -uo pipefail
cd "$(dirname "$0")/.."
ARM=$1
ABL=$ARM; [ "$ARM" = none ] && ABL=
mkdir -p benchmarks/results/vision_ablation
CACHALOT_SNAPSHOT_DIR= CACHALOT_VISION_ABLATE=$ABL ./serve.sh > benchmarks/results/vision_ablation/serve_$ARM.log 2>&1 &
pid=$!
until curl -s -m 2 http://127.0.0.1:8011/v1/models >/dev/null 2>&1; do
    sleep 2; kill -0 $pid 2>/dev/null || { echo "server died"; tail -5 benchmarks/results/vision_ablation/serve_$ARM.log; exit 1; }
done
~/venvs/deepseek-v41/bin/python benchmarks/vision_ablation.py --arm "$ARM"
kill $pid; wait $pid 2>/dev/null
