#!/bin/bash
# The shipped interactive configuration, as one command that cannot be split.
#
# The equivalent one-liner in docs/HANDOFF.md section 4 carries eight
# environment variables in front of the interpreter. Pasted into a terminal
# that wraps it, zsh runs each wrapped line as its own command: the assignments
# on the leading lines become shell parameters that are never exported, the
# runtime starts without CACHALOT_EXPERT_BANK and silently serves FP4 off the
# USB drive, and the arguments on the trailing line come back as
# "command not found: --max-seq-len". That happened on 2026-09-21 and the
# session read 0.6 tok/s against the configuration's 7.6.
#
# Any argument given here is passed through to `cachalot.cli chat`, so
#   ./chat.sh --temperature 0.2
# overrides the default below (the last value of a repeated flag wins).
#
# --max-new-tokens is 2000 because 1024 cut a coding turn mid-method in three
# separate sessions, and a stop=length reply is not a quality signal. At 2000
# the two Objective-C turns of 2026-09-21 finished on stop=stop at 1,483 and
# 1,788 tokens (HANDOFF section 7.2.4).
set -euo pipefail
cd "$(dirname "$0")"

if pgrep -fl "deepseek-v41/bin/python|cachalot\.cli" >/dev/null 2>&1; then
    echo "a runtime is already running; not starting a second one:" >&2
    pgrep -fl "deepseek-v41/bin/python|cachalot\.cli" >&2
    exit 1
fi

export CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash
export CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128
export CACHALOT_PAGE_CACHE=1
export CACHALOT_MLX_WIRED_LIMIT_GIB=${CACHALOT_MLX_WIRED_LIMIT_GIB:-72}
export CACHALOT_HOTLIST=/Users/hamedprooshani/cachalot-hotlist.json
export CACHALOT_HOTLIST_GIB=8
export PYTHONPATH=src

exec ~/venvs/deepseek-v41/bin/python -m cachalot.cli chat \
    --expert-budget-gib 44 \
    --max-seq-len 32768 \
    --max-new-tokens 2000 \
    --temperature 0.6 \
    "$@"
