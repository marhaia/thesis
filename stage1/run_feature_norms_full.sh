#!/usr/bin/env bash
# Crash-resilient runner for the GUI feature-norm batch.
#
# The Python builder writes every processed image to the per-image CSV
# immediately and resumes from it on restart. This wrapper relaunches the
# builder whenever it exits non-zero (e.g. an OOM kill), so repeated crashes
# only cost a few images of recompute instead of the whole run. It stops as
# soon as the builder finishes cleanly (exit 0) or the retry cap is reached.
#
# Usage:
#   bash stage1/run_feature_norms_full.sh            # all GUI types
#   bash stage1/run_feature_norms_full.sh web        # one folder/type only
#   bash stage1/run_feature_norms_full.sh mobile
#   bash stage1/run_feature_norms_full.sh desktop

set -u

cd "$(dirname "$0")/.." || exit 1

# shellcheck disable=SC1091
source venv/bin/activate

CATEGORY="${1:-}"
CAT_ARG=()
if [ -n "$CATEGORY" ]; then
    CAT_ARG=(--category "$CATEGORY")
fi

LOG="stage1/data/results/feature_norms_run.log"
MAX_RETRIES=200
attempt=0

echo "=== Auto-restart runner started $(date) (category='${CATEGORY:-ALL}') ===" | tee -a "$LOG"

while :; do
    attempt=$((attempt + 1))
    echo "" | tee -a "$LOG"
    echo "=== Attempt ${attempt} at $(date) ===" | tee -a "$LOG"

    python3 stage1/build_feature_norms.py "${CAT_ARG[@]}" >>"$LOG" 2>&1
    code=$?

    if [ "$code" -eq 0 ]; then
        echo "=== Builder finished successfully (exit 0) at $(date) ===" | tee -a "$LOG"
        break
    fi

    echo "=== Builder exited with code ${code}; resuming in 5s (attempt ${attempt}/${MAX_RETRIES}) ===" | tee -a "$LOG"

    if [ "$attempt" -ge "$MAX_RETRIES" ]; then
        echo "=== Reached MAX_RETRIES (${MAX_RETRIES}); giving up. ===" | tee -a "$LOG"
        break
    fi
    sleep 5
done

echo "=== Runner done $(date) ===" | tee -a "$LOG"
