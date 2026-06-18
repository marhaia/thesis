#!/usr/bin/env bash
# Quick status check for the GUI feature-norm batch run.
# Run anytime:  bash stage1/status_feature_norms.sh
# Shows whether the builder is running, how many of the 1,485 images are done
# (overall and per GUI type), and the last few log lines.

cd "$(dirname "$0")/.." || exit 1

CSV="stage1/data/results/gui_reference_features.csv"
LOG="stage1/data/results/feature_norms_run.log"
TOTAL=1485

echo "============================================================"
echo " Feature-Norm Batch — Status  ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "============================================================"

# 1) Is a run active?
if pgrep -fl "build_feature_norms|run_feature_norms_full" >/dev/null 2>&1; then
    echo "Status      : RUNNING"
    pgrep -fl "build_feature_norms|run_feature_norms_full" | sed 's/^/              /'
else
    echo "Status      : not running"
fi

# 2) How many images are in the CSV (minus header)?
if [ -f "$CSV" ]; then
    done=$(( $(wc -l < "$CSV") - 1 ))
    [ "$done" -lt 0 ] && done=0
    pct=$(awk "BEGIN{printf \"%.1f\", ($done/$TOTAL)*100}")
    echo "Progress    : ${done} / ${TOTAL} images (${pct}%)"

    # Per-category breakdown (column 2 = category).
    echo "Per type    :"
    tail -n +2 "$CSV" | awk -F',' '{c[$2]++} END{for (k in c) printf "              %-9s %d\n", k, c[k]}' | sort
else
    echo "Progress    : no CSV yet (0 images)"
fi

# 3) Last log lines (current image / last crash-restart).
echo "------------------------------------------------------------"
echo "Last log lines:"
if [ -f "$LOG" ]; then
    tail -6 "$LOG" | sed 's/^/  /'
else
    echo "  (no log file yet)"
fi
echo "============================================================"
