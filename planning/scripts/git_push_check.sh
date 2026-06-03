#!/usr/bin/env bash
# planning/scripts/git_push_check.sh
# Wird beim Terminal-Start aus .zshrc aufgerufen.
# Warnt, wenn seit mehr als 24h nicht gepusht wurde.

THESIS_DIR="$HOME/Thesis_G"

# Nur prüfen wenn wir im Thesis-Verzeichnis sind oder git-Repo vorhanden
if ! git -C "$THESIS_DIR" rev-parse --git-dir > /dev/null 2>&1; then
    return 0 2>/dev/null || exit 0
fi

# Letzten Push-Zeitpunkt ermitteln
LAST_COMMIT_TS=$(git -C "$THESIS_DIR" log -1 --format="%ct" 2>/dev/null || echo 0)
NOW_TS=$(date +%s)
HOURS_AGO=$(( (NOW_TS - LAST_COMMIT_TS) / 3600 ))

# Unpushed commits zählen
UNPUSHED=$(git -C "$THESIS_DIR" log "origin/$(git -C "$THESIS_DIR" branch --show-current)"..HEAD --oneline 2>/dev/null | wc -l | tr -d ' ')

if [ "$UNPUSHED" -gt 0 ]; then
    echo ""
    echo "  📌 THESIS GIT REMINDER"
    echo "  ┌─────────────────────────────────────────────┐"
    echo "  │  $UNPUSHED unpushed commit(s) in ~/Thesis_G       │"
    echo "  │  Letzter Commit: vor $HOURS_AGO Stunden             │"
    echo "  │                                             │"
    echo "  │  → cd ~/Thesis_G && git push               │"
    echo "  └─────────────────────────────────────────────┘"
    echo ""
elif [ "$HOURS_AGO" -gt 24 ]; then
    echo ""
    echo "  📌 THESIS GIT REMINDER: Seit $HOURS_AGO h kein Commit in ~/Thesis_G"
    echo ""
fi
