#!/usr/bin/env bash
# planning/scripts/morning.sh
# Morgen-Skript: Tagesplan anzeigen + Git-Status checken
# Aufruf: bash ~/Thesis_G/planning/scripts/morning.sh

set -euo pipefail

THESIS_DIR="$HOME/Thesis_G"
PLANNING_DIR="$THESIS_DIR/planning"
TODAY=$(date +%Y-%m-%d)
WEEKDAY=$(date +%A)
KW=$(date +%V)

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   🌅  GUTEN MORGEN — $TODAY ($WEEKDAY)   ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── Git-Status ────────────────────────────────────────────────
echo "── GIT STATUS ──────────────────────────────────────────"
cd "$THESIS_DIR"
LAST_PUSH=$(git log --oneline -1 --format="%ar | %s" 2>/dev/null || echo "kein git-log verfügbar")
UNPUSHED=$(git log origin/$(git branch --show-current)..HEAD --oneline 2>/dev/null | wc -l | tr -d ' ')
UNSTAGED=$(git status --short 2>/dev/null | wc -l | tr -d ' ')

echo "  Letzter Commit:  $LAST_PUSH"
echo "  Unpushed Commits: $UNPUSHED"
echo "  Ungespeicherte Änderungen: $UNSTAGED Dateien"
if [ "$UNPUSHED" -gt 0 ] || [ "$UNSTAGED" -gt 0 ]; then
    echo ""
    echo "  ⚠️  REMINDER: Bitte heute pushen!"
fi
echo ""

# ── Tagesplan ─────────────────────────────────────────────────
echo "── TAGESPLAN ─────────────────────────────────────────────"
DAILY_FILE="$PLANNING_DIR/daily/$TODAY.md"

if [ -f "$DAILY_FILE" ]; then
    echo "  Datei: $DAILY_FILE"
    echo ""
    # Must-Do Tasks anzeigen
    echo "  🔴 Must-Do heute:"
    grep -A 10 "Must-Do" "$DAILY_FILE" | grep "^\- \[" | head -5 | sed 's/^/    /'
    echo ""
    echo "  🟡 Should-Do:"
    grep -A 10 "Should-Do" "$DAILY_FILE" | grep "^\- \[" | head -5 | sed 's/^/    /'
else
    echo "  ⚠️  Noch kein Tagesplan für heute!"
    echo "  → Template kopieren:"
    echo "     cp $PLANNING_DIR/daily/TEMPLATE_day.md $DAILY_FILE"
fi
echo ""

# ── Wochenplan-Reminder ────────────────────────────────────────
echo "── WOCHE KW$KW ─────────────────────────────────────────"
WEEK_FILE="$PLANNING_DIR/weekly/$(date +%Y)-KW$KW.md"
if [ -f "$WEEK_FILE" ]; then
    echo "  Wochenziel:"
    grep -A 3 "## Wochenziel" "$WEEK_FILE" | grep "^\- \[" | head -3 | sed 's/^/    /'
else
    echo "  ⚠️  Kein Wochenplan für KW$KW!"
    echo "  → Template kopieren:"
    echo "     cp $PLANNING_DIR/weekly/TEMPLATE_week.md $WEEK_FILE"
fi
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  VS Code öffnen:  code $THESIS_DIR"
echo "  Tagesplan:       code $DAILY_FILE"
echo "═══════════════════════════════════════════════════════"
echo ""
