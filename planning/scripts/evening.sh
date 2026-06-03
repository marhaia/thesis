#!/usr/bin/env bash
# planning/scripts/evening.sh
# Abend-Skript: Status-Check + Git-Push-Reminder + Tagesplan-Update-Prompt
# Aufruf: bash ~/Thesis_G/planning/scripts/evening.sh

set -euo pipefail

THESIS_DIR="$HOME/Thesis_G"
PLANNING_DIR="$THESIS_DIR/planning"
TODAY=$(date +%Y-%m-%d)
TOMORROW=$(date -v+1d +%Y-%m-%d 2>/dev/null || date -d "+1 day" +%Y-%m-%d)

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   🌙  ABEND-CHECK — $TODAY              ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── Git Push Check ─────────────────────────────────────────────
echo "── GIT STATUS ──────────────────────────────────────────"
cd "$THESIS_DIR"
UNPUSHED=$(git log origin/$(git branch --show-current)..HEAD --oneline 2>/dev/null | wc -l | tr -d ' ')
UNSTAGED=$(git status --short 2>/dev/null | wc -l | tr -d ' ')

if [ "$UNPUSHED" -gt 0 ] || [ "$UNSTAGED" -gt 0 ]; then
    echo ""
    echo "  🚨  ACHTUNG: Noch nicht gepusht!"
    echo "  Unpushed Commits: $UNPUSHED"
    echo "  Geänderte Dateien: $UNSTAGED"
    echo ""
    echo "  Jetzt pushen? Ausführen:"
    echo "  cd $THESIS_DIR && git add . && git commit -m 'chore: daily update $TODAY' && git push"
    echo ""
else
    echo "  ✅ Git ist aktuell — gut gemacht!"
fi
echo ""

# ── Tagesplan Status ───────────────────────────────────────────
echo "── WAS HAST DU HEUTE GESCHAFFT? ───────────────────────"
DAILY_FILE="$PLANNING_DIR/daily/$TODAY.md"
if [ -f "$DAILY_FILE" ]; then
    DONE=$(grep -c "\- \[x\]" "$DAILY_FILE" 2>/dev/null || echo 0)
    OPEN=$(grep -c "\- \[ \]" "$DAILY_FILE" 2>/dev/null || echo 0)
    echo "  ✅ Erledigt:   $DONE Tasks"
    echo "  ❌ Offen:      $OPEN Tasks"
    echo ""
    if [ "$OPEN" -gt 0 ]; then
        echo "  Offene Tasks:"
        grep "\- \[ \]" "$DAILY_FILE" | head -5 | sed 's/^/    /'
    fi
    echo ""
    echo "  → Tagesplan aktualisieren: code $DAILY_FILE"
else
    echo "  ⚠️  Kein Tagesplan für heute gefunden."
fi
echo ""

# ── Morgen vorbereiten ─────────────────────────────────────────
echo "── MORGEN: $TOMORROW ─────────────────────────────────"
TOMORROW_FILE="$PLANNING_DIR/daily/$TOMORROW.md"
if [ ! -f "$TOMORROW_FILE" ]; then
    echo "  Tagesplan für morgen erstellen:"
    echo "  cp $PLANNING_DIR/daily/TEMPLATE_day.md $TOMORROW_FILE"
    echo "  code $TOMORROW_FILE"
    echo ""
    read -p "  Soll die Datei jetzt erstellt werden? [j/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Jj]$ ]]; then
        cp "$PLANNING_DIR/daily/TEMPLATE_day.md" "$TOMORROW_FILE"
        # Datum eintragen
        sed -i '' "s/\[DATUM\]/$TOMORROW/g" "$TOMORROW_FILE" 2>/dev/null || \
        sed -i "s/\[DATUM\]/$TOMORROW/g" "$TOMORROW_FILE"
        echo "  ✅ Erstellt: $TOMORROW_FILE"
    fi
else
    echo "  ✅ Tagesplan für morgen existiert bereits"
fi
echo ""

# ── Freitag: Wochenlog-Reminder ────────────────────────────────
if [ "$(date +%u)" -eq 5 ]; then
    KW=$(date +%V)
    WEEK_FILE="$PLANNING_DIR/weekly/$(date +%Y)-KW$KW.md"
    echo "── 📋 FREITAG: WOCHENREVIEW ────────────────────────────"
    echo "  Wochenplan ausfüllen: code $WEEK_FILE"
    echo "  Nächste Woche planen: KW$((KW + 1))"
    echo ""
fi

echo "═══════════════════════════════════════════════════════"
echo "  Bis morgen! 👋"
echo "═══════════════════════════════════════════════════════"
echo ""
