#!/usr/bin/env bash
# planning/scripts/auto_push.sh
# Committet automatisch ALLE Änderungen in ~/Thesis_G und pusht sie.
# Wird vom LaunchAgent com.thesis.autopush täglich um 22:00 ausgeführt.
#
# Manuell testen:
#   bash ~/Thesis_G/planning/scripts/auto_push.sh

set -uo pipefail

THESIS_DIR="$HOME/Thesis_G"
LOG_FILE="$THESIS_DIR/planning/scripts/auto_push.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"
}

# Nur fortfahren, wenn es ein Git-Repo ist
if ! git -C "$THESIS_DIR" rev-parse --git-dir > /dev/null 2>&1; then
    log "FEHLER: $THESIS_DIR ist kein Git-Repo. Abbruch."
    exit 1
fi

cd "$THESIS_DIR" || { log "FEHLER: cd nach $THESIS_DIR fehlgeschlagen."; exit 1; }

BRANCH=$(git branch --show-current 2>/dev/null || echo "main")

# Gibt es überhaupt etwas zu tun? (uncommittete Änderungen ODER unpushed commits)
HAS_CHANGES=$(git status --porcelain | wc -l | tr -d ' ')
UNPUSHED=$(git log "origin/${BRANCH}..HEAD" --oneline 2>/dev/null | wc -l | tr -d ' ')

if [ "$HAS_CHANGES" -eq 0 ] && [ "$UNPUSHED" -eq 0 ]; then
    log "Nichts zu tun – keine Änderungen, keine unpushed commits."
    exit 0
fi

# 1. Uncommittete Änderungen committen
if [ "$HAS_CHANGES" -gt 0 ]; then
    git add -A
    MSG="Auto-commit $(date '+%Y-%m-%d %H:%M') ($HAS_CHANGES Datei(en))"
    if git commit -m "$MSG" >> "$LOG_FILE" 2>&1; then
        log "Committet: $MSG"
    else
        log "FEHLER beim Commit."
        exit 1
    fi
fi

# 2. Pushen (Token kommt non-interaktiv aus dem macOS-Schlüsselbund)
if git push origin "$BRANCH" >> "$LOG_FILE" 2>&1; then
    log "✅ Push nach origin/$BRANCH erfolgreich."
else
    log "❌ Push fehlgeschlagen (Netzwerk? Anmeldedaten?). Versuche es morgen erneut."
    # Optional: macOS-Benachrichtigung
    osascript -e 'display notification "Auto-Push fehlgeschlagen – siehe auto_push.log" with title "Thesis Git" sound name "Basso"' 2>/dev/null || true
    exit 1
fi
