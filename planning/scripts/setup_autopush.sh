#!/usr/bin/env bash
# planning/scripts/setup_autopush.sh
# Richtet einen macOS LaunchAgent ein, der täglich um 22:00 automatisch
# alle Änderungen in ~/Thesis_G committet und pusht.
#
# Einmalig ausführen:
#   bash ~/Thesis_G/planning/scripts/setup_autopush.sh
#
# Wieder entfernen:
#   launchctl unload ~/Library/LaunchAgents/com.thesis.autopush.plist
#   rm ~/Library/LaunchAgents/com.thesis.autopush.plist

set -euo pipefail

THESIS_DIR="$HOME/Thesis_G"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
PLIST="$LAUNCH_AGENTS/com.thesis.autopush.plist"
SCRIPT="$THESIS_DIR/planning/scripts/auto_push.sh"

mkdir -p "$LAUNCH_AGENTS"
chmod +x "$SCRIPT"

echo "🔧 Setup: Auto-Push LaunchAgent (täglich 22:00)..."

cat > "$PLIST" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.thesis.autopush</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>-l</string>
        <string>${SCRIPT}</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>22</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>${THESIS_DIR}/planning/scripts/auto_push.log</string>
    <key>StandardErrorPath</key>
    <string>${THESIS_DIR}/planning/scripts/auto_push.log</string>
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
EOF

# Neu laden (erst entladen, falls schon vorhanden)
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"

echo "✅ Auto-Push eingerichtet:"
echo "   22:00 → committet alle Änderungen in ~/Thesis_G und pusht sie"
echo ""
echo "ℹ️  War der Laptop um 22:00 aus, läuft der Job beim nächsten Aufwachen nach."
echo ""
echo "Manuell testen:"
echo "   bash $SCRIPT"
echo "Log ansehen:"
echo "   tail -f $THESIS_DIR/planning/scripts/auto_push.log"
