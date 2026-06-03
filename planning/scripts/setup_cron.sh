#!/usr/bin/env bash
# planning/scripts/setup_cron.sh
# Richtet macOS LaunchAgents für Morgen (08:00) und Abend (19:00) ein.
# Einmalig ausführen: bash ~/Thesis_G/planning/scripts/setup_cron.sh

set -euo pipefail

THESIS_DIR="$HOME/Thesis_G"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
mkdir -p "$LAUNCH_AGENTS"

echo "🔧 Setup: LaunchAgents für Thesis-Reminder..."

# ── Morgen-Agent (08:00) ───────────────────────────────────────
cat > "$LAUNCH_AGENTS/com.thesis.morning.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.thesis.morning</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>-l</string>
        <string>-c</string>
        <string>osascript -e 'display notification "Tagesplan öffnen: Thesis_G/planning/daily/" with title "Thesis Morning Briefing" subtitle "bash ~/Thesis_G/planning/scripts/morning.sh" sound name "Glass"'</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>8</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
EOF

# ── Abend-Agent (19:00) ────────────────────────────────────────
cat > "$LAUNCH_AGENTS/com.thesis.evening.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.thesis.evening</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>-l</string>
        <string>-c</string>
        <string>osascript -e 'display notification "bash ~/Thesis_G/planning/scripts/evening.sh" with title "Thesis Evening Check-In" subtitle "Status update + Git push reminder" sound name "Glass"'</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>19</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
EOF

# LaunchAgents laden
launchctl load "$LAUNCH_AGENTS/com.thesis.morning.plist" 2>/dev/null || true
launchctl load "$LAUNCH_AGENTS/com.thesis.evening.plist" 2>/dev/null || true

echo "✅ LaunchAgents eingerichtet:"
echo "   08:00 → Morgen-Benachrichtigung"
echo "   19:00 → Abend-Check-In-Benachrichtigung"
echo ""
echo "Skripte manuell ausführen:"
echo "  bash ~/Thesis_G/planning/scripts/morning.sh"
echo "  bash ~/Thesis_G/planning/scripts/evening.sh"
