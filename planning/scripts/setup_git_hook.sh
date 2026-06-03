#!/usr/bin/env bash
# planning/scripts/setup_git_hook.sh
# Richtet einen post-commit hook ein, der an push erinnert.
# Einmalig ausführen: bash ~/Thesis_G/planning/scripts/setup_git_hook.sh

set -euo pipefail

THESIS_DIR="$HOME/Thesis_G"
HOOKS_DIR="$THESIS_DIR/.git/hooks"

echo "🔧 Setup: Git post-commit Hook..."

cat > "$HOOKS_DIR/post-commit" << 'EOF'
#!/usr/bin/env bash
# post-commit hook: erinnert ans Pushen

BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
UNPUSHED=$(git log "origin/$BRANCH"..HEAD --oneline 2>/dev/null | wc -l | tr -d ' ')

if [ "$UNPUSHED" -gt 0 ]; then
    echo ""
    echo "  ✅ Commit gespeichert!"
    echo "  📌 $UNPUSHED Commit(s) noch nicht gepusht."
    echo "  → git push"
    echo ""
fi
EOF

chmod +x "$HOOKS_DIR/post-commit"

echo "✅ Git Hook eingerichtet: $HOOKS_DIR/post-commit"
echo "   → Erinnert nach jedem commit ans Pushen"
