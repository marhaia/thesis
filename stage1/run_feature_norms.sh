#!/bin/zsh
# Launcher for the full GUI feature-norm build.
# Runs detached so the long (~3-4h) job survives terminal churn.
cd "$(dirname "$0")/.."          # repo root (script lives in stage1/)
source venv/bin/activate
exec python3 -u stage1/build_feature_norms.py
