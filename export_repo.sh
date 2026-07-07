#!/bin/bash
# Safe zip export of this repo -- excludes secrets, caches, and generated
# results so the .env (API keys) never accidentally ends up in the archive.
#
# Usage: ./export_repo.sh [output.zip]
set -euo pipefail

OUTPUT="${1:-honest_agent_export.zip}"

zip -r "$OUTPUT" . \
  -x ".env" \
  -x "*.env" \
  -x "*__pycache__/*" \
  -x "*.pyc" \
  -x ".git/*" \
  -x ".pytest_cache/*" \
  -x "results/*" \
  -x "*.DS_Store"

echo "Wrote $OUTPUT (excluding .env, caches, .git, results)"
