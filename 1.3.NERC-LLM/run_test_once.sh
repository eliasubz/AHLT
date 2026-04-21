#!/usr/bin/env bash
# Single **test.xml** run with a frozen config (source ``FROZEN_CONFIG.env`` first or set ``LLM_FROZEN_CONFIG``).
# Usage: ./run_test_once.sh [path/to/result.out]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABS="$(cd "$REPO/../.." && pwd)"
TEST_XML="$LABS/data/test.xml"
OUT="${1:-$REPO/results/test.result.out}"

CONFIG="${LLM_FROZEN_CONFIG:-$REPO/FROZEN_CONFIG.env}"
if [[ -f "$CONFIG" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$CONFIG"
  set +a
fi

python3 "$REPO/llm-NERC.py" "$TEST_XML" "$OUT"
