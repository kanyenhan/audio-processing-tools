#!/usr/bin/env bash
# ============================================================
# Start Whisper Live Transcriber (macOS).
#   1. create .venv (if missing)
#   2. install requirements
#   3. start the app
#   4. open the page in the default browser once the server answers
# ============================================================
set -euo pipefail

# Always run from the repository root, regardless of where invoked.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

HOST=127.0.0.1
PORT=8080
URL="http://${HOST}:${PORT}"

# --- 1. virtual environment --------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 not found on PATH (Python 3.10+ required)" >&2
  exit 1
fi
if [ ! -x .venv/bin/python ]; then
  echo "Creating virtual environment in .venv ..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
PY=.venv/bin/python

# --- 2. dependencies -----------------------------------------------------
echo "Installing requirements (safe to re-run) ..."
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -r requirements.txt

# --- 4. open the browser once the server answers (background) -----------
(
  for _ in $(seq 1 240); do            # up to ~2 minutes
    if "$PY" -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=1)' "$URL" >/dev/null 2>&1; then
      open "$URL"
      exit 0
    fi
    sleep 0.5
  done
  echo "warning: ${URL} did not come up in time — open it manually" >&2
) &
OPENER_PID=$!
trap 'kill "$OPENER_PID" 2>/dev/null || true' EXIT

# --- 3. start the app -----------------------------------------------------
echo "Starting Whisper Live Transcriber at ${URL}"
"$PY" -m uvicorn app.main:app --host "$HOST" --port "$PORT"
