#!/usr/bin/env bash
# ============================================================
# Start Whisper Live Transcriber (Linux).
#   0. open a dedicated terminal window (closes when the app quits)
#   1. create .venv (if missing)
#   2. install requirements
#   3. start the app
#   4. open the page in a new browser window once the server answers
# ============================================================
set -euo pipefail

# Always run from the repository root, regardless of where invoked.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- 0. dedicated terminal window ----------------------------------------
# First invocation relaunches itself in a new terminal window and exits,
# so the app runs in its own window that closes automatically once the
# process exits.  The guard flag is embedded *inline* in the spawned
# command on purpose: a fresh terminal does NOT inherit env vars exported
# by the parent shell.  `exec` makes the window's process the app script
# itself, so the window dies with the app.  Without a known terminal
# emulator we fall back to running in place.
if [ -z "${WHISPER_LIVE_DEDICATED:-}" ]; then
  SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  CMD="WHISPER_LIVE_DEDICATED=1 exec bash $(printf '%q' "$SCRIPT_PATH")"
  if command -v gnome-terminal >/dev/null 2>&1; then
    echo "Opening a dedicated terminal window for Whisper Live Transcriber ..."
    gnome-terminal --title="Whisper Live Transcriber" -- bash -c "$CMD"
    exit 0
  elif command -v x-terminal-emulator >/dev/null 2>&1; then
    echo "Opening a dedicated terminal window for Whisper Live Transcriber ..."
    x-terminal-emulator --title="Whisper Live Transcriber" -- bash -c "$CMD"
    exit 0
  else
    echo "notice: no terminal emulator found (gnome-terminal / x-terminal-emulator)" >&2
    echo "running in this terminal instead (window will not auto-close)" >&2
  fi
fi

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
# </dev/null: with a TTY on stdin, pip's readline can hang on some
# systems; no interactive input is needed here.
"$PY" -m pip install --quiet --upgrade pip </dev/null
"$PY" -m pip install --quiet -r requirements.txt </dev/null

# --- 4. open the browser once the server answers (background) -----------
# Open the page in a *new browser window* instead of a new tab: xdg-open
# cannot force this, so when the default browser is a well-known one, call
# it with its own new-window flag; otherwise fall back to xdg-open.
open_in_new_window() {
  local def
  def="$(xdg-mime query default x-scheme-handler/http 2>/dev/null || true)"
  case "$def" in
    firefox*.desktop)
      firefox -new-window "$URL" 2>/dev/null || xdg-open "$URL" ;;
    microsoft-edge*.desktop)
      msedge --new-window "$URL" 2>/dev/null || xdg-open "$URL" ;;
    google-chrome*.desktop|chromium*.desktop|brave*.desktop|vivaldi*.desktop)
      local bin="${def%.desktop}"
      command -v "$bin" >/dev/null 2>&1 && "$bin" --new-window "$URL" || xdg-open "$URL" ;;
    *)
      if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$URL"
      else
        echo "warning: xdg-open not found — please open ${URL} in a browser" >&2
      fi ;;
  esac
}

(
  for _ in $(seq 1 240); do            # up to ~2 minutes
    if "$PY" -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=1)' "$URL" >/dev/null 2>&1; then
      open_in_new_window
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

# The app exited (Quit button or Ctrl-C): close this dedicated window.
echo
echo "App stopped. Closing this window in 3 seconds..."
sleep 3
