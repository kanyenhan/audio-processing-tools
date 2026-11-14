#!/usr/bin/env bash
# ============================================================
# Start Whisper Live Transcriber (macOS).
#   0. open a dedicated Terminal window (closes when the app quits)
#   1. create .venv (if missing)
#   2. install requirements
#   3. start the app
#   4. open the page in a new browser window once the server answers
# ============================================================
set -euo pipefail

# Always run from the repository root, regardless of where invoked.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- 0. dedicated terminal window ----------------------------------------
# First invocation relaunches itself in a *new* Terminal window and exits,
# so the app runs in its own window that closes automatically once the
# process exits (Terminal setting "When the shell exits" → "Close the
# window", the default).  The guard flag is embedded *inline* in the
# do-script command on purpose: a fresh Terminal window does NOT inherit
# env vars exported by the parent shell, so setting it in this shell would
# just relaunch forever.
if [ -z "${WHISPER_LIVE_DEDICATED:-}" ]; then
  SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  echo "Opening a dedicated Terminal window for Whisper Live Transcriber ..."
  # 'with timeout': a background Terminal app can be slow to answer
  # do-script (AppleEvent -1712) even though the window does open.
  # `exec` replaces the new window's login shell with the script, so when
  # the app quits the shell has actually exited and Terminal closes the
  # window (default: "When the shell exits" → "Close the window").
  osascript <<EOF
with timeout of 180 seconds
  tell application "Terminal"
    activate
    do script "WHISPER_LIVE_DEDICATED=1 exec bash $(printf '%q' "$SCRIPT_PATH")"
  end tell
end timeout
EOF
  exit 0
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
# Open the page in a *new browser window* instead of a new tab in an
# existing one.  Each well-known browser gets its own new-window flag,
# passed to a fresh app instance (which forwards the flag to the running
# browser, so the window appears in the same session); anything else falls
# back to `open -n`, which starts a fresh app instance/window.
detect_default_browser() {
  "$PY" - <<'PYEOF' 2>/dev/null || true
import plistlib, subprocess, sys
try:
    out = subprocess.run(
        ["/usr/bin/defaults", "export",
         "com.apple.LaunchServices/com.apple.launchservices.secure", "-"],
        capture_output=True, timeout=10)
    pl = plistlib.loads(out.stdout)
except Exception:
    sys.exit(0)
for h in pl.get("LSHandlers", []):
    # The scheme may be stored as the plural list key (user-chosen default)
    # or the singular string key (system default); bundle ids can be lowercase.
    schemes = [s.lower() for s in h.get("LSHandlerURLSchemes", [])]
    single = h.get("LSHandlerURLScheme")
    if isinstance(single, str):
        schemes.append(single.lower())
    if "http" in schemes or "https" in schemes:
        print(h.get("LSHandlerRoleAll", ""))
        break
PYEOF
}

open_in_new_window() {
  local bundle
  bundle="$(detect_default_browser)"
  case "$(printf '%s' "$bundle" | tr '[:upper:]' '[:lower:]')" in
    com.google.chrome)      open -na "Google Chrome" --args --new-window "$URL" ;;
    com.google.chromium)    open -na "Chromium" --args --new-window "$URL" ;;
    com.brave.browser)      open -na "Brave Browser" --args --new-window "$URL" ;;
    com.microsoft.edgemac)  open -na "Microsoft Edge" --args --new-window "$URL" ;;
    com.vivaldi.vivaldi)    open -na "Vivaldi" --args --new-window "$URL" ;;
    org.mozilla.firefox)    open -na "Firefox" --args -new-window "$URL" ;;
    *)                      open -n "$URL" ;;
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

# The app exited (Quit button or Ctrl-C): close the browser tab the app
# opened, then close this dedicated window.  window.close() from the page
# is blocked by browsers for tabs not opened by a script, so we close the
# matching tab from here via AppleScript — but only in browsers that are
# already running (never launching one just to check).  Missing browsers,
# or a denied Automation permission, are not a problem: `|| true` keeps
# the shutdown going.  The first run triggers a one-time macOS prompt
# ("Terminal is trying to control <browser>"); click Allow once.
osascript - "$URL" <<'APPLESCRIPT' || true
on run argv
  set urlPrefix to item 1 of argv
  set targetApps to {"Safari", "Google Chrome", "Chromium", "Brave Browser", "Microsoft Edge", "Vivaldi"}
  repeat with appName in targetApps
    try
      if application appName is running then
        -- (appName) needs parens in a dynamic tell, and tabs are addressed
        -- via a fetched list (indexed refs like "tab i of w" do not parse
        -- there); closing in reverse order keeps indices valid
        tell application (appName)
          repeat with w in (get windows)
            set tabList to (get tabs of w)
            repeat with i from (count of tabList) to 1 by -1
              set t to (item i of tabList)
              if ((URL of t) as text) starts with urlPrefix then
                close t
              end if
            end repeat
          end repeat
        end tell
      end if
    end try
  end repeat
end run
APPLESCRIPT

echo
echo "App stopped. Closing this window in 3 seconds..."
sleep 3
