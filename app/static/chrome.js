// App chrome: the Quit button (stops the server, closing its start-script
// terminal window and browser tab) and the log console (show/hide toggle +
// draggable splitter).

import { state, quitBtn, logsVisibleChk, splitterEl } from "./state.js";
import { log, setStatus } from "./ui.js";
import { stopAudio } from "./audio.js";

// ---------- quit the app (closes its start-script window) ----------

quitBtn.addEventListener("click", async () => {
  if (!confirm("Quit Whisper Live Transcriber?\nThe app will stop and its terminal window and browser tab will close.")) return;
  state.userQuit = true;
  if (state.running) stopAudio();
  setStatus("idle", "Quitting…");
  quitBtn.disabled = true;
  try {
    const res = await fetch("/api/quit", { method: "POST" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    log("App stopped");
    // Best-effort self-close: works when this tab was opened by a script
    // (window.open). For a tab opened by the start script's
    // `open`/`xdg-open`, browsers silently block window.close(), so the
    // start script (macOS) also closes the matching tab once the server
    // has exited.
    window.close();
  } catch (err) {
    log(`Quit failed: ${err}`, "err");
    setStatus("error", "Quit failed");
    quitBtn.disabled = false;
    state.userQuit = false;
  }
});

// ---------- log console: show/hide toggle + draggable splitter ----------

function setLogsVisible(visible) {
  document.body.classList.toggle("logs-hidden", !visible);
}
setLogsVisible(logsVisibleChk.checked); // HTML default: hidden
logsVisibleChk.addEventListener("change", () => {
  setLogsVisible(logsVisibleChk.checked);
  log(logsVisibleChk.checked ? "Log console shown" : "Log console hidden");
});

const MIN_TRANSCRIPT_PX = 80; // mirror the CSS min-heights so the clamp
const MIN_LOG_PX = 44;        // never fights the layout
let splitDragging = false;

splitterEl.addEventListener("pointerdown", (e) => {
  if (document.body.classList.contains("logs-hidden")) return;
  splitDragging = true;
  splitterEl.classList.add("active");
  document.body.classList.add("resizing");
  splitterEl.setPointerCapture(e.pointerId);
  e.preventDefault();
});

splitterEl.addEventListener("pointermove", (e) => {
  if (!splitDragging) return;
  const split = splitterEl.parentElement;
  const rect = split.getBoundingClientRect();
  const available = rect.height - splitterEl.offsetHeight;
  const logH = Math.max(
    MIN_LOG_PX,
    Math.min(Math.round(rect.bottom - e.clientY), available - MIN_TRANSCRIPT_PX),
  );
  // Flex-grow values, so the chosen ratio also survives window resizes.
  split.style.setProperty("--split-log", String(logH));
  split.style.setProperty("--split-transcript", String(available - logH));
});

function endSplitDrag(e) {
  if (!splitDragging) return;
  splitDragging = false;
  splitterEl.classList.remove("active");
  document.body.classList.remove("resizing");
  if (splitterEl.hasPointerCapture(e.pointerId)) {
    splitterEl.releasePointerCapture(e.pointerId);
  }
}
splitterEl.addEventListener("pointerup", endSplitDrag);
splitterEl.addEventListener("pointercancel", endSplitDrag);
