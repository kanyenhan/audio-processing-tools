// Live (mic) mode: start/stop the mic session and the relay
// "start"/"stop" control messages.

import {
  state,
  startBtn,
  stopBtn,
  modelSelect,
  langSelect,
  translateChk,
} from "./state.js";
import { log, setStatus, setMascotActive } from "./ui.js";
import { ensureWsOpen, sendCmd, sendCmdIfOpen } from "./ws.js";
import { startAudio, stopAudio } from "./audio.js";
import { setBusyControls } from "./file-mode.js";

async function start() {
  if (state.running) return;
  const model = modelSelect.value;
  const language = langSelect.value;
  const translate = translateChk.checked;

  startBtn.disabled = true;
  modelSelect.disabled = true;
  langSelect.disabled = true;
  // translateChk stays enabled: it can be toggled mid-session.
  setStatus("connecting", "Connecting…");

  // Wait for the socket to open before sending the start command.
  await ensureWsOpen();

  sendCmd({ type: "start", model, language, translate });

  try {
    await startAudio();
    state.running = true;
    stopBtn.disabled = false;
    setMascotActive(true); // GIF while transcribing
    // Status flips to "recording" when the local model is ready;
    // otherwise an error (or loading_model) arrives.
  } catch (err) {
    log(`Microphone access failed: ${err.message}`, "err");
    setStatus("error", "Microphone error");
    startBtn.disabled = false;
    modelSelect.disabled = false;
    langSelect.disabled = false;
  }
}

export function stopAll(reason) {
  const wasRunning = state.running;
  stopAudio();
  setMascotActive(false); // back to the idle PNG
  if (wasRunning) {
    sendCmdIfOpen({ type: "stop" });
  } else if (reason && state.localWs && state.localWs.readyState === WebSocket.OPEN) {
    // Whisper died mid-session: tell the backend to reset.
    sendCmd({ type: "stop" });
  }
  if (!reason) setStatus("idle", "Idle");
  startBtn.disabled = false;
  modelSelect.disabled = false;
  langSelect.disabled = false;
  stopBtn.disabled = true;
  setBusyControls(false); // also resets file-mode busy state
}

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", () => {
  stopAll(undefined);
  // stopAll only sends "stop" for live runs; send it anyway so a
  // file-mode run is released too (a no-op when no session is open).
  sendCmdIfOpen({ type: "stop" });
});
translateChk.addEventListener("change", () => {
  // Toggle mid-session: applies to segments from now on.
  if (state.localWs && state.localWs.readyState === WebSocket.OPEN) {
    sendCmd({ type: "set_translate", translate: translateChk.checked });
    log(`Translation to English ${translateChk.checked ? "enabled" : "disabled"}`);
  }
});
