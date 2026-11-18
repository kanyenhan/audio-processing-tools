// Local websocket (browser <-> FastAPI app): connection lifecycle with
// auto-reconnect, plus dispatch of the server messages to the right
// feature module.

import { state, transcriptEl } from "./state.js";
import {
  appendTranscript,
  appendFileTranslation,
  formatSeconds,
  log,
  setFileStatus,
  setStatus,
} from "./ui.js";
import { stopAll } from "./live-mode.js";
import { setBusyControls } from "./file-mode.js";
import { stopAudio } from "./audio.js";

function resolveConnected() {
  if (state.connectedWaiter) {
    const w = state.connectedWaiter;
    state.connectedWaiter = null;
    w();
  }
}

export function connectLocalWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  state.localWs = new WebSocket(`${proto}://${location.host}/ws`);
  state.localWs.binaryType = "arraybuffer";

  state.localWs.onopen = () => log("Connected to app relay");
  state.localWs.onclose = () => {
    if (state.running) stopAudio();
    if (state.userQuit) {
      // The app is shutting down (Quit button) — don't reconnect or we
      // would spam the dead port forever.
      setStatus("idle", "App stopped");
      log("App stopped; not reconnecting");
      return;
    }
    log("App relay closed; reconnecting in 2s", "warn");
    setTimeout(() => {
      const ws = state.localWs;
      if (!ws || ws.readyState > 1) connectLocalWs();
    }, 2000);
  };
  state.localWs.onerror = () => log("App relay error", "err");
  state.localWs.onmessage = (ev) => {
    if (typeof ev.data === "string") {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      handleServerMessage(msg);
    }
  };
}

export function handleServerMessage(msg) {
  switch (msg.type) {
    case "status":
      if (msg.status === "connected") {
        resolveConnected();
        if (state.appMode === "file") {
          setStatus("connecting", "Model ready");
        } else {
          setStatus("recording", `Recording → ${msg.url}`);
        }
        log(`Transcribing locally with ${msg.url}`);
      } else if (msg.status === "processing_file") {
        setStatus("recording", msg.message || "Processing file…");
        if (msg.message) setFileStatus(msg.message);
      } else if (msg.status === "loading_model") {
        setStatus("connecting", msg.message || "Loading model…");
        log(`Model ${msg.model} is not loaded yet: ${msg.message || "downloading/loading"}`);
      } else if (msg.status === "disconnected") {
        setStatus("error", "Whisper server disconnected");
        log(msg.message || "Whisper server disconnected", "err");
        stopAll("Whisper server disconnected");
      } else if (msg.status === "idle") {
        setStatus("idle", "Idle");
        log("Stopped");
      }
      break;
    case "error":
      resolveConnected();
      setStatus("error", "Error");
      log(msg.message, "err");
      stopAll(msg.message);
      break;
    case "warning":
      log(msg.message, "warn");
      // Surface actionable warnings (e.g. model cannot translate) in the
      // status bar, not just the log.
      setStatus("connecting", msg.message);
      break;
    case "translation":
      if (msg.text) {
        if (state.appMode === "file") {
          // File mode: accumulate into one continuous translation paragraph
          // (arrival order equals segment order).
          appendFileTranslation(msg.text.trim());
        } else {
          // Follow-up to a segment message (matched by ts): append the
          // English translation under that segment in place.  Silently
          // ignored if the user cleared the transcript in the meantime.
          if (msg.ts) {
            const seg = transcriptEl.querySelector(`.segment[data-ts="${msg.ts}"]`);
            if (seg) {
              const tr = document.createElement("div");
              tr.className = "translation";
              tr.textContent = msg.text.trim();
              seg.appendChild(tr);
              transcriptEl.scrollTop = transcriptEl.scrollHeight;
            }
          }
        }
      }
      break;
    case "file_done":
      setBusyControls(false);
      setStatus("idle", "Idle");
      setFileStatus(
        `${msg.filename}: ${msg.segments} segment(s) · ${formatSeconds(msg.duration_s)} audio`
      );
      log(`Done: ${msg.filename} — ${msg.segments} segment(s), ${msg.duration_s}s`);
      break;
    default:
      // Transcript relayed from the whisper server (tolerant parsing).
      appendTranscript(msg);
  }
}

export function sendCmd(obj) {
  state.localWs.send(JSON.stringify(obj));
}

export function sendCmdIfOpen(obj) {
  const ws = state.localWs;
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

export function ensureWsOpen() {
  return new Promise((resolve) => {
    if (state.localWs && state.localWs.readyState === WebSocket.OPEN) return resolve();
    connectLocalWs();
    const check = () => {
      if (state.localWs && state.localWs.readyState === WebSocket.OPEN) resolve();
      else setTimeout(check, 100);
    };
    check();
  });
}
