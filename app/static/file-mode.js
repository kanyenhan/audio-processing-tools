// File mode: transcribe (and translate) an audio file, either referenced
// by a typed path or picked via Browse (uploaded to the app first).  The
// app decodes it server-side and streams the result back through the same
// per-utterance pipeline as live mode.

import {
  state,
  modelSelect,
  langSelect,
  translateChk,
  modeLiveBtn,
  modeFileBtn,
  livePanel,
  filePanel,
  transcribePanel,
  filePathEl,
  fileInput,
  browseBtn,
  transcribeBtn,
} from "./state.js";
import { log, resetTranscript, setFileStatus, setStatus, setMascotActive } from "./ui.js";
import { ensureWsOpen, sendCmd, sendCmdIfOpen } from "./ws.js";
import { stopAll } from "./live-mode.js";

let selectedFile = null; // File object picked via Browse
let fileBusy = false;    // a file transcription is in flight

export function setBusyControls(busy) {
  fileBusy = busy;
  modelSelect.disabled = busy;
  langSelect.disabled = busy;
  transcribeBtn.disabled = busy;
  browseBtn.disabled = busy;
  filePathEl.disabled = busy;
  transcribeBtn.textContent = busy ? "Processing…" : "Transcribe";
}

async function startFileTranscript() {
  if (fileBusy) return;
  let path = filePathEl.value.trim();
  let filename = path ? path.split("/").pop() : null;

  if (!path && !selectedFile) {
    setFileStatus("Enter a file path or choose a file with Browse first.", "warn");
    return;
  }

  setBusyControls(true);
  setMascotActive(true); // GIF while processing
  setStatus("connecting", "Connecting…");

  try {
    // Browse-picked files are uploaded to the app first; the server then
    // processes the temp path exactly like a typed one.
    if (selectedFile) {
      setFileStatus(`Uploading ${selectedFile.name}…`);
      const form = new FormData();
      form.append("file", selectedFile);
      const res = await fetch("/api/upload", { method: "POST", body: form });
      if (!res.ok) throw new Error(`Upload failed (HTTP ${res.status})`);
      const uploaded = await res.json();
      path = uploaded.path;
      filename = uploaded.filename;
    }

    await ensureWsOpen();
    const connected = new Promise((resolve) => { state.connectedWaiter = resolve; });
    sendCmd({
      type: "start",
      model: modelSelect.value,
      language: langSelect.value,
      translate: translateChk.checked,
    });
    await connected;
    if (!fileBusy) return; // an error arrived; already handled

    // Each file renders as fresh continuous paragraphs.
    resetTranscript();
    setFileStatus(`Transcribing ${filename}…`);
    sendCmd({ type: "file", path, filename });
    log(`Submitting ${filename} for transcription (model=${modelSelect.value}, translate=${translateChk.checked})`);
  } catch (err) {
    log(`File transcription failed: ${err.message}`, "err");
    setStatus("error", "Error");
    setBusyControls(false);
    setMascotActive(false);
    setFileStatus("Failed — see log.", "warn");
    sendCmdIfOpen({ type: "stop" });
  }
}

function setMode(mode) {
  if (mode === state.appMode) return;
  if (fileBusy) {
    setFileStatus("Wait for the current file to finish (or press Stop).", "warn");
    return;
  }
  const isLive = mode === "live";
  state.appMode = mode;
  livePanel.hidden = !isLive;
  filePanel.hidden = isLive;
  transcribePanel.hidden = isLive;
  modeLiveBtn.classList.toggle("active", isLive);
  modeFileBtn.classList.toggle("active", !isLive);
  if (isLive) {
    // Leaving file mode: drop the picked/typed file.
    fileInput.value = "";
    selectedFile = null;
    filePathEl.value = "";
    transcribeBtn.disabled = true;
    setFileStatus("");
  } else if (state.running) {
    // A live mic session can't share the session with a file run.
    stopAll(undefined);
  }
}

modeLiveBtn.addEventListener("click", () => setMode("live"));
modeFileBtn.addEventListener("click", () => setMode("file"));
browseBtn.addEventListener("click", () => { if (!fileBusy) fileInput.click(); });
fileInput.addEventListener("change", () => {
  selectedFile = fileInput.files[0] || null;
  if (selectedFile) {
    filePathEl.value = selectedFile.name;
    transcribeBtn.disabled = false;
  } else {
    transcribeBtn.disabled = !filePathEl.value.trim();
  }
});
filePathEl.addEventListener("input", () => {
  // Typing a path supersedes a browsed file (and vice versa).
  selectedFile = null;
  fileInput.value = "";
  transcribeBtn.disabled = fileBusy || !filePathEl.value.trim();
});
transcribeBtn.addEventListener("click", startFileTranscript);
