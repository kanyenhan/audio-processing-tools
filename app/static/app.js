// Whisper Live Transcriber — browser client.
// Captures microphone audio as 16 kHz mono PCM16 and streams it to the
// local FastAPI app, which transcribes it in-process via whisper-ctranslate2.

const TARGET_SAMPLE_RATE = 16000;

const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const clearBtn = document.getElementById("clear-btn");
const modelSelect = document.getElementById("model");
const langSelect = document.getElementById("language");
const translateChk = document.getElementById("translate-en");
const transcriptEl = document.getElementById("transcript");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const logEl = document.getElementById("log");
const modeLiveBtn = document.getElementById("mode-live");
const modeFileBtn = document.getElementById("mode-file");
const livePanel = document.getElementById("live-panel");
const filePanel = document.getElementById("file-panel");
const transcribePanel = document.getElementById("transcribe-panel");
const filePathEl = document.getElementById("file-path");
const fileInput = document.getElementById("file-input");
const browseBtn = document.getElementById("browse-btn");
const transcribeBtn = document.getElementById("transcribe-btn");
const fileStatusEl = document.getElementById("file-status");
const logsVisibleChk = document.getElementById("logs-visible");
const splitterEl = document.getElementById("splitter");
const exportBtn = document.getElementById("export-btn");
const quitBtn = document.getElementById("quit-btn");

let localWs = null;
let running = false;
let userQuit = false;       // true after the user pressed Quit — no reconnects
let partialEl = null;
let appMode = "live";        // "live" | "file"
let selectedFile = null;     // File object picked via Browse
let fileBusy = false;        // a file transcription is in flight
let fileTextEl = null;       // continuous-paragraph elements (file mode)
let fileTrEl = null;
let connectedWaiter = null;  // resolved when a "connected" status arrives

let audioCtx = null;
let mediaStream = null;
let audioSource = null;
let scriptNode = null;

// ---------- logging / status ----------

function log(message, cls = "") {
  const line = document.createElement("div");
  if (cls) line.className = cls;
  const ts = new Date().toLocaleTimeString();
  line.textContent = `[${ts}] ${message}`;
  logEl.appendChild(line);
  while (logEl.children.length > 60) logEl.removeChild(logEl.firstChild);
  logEl.scrollTop = logEl.scrollHeight;
}

function setStatus(state, text) {
  statusDot.className = `dot ${state}`;
  statusText.textContent = text;
}

function resolveConnected() {
  if (connectedWaiter) {
    const w = connectedWaiter;
    connectedWaiter = null;
    w();
  }
}

// ---------- local websocket (browser <-> FastAPI app) ----------

function connectLocalWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  localWs = new WebSocket(`${proto}://${location.host}/ws`);
  localWs.binaryType = "arraybuffer";

  localWs.onopen = () => log("Connected to app relay");
  localWs.onclose = () => {
    if (running) stopAudio();
    if (userQuit) {
      // The app is shutting down (Quit button) — don't reconnect or we
      // would spam the dead port forever.
      setStatus("idle", "App stopped");
      log("App stopped; not reconnecting");
      return;
    }
    log("App relay closed; reconnecting in 2s", "warn");
    setTimeout(() => { if (!localWs || localWs.readyState > 1) connectLocalWs(); }, 2000);
  };
  localWs.onerror = () => log("App relay error", "err");
  localWs.onmessage = (ev) => {
    if (typeof ev.data === "string") {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      handleServerMessage(msg);
    }
  };
}

function handleServerMessage(msg) {
  switch (msg.type) {
    case "status":
      if (msg.status === "connected") {
        resolveConnected();
        if (appMode === "file") {
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
        if (appMode === "file") {
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

// ---------- transcript rendering ----------

function textOf(msg) {
  return msg.text ?? msg.transcript ?? msg.message ?? null;
}

function formatSeconds(value) {
  return `${(Math.round(value * 10) / 10).toFixed(1)}s`;
}

function segmentMeta(msg) {
  // "[19:46:15] en · 6.5s audio · VAD −1.4s"
  const bits = [];
  if (msg.ts) bits.push(new Date(msg.ts).toLocaleTimeString());
  const lang = msg.language || "auto";
  bits.push(lang);
  if (typeof msg.duration === "number") bits.push(`${formatSeconds(msg.duration)} audio`);
  if (typeof msg.vad_removed === "number" && msg.vad_removed > 0) {
    bits.push(`VAD −${formatSeconds(msg.vad_removed)}`);
  }
  return `[${bits.join(" · ")}]`;
}

function appendTranscript(msg) {
  const text = textOf(msg);
  if (text === null || text === undefined || text === "") return;

  const isPartial = msg.partial === true || msg.is_final === false;
  if (isPartial) {
    if (!partialEl) {
      partialEl = document.createElement("span");
      partialEl.className = "partial";
      transcriptEl.appendChild(partialEl);
    }
    partialEl.textContent = text;
  } else {
    // Each completed utterance is its own paragraph on a new line.
    if (partialEl) {
      partialEl.remove();
      partialEl = null;
    }
    if (appMode === "file") {
      // File mode: one continuous paragraph, no per-utterance timestamps.
      appendFileText(text);
    } else {
      if (transcriptEl.firstElementChild?.classList.contains("muted")) {
        transcriptEl.innerHTML = "";
      }
      const seg = document.createElement("div");
      seg.className = "segment";
      // Translation follow-up messages match their segment on this ts.
      if (msg.ts) seg.dataset.ts = msg.ts;
      const meta = document.createElement("span");
      meta.className = "meta";
      meta.textContent = segmentMeta(msg);
      seg.appendChild(meta);
      seg.appendChild(document.createTextNode(` ${text.trim()}`));
      transcriptEl.appendChild(seg);
    }
  }
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
  refreshExportBtn();
}

// ---------- file-mode rendering (continuous paragraphs) ----------

// Text that ends in CJK (incl. CJK/full-width punctuation) — join without a
// space so Chinese output reads as one flowing paragraph.
const CJK_TAIL = /[\u2E80-\u9FFF\uF900-\uFAFF\u3000-\u303F\uFF00-\uFFEF]$/;

function appendFileText(text) {
  if (!fileTextEl) {
    if (transcriptEl.firstElementChild?.classList.contains("muted")) {
      transcriptEl.innerHTML = "";
    }
    fileTextEl = document.createElement("div");
    fileTextEl.className = "file-paragraph";
    transcriptEl.appendChild(fileTextEl);
  }
  const prev = fileTextEl.textContent;
  const sep = prev && !CJK_TAIL.test(prev) ? " " : "";
  fileTextEl.textContent = prev ? prev + sep + text.trim() : text.trim();
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
  refreshExportBtn();
}

function appendFileTranslation(text) {
  if (!fileTextEl) return; // transcript was cleared in the meantime
  if (!fileTrEl) {
    fileTrEl = document.createElement("div");
    fileTrEl.className = "file-translation";
    transcriptEl.appendChild(fileTrEl);
  }
  fileTrEl.textContent += (fileTrEl.textContent ? " " : "") + text;
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
  refreshExportBtn();
}

function setFileStatus(text, cls) {
  fileStatusEl.textContent = text || "";
  fileStatusEl.className = `file-status ${cls || ""}`.trim();
}

function setBusyControls(busy) {
  fileBusy = busy;
  modelSelect.disabled = busy;
  langSelect.disabled = busy;
  transcribeBtn.disabled = busy;
  browseBtn.disabled = busy;
  filePathEl.disabled = busy;
  transcribeBtn.textContent = busy ? "Processing…" : "Transcribe";
}

function resetTranscript() {
  partialEl = null;
  fileTextEl = null;
  fileTrEl = null;
  transcriptEl.innerHTML = '<span class="muted">Transcription will appear here…</span>';
  refreshExportBtn();
}

clearBtn.addEventListener("click", resetTranscript);

// ---------- transcript export ----------

function transcriptIsEmpty() {
  // Only the empty-state placeholder and/or the in-flight partial remain.
  return [...transcriptEl.children].every(
    (n) => n.classList.contains("partial") || n.classList.contains("muted"),
  );
}

function refreshExportBtn() {
  exportBtn.disabled = transcriptIsEmpty();
}
refreshExportBtn();

exportBtn.addEventListener("click", () => {
  const parts = [];
  for (const node of transcriptEl.children) {
    // Skip the empty-state placeholder and the tentative in-flight line.
    if (node.classList.contains("partial") || node.classList.contains("muted")) continue;
    const text = node.innerText.trim().split("\n").map((l) => l.trim()).join("\n");
    if (text) parts.push(text);
  }
  const content = parts.join("\n\n");
  if (!content) {
    log("Nothing to export yet — the transcript is empty", "warn");
    return;
  }
  const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const filename = `transcript-${stamp}.txt`;
  const url = URL.createObjectURL(
    new Blob([content + "\n"], { type: "text/plain;charset=utf-8" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a); // required for Firefox
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  log(`Transcript exported → ${filename}`);
});

// ---------- audio capture ----------

function resampleLinear(input, inRate, outRate) {
  if (inRate === outRate) return input;
  const ratio = inRate / outRate;
  const outLen = Math.max(0, Math.floor(input.length / ratio));
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const idx = i * ratio;
    const i0 = Math.floor(idx);
    const i1 = Math.min(i0 + 1, input.length - 1);
    const frac = idx - i0;
    out[i] = input[i0] * (1 - frac) + input[i1] * frac;
  }
  return out;
}

function floatTo16(input) {
  const out = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

async function startAudio() {
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      sampleRate: TARGET_SAMPLE_RATE,
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
    },
  });

  const AC = window.AudioContext || window.webkitAudioContext;
  // Browsers may ignore the sample rate and use 44.1/48 kHz instead —
  // we resample in onaudioprocess to 16 kHz either way.
  audioCtx = new AC({ sampleRate: TARGET_SAMPLE_RATE });
  if (audioCtx.state === "suspended") await audioCtx.resume();
  log(`AudioContext sample rate: ${audioCtx.sampleRate} Hz`);

  audioSource = audioCtx.createMediaStreamSource(mediaStream);
  scriptNode = audioCtx.createScriptProcessor(4096, 1, 1);
  scriptNode.onaudioprocess = (e) => {
    if (!running) return;
    if (localWs === null || localWs.readyState !== WebSocket.OPEN) return;
    const samples = resampleLinear(
      e.inputBuffer.getChannelData(0),
      audioCtx.sampleRate,
      TARGET_SAMPLE_RATE
    );
    if (samples.length === 0) return;
    const pcm16 = floatTo16(samples);
    localWs.send(pcm16.buffer);
  };

  // ScriptProcessor must be connected to the destination to run; mute it
  // to avoid feedback.
  const mute = audioCtx.createGain();
  mute.gain.value = 0;
  audioSource.connect(scriptNode);
  scriptNode.connect(mute);
  mute.connect(audioCtx.destination);
}

function stopAudio() {
  running = false;
  if (scriptNode) {
    scriptNode.onaudioprocess = null;
    try { scriptNode.disconnect(); } catch { /* ignore */ }
    scriptNode = null;
  }
  if (audioSource) {
    try { audioSource.disconnect(); } catch { /* ignore */ }
    audioSource = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  if (audioCtx) {
    audioCtx.close().catch(() => {});
    audioCtx = null;
  }
}

// ---------- file mode ----------

function ensureWsOpen() {
  return new Promise((resolve) => {
    if (localWs && localWs.readyState === WebSocket.OPEN) return resolve();
    connectLocalWs();
    const check = () => {
      if (localWs && localWs.readyState === WebSocket.OPEN) resolve();
      else setTimeout(check, 100);
    };
    check();
  });
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
    const connected = new Promise((resolve) => { connectedWaiter = resolve; });
    localWs.send(JSON.stringify({
      type: "start",
      model: modelSelect.value,
      language: langSelect.value,
      translate: translateChk.checked,
    }));
    await connected;
    if (!fileBusy) return; // an error arrived; already handled

    // Each file renders as fresh continuous paragraphs.
    resetTranscript();
    setFileStatus(`Transcribing ${filename}…`);
    localWs.send(JSON.stringify({ type: "file", path, filename }));
    log(`Submitting ${filename} for transcription (model=${modelSelect.value}, translate=${translateChk.checked})`);
  } catch (err) {
    log(`File transcription failed: ${err.message}`, "err");
    setStatus("error", "Error");
    setBusyControls(false);
    setFileStatus("Failed — see log.", "warn");
    if (localWs && localWs.readyState === WebSocket.OPEN) {
      localWs.send(JSON.stringify({ type: "stop" }));
    }
  }
}

function setMode(mode) {
  if (mode === appMode) return;
  if (fileBusy) {
    setFileStatus("Wait for the current file to finish (or press Stop).", "warn");
    return;
  }
  const isLive = mode === "live";
  appMode = mode;
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
  } else if (running) {
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

// ---------- start / stop ----------

async function start() {
  if (running) return;
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

  localWs.send(JSON.stringify({ type: "start", model, language, translate }));

  try {
    await startAudio();
    running = true;
    stopBtn.disabled = false;
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

function stopAll(reason) {
  if (!running && localWs && reason) {
    // Server-side failure: still release the relay session if it was open.
  }
  const wasRunning = running;
  stopAudio();
  if (wasRunning) {
    localWs?.send(JSON.stringify({ type: "stop" }));
  } else if (reason && localWs && localWs.readyState === WebSocket.OPEN) {
    // Whisper died mid-session: tell the backend to reset.
    localWs.send(JSON.stringify({ type: "stop" }));
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
  if (localWs && localWs.readyState === WebSocket.OPEN) {
    localWs.send(JSON.stringify({ type: "stop" }));
  }
});
translateChk.addEventListener("change", () => {
  // Toggle mid-session: applies to segments from now on.
  if (localWs && localWs.readyState === WebSocket.OPEN) {
    localWs.send(JSON.stringify({ type: "set_translate", translate: translateChk.checked }));
    log(`Translation to English ${translateChk.checked ? "enabled" : "disabled"}`);
  }
});

// ---------- quit the app (closes its start-script window) ----------

quitBtn.addEventListener("click", async () => {
  if (!confirm("Quit Whisper Live Transcriber?\nThe app will stop and its terminal window and browser tab will close.")) return;
  userQuit = true;
  if (running) stopAudio();
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
    userQuit = false;
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

connectLocalWs();
setStatus("idle", "Idle — press Start (mic permission will be requested)");

