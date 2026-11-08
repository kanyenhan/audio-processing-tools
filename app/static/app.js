// Whisper Live Transcriber — browser client.
// Captures microphone audio as 16 kHz mono PCM16 and streams it to the
// local FastAPI app, which transcribes it in-process via whisper-ctranslate2.

const TARGET_SAMPLE_RATE = 16000;

const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const clearBtn = document.getElementById("clear-btn");
const modelSelect = document.getElementById("model");
const langSelect = document.getElementById("language");
const transcriptEl = document.getElementById("transcript");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const logEl = document.getElementById("log");

let localWs = null;
let running = false;
let committedText = "";
let partialEl = null;

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

// ---------- local websocket (browser <-> FastAPI app) ----------

function connectLocalWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  localWs = new WebSocket(`${proto}://${location.host}/ws`);
  localWs.binaryType = "arraybuffer";

  localWs.onopen = () => log("Connected to app relay");
  localWs.onclose = () => {
    log("App relay closed; reconnecting in 2s", "warn");
    if (running) stopAudio();
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
        setStatus("recording", `Recording → ${msg.url}`);
        log(`Transcribing locally with ${msg.url}`);
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
      setStatus("error", "Error");
      log(msg.message, "err");
      stopAll(msg.message);
      break;
    case "warning":
      log(msg.message, "warn");
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
    // Final transcript: commit it and drop any pending partial.
    if (partialEl) {
      partialEl.remove();
      partialEl = null;
    }
    if (committedText === "" && transcriptEl.firstElementChild?.classList.contains("muted")) {
      transcriptEl.innerHTML = "";
    }
    committedText += text;
    transcriptEl.appendChild(document.createTextNode(text));
  }
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

clearBtn.addEventListener("click", () => {
  committedText = "";
  partialEl = null;
  transcriptEl.innerHTML = '<span class="muted">Transcription will appear here…</span>';
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

// ---------- start / stop ----------

async function start() {
  if (running) return;
  const model = modelSelect.value;
  const language = langSelect.value;

  startBtn.disabled = true;
  modelSelect.disabled = true;
  langSelect.disabled = true;
  setStatus("connecting", "Connecting…");

  if (!localWs || localWs.readyState === WebSocket.CLOSED) {
    connectLocalWs();
    // Wait for the socket to open before sending the start command.
    await new Promise((resolve) => {
      const check = () => {
        if (localWs && localWs.readyState === WebSocket.OPEN) resolve();
        else setTimeout(check, 100);
      };
      check();
    });
  }

  localWs.send(JSON.stringify({ type: "start", model, language }));

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
}

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", () => stopAll(undefined));

connectLocalWs();
setStatus("idle", "Idle — press Start (mic permission will be requested)");

