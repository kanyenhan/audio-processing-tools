// Microphone capture: 16 kHz mono PCM16 frames pushed to the local
// websocket, with linear resampling from the AudioContext's native rate
// (browsers may ignore the requested sample rate).

import { state, TARGET_SAMPLE_RATE } from "./state.js";

// Audio graph nodes for the current capture session (all null when idle).
let audioCtx = null;
let mediaStream = null;
let audioSource = null;
let scriptNode = null;
import { log } from "./ui.js";

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

export async function startAudio() {
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
    if (!state.running) return;
    if (state.localWs === null || state.localWs.readyState !== WebSocket.OPEN) return;
    const samples = resampleLinear(
      e.inputBuffer.getChannelData(0),
      audioCtx.sampleRate,
      TARGET_SAMPLE_RATE
    );
    if (samples.length === 0) return;
    const pcm16 = floatTo16(samples);
    state.localWs.send(pcm16.buffer);
  };

  // ScriptProcessor must be connected to the destination to run; mute it
  // to avoid feedback.
  const mute = audioCtx.createGain();
  mute.gain.value = 0;
  audioSource.connect(scriptNode);
  scriptNode.connect(mute);
  mute.connect(audioCtx.destination);
}

export function stopAudio() {
  state.running = false;
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
