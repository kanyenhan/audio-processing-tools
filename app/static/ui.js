// Transcript / log / status rendering + the transcript header controls
// (font size, meta toggle, export).  Pure DOM helpers: no websocket or
// audio knowledge here.

import {
  transcriptEl,
  statusDot,
  statusText,
  mascotEl,
  logEl,
  clearBtn,
  fontIncBtn,
  fontDecBtn,
  metaToggleChk,
  exportBtn,
  fileStatusEl,
  state,
} from "./state.js";

// ---------- logging / status ----------

export function log(message, cls = "") {
  const line = document.createElement("div");
  if (cls) line.className = cls;
  const ts = new Date().toLocaleTimeString();
  line.textContent = `[${ts}] ${message}`;
  logEl.appendChild(line);
  while (logEl.children.length > 60) logEl.removeChild(logEl.firstChild);
  logEl.scrollTop = logEl.scrollHeight;
}

export function setStatus(level, text) {
  statusDot.className = `dot ${level}`;
  statusText.textContent = text;
}

// ---------- transcript rendering ----------

export function textOf(msg) {
  return msg.text ?? msg.transcript ?? msg.message ?? null;
}

export function formatSeconds(value) {
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

let partialEl = null;

export function appendTranscript(msg) {
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
    if (state.appMode === "file") {
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
  pinLlmOutput();
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
  refreshExportBtn();
}

// Keep the LLM result (if any) pinned directly below the newest transcript
// content — new segments/paragraphs are appended after it as they arrive.
function pinLlmOutput() {
  if (state.llmOutputEl && state.llmOutputEl.parentNode === transcriptEl) {
    transcriptEl.appendChild(state.llmOutputEl);
  }
}

// ---------- file-mode rendering (continuous paragraphs) ----------

// Text that ends in CJK (incl. CJK/full-width punctuation) — join without a
// space so Chinese output reads as one flowing paragraph.
const CJK_TAIL = /[\u2E80-\u9FFF\uF900-\uFAFF\u3000-\u303F\uFF00-\uFFEF]$/;

let fileTextEl = null;
let fileTrEl = null;

export function appendFileText(text) {
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

export function appendFileTranslation(text) {
  if (!fileTextEl) return; // transcript was cleared in the meantime
  if (!fileTrEl) {
    fileTrEl = document.createElement("div");
    fileTrEl.className = "file-translation";
    transcriptEl.appendChild(fileTrEl);
  }
  fileTrEl.textContent += (fileTrEl.textContent ? " " : "") + text;
  pinLlmOutput();
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
  refreshExportBtn();
}

export function setFileStatus(text, cls) {
  fileStatusEl.textContent = text || "";
  fileStatusEl.className = `file-status ${cls || ""}`.trim();
}

export function resetTranscript() {
  partialEl = null;
  fileTextEl = null;
  fileTrEl = null;
  transcriptEl.innerHTML = '<span class="muted">Transcription will appear here…</span>';
  refreshExportBtn();
}

clearBtn.addEventListener("click", resetTranscript);

// ---------- transcript font size (−/+ pill in the transcript header) ----------
// All transcript font sizes derive from the --transcript-size CSS variable
// (see css/transcript.css), so adjusting it scales the whole panel
// proportionally.
const FONT_MIN_PX = 12;
const FONT_MAX_PX = 28;
const FONT_STEP_PX = 2;
let transcriptFontPx = 16; // matches the CSS default

function applyTranscriptFont() {
  document.documentElement.style.setProperty(
    "--transcript-size",
    `${transcriptFontPx}px`,
  );
  fontDecBtn.disabled = transcriptFontPx <= FONT_MIN_PX;
  fontIncBtn.disabled = transcriptFontPx >= FONT_MAX_PX;
}

fontDecBtn.addEventListener("click", () => {
  transcriptFontPx = Math.max(FONT_MIN_PX, transcriptFontPx - FONT_STEP_PX);
  applyTranscriptFont();
});

fontIncBtn.addEventListener("click", () => {
  transcriptFontPx = Math.min(FONT_MAX_PX, transcriptFontPx + FONT_STEP_PX);
  applyTranscriptFont();
});

applyTranscriptFont();

// ---------- per-segment meta line (timestamp · language · length · VAD) ----------
// The "Timestamps" switch in the transcript header toggles it.  When off, the
// transcript shows only the transcribed text: the .meta spans keep being
// created (so flipping back on restores the info for old segments too) but
// the body class below hides them via CSS.  Export picks this up for free —
// innerText skips display:none nodes, so a saved .txt then contains just the
// text.  The choice persists in localStorage like the other view settings.
const META_KEY = "whisper-live.meta-hidden";
let metaHidden = false;
try {
  metaHidden = localStorage.getItem(META_KEY) === "1";
} catch { /* ignore */ }

function applyMetaToggle() {
  document.body.classList.toggle("transcript-meta-hidden", metaHidden);
  metaToggleChk.checked = !metaHidden;
}

metaToggleChk.addEventListener("change", () => {
  metaHidden = !metaToggleChk.checked;
  try {
    localStorage.setItem(META_KEY, metaHidden ? "1" : "0");
  } catch { /* ignore */ }
  applyMetaToggle();
});

applyMetaToggle();

// ---------- transcript export ----------

function transcriptIsEmpty() {
  // Only the empty-state placeholder, the in-flight partial, and/or the LLM
  // result block (post-processing, not transcription) remain.
  return [...transcriptEl.children].every(
    (n) =>
      n.classList.contains("partial") ||
      n.classList.contains("muted") ||
      n.classList.contains("llm-output"),
  );
}

export function refreshExportBtn() {
  exportBtn.disabled = transcriptIsEmpty();
}
refreshExportBtn();

exportBtn.addEventListener("click", () => {
  const parts = [];
  for (const node of transcriptEl.children) {
    // Skip the empty-state placeholder, the tentative in-flight line, and
    // the LLM result (the export stays the transcription itself).
    if (
      node.classList.contains("partial") ||
      node.classList.contains("muted") ||
      node.classList.contains("llm-output")
    ) continue;
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

// ---------- mascot art (left gutter) ----------
// Static mascot.png at idle; while a transcription is in flight (live
// mic session or file processing) the src swaps to the looping
// mascot-active.gif.  Browsers loop GIFs natively, so the animation needs
// no extra code: the call sites (live-mode.js, file-mode.js, ws.js) just
// toggle it at their start/stop points.
const MASCOT_IDLE_SRC = "/static/assets/mascot.png";
const MASCOT_ACTIVE_SRC = "/static/assets/mascot-active.gif";

// Preload the animated version so it starts on the first frame of
// activity, not after a fetch.
new Image().src = MASCOT_ACTIVE_SRC;

export function setMascotActive(active) {
  const src = active ? MASCOT_ACTIVE_SRC : MASCOT_IDLE_SRC;
  if (mascotEl.getAttribute("src") !== src) mascotEl.setAttribute("src", src);
}

