// Shared DOM element references and mutable app state.
//
// Every feature module imports what it needs from here, so there is one
// source of truth for both and no module relies on script load order.
// The `state` object holds everything that changes at runtime; plain
// exported consts are DOM elements and fixed settings.

export const TARGET_SAMPLE_RATE = 16000;

// --- top bar ----------------------------------------------------------------
export const statusDot = document.getElementById("status-dot");
export const statusText = document.getElementById("status-text");
export const quitBtn = document.getElementById("quit-btn");

// --- mode bar + control panels ----------------------------------------------
export const modeLiveBtn = document.getElementById("mode-live");
export const modeFileBtn = document.getElementById("mode-file");
export const livePanel = document.getElementById("live-panel");
export const filePanel = document.getElementById("file-panel");
export const transcribePanel = document.getElementById("transcribe-panel");

// --- live mode controls -------------------------------------------------------
export const startBtn = document.getElementById("start-btn");
export const stopBtn = document.getElementById("stop-btn");
export const modelSelect = document.getElementById("model");
export const langSelect = document.getElementById("language");
export const translateChk = document.getElementById("translate-en");

// --- file mode controls --------------------------------------------------------
export const filePathEl = document.getElementById("file-path");
export const fileInput = document.getElementById("file-input");
export const browseBtn = document.getElementById("browse-btn");
export const transcribeBtn = document.getElementById("transcribe-btn");
export const fileStatusEl = document.getElementById("file-status");

// --- transcript panel ------------------------------------------------------------
export const transcriptEl = document.getElementById("transcript");
export const clearBtn = document.getElementById("clear-btn");
export const fontIncBtn = document.getElementById("font-inc-btn");
export const fontDecBtn = document.getElementById("font-dec-btn");
export const metaToggleChk = document.getElementById("meta-toggle");
export const exportBtn = document.getElementById("export-btn");

// --- log console -------------------------------------------------------------------
export const logEl = document.getElementById("log");
export const logsVisibleChk = document.getElementById("logs-visible");
export const splitterEl = document.getElementById("splitter");

// --- advanced panel (LLM post-processing) ---------------------------------------------
export const advancedBtn = document.getElementById("advanced-btn");
export const advancedPanel = document.getElementById("advanced-panel");
export const advancedCloseBtn = document.getElementById("advanced-close");
export const llmUrlEl = document.getElementById("llm-url");
export const llmModelSel = document.getElementById("llm-model");
export const llmLangSel = document.getElementById("llm-language");
export const llmFetchModelsBtn = document.getElementById("llm-fetch-models");
export const summarizeBtn = document.getElementById("summarize-btn");
export const translateBtn = document.getElementById("translate-btn");
export const customPromptEl = document.getElementById("llm-custom-prompt");
export const customPromptSubmitBtn = document.getElementById("custom-prompt-submit");
export const llmStatusEl = document.getElementById("llm-status");

// --- mutable state shared between modules ----------------------------------------------
export const state = {
  localWs: null,        // the local websocket to the FastAPI app
  running: false,       // a live mic session is active
  userQuit: false,      // true after the user pressed Quit — no reconnects
  appMode: "live",      // "live" | "file"
  connectedWaiter: null, // resolved when a "connected" status arrives
  // LLM result block, created on demand and pinned directly below the
  // newest transcript content (ui.js keeps it pinned, llm.js fills it).
  llmOutputEl: null,
};
