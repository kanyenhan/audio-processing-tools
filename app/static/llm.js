// Advanced panel: LLM post-processing of the transcript (summarize /
// translate / custom prompt) via a local OpenAI-compatible server.
// Calls go through the app's /api/llm/* proxy endpoints (no CORS).

import {
  state,
  transcriptEl,
  advancedBtn,
  advancedPanel,
  advancedCloseBtn,
  llmUrlEl,
  llmModelSel,
  llmLangSel,
  llmFetchModelsBtn,
  summarizeBtn,
  translateBtn,
  customPromptEl,
  customPromptSubmitBtn,
  llmStatusEl,
  langSelect,
} from "./state.js";
import { log } from "./ui.js";

const LLM_URL_KEY = "whisper-live.llm-url";
const LLM_MODEL_KEY = "whisper-live.llm-model";
const LLM_LANG_KEY = "whisper-live.llm-language";
let llmBusy = false;

// The advanced panel's language dropdown must offer exactly the same
// options as the main page's #language select — copy them once at startup
// (the main list is the single source of truth).
function syncLlmLanguageOptions() {
  llmLangSel.innerHTML = "";
  for (const opt of langSelect.options) {
    const copy = document.createElement("option");
    copy.value = opt.value;
    copy.textContent = opt.textContent;
    llmLangSel.appendChild(copy);
  }
}
syncLlmLanguageOptions();

try {
  const savedUrl = localStorage.getItem(LLM_URL_KEY);
  if (savedUrl) llmUrlEl.value = savedUrl;
  const savedLang = localStorage.getItem(LLM_LANG_KEY);
  if (savedLang && llmLangSel.querySelector(`option[value="${savedLang}"]`)) {
    llmLangSel.value = savedLang;
  }
} catch { /* ignore (private mode etc.) */ }

function setAdvancedOpen(open) {
  advancedPanel.classList.toggle("open", open);
  advancedPanel.setAttribute("aria-hidden", String(!open));
  // Row class: CSS uses it to square the transcript's right corners
  // while the panel is open
  advancedPanel.parentElement.classList.toggle("adv-open", open);
}

function setLlmStatus(text, cls) {
  llmStatusEl.textContent = text || "";
  llmStatusEl.className = ("file-status " + (cls || "")).trim();
}

advancedBtn.addEventListener("click", () => {
  setAdvancedOpen(!advancedPanel.classList.contains("open"));
});
advancedCloseBtn.addEventListener("click", () => setAdvancedOpen(false));

// Show the LLM reply directly under the original transcription.  The block
// is created once and reused for subsequent runs (its content is replaced).
// ui.js keeps it pinned below the newest transcript content.
function showLlmOutput(text, label) {
  if (!state.llmOutputEl) {
    state.llmOutputEl = document.createElement("div");
    state.llmOutputEl.className = "llm-output";
    const head = document.createElement("div");
    head.className = "llm-output-label";
    const body = document.createElement("div");
    body.className = "llm-output-text";
    state.llmOutputEl.appendChild(head);
    state.llmOutputEl.appendChild(body);
  }
  state.llmOutputEl.querySelector(".llm-output-label").textContent = label;
  state.llmOutputEl.querySelector(".llm-output-text").textContent = text;
  transcriptEl.appendChild(state.llmOutputEl);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

// Raw transcript text for the LLM: live-mode segments without their meta
// line and without whisper translation children, or the file-mode continuous
// paragraph.  Skips the in-flight partial and the empty placeholder.
function collectTranscriptText() {
  const parts = [];
  for (const node of transcriptEl.children) {
    if (node.classList.contains("partial") || node.classList.contains("muted")) continue;
    if (node.classList.contains("file-translation")) continue;
    if (node.classList.contains("llm-output")) continue; // don't feed LLM results back in
    let text;
    if (node.classList.contains("segment")) {
      const clone = node.cloneNode(true);
      clone.querySelectorAll(".meta, .translation").forEach((el) => el.remove());
      text = clone.textContent.trim();
    } else {
      text = node.textContent.trim();
    }
    if (text) parts.push(text);
  }
  return parts.join("\n\n");
}

async function fetchLlmModels() {
  const url = llmUrlEl.value.trim();
  if (!url) {
    setLlmStatus("Enter the LLM API URL first.", "warn");
    return;
  }
  try { localStorage.setItem(LLM_URL_KEY, url); } catch { /* ignore */ }
  llmFetchModelsBtn.disabled = true;
  setLlmStatus("Fetching model list...");
  try {
    const res = await fetch("/api/llm/models?url=" + encodeURIComponent(url));
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || "HTTP " + res.status);
    const models = data.models || [];
    if (!models.length) throw new Error("Server returned no models.");
    llmModelSel.innerHTML = "";
    for (const id of models) {
      const opt = document.createElement("option");
      opt.value = id;
      opt.textContent = id;
      llmModelSel.appendChild(opt);
    }
    try {
      const saved = localStorage.getItem(LLM_MODEL_KEY);
      if (saved && models.includes(saved)) llmModelSel.value = saved;
    } catch { /* ignore */ }
    setLlmStatus(models.length + " model(s) found.");
    log("LLM: " + models.length + " model(s) from " + url);
  } catch (err) {
    setLlmStatus("Fetch failed: " + err.message, "err");
    log("LLM model fetch failed: " + err.message, "err");
  } finally {
    llmFetchModelsBtn.disabled = false;
  }
}

// action: "summarize" | "translate" | "custom".  For "custom" the prompt
// (from the collapsible Custom prompt field) is sent to the model verbatim
// together with the transcript.
async function runLlmAction(action, customPrompt) {
  if (llmBusy) return;
  const url = llmUrlEl.value.trim();
  const model = llmModelSel.value;
  if (!url) {
    setLlmStatus("Enter the LLM API URL first.", "warn");
    return;
  }
  if (!model) {
    setLlmStatus("Fetch the model list and pick a model first.", "warn");
    return;
  }
  customPrompt = (customPrompt || "").trim();
  if (action === "custom" && !customPrompt) {
    setLlmStatus("Enter a custom prompt first.", "warn");
    return;
  }
  const text = collectTranscriptText();
  if (!text) {
    setLlmStatus("The transcript is empty - nothing to process.", "warn");
    return;
  }
  llmBusy = true;
  summarizeBtn.disabled = true;
  translateBtn.disabled = true;
  customPromptSubmitBtn.disabled = true;
  if (state.llmOutputEl) state.llmOutputEl.remove(); // a new result replaces the old one
  const verb = action === "summarize"
    ? "Summarizing"
    : action === "translate"
      ? "Translating"
      : "Running custom prompt";
  setLlmStatus(verb + " with " + model + "... (local models can take a while)");
  log("LLM " + action + ": " + text.length + " chars to " + model + " @ " + url);
  const started = Date.now();
  try {
    const res = await fetch("/api/llm/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url, model, action, text,
        language: llmLangSel.value,
        prompt: customPrompt,
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || "HTTP " + res.status);
    const secs = ((Date.now() - started) / 1000).toFixed(1);
    const label = (action === "summarize" ? "Summary"
      : action === "translate" ? "Translation"
      : "Custom prompt") + " · " + model;
    showLlmOutput((data.text || "").trim() || "(empty response)", label);
    setLlmStatus("Done in " + secs + "s.");
    log("LLM " + action + " done (" + secs + "s)");
  } catch (err) {
    setLlmStatus(action + " failed: " + err.message, "err");
    log("LLM " + action + " failed: " + err.message, "err");
  } finally {
    llmBusy = false;
    summarizeBtn.disabled = false;
    translateBtn.disabled = false;
    customPromptSubmitBtn.disabled = false;
  }
}

summarizeBtn.addEventListener("click", () => runLlmAction("summarize"));
translateBtn.addEventListener("click", () => runLlmAction("translate"));
customPromptSubmitBtn.addEventListener("click", () =>
  runLlmAction("custom", customPromptEl.value),
);

// ---------- custom prompt presets (Pirate / Caveman / Rapper) ----------
// One-click "character" profiles that sit above the Custom prompt box.  A
// click fills the box with a ready-made instruction so the local model
// rewrites the transcription in that voice; the user can still edit the text
// before pressing Submit.  The backend sends the custom prompt verbatim as the
// system message, so each preset fully describes the persona and the task.
const PROMPT_PRESETS = {
  pirate:
    "You are a charming, grizzled pirate — an old salt of the seven seas. " +
    "Rewrite the transcription the user provides entirely in a pirate's " +
    "voice: use nautical slang (\"arr\", \"matey\", \"ye scurvy dog\", \"me " +
    "hearty\", \"the briny deep\", \"plunder\", \"landlubber\"), a " +
    "swashbuckling tone and the occasional sea-shanty flourish. Keep the " +
    "underlying meaning, facts and names intact. Reply with the pirate-style " +
    "rewrite only — no preamble, no commentary.",
  caveman:
    "You are a grunting prehistoric caveman. Rewrite the transcription the " +
    "user provides the way a caveman would speak: short, blunt, simple words " +
    "(\"me\", \"we\", \"stone\", \"fire\", \"big\", \"good\"), minimal grammar and " +
    "the occasional \"Ugh\", but keep the underlying meaning clear enough to " +
    "understand. Reply with the caveman-style rewrite only — no preamble, no " +
    "commentary.",
  rapper:
    "You are a confident modern rapper with a smooth flow. Turn the " +
    "transcription the user provides into a rap: give it a clear rhyme " +
    "scheme, rhythm and swagger, add wordplay and a few punchlines where they " +
    "fit, but keep the underlying meaning and facts intact. Reply with the " +
    "rap only — no preamble, no commentary.",
};

// The preset buttons live in the static HTML above the prompt box; clicking
// one fills the box and marks it as the active profile.
const presetBtns = document.querySelectorAll(".prompt-preset");
function clearPresetActive() {
  for (const b of presetBtns) b.classList.remove("active");
}
for (const btn of presetBtns) {
  btn.addEventListener("click", () => {
    const preset = PROMPT_PRESETS[btn.dataset.preset];
    if (!preset) return;
    customPromptEl.value = preset;
    clearPresetActive();
    btn.classList.add("active");
    customPromptEl.focus();
  });
}
// Editing the box by hand means no preset is "applied" any more.
customPromptEl.addEventListener("input", clearPresetActive);

llmFetchModelsBtn.addEventListener("click", fetchLlmModels);
llmModelSel.addEventListener("change", () => {
  try {
    if (llmModelSel.value) localStorage.setItem(LLM_MODEL_KEY, llmModelSel.value);
  } catch { /* ignore */ }
});
llmLangSel.addEventListener("change", () => {
  try {
    localStorage.setItem(LLM_LANG_KEY, llmLangSel.value);
  } catch { /* ignore */ }
});
