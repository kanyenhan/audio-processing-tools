// Whisper Live Transcriber — browser client (ES module entry point).
//
// Captures microphone audio as 16 kHz mono PCM16 and streams it to the
// local FastAPI app, which transcribes it in-process via
// whisper-ctranslate2.  Feature modules (each wires its own DOM listeners
// at import time):
//
//   state.js     shared DOM element refs + mutable app state
//   ui.js        transcript / log / status rendering, transcript header
//   ws.js        local websocket + server-message dispatch
//   audio.js     microphone capture (16 kHz mono PCM16)
//   live-mode.js live mic session: start/stop
//   file-mode.js file transcription (typed path or Browse upload)
//   llm.js       advanced panel: LLM post-processing
//   chrome.js    Quit button, log console, splitter

import "./live-mode.js";
import "./file-mode.js";
import "./llm.js";
import "./chrome.js";

import { connectLocalWs } from "./ws.js";
import { setStatus } from "./ui.js";

connectLocalWs();
setStatus("idle", "Idle — press Start (mic permission will be requested)");
