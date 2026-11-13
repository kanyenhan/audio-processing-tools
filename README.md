# audio-processing-tools

AI assisted tools for transcribing or translating audio through clips or live audio stream.

## Whisper Live Transcriber (web app)

A Python web app that captures microphone audio in the browser (16 kHz mono
PCM16), streams it via websocket to a local FastAPI app, and transcribes it
**in-process** with [`whisper-ctranslate2`](https://pypi.org/project/whisper-ctranslate2/)
(CTranslate2 + faster-whisper — up to 4x faster than openai/whisper at the
same accuracy while using less memory).

### Features

- **Start / Stop** button to begin and end transcription
- **Model selector** (`turbo` recommended, plus `large-v3`, `medium`,
  `small`, `base`, `tiny`) — all run 100% locally via CTranslate2
  (int8 on CPU)
- **Language selector** (default: auto-detect)
- **Translate to English** toggle (can be flipped mid-session): non-English
  segments show the English translation below the original transcript
  - Chinese output is normalized to Traditional via OpenCC (`s2t`), and a
    prompt nudges Whisper into emitting full-width Chinese punctuation
  - `turbo` has no translation decoder, so it can't translate natively.
    Either pick a non-turbo model, or set
    `WHISPER_LIVE_TRANSLATE_MODEL=large-v3` (any translate-capable model) to
    use a second model for the translation pass while turbo does the
    transcription; without it the UI warns instead of translating
   - Transcription and translation are **pipelined**: each pass runs on its
     own model instance (separate lock, worker thread), so the original
     transcript appears as soon as the fast pass finishes and its translation
     arrives ~1 s later as a follow-up while the next utterance is already
     being transcribed. The fallback model is preloaded at session start so
     the first segment doesn't pay the load time
- **File mode**: transcribe an audio file instead of the mic — type a path
  or use **Browse** (the file is uploaded to the app), with the same
  Model / Language / Translate-to-English settings. The file is decoded
  server-side (PyAV/ffmpeg: wav, mp3, m4a, flac, ogg, …) and runs through
  the same segmentation + VAD pipeline; transcripts stream back utterance
  by utterance and render as **continuous paragraphs** (no per-utterance
  timestamps), with the English translation as a second continuous
  paragraph. A typed path is read by the local app, so it can point at any
  file on your own machine
- **Live utterance segmentation**: audio is buffered in 30 ms blocks and
  segmented on speech activity (RMS volume threshold + vocal-frequency
  check), so the model only runs on actual utterances
- **Quit** button in the header (top-right): gracefully stops the app
  (`SIGTERM` — the same as `Ctrl-C` in the terminal) and closes the app's
  browser tab (on macOS, via AppleScript; see "Dedicated window" below)
- **Advanced** panel (header, left of Quit): slide-out panel that sends the
  transcript to a local OpenAI-compatible LLM (e.g. LM Studio at
  `http://127.0.0.1:1234/v1`) - enter the server URL, fetch its model list,
  pick a model, choose the **output language** (the same options as the main
  page's Language dropdown; applies to both Summarize and Translate), then
  Summarize or Translate the transcript
- **Live transcript** text area fed by the transcriber
- **Resizable transcript / log split**: drag the divider between the two
  panels to resize them; a **Logs** slide toggle in the bottom-right corner
  shows/hides the log console (hidden by default)
- No audio is stored — frames are processed in memory only

### Requirements

- Python 3.10+ (tested on 3.14; CTranslate2 ships arm64/x86_64 wheels for
  macOS, Linux and Windows)
- `whisper-ctranslate2` (installed via requirements.txt)

### Run

**One-click start** — creates `.venv` (if missing), installs requirements,
starts the app and opens the page in your browser:

| Platform | Script |
| --- | --- |
| macOS | `scripts/start-macos.sh` |
| Linux | `scripts/start-linux.sh` |
| Windows | `scripts\start-windows.bat` (double-click) |

**Dedicated window (auto-closing)**: on macOS and Linux the start script
opens the app in its own terminal window (a new Terminal / gnome-terminal
window) and prints the logs there. When the app quits — **Quit** button or
`Ctrl-C` — the window closes automatically after a short farewell. This
relies on your terminal's "close the window when the shell exits" setting
being at its default. A `WHISPER_LIVE_DEDICATED` guard prevents relaunch
loops (re-running the script just opens one fresh window). On Windows a
double-clicked console window already closes on its own when the bat ends.

**Browser tab auto-close**: browsers only let a page close a tab it opened
itself (`window.close()` is blocked for ordinary tabs), so the macOS start
script closes the app's tab from the shell via AppleScript — any tab whose
URL is the app's URL, in whichever of the common browsers (Safari, Chrome,
Chromium, Brave, Edge, Vivaldi) are already running. Nothing launches, and
a missing browser or a denied permission is skipped silently. The first
time, macOS shows a one-time Automation prompt ("Terminal is trying to
control \<browser\>") — click **Allow** once. On Linux/Windows, or when
running `uvicorn` directly (bypassing the start script), the tab is left
open showing "App stopped" and can be closed manually.

> Caveat: if you run `uvicorn` directly in your own terminal (bypassing the
> start script), **Quit** stops the app and returns you to your shell
> prompt — we intentionally never kill a user-owned shell.

Manual:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --host 127.0.0.1 --port 8080
```

Open **http://localhost:8080**, pick a model (and language if desired),
allow microphone access, and press **Start**.

> Note: `localhost` is a secure context, so browser microphone access works
> over plain `http://`.

**First run downloads the model** (e.g. `turbo` ≈ 1.6 GB) from Hugging Face
into `.cache/whisper-ctranslate2/` (gitignored); the UI shows a
“Loading model” status until it is ready. Subsequent starts load it from
cache.

### How it works

```
Browser (getUserMedia → AudioContext → 16kHz PCM16)
   │  ws (binary audio up, JSON transcripts down)
   ▼
FastAPI app (app/main.py)
   │  SpeechSegmenter (app/live_buffer.py)  — 30ms blocks, RMS + vocal-band
   │  gating, emits utterance on ~2s silence / 30s max
   ▼
whisper-ctranslate2 (app/transcriber.py)   — CTranslate2 + faster-whisper,
   int8 on CPU, VAD filter (silero), one asyncio lock per model instance
   (transcribe/translate overlap), each inference runs in a worker thread

File mode: the file is decoded to 16 kHz mono (faster-whisper's
decode_audio / PyAV) and fed through the same SpeechSegmenter, so the
whole pipeline (segmentation, VAD, parallel transcribe + pipelined
translation) is identical to live mode — only the rendering differs
(continuous paragraphs instead of timestamped segments).
```

- Browser → app: `{"type":"start","model":"turbo","language":...}`,
  `{"type":"stop"}`, and binary PCM16 frames (~4 KB / ~250 ms chunks).
- App → browser: `{"type":"status", ...}` (`loading_model`, `connected`,
  `idle`, `disconnected`), `{"type":"error"|"warning", ...}`, plus one
  `{"text": "...", "is_final": true, "language": "en", "duration": 6.5,
  "vad_removed": 1.4, "ts": <epoch ms>}` per completed utterance — rendered
  in the UI as separate paragraphs, each prefixed with
  `[HH:MM:SS] lang · Xs audio · VAD −Ys`. When translation is enabled and
  the utterance is non-English, the transcript is sent as soon as the fast
  transcribe pass finishes and the English translation follows a moment
  later as `{"type": "translation", "ts": <same epoch ms>, "text": "..."}`,
  which the UI attaches in place under its segment.
- File mode: after `start` (with the chosen model settings) the browser
  sends `{"type": "file", "path": ..., "filename": ...}` — Browse first
  uploads via `POST /api/upload`, which returns the temp-file path — and
  receives one `{"type": "file_done", "filename": ..., "duration_s": ...,
  "segments": N}` once the whole file has been processed (the per-utterance
  messages above keep streaming until then).

The live/VAD tuning follows
[Run Local Speech-to-Text Transcription](https://simonlermen.substack.com/p/run-local-speech-to-text-transcription):

| Setting | Value | Where |
| --- | --- | --- |
| Volume (speech) threshold | `0.02` | `app/live_buffer.py` (`DEFAULT_VOLUME_THRESHOLD`) |
| VAD filter | on (Silero) | `app/transcriber.py` (`live_options()`) |
| Min speech duration | `1500 ms` | both (segments shorter are discarded) |
| Min silence duration | `900 ms` | `app/transcriber.py` |
| Max speech duration | `30 s` | both (longer utterances are flushed) |

### Tests

```bash
# Fake transcriber (no model download) + real segmenter, end-to-end via websocket
WHISPER_LIVE_FAKE_TRANSCRIBER=1 python tests/smoke_test.py
```

Set `WHISPER_LIVE_FAKE_TRANSCRIBER=1` any time you want to exercise the app
without loading a model.


