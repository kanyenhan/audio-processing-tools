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
- **Live utterance segmentation**: audio is buffered in 30 ms blocks and
  segmented on speech activity (RMS volume threshold + vocal-frequency
  check), so the model only runs on actual utterances
- **Live transcript** text area fed by the transcriber
- No audio is stored — frames are processed in memory only

### Requirements

- Python 3.10+ (tested on 3.14; CTranslate2 ships arm64/x86_64 wheels for
  macOS, Linux and Windows)
- `whisper-ctranslate2` (installed via requirements.txt)

### Run

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


