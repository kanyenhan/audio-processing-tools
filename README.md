# audio-processing-tools

AI assisted tools for transcribing or translating audio through clips or live audio stream.

## Whisper Live Transcriber (web app)

A Python web app that captures microphone audio in the browser (16 kHz mono
PCM16), streams it via websocket through a local FastAPI relay to a locally
running Whisper (large-v3) websocket server, and displays transcript JSON
replies live in the UI.

### Features

- **Start / Stop** button to begin and end transcription
- **Whisper server URL** field (default `ws://127.0.0.1:1025/ws`)
- **Language selector** (default: auto-detect) — sent to the server as a
  `{"language": "..."}` frame when not auto
- **Live transcript** text area fed by the whisper server's websocket replies
- No audio is stored — frames are relayed in memory only

### Requirements

- Python 3.10+
- A locally running Whisper websocket server that accepts **raw binary
  PCM16, 16 kHz, mono** audio frames and replies with **JSON** transcript
  messages (e.g. `{"text": "...", "is_final": true}`).

### Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# optional: start the mock whisper server (no real model needed)
python tests/mock_whisper_server.py &

uvicorn app.main:app --host 127.0.0.1 --port 8080
```

Open **http://localhost:8080**, set the Whisper URL (and language if desired),
allow microphone access, and press **Start**.

> Note: `localhost` is a secure context, so browser microphone access works
> over plain `http://`.

### How it works

```
Browser (getUserMedia → AudioContext → 16kHz PCM16)
   │  ws (binary audio up, JSON transcripts down)
   ▼
FastAPI app (app/main.py)  ── /ws endpoint, pure relay
   │  outbound ws client
   ▼
Local whisper large-v3 server (user-configurable URL)
```

- Browser → app: `{"type":"start","whisper_url":...,"language":...}`,
  `{"type":"stop"}`, and binary PCM16 frames (~4 KB / ~250 ms chunks).
- App → browser: `{"type":"status", ...}`, `{"type":"error"|"warning", ...}`,
  plus transcript messages relayed unchanged.
- If the server does not accept the JSON language frame, the app reconnects
  and warns instead of failing (audio-only servers still work).

### Tests

```bash
python tests/smoke_test.py   # boots mock server + app, verifies relay end-to-end
```

