"""Live Whisper transcription web app.

Serves a browser UI that captures microphone audio as 16 kHz mono PCM16
and relays it, over websockets, to a local FastAPI relay which then
transcribes it via the LocalAI REST API. Transcript JSON replies are
relayed back to the browser.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import struct
from pathlib import Path

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("whisper-live")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"

app = FastAPI(title="Whisper Live Transcriber")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Audio constants (must match the browser client)
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2   # PCM16
CHANNELS = 1
MIN_AUDIO_SECONDS = 0.5
MIN_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * CHANNELS * MIN_AUDIO_SECONDS)
TRANSCRIBE_TIMEOUT = 60.0


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(TEMPLATE_DIR / "index.html")


def pcm16_to_wav(pcm: bytes) -> bytes:
    """Wrap raw PCM16 mono data in a WAV container (in memory)."""
    data_size = len(pcm)
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack(
        "<IHHIIHH",
        16,                      # fmt chunk size
        1,                       # audio format (1 = PCM)
        CHANNELS,                # channels
        SAMPLE_RATE,             # sample rate
        SAMPLE_RATE * BYTES_PER_SAMPLE * CHANNELS,  # byte rate
        BYTES_PER_SAMPLE * CHANNELS,  # block align
        16,                      # bits per sample
    ))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm)
    return buf.getvalue()


class TranscriptionSession:
    """Buffers incoming PCM16 audio and periodically transcribes it
    via the LocalAI REST API, pushing results back over the browser
    websocket."""

    def __init__(
        self,
        browser_ws: WebSocket,
        base_url: str,
        model: str,
        language: str | None,
    ) -> None:
        self.browser_ws = browser_ws
        self.base_url = base_url.rstrip("/")
        self.transcribe_url = f"{self.base_url}/v1/audio/transcriptions"
        self.model = model
        self.language = language
        self._buffer = bytearray()
        self._event = asyncio.Event()
        self._running = False
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._running = False
        self._event.set()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        self._task = None
        self._buffer.clear()

    def feed(self, pcm_bytes: bytes) -> None:
        """Called for every binary frame from the browser."""
        if self._running:
            self._buffer.extend(pcm_bytes)
            self._event.set()

    async def _loop(self) -> None:
        async with httpx.AsyncClient(timeout=TRANSCRIBE_TIMEOUT) as client:
            while self._running:
                try:
                    await asyncio.wait_for(self._event.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                self._event.clear()

                if not self._running:
                    break

                if len(self._buffer) < MIN_BYTES:
                    continue

                chunk = bytes(self._buffer)
                self._buffer.clear()

                wav = pcm16_to_wav(chunk)
                duration = len(chunk) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
                log.info(
                    "Transcribing %.1fs audio (%d bytes WAV) via %s model=%s",
                    duration, len(wav), self.transcribe_url, self.model,
                )

                result = await self._transcribe(client, wav)
                if result is not None:
                    try:
                        await self.browser_ws.send_json(result)
                    except Exception:
                        log.info("Browser gone; stopping session")
                        self._running = False
                        break

    async def _transcribe(
        self, client: httpx.AsyncClient, wav: bytes
    ) -> dict | None:
        form: dict[str, str] = {"model": self.model}
        if self.language and self.language != "auto":
            form["language"] = self.language

        try:
            resp = await client.post(
                self.transcribe_url,
                files={"file": ("audio.wav", wav, "audio/wav")},
                data=form,
            )
            resp.raise_for_status()
            body = resp.json()
            text = body.get("text", "")
            if text:
                return {"text": text, "is_final": True}
        except httpx.HTTPStatusError as exc:
            log.warning(
                "LocalAI HTTP %s: %s",
                exc.response.status_code, exc.response.text[:300],
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Transcription request failed: %s", exc)
        return None

@app.websocket("/ws")
async def browser_ws_endpoint(browser_ws: WebSocket) -> None:
    """Browser-side websocket: control messages + audio relay.

    Browser -> app messages:
      {"type": "start", "whisper_url": "http://...", "language": "auto"|code, "model": "whisper-large"}
      {"type": "stop"}
      binary frames (PCM16 16kHz mono audio chunks)
    App -> browser messages:
      {"type": "status", "status": "connected"|"disconnected"|"idle", ...}
      {"type": "error"|"warning", "message": "..."}
      {"text": "...", "is_final": true}  (transcription results)
    """
    await browser_ws.accept()
    session: TranscriptionSession | None = None
    log.info("Browser connected")
    try:
        while True:
            message = await browser_ws.receive()

            if "bytes" in message and message["bytes"] is not None:
                if session is not None:
                    session.feed(message["bytes"])
                continue

            if "text" not in message or message["text"] is None:
                break

            try:
                data = json.loads(message["text"])
            except json.JSONDecodeError:
                continue

            mtype = data.get("type")

            if mtype == "start":
                base_url = (data.get("whisper_url") or "http://localhost:8080").strip()
                model = (data.get("model") or "whisper-large").strip()
                language = (data.get("language") or "auto").strip()

                if session is not None:
                    await session.stop()
                    session = None

                log.info("Connecting to LocalAI at %s (model=%s)", base_url, model)

                try:
                    async with httpx.AsyncClient(timeout=5.0) as check:
                        resp = await check.get(f"{base_url.rstrip('/')}/v1/models")
                        if resp.status_code != 200:
                            raise ConnectionError(
                                f"LocalAI returned HTTP {resp.status_code}"
                            )
                except Exception as exc:
                    await browser_ws.send_json(
                        {
                            "type": "error",
                            "message": f"Cannot reach LocalAI at {base_url}: {exc}",
                        }
                    )
                    continue

                await browser_ws.send_json(
                    {"type": "status", "status": "connected", "url": base_url}
                )

                session = TranscriptionSession(browser_ws, base_url, model, language)
                session.start()

            elif mtype == "stop":
                if session is not None:
                    await session.stop()
                    session = None
                await browser_ws.send_json(
                    {"type": "status", "status": "idle", "message": "Stopped"}
                )
            else:
                await browser_ws.send_json(
                    {"type": "error", "message": f"Unknown message type: {mtype}"}
                )
    except WebSocketDisconnect:
        log.info("Browser disconnected")
    except Exception as exc:  # noqa: BLE001
        log.warning("Browser websocket error: %s", exc)
    finally:
        if session is not None:
            await session.stop()


