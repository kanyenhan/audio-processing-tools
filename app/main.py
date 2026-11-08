"""Live Whisper transcription web app.

Serves a browser UI that captures microphone audio as 16 kHz mono PCM16
and streams it over websockets to this FastAPI app.  Audio is segmented
into utterances (see app.live_buffer) and transcribed in-process with
whisper-ctranslate2 (CTranslate2 + faster-whisper, see app.transcriber).
Transcript results are sent back to the browser as JSON.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from whisper_ctranslate2.commandline import MODEL_NAMES

from .live_buffer import SpeechSegmenter
from .transcriber import DEFAULT_MODEL, transcriber_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("whisper-live")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"

app = FastAPI(title="Whisper Live Transcriber")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(TEMPLATE_DIR / "index.html")


class TranscriptionSession:
    """Buffers incoming PCM16 audio, segments it into utterances, and
    transcribes each utterance with the in-process whisper-ctranslate2
    backend, pushing results back over the browser websocket."""

    def __init__(
        self,
        browser_ws: WebSocket,
        transcriber,
        language: str | None,
    ) -> None:
        self.browser_ws = browser_ws
        self.transcriber = transcriber
        self.language = language
        self.segmenter = SpeechSegmenter()
        self._running = False
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        self._task = None
        # Transcribe any utterance still in the segmenter's buffer.
        remaining = self.segmenter.flush()
        if remaining is not None:
            asyncio.ensure_future(self._process(remaining))

    def feed(self, pcm_bytes: bytes) -> None:
        """Called for every binary frame from the browser."""
        if not self._running:
            return
        for segment in self.segmenter.feed(pcm_bytes):
            asyncio.ensure_future(self._process(segment))

    async def _loop(self) -> None:
        """Keeps the session alive until stopped."""
        try:
            while self._running:
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass

    async def _process(self, segment) -> None:
        duration = len(segment) / 16000.0
        log.info(
            "Transcribing %.1fs segment (model=%s)", duration, self.transcriber.model
        )
        try:
            text = await self.transcriber.transcribe(segment, self.language)
        except Exception as exc:  # noqa: BLE001
            log.warning("Transcription failed: %s", exc)
            return
        if not text:
            return
        try:
            await self.browser_ws.send_json({"text": text, "is_final": True})
        except Exception:
            log.info("Browser gone; stopping session")
            self._running = False


@app.websocket("/ws")
async def browser_ws_endpoint(browser_ws: WebSocket) -> None:
    """Browser-side websocket: control messages + streaming audio.

    Browser -> app messages:
      {"type": "start", "model": "turbo", "language": "auto"|code}
      {"type": "stop"}
      binary frames (PCM16 16kHz mono audio chunks)
    App -> browser messages:
      {"type": "status", "status": "loading_model"|"connected"|"disconnected"|"idle", ...}
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
                model = (data.get("model") or DEFAULT_MODEL).strip()
                language = (data.get("language") or "auto").strip()

                if model not in MODEL_NAMES:
                    await browser_ws.send_json(
                        {
                            "type": "error",
                            "message": (
                                f"Unknown model {model!r}. Available: "
                                + ", ".join(MODEL_NAMES)
                            ),
                        }
                    )
                    continue

                if session is not None:
                    await session.stop()
                    session = None

                transcriber = await transcriber_cache.get(model)

                if not transcriber.ready:
                    await browser_ws.send_json(
                        {
                            "type": "status",
                            "status": "loading_model",
                            "model": model,
                            "message": "Loading model (first run downloads it)…",
                        }
                    )
                    try:
                        await transcriber.ensure_loaded()
                    except Exception as exc:  # noqa: BLE001
                        log.warning("Model load failed: %s", exc)
                        await browser_ws.send_json(
                            {
                                "type": "error",
                                "message": f"Failed to load model {model!r}: {exc}",
                            }
                        )
                        continue

                log.info("Session started (model=%s, language=%s)", model, language)
                await browser_ws.send_json(
                    {
                        "type": "status",
                        "status": "connected",
                        "model": model,
                        "url": f"whisper-ctranslate2/{model}",
                    }
                )

                session = TranscriptionSession(browser_ws, transcriber, language)
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

