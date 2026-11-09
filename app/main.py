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
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from whisper_ctranslate2.commandline import MODEL_NAMES

from .live_buffer import SpeechSegmenter
from .transcriber import DEFAULT_MODEL, translation_available, transcriber_cache

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
        translate: bool = False,
    ) -> None:
        self.browser_ws = browser_ws
        self.transcriber = transcriber
        self.language = language
        #: When enabled, non-English segments are also translated to English.
        self.translate = translate
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
            result = await self.transcriber.transcribe(
                segment, self.language, translate=self.translate
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Transcription failed: %s", exc)
            return
        text = result.get("text")
        if not text:
            return
        message = {
            "text": text,
            "is_final": True,
            "language": result.get("language") or self.language,
            "duration": result.get("duration", round(duration, 3)),
            "vad_removed": result.get("vad_removed"),
            "ts": int(time.time() * 1000),
        }
        if result.get("translation"):
            message["translation"] = result["translation"]
        try:
            await self.browser_ws.send_json(message)
        except Exception:
            log.info("Browser gone; stopping session")
            self._running = False


@app.websocket("/ws")
async def browser_ws_endpoint(browser_ws: WebSocket) -> None:
    """Browser-side websocket: control messages + streaming audio.

    Browser -> app messages:
      {"type": "start", "model": "turbo", "language": "auto"|code,
       "translate": bool}
      {"type": "stop"}
      {"type": "set_translate", "translate": bool}
      binary frames (PCM16 16kHz mono audio chunks)
    App -> browser messages:
      {"type": "status", "status": "loading_model"|"connected"|"disconnected"|"idle", ...}
      {"type": "error"|"warning", "message": "..."}
      {"text": "...", "is_final": true, "language": "en", "duration": 6.5,
       "vad_removed": 1.4, "translation": "<English, only for non-English
       segments when translation is enabled>", "ts": 1794106083000}
       (one per completed utterance)
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
                translate = bool(data.get("translate", False))

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

                if translate and not translation_available(model):
                    await browser_ws.send_json(
                        {
                            "type": "warning",
                            "message": (
                                f"Model {model!r} cannot translate to English. "
                                "Pick a non-turbo model, or set "
                                "WHISPER_LIVE_TRANSLATE_MODEL (e.g. large-v3) "
                                "and restart the app."
                            ),
                        }
                    )

                log.info(
                    "Session started (model=%s, language=%s, translate=%s)",
                    model, language, translate,
                )
                await browser_ws.send_json(
                    {
                        "type": "status",
                        "status": "connected",
                        "model": model,
                        "url": f"whisper-ctranslate2/{model}",
                    }
                )

                session = TranscriptionSession(
                    browser_ws, transcriber, language, translate=translate
                )
                session.start()

            elif mtype == "set_translate":
                if session is not None:
                    session.translate = bool(data.get("translate", False))
                    log.info("Translation to English %s",
                             "enabled" if session.translate else "disabled")
                    if session.translate and not translation_available(
                        session.transcriber.model
                    ):
                        await browser_ws.send_json(
                            {
                                "type": "warning",
                                "message": (
                                    f"Model {session.transcriber.model!r} cannot "
                                    "translate to English. Pick a non-turbo model, "
                                    "or set WHISPER_LIVE_TRANSLATE_MODEL (e.g. "
                                    "large-v3) and restart the app."
                                ),
                            }
                        )

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

