"""Live Whisper transcription web app.

Serves a browser UI that captures microphone audio as 16 kHz mono PCM16
and streams it over websockets to this FastAPI app.  Audio is segmented
into utterances (see app.live_buffer) and transcribed in-process with
whisper-ctranslate2 (CTranslate2 + faster-whisper, see app.transcriber).
Transcript results are sent back to the browser as JSON.

File mode: an audio file is either referenced by path (or uploaded to
POST /api/upload) and processed the same way — decoded server-side to
16 kHz mono PCM16, segmented, and streamed back utterance by utterance.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import time
from pathlib import Path

import numpy as np

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper.audio import decode_audio
from whisper_ctranslate2.commandline import MODEL_NAMES

from .live_buffer import SpeechSegmenter
from .transcriber import (
    DEFAULT_MODEL,
    can_translate,
    get_fallback_translator,
    translation_available,
    transcriber_cache,
)

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
        #: True while a file-mode transcription is in flight (one at a time).
        self._processing_file = False
        #: Number of segments successfully transcribed this session.
        self._processed = 0

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
            result = await self.transcriber.transcribe(segment, self.language)
        except Exception as exc:  # noqa: BLE001
            log.warning("Transcription failed: %s", exc)
            return
        text = result.get("text")
        if not text:
            return
        self._processed += 1
        language = result.get("language") or self.language
        ts = int(time.time() * 1000)
        message = {
            "text": text,
            "is_final": True,
            "language": language,
            "duration": result.get("duration", round(duration, 3)),
            "vad_removed": result.get("vad_removed"),
            "ts": ts,
        }
        # Send the transcript immediately; the (slower) English translation
        # follows as a {"type": "translation"} message once ready.  The two
        # passes use different model instances, so while large-v3 translates
        # this segment, turbo is already transcribing the next one.
        try:
            await self.browser_ws.send_json(message)
        except Exception:
            log.info("Browser gone; stopping session")
            self._running = False
            return
        if self.translate and (language or "").lower() != "en":
            translation = await self._translate_segment(language, segment)
            if translation:
                try:
                    await self.browser_ws.send_json(
                        {"type": "translation", "ts": ts, "text": translation}
                    )
                except Exception:
                    log.info("Browser gone; stopping session")
                    self._running = False

    async def _translate_segment(
        self, language: str, segment
    ) -> str | None:
        """Run the translate pass, picking the right model for it.

        Uses the session's own model when it supports translate, otherwise
        the WHISPER_LIVE_TRANSLATE_MODEL fallback (turbo has no translation
        decoder).  Runs on the translator's own lock, so it overlaps with
        transcription of other segments.
        """
        try:
            if can_translate(self.transcriber.model):
                translator = self.transcriber
            else:
                translator = await get_fallback_translator()
                if translator is None:
                    return None
            return await translator.translate(language, segment)
        except Exception as exc:  # noqa: BLE001
            log.warning("Translation failed: %s", exc)
            return None


#: Root for files picked via the UI's "Browse" button; swept on each upload.
UPLOAD_DIR = Path(tempfile.gettempdir()) / "whisper-live-uploads"
UPLOAD_MAX_AGE_S = 3600  # delete uploaded files older than 1 hour


def _decode_to_pcm16(path: str) -> bytes:
    """Decode an audio file to 16 kHz mono PCM16 bytes (what the
    segmenter expects).  faster-whisper's ``decode_audio`` resamples to
    16 kHz mono and returns float32 in [-1, 1], so convert to int16."""
    audio = decode_audio(path, sampling_rate=16000)
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def _sweep_uploads() -> None:
    """Remove uploaded files that have not been used recently."""
    try:
        cutoff = time.time() - UPLOAD_MAX_AGE_S
        for old in UPLOAD_DIR.glob("upload_*"):
            if old.is_file() and old.stat().st_mtime < cutoff:
                old.unlink(missing_ok=True)
    except OSError:
        pass


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)) -> JSONResponse:
    """Save a browser-picked audio file to a temp file for file-mode
    transcription.  Returns the server-side path; the browser then sends
    ``{"type": "file", "path": …}`` over the websocket like a typed path."""
    _sweep_uploads()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "audio").suffix.lower()
    dest = UPLOAD_DIR / f"upload_{int(time.time() * 1000)}{suffix}"
    dest.write_bytes(await file.read())
    return JSONResponse({"path": str(dest), "filename": file.filename or dest.name})


async def transcribe_file(
    browser_ws: WebSocket,
    session: "TranscriptionSession",
    path: str,
    filename: str,
) -> None:
    """Decode an audio file and run it through the session's existing
    pipeline: 16 kHz decode → SpeechSegmenter → per-segment _process (so
    transcript + pipelined translation follow-ups stream back exactly like
    live mode), ending with a ``file_done`` message."""
    file = Path(path).expanduser()
    if not file.is_file():
        await browser_ws.send_json(
            {"type": "error", "message": f"File not found: {path}"}
        )
        return
    if session._processing_file:
        await browser_ws.send_json(
            {
                "type": "warning",
                "message": "Another file is already being processed.",
            }
        )
        return
    session._processing_file = True
    try:
        await browser_ws.send_json(
            {
                "type": "status",
                "status": "processing_file",
                "message": f"Decoding {filename}…",
            }
        )
        # Decode (and resample to 16 kHz mono PCM16) in a worker thread.
        try:
            pcm16 = await asyncio.to_thread(_decode_to_pcm16, str(file))
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to decode %s: %s", path, exc)
            await browser_ws.send_json(
                {
                    "type": "error",
                    "message": (
                        f"Could not decode {filename!r}: {exc} "
                        "(expected wav, mp3, m4a, flac, ogg, …)"
                    ),
                }
            )
            return
        duration_s = len(pcm16) / 2 / 16000.0
        log.info(
            "Transcribing file %s (%.1fs, model=%s)",
            filename, duration_s, session.transcriber.model,
        )
        # Feed the segmenter in chunks (it re-blocks internally); each
        # emitted utterance is processed concurrently, same as live mode.
        chunk = 256 * 1024
        pending: list[asyncio.Task] = []
        for i in range(0, len(pcm16), chunk):
            if not session._running:
                break
            for segment in session.segmenter.feed(pcm16[i : i + chunk]):
                pending.append(asyncio.ensure_future(session._process(segment)))
        if session._running:
            # Transcribe any utterance still open at end of file.
            remaining = session.segmenter.flush()
            if remaining is not None:
                pending.append(asyncio.ensure_future(session._process(remaining)))
        # Wait for all segments (and their translations) so file_done is
        # accurate and no transcript message arrives after it.
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await browser_ws.send_json(
            {
                "type": "file_done",
                "filename": filename,
                "duration_s": round(duration_s, 1),
                "segments": session._processed,
            }
        )
        if session._processed == 0:
            await browser_ws.send_json(
                {
                    "type": "warning",
                    "message": f"No speech detected in {filename}.",
                }
            )
    finally:
        session._processing_file = False


@app.websocket("/ws")
async def browser_ws_endpoint(browser_ws: WebSocket) -> None:
    """Browser-side websocket: control messages + streaming audio.

    Browser -> app messages:
      {"type": "start", "model": "turbo", "language": "auto"|code,
       "translate": bool}
      {"type": "file", "path": "/path/or/upload/audio.mp3",
       "filename": "audio.mp3"}   (file mode; after a successful start)
      {"type": "stop"}
      {"type": "set_translate", "translate": bool}
      binary frames (PCM16 16kHz mono audio chunks; live mode)
    App -> browser messages:
      {"type": "status", "status": "loading_model"|"connected"|"disconnected"|"idle"|
       "processing_file", ...}
      {"type": "error"|"warning", "message": "..."}
      {"type": "file_done", "filename": "a.mp3", "duration_s": 61.2,
       "segments": 7}
       (file mode: sent once the whole file has been fed through the
       pipeline; the per-utterance messages below keep streaming)
      {"text": "...", "is_final": true, "language": "en", "duration": 6.5,
       "vad_removed": 1.4, "ts": 1794106083000}
       (one per completed utterance; sent as soon as the fast transcribe
       pass finishes)
      {"type": "translation", "ts": 1794106083000, "text": "<English>"}
       (follow-up, only for non-English segments when translation is
       enabled; ts matches the segment it belongs to and arrives once the
       slower translate pass completes)
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

                if translate and not can_translate(model):
                    # Preload the fallback translator (if configured) before
                    # starting the session so the first segment's translation
                    # doesn't pay the model-load time.
                    try:
                        await get_fallback_translator()
                    except Exception as exc:  # noqa: BLE001
                        log.warning("Fallback translator preload failed: %s", exc)
                        await browser_ws.send_json(
                            {
                                "type": "warning",
                                "message": (
                                    "Translation model failed to load; "
                                    "translations will be retried per segment."
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

            elif mtype == "file":
                # File mode: transcribe (and translate) an audio file that
                # was either referenced by path or uploaded via /api/upload.
                # Must follow a successful "start" (same model settings).
                if session is None:
                    await browser_ws.send_json(
                        {"type": "error", "message": "No session; send start first."}
                    )
                    continue
                path = str(data.get("path") or "").strip()
                if not path:
                    await browser_ws.send_json(
                        {"type": "error", "message": "Missing file path."}
                    )
                    continue
                filename = str(data.get("filename") or Path(path).name)
                # Run in its own task so further messages (stop, set_translate)
                # are still served while the file is being processed.
                asyncio.ensure_future(
                    transcribe_file(browser_ws, session, path, filename)
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

