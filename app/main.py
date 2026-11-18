"""Live Whisper transcription web app (FastAPI entry point).

Serves a browser UI that captures microphone audio as 16 kHz mono PCM16
and streams it over websockets to this app.  Audio is segmented into
utterances (app.live_buffer) and transcribed in-process with
whisper-ctranslate2 (CTranslate2 + faster-whisper, app.transcriber).
Transcript results are sent back to the browser as JSON.

File mode: an audio file is either referenced by path (or uploaded to
POST /api/upload) and processed the same way — decoded server-side to
16 kHz mono PCM16, segmented, and streamed back utterance by utterance.

Module layout:
  app.transcriber   in-process Whisper backend (model cache, VAD options)
  app.live_buffer   speech segmentation of streamed PCM16 audio
  app.session       per-browser session: live loop + file-mode pipeline
  app.file_audio    audio-file decoding + upload bookkeeping
  app.llm           LLM post-processing proxy helpers (advanced panel)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from whisper_ctranslate2.commandline import MODEL_NAMES

from .file_audio import UPLOAD_DIR, sweep_uploads
from .llm import (
    LLM_LANGUAGES,
    http_json,
    llm_prompt,
    normalize_llm_base_url,
)
from .session import TranscriptionSession, transcribe_file
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


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)) -> JSONResponse:
    """Save a browser-picked audio file to a temp file for file-mode
    transcription.  Returns the server-side path; the browser then sends
    ``{"type": "file", "path": …}`` over the websocket like a typed path."""
    sweep_uploads()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "audio").suffix.lower()
    dest = UPLOAD_DIR / f"upload_{int(time.time() * 1000)}{suffix}"
    dest.write_bytes(await file.read())
    return JSONResponse({"path": str(dest), "filename": file.filename or dest.name})


# ---------------------------------------------------------------------------
# Advanced: LLM post-processing of the transcript (summarize / translate)
# via any local OpenAI-compatible server (e.g. LM Studio at 127.0.0.1:1234).
# The app proxies the calls so the browser never makes cross-origin requests
# to the LLM server (no CORS concerns) and the UI stays single-origin.
# Helpers live in app.llm.
# ---------------------------------------------------------------------------


@app.get("/api/llm/models")
async def llm_models(url: str) -> JSONResponse:
    """Proxy the OpenAI-compatible ``/models`` endpoint of a local LLM
    server so the UI can offer its models as a drop-down."""
    base = normalize_llm_base_url(url)
    try:
        data = await asyncio.to_thread(http_json, "GET", f"{base}/models")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"Cannot reach LLM server at {base}: {exc}"
        ) from exc
    models = [str(m.get("id")) for m in data.get("data", []) if m.get("id")]
    log.info("LLM models from %s: %d", base, len(models))
    return JSONResponse({"models": models})


@app.post("/api/llm/run")
async def llm_run(payload: dict) -> JSONResponse:
    """Send the transcript to the local LLM with a summarize/translate
    prompt and return its reply.  Generous timeout: a local large model
    can take minutes to generate."""
    action = str(payload.get("action") or "").strip()
    model = str(payload.get("model") or "").strip()
    text = str(payload.get("text") or "").strip()
    language = str(payload.get("language") or "auto").strip() or "auto"
    if action not in ("summarize", "translate", "custom"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown action {action!r}; use summarize, translate or custom.",
        )
    custom_prompt = str(payload.get("prompt") or "").strip()
    if action == "custom" and not custom_prompt:
        raise HTTPException(status_code=400, detail="Missing custom prompt.")
    if language != "auto" and language not in LLM_LANGUAGES:
        raise HTTPException(
            status_code=400, detail=f"Unknown language {language!r}."
        )
    if not model:
        raise HTTPException(status_code=400, detail="Missing LLM model.")
    if not text:
        raise HTTPException(status_code=400, detail="Transcript is empty.")
    base = normalize_llm_base_url(str(payload.get("url") or ""))
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": llm_prompt(action, language, custom_prompt)},
            {"role": "user", "content": f"Transcription:\n{text}"},
        ],
        "temperature": 0.3,
        "stream": False,
    }
    try:
        data = await asyncio.to_thread(
            http_json, "POST", f"{base}/chat/completions", body, 600.0
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc
    try:
        result_text = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        raise HTTPException(
            status_code=502, detail=f"Unexpected LLM response: {str(data)[:200]}"
        ) from None
    log.info(
        "LLM %s via %s/%s (lang=%s): %d chars in, %d out",
        action, base, model, language, len(text), len(result_text),
    )
    return JSONResponse({"action": action, "model": model, "text": result_text})


def self_terminate() -> None:
    """Let the /api/quit response flush, then SIGTERM our own process:
    uvicorn shuts down gracefully (flushes in-flight responses, closes
    websockets, runs lifespan cleanup) and the start script's terminal
    window closes automatically once this process exits."""
    time.sleep(0.2)
    os.kill(os.getpid(), signal.SIGTERM)


# --- dedicated-window auto-stop -------------------------------------------
# When started via the start scripts, the app runs in a dedicated terminal
# window together with its own dedicated browser window.  The page keeps a
# websocket open for its whole lifetime (and reconnects within ~2 s on a
# reload), so "no browser connection left" is a reliable signal that the
# browser window was closed: stop the app so the terminal window closes too.
DEDICATED_MODE = os.environ.get("WHISPER_LIVE_DEDICATED", "") == "1"
#: How long to wait after the last disconnect before quitting.  The browser
#: client reconnects after 2 s, so a page reload or transient blip is
#: cancelled out; a genuinely closed window has nothing coming back.
BROWSER_CLOSE_GRACE_S = 5.0

_browser_conns = 0
_shutting_down = False
_close_watcher: asyncio.Task | None = None


def _cancel_close_watcher() -> None:
    global _close_watcher
    if _close_watcher is not None and not _close_watcher.done():
        _close_watcher.cancel()
    _close_watcher = None


async def _close_watcher_loop() -> None:
    global _shutting_down
    await asyncio.sleep(BROWSER_CLOSE_GRACE_S)
    if _shutting_down or _browser_conns > 0:
        return
    log.info(
        "Browser window closed (no connections after %.0fs grace) — "
        "stopping the app; the dedicated terminal window will close",
        BROWSER_CLOSE_GRACE_S,
    )
    _shutting_down = True
    # uvicorn's SIGTERM handler runs on the event loop (add_signal_handler),
    # so sending the signal from this task is safe and triggers the same
    # graceful shutdown as the Quit button / Ctrl-C.
    os.kill(os.getpid(), signal.SIGTERM)


def _note_browser_left() -> None:
    """Called after a browser websocket ends.  In dedicated mode, arm the
    quit-on-idle watcher when this was the last connection; the watcher
    re-checks the connection count when the grace period expires, so a
    reconnect within the window (page reload, quick reopen) aborts it."""
    if not DEDICATED_MODE or _shutting_down or _browser_conns > 0:
        _cancel_close_watcher()
        return
    global _close_watcher
    if _close_watcher is None or _close_watcher.done():
        _close_watcher = asyncio.create_task(_close_watcher_loop())


@app.post("/api/quit")
async def quit_app() -> JSONResponse:
    """Stop the app from the UI; the start script's terminal window
    closes automatically once this process exits."""
    global _shutting_down
    _shutting_down = True  # the shutdown-triggered disconnects must not
    # re-arm the close watcher (see _note_browser_left)
    log.info("Quit requested from UI — shutting down")
    return JSONResponse({"status": "quitting"},
                        background=BackgroundTask(self_terminate))


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
    global _browser_conns
    await browser_ws.accept()
    _browser_conns += 1
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
        _browser_conns -= 1
        _note_browser_left()
