"""Per-browser transcription session and file-mode pipeline.

A session buffers incoming PCM16 audio (live mic frames, or chunks of a
decoded file), segments it into utterances (see app.live_buffer) and
transcribes each utterance with the in-process whisper-ctranslate2
backend (see app.transcriber), pushing results back over the browser
websocket.  File mode reuses the exact same per-utterance pipeline, so
transcript + pipelined-translation messages stream back just like live
mode (see ``transcribe_file``).
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from fastapi import WebSocket

from .file_audio import decode_to_pcm16
from .live_buffer import SpeechSegmenter
from .transcriber import can_translate, get_fallback_translator

log = logging.getLogger("whisper-live")


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


async def transcribe_file(
    browser_ws: WebSocket,
    session: TranscriptionSession,
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
            pcm16 = await asyncio.to_thread(decode_to_pcm16, str(file))
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
