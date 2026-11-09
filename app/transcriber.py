"""In-process Whisper transcription backend built on whisper-ctranslate2.

``whisper-ctranslate2`` wraps CTranslate2 + faster-whisper (up to 4x faster
than openai/whisper at the same accuracy).  A single model is loaded per
model name and shared across sessions; inference is serialized with an
async lock because the CTranslate2 transcriber is not safe for concurrent
use, and each blocking ``inference()`` call runs in a worker thread so the
FastAPI event loop stays responsive.

VAD / live tuning follows the settings recommended in
https://simonlermen.substack.com/p/run-local-speech-to-text-transcription :
  --live_volume_threshold 0.02 (handled by app.live_buffer.SpeechSegmenter)
  --vad_filter True
  --vad_min_speech_duration_ms 1500
  --vad_min_silence_duration_ms 900
  --vad_max_speech_duration_s 30
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
from pathlib import Path

import numpy as np
from whisper_ctranslate2.transcribe import Transcribe, TranscriptionOptions

log = logging.getLogger("whisper-live")

DEFAULT_MODEL = "turbo"  # best accuracy/speed on CPU (see article)
DEFAULT_COMPUTE_TYPE = "int8"  # best CPU performance per the package README
DEFAULT_DEVICE = "cpu"  # macOS Apple Silicon: CTranslate2 uses Accelerate
DEFAULT_CPU_THREADS = 8

#: Cache models under the repo's .cache/whisper-ctranslate2 (gitignored).
MODEL_CACHE_DIR = str(
    Path(__file__).resolve().parent.parent / ".cache" / "whisper-ctranslate2"
)

_FAKE = os.environ.get("WHISPER_LIVE_FAKE_TRANSCRIBER", "").lower() in (
    "1", "true", "yes",
)

#: Whisper's turbo (large-v3-turbo) dropped the translation decoder, so it
#: only supports transcribe — its translate task silently returns empty text.
#: When set, this model is used (lazily loaded, shared) for the translate
#: pass instead, e.g. WHISPER_LIVE_TRANSLATE_MODEL=large-v3.
TRANSLATE_FALLBACK_MODEL = os.environ.get(
    "WHISPER_LIVE_TRANSLATE_MODEL", ""
).strip()


def can_translate(model: str) -> bool:
    """Whether ``model`` supports the Whisper translate task (turbo does not)."""
    return "turbo" not in model.lower()


def translation_available(model: str) -> bool:
    """Whether an English translation can be produced for ``model``:
    natively, or via the WHISPER_LIVE_TRANSLATE_MODEL fallback."""
    return can_translate(model) or bool(TRANSLATE_FALLBACK_MODEL)


_FAKE_WORDS = ["this", "is", "a", "mock", "transcription", "from", "the", "test"]
_fake_counter = 0


def _parse_timestamp(value: str) -> float:
    """Parse a faster-whisper ``format_timestamp`` string (MM:SS.mmm or
    HH:MM:SS.mmm) into seconds."""
    seconds = 0.0
    for part in value.split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


class _VadLogHandler(logging.Handler):
    """Capture faster-whisper's ``VAD filter removed …`` log line.

    faster-whisper (used by whisper-ctranslate2) only reports the VAD
    result through its ``faster_whisper`` logger, so we tap it here.
    Inference runs in worker threads, so the captured value is kept in
    thread-local storage.
    """

    def emit(self, record: logging.LogRecord) -> None:
        if not record.getMessage().startswith("VAD filter removed"):
            return
        match = re.search(r"(\d+(?::\d+)*\.\d+)", record.getMessage())
        if match:
            try:
                _vad_capture.vad_removed = _parse_timestamp(match.group(1))
            except ValueError:
                pass


_s2t_converter: "OpenCC | None" = None
_s2t_lock = threading.Lock()


def _get_s2t() -> "OpenCC":
    """Lazily create the shared Simplified -> Traditional OpenCC converter.

    Created on first use (and only for zh segments) so importing this module
    or running in fake-transcriber mode never pays the initialization cost.
    OpenCC converters are read-only after construction, so sharing one
    instance across inference worker threads is safe.
    """
    global _s2t_converter
    if _s2t_converter is None:
        with _s2t_lock:
            if _s2t_converter is None:
                from opencc import OpenCC  # opencc-python-reimplemented

                _s2t_converter = OpenCC("s2t")
    return _s2t_converter


def to_traditional(text: str, language: str | None) -> str:
    """Convert Simplified Chinese to Traditional when the segment is zh.

    whisper-ctranslate2 cannot be told to emit Traditional Chinese: the
    ``zh`` language code covers both variants and the model tends to output
    Simplified.  ``s2t`` is a near-identity mapping on already-Traditional
    text, so converting unconditionally for ``zh`` is safe.
    """
    if (language or "").lower() == "zh" and text:
        return _get_s2t().convert(text)
    return text


_vad_capture = threading.local()
# faster-whisper logs VAD results through the "faster_whisper" logger; attach
# the capture handler directly to it (and pin its level) so the lines are not
# gated out by the root/uvicorn log level.
_whisper_logger = logging.getLogger("faster_whisper")
if not any(isinstance(h, _VadLogHandler) for h in _whisper_logger.handlers):
    _whisper_logger.addHandler(_VadLogHandler())
_whisper_logger.setLevel(logging.INFO)


#: Whisper rarely emits punctuation on its own; an initial prompt nudges the
#: output to include full-width Chinese punctuation when the language is zh
#: (kept in Traditional to match the s2t-converted transcript).  The prompt
#: is conditional in its own wording, so it is safe to send for every
#: language, including auto-detect.
INITIAL_PROMPT = "如果語言是中文(zh)，請用全形標點符號，例如：，。：「」！？等，不使用半形標點如：,.:!?[]"


def live_options() -> TranscriptionOptions:
    """Transcription options tuned for live streaming (see module docstring)."""
    return TranscriptionOptions(
        beam_size=5,
        best_of=5,
        patience=1.0,
        length_penalty=1.0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        condition_on_previous_text=False,  # live: avoid looping on prior output
        prompt_reset_on_temperature=0.5,
        temperature=(0.0,),
        initial_prompt=INITIAL_PROMPT,
        prefix=None,
        hotwords=None,
        suppress_blank=True,
        suppress_tokens=[-1],
        word_timestamps=False,
        print_colors=False,
        prepend_punctuations='"' '‘“¿([{-',
        append_punctuations='"' ".。,，!！?？:：”)]}、",
        hallucination_silence_threshold=None,
        vad_filter=True,
        vad_threshold=None,
        vad_min_speech_duration_ms=1500,
        vad_max_speech_duration_s=30,
        vad_min_silence_duration_ms=900,
        multilingual=True,
    )



class LocalTranscriber:
    """Wraps whisper-ctranslate2's Transcribe for the live web app."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = DEFAULT_DEVICE,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
        cpu_threads: int = DEFAULT_CPU_THREADS,
        cache_dir: str = MODEL_CACHE_DIR,
        local_files_only: bool = False,
    ) -> None:
        self.model = model
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self._transcribe: Transcribe | None = None
        self._fake_ready = False
        self._lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return self._transcribe is not None or self._fake_ready

    async def ensure_loaded(self) -> None:
        """Load (and if needed download) the model.  Idempotent."""
        if self.ready:
            return
        if _FAKE:
            log.warning("Fake transcriber active (WHISPER_LIVE_FAKE_TRANSCRIBER)")
            self._fake_ready = True
            return
        log.info(
            "Loading whisper-ctranslate2 model %r (device=%s, compute_type=%s, "
            "cache=%s)… first run downloads it from Hugging Face",
            self.model, self.device, self.compute_type, self.cache_dir,
        )
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        self._transcribe = Transcribe(
            self.model,
            self.device,
            0,          # device_index
            self.compute_type,
            self.cpu_threads,
            self.cache_dir,
            self.local_files_only,
            False,      # batched
        )
        log.info("Model %r ready", self.model)

    async def transcribe(self, audio: np.ndarray, language: str | None) -> dict:
        """Transcribe one segment (transcribe task only — fast path).

        Returns ``{"text", "language", "duration", "vad_removed"}`` where
        ``language`` is the detected (or requested) ISO code and
        ``vad_removed`` is the number of seconds the VAD filter dropped
        (None when unknown or in fake mode).  English translation is a
        separate pass (:meth:`translate`) so callers can show the transcript
        immediately and stream the translation once it is ready — the two
        passes run on different model instances and therefore overlap
        (turbo transcribing the next utterance while large-v3 translates the
        previous one).
        """
        async with self._lock:
            return await asyncio.to_thread(self._infer_sync, audio, language)

    async def translate(
        self, language: str | None, audio: np.ndarray
    ) -> str | None:
        """Run the translate-to-English pass for one segment with this model.

        Independent of :meth:`transcribe` (separate lock; possibly a
        separate model instance, see :data:`TRANSLATE_FALLBACK_MODEL`), so
        it can run concurrently.  Returns None for English input or when
        this model cannot translate; in fake mode it returns a placeholder
        so the transcribe/translate protocol can be exercised end-to-end
        without models.
        """
        if not can_translate(self.model):
            return None
        if (language or "").lower() == "en":
            return None
        async with self._lock:
            return await asyncio.to_thread(self._translate_sync, audio)

    def _infer_sync(self, audio: np.ndarray, language: str | None) -> dict:
        global _fake_counter
        duration = round(len(audio) / 16000.0, 3)
        if _FAKE:
            _fake_counter += 1
            return {
                "text": f"{_FAKE_WORDS[_fake_counter % len(_FAKE_WORDS)]} ",
                "language": language if language and language != "auto" else None,
                "duration": duration,
                "vad_removed": None,
            }
        assert self._transcribe is not None, "model not loaded"
        _vad_capture.vad_removed = None
        lang = language if language and language != "auto" else None
        result = self._transcribe.inference(
            audio=audio.flatten().astype("float32"),
            task="transcribe",
            language=lang,
            verbose=False,
            live=True,
            options=live_options(),
        )
        return {
            "text": to_traditional(
                (result.get("text") or "").strip(), result.get("language")
            ),
            "language": result.get("language") or lang,
            "duration": duration,
            "vad_removed": getattr(_vad_capture, "vad_removed", None),
        }

    def _translate_sync(self, audio: np.ndarray) -> str | None:
        if _FAKE:
            return f"[en] fake translation #{_fake_counter}"
        assert self._transcribe is not None, "model not loaded"
        # No initial prompt here: the Chinese punctuation prompt is only
        # useful for the transcribe pass and pulls the translate output back
        # toward Chinese.
        translated = self._transcribe.inference(
            audio=audio.flatten().astype("float32"),
            task="translate",
            language="en",
            verbose=False,
            live=True,
            options=live_options()._replace(initial_prompt=None),
        )
        return (translated.get("text") or "").strip() or None


_fallback_translator: "LocalTranscriber | None" = None
_fallback_lock = asyncio.Lock()


async def get_fallback_translator() -> "LocalTranscriber | None":
    """Lazily load the WHISPER_LIVE_TRANSLATE_MODEL translator (shared).

    Returns None when the fallback is not configured, in which case callers
    skip the translate pass and the UI warns instead.  In fake mode the
    loaded instance does no real work, so the transcribe/translate protocol
    can be tested end-to-end without models.
    """
    global _fallback_translator
    if not TRANSLATE_FALLBACK_MODEL:
        return None
    if _fallback_translator is None:
        async with _fallback_lock:
            if _fallback_translator is None:
                log.info(
                    "Loading fallback translation model %r "
                    "(WHISPER_LIVE_TRANSLATE_MODEL)",
                    TRANSLATE_FALLBACK_MODEL,
                )
                _fallback_translator = LocalTranscriber(
                    model=TRANSLATE_FALLBACK_MODEL
                )
                await _fallback_translator.ensure_loaded()
    return _fallback_translator


class TranscriberCache:
    """Caches one LocalTranscriber per model name (models are expensive)."""

    def __init__(self) -> None:
        self._cache: dict[str, LocalTranscriber] = {}
        self._lock = asyncio.Lock()

    async def get(self, model: str) -> LocalTranscriber:
        async with self._lock:
            t = self._cache.get(model)
            if t is None:
                t = LocalTranscriber(model=model)
                self._cache[model] = t
            return t


transcriber_cache = TranscriberCache()
