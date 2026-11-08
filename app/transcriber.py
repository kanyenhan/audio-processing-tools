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
_FAKE_WORDS = ["this", "is", "a", "mock", "transcription", "from", "the", "test"]
_fake_counter = 0


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
        initial_prompt=None,
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

    async def transcribe(self, audio: np.ndarray, language: str | None) -> str:
        """Transcribe one segment; returns the text (may be empty)."""
        async with self._lock:
            return await asyncio.to_thread(self._infer_sync, audio, language)

    def _infer_sync(self, audio: np.ndarray, language: str | None) -> str:
        global _fake_counter
        if _FAKE:
            _fake_counter += 1
            return f"{_FAKE_WORDS[_fake_counter % len(_FAKE_WORDS)]} "
        assert self._transcribe is not None, "model not loaded"
        lang = language if language and language != "auto" else None
        result = self._transcribe.inference(
            audio=audio.flatten().astype("float32"),
            task="transcribe",
            language=lang,
            verbose=False,
            live=True,
            options=live_options(),
        )
        return (result.get("text") or "").strip()


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
