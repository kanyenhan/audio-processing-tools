"""File-mode audio support.

Decodes arbitrary audio files (wav, mp3, m4a, flac, ogg, …) to 16 kHz mono
PCM16 bytes — the format the SpeechSegmenter expects — and keeps
bookkeeping for files picked via the UI's "Browse" button (saved under the
system temp dir and swept periodically).
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
from faster_whisper.audio import decode_audio

#: Root for files picked via the UI's "Browse" button; swept on each upload.
UPLOAD_DIR = Path(tempfile.gettempdir()) / "whisper-live-uploads"
UPLOAD_MAX_AGE_S = 3600  # delete uploaded files older than 1 hour


def decode_to_pcm16(path: str) -> bytes:
    """Decode an audio file to 16 kHz mono PCM16 bytes (what the
    segmenter expects).  faster-whisper's ``decode_audio`` resamples to
    16 kHz mono and returns float32 in [-1, 1], so convert to int16."""
    audio = decode_audio(path, sampling_rate=16000)
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def sweep_uploads() -> None:
    """Remove uploaded files that have not been used recently."""
    try:
        cutoff = time.time() - UPLOAD_MAX_AGE_S
        for old in UPLOAD_DIR.glob("upload_*"):
            if old.is_file() and old.stat().st_mtime < cutoff:
                old.unlink(missing_ok=True)
    except OSError:
        pass
