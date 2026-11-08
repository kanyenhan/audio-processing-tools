"""Speech segmenter for live (streaming) transcription.

Ports the buffering strategy from ``whisper_ctranslate2.live.Live``:
incoming audio is split into fixed-size blocks (30 ms); a block counts as
speech when its RMS volume is above a threshold and its dominant frequency
falls inside the typical vocal range.  A completed speech segment is
emitted either when enough silence follows the speech, or when the
maximum segment length is reached.  Segments shorter than the minimum
speech duration are discarded.
"""

from __future__ import annotations

import numpy as np

BLOCK_MS = 30                 # block size in milliseconds
VOCALS = (50, 1000)           # Hz range of frequencies that could be speech
END_SILENCE_MS = 2000         # silence to wait after speech before emitting
MAX_SPEECH_MS = 30000         # hard cap on a single segment (article: 30 s)
MIN_SPEECH_MS = 1500          # shorter utterances are discarded (article: 1500 ms)
DEFAULT_VOLUME_THRESHOLD = 0.02
SAMPLE_RATE = 16000           # must match the browser client and Whisper


class SpeechSegmenter:
    """Buffers PCM16 audio frames and emits completed speech segments.

    All audio is 16 kHz mono, matching the browser client and Whisper.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        volume_threshold: float = DEFAULT_VOLUME_THRESHOLD,
        min_speech_ms: int = MIN_SPEECH_MS,
        max_speech_ms: int = MAX_SPEECH_MS,
        end_silence_ms: int = END_SILENCE_MS,
    ) -> None:
        self.sample_rate = sample_rate
        self.volume_threshold = volume_threshold
        self.block_samples = max(1, int(sample_rate * BLOCK_MS / 1000))
        self.min_speech_blocks = max(1, int(min_speech_ms / BLOCK_MS))
        self.max_speech_blocks = max(
            self.min_speech_blocks, int(max_speech_ms / BLOCK_MS)
        )
        self.end_silence_blocks = max(1, int(end_silence_ms / BLOCK_MS))

        self._pending = np.zeros(0, dtype=np.float32)
        self._prev_block = np.zeros(0, dtype=np.float32)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._speaking = False
        self._waiting = 0
        self._blocks_speaking = 0
        self._speech_blocks = 0

    def feed(self, pcm16: bytes) -> list[np.ndarray]:
        """Append a PCM16 mono frame; return completed segments (float32)."""
        if not pcm16:
            return []
        samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        self._pending = np.concatenate((self._pending, samples))

        segments: list[np.ndarray] = []
        while len(self._pending) >= self.block_samples:
            block = self._pending[: self.block_samples]
            self._pending = self._pending[self.block_samples:]
            if self._process_block(block):
                segments.append(self._completed())
        return segments

    def flush(self) -> np.ndarray | None:
        """Emit any in-flight segment (e.g. when the session stops)."""
        self._pending = np.zeros(0, dtype=np.float32)
        if self._speaking and self._speech_blocks >= self.min_speech_blocks:
            return self._completed()
        self._reset()
        return None

    # ------------------------------------------------------------------

    def _process_block(self, block: np.ndarray) -> bool:
        """Run the speech state machine on one 30 ms block.

        Returns True when a finished, acceptable segment is complete and
        waiting to be collected via ``_completed()``.
        """
        voice = self._is_there_voice(block)

        if voice:
            # Prepend the last silent block as pre-roll on utterance start.
            if not self._speaking:
                self._buffer = self._prev_block.copy()
                self._blocks_speaking = self.max_speech_blocks
            self._buffer = np.concatenate((self._buffer, block))
            self._waiting = self.end_silence_blocks
            self._speaking = True
            self._speech_blocks += 1
        elif self._speaking:  # silence while an utterance is open
            self._waiting -= 1
            if self._waiting <= 0:
                return self._finalize()
            self._buffer = np.concatenate((self._buffer, block))

        # Hard cap: utterance longer than max_speech, flush what we have.
        if self._speaking:
            self._blocks_speaking -= 1
            if self._blocks_speaking <= 0:
                return self._finalize()

        self._prev_block = block
        return False

    def _finalize(self) -> bool:
        ok = self._speech_blocks >= self.min_speech_blocks
        segment = self._buffer
        self._reset()
        if ok:
            self._buffer = segment
            return True
        return False

    def _completed(self) -> np.ndarray:
        segment = self._buffer.astype(np.float32)
        self._buffer = np.zeros(0, dtype=np.float32)
        return segment

    def _reset(self) -> None:
        self._speaking = False
        self._waiting = 0
        self._blocks_speaking = 0
        self._speech_blocks = 0
        self._buffer = np.zeros(0, dtype=np.float32)

    def _is_there_voice(self, block: np.ndarray) -> bool:
        """Volume + dominant-frequency check (ported from whisper-ctranslate2)."""
        if not np.any(block):
            return False
        frames = len(block)
        freq = int(
            np.argmax(np.abs(np.fft.rfft(block))) * self.sample_rate / frames
        )
        volume = float(np.sqrt(np.mean(block**2)))
        return volume > self.volume_threshold and VOCALS[0] <= freq <= VOCALS[1]

