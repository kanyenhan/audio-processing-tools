"""Smoke test for the live transcription relay.

Starts the FastAPI app (with the fake whisper-ctranslate2 transcriber
enabled via WHISPER_LIVE_FAKE_TRANSCRIBER so no model download is needed),
then simulates the browser: connect to the app's /ws, send a start command,
stream audio that the segmenter will treat as an utterance followed by
silence, and verify that transcript replies flow back to the browser.

Run:  python tests/smoke_test.py
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import websockets  # noqa: E402

APP_WS = "ws://127.0.0.1:8090/ws"
SAMPLE_RATE = 16000


def sine_pcm16(seconds: float, freq: float = 300.0, amp: float = 0.5) -> bytes:
    """Render a sine tone (inside the vocal band, loud enough to trigger
    the segmenter) as 16 kHz mono PCM16."""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    samples = amp * np.sin(2 * np.pi * freq * t)
    return (samples * 32767).astype("<i2").tobytes()


def silence_pcm16(seconds: float) -> bytes:
    return b"\x00\x00" * int(seconds * SAMPLE_RATE)


def write_wav(path, seconds_voice: float, seconds_silence: float) -> None:
    """Write a 16 kHz mono PCM16 wav: vocal-band tone, then silence."""
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(sine_pcm16(seconds_voice) + silence_pcm16(seconds_silence))


async def wait_for(port, what):
    for _ in range(100):
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            return
        except OSError:
            await asyncio.sleep(0.2)
    raise RuntimeError(f"{what} did not come up on port {port}")


def frame_chunks(pcm: bytes, chunk: int = 4096):
    for i in range(0, len(pcm), chunk):
        yield pcm[i : i + chunk]


async def main():
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app",
         "--host", "127.0.0.1", "--port", "8090", "--log-level", "warning"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={
            **os.environ,
            "WHISPER_LIVE_FAKE_TRANSCRIBER": "1",
            "WHISPER_LIVE_TRANSLATE_MODEL": "large-v3",
        },
    )
    ok = True
    try:
        await wait_for(8090, "app")

        async with websockets.connect(APP_WS) as ws:
            # 1. start session (model is validated against MODEL_NAMES)
            await ws.send(json.dumps(
                {"type": "start", "model": "turbo", "language": "en"}
            ))
            status = None
            deadline = time.time() + 10
            while time.time() < deadline:
                msg = json.loads(await asyncio.wait_for(ws.recv(), 10))
                if msg.get("type") != "status":
                    continue
                print("status:", msg.get("status"), msg.get("message", ""))
                status = msg
                if msg.get("status") == "connected":
                    break
            assert status and status.get("status") == "connected", \
                "expected connected status"

            # 2. stream ~2s of speech then ~2s of silence to complete a
            #    segment (needs >=1.5s speech and ~2s trailing silence).
            replies = 0
            for frame in frame_chunks(sine_pcm16(2.0)):
                await ws.send(frame)
            for frame in frame_chunks(silence_pcm16(2.0)):
                await ws.send(frame)

            deadline = time.time() + 20
            while replies < 1 and time.time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), 1.0)
                except asyncio.TimeoutError:
                    continue
                msg = json.loads(raw)
                if msg.get("type") in ("status", "error", "warning"):
                    print("control:", msg)
                    if msg.get("type") == "error":
                        ok = False
                else:
                    replies += 1
                    print(f"transcript[{msg.get('is_final', True)}]: {msg.get('text')!r}")
                    print(
                        "  fields:",
                        {k: msg.get(k) for k in ("language", "duration", "vad_removed", "ts")},
                    )
                    assert msg.get("is_final") is True, "expected final transcript"
                    assert msg.get("language"), "missing language field"
                    assert isinstance(msg.get("duration"), (int, float)) and msg["duration"] > 0, \
                        "missing/invalid duration field"
                    assert "vad_removed" in msg, "missing vad_removed field"
                    assert isinstance(msg.get("ts"), int), "missing ts field"
            assert replies >= 1, "no transcript replies received"

            # 2b. translation follow-up: restart with a non-English language
            #     and translate enabled (fallback translator is faked via
            #     WHISPER_LIVE_TRANSLATE_MODEL), so a {"type": "translation"}
            #     message with the same ts must follow the segment message.
            await ws.send(json.dumps(
                {"type": "start", "model": "turbo", "language": "zh",
                 "translate": True}
            ))
            deadline = time.time() + 10
            connected = False
            while time.time() < deadline:
                msg = json.loads(await asyncio.wait_for(ws.recv(), 10))
                if msg.get("type") != "status":
                    print("control:", msg)
                    continue
                if msg.get("status") == "connected":
                    connected = True
                    break
            assert connected, "expected connected status after restart"

            for frame in frame_chunks(sine_pcm16(2.0)):
                await ws.send(frame)
            for frame in frame_chunks(silence_pcm16(2.0)):
                await ws.send(frame)

            segment_ts = None
            translation = None
            deadline = time.time() + 20
            while translation is None and time.time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), 1.0)
                except asyncio.TimeoutError:
                    continue
                msg = json.loads(raw)
                if msg.get("type") in ("status", "error", "warning"):
                    print("control:", msg)
                    continue
                if msg.get("type") == "translation":
                    translation = msg
                else:
                    print(f"transcript[{msg.get('is_final', True)}]: {msg.get('text')!r}")
                    segment_ts = msg.get("ts")
            assert segment_ts is not None, "no segment message received"
            assert translation is not None, "no translation follow-up received"
            print("translation:", translation)
            assert translation.get("ts") == segment_ts, \
                "translation ts does not match its segment"

            # 2c. file mode: decode + segment + stream, then file_done
            wav_path = os.path.join(tempfile.gettempdir(), "whisper-live-smoke.wav")
            write_wav(wav_path, 2.0, 2.0)
            try:
                await ws.send(json.dumps(
                    {"type": "start", "model": "turbo", "language": "en"}
                ))
                deadline = time.time() + 10
                connected = False
                while time.time() < deadline:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), 10))
                    if msg.get("type") == "error":
                        raise AssertionError(msg)
                    if msg.get("type") == "status" and msg.get("status") == "connected":
                        connected = True
                        break
                assert connected, "expected connected status for file session"

                await ws.send(json.dumps({"type": "file", "path": wav_path}))

                file_transcripts = 0
                file_done = None
                deadline = time.time() + 20
                while file_done is None and time.time() < deadline:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), 1.0)
                    except asyncio.TimeoutError:
                        continue
                    msg = json.loads(raw)
                    if msg.get("type") in ("status", "error", "warning"):
                        print("control:", msg)
                        if msg.get("type") == "error":
                            ok = False
                    elif msg.get("type") == "file_done":
                        file_done = msg
                    else:
                        file_transcripts += 1
                        print(f"file transcript: {msg.get('text')!r}")
                assert file_done is not None, "no file_done received"
                print("file_done:", file_done)
                assert file_done.get("segments", 0) >= 1, "expected >=1 segment"
                assert file_transcripts >= 1, "no transcript for file"
                assert file_done.get("duration_s", 0) > 0, "missing duration_s"

                # 2d. file mode error path: nonexistent path
                await ws.send(json.dumps(
                    {"type": "file", "path": "/nonexistent/nope.wav"}
                ))
                deadline = time.time() + 10
                got_error = False
                while time.time() < deadline:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), 10))
                    if msg.get("type") == "error" and "File not found" in msg.get("message", ""):
                        got_error = True
                        break
                    print("control:", msg)
                assert got_error, "expected File-not-found error"
            finally:
                os.unlink(wav_path)

            # 3. stop session
            await ws.send(json.dumps({"type": "stop"}))
            status = json.loads(await asyncio.wait_for(ws.recv(), 10))
            print("status after stop:", status)
            assert status.get("status") == "idle", "expected idle status"

            # 4. error path: unknown model name
            await ws.send(json.dumps(
                {"type": "start", "model": "nonexistent-model", "language": "auto"}
            ))
            err = json.loads(await asyncio.wait_for(ws.recv(), 15))
            print("bad model reply:", err)
            assert err.get("type") == "error", "expected error for bad model"

        print("\nSMOKE TEST PASSED" if ok else "\nSMOKE TEST FAILED (see errors)")
        sys.exit(0 if ok else 1)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    asyncio.run(main())

