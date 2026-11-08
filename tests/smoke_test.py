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
import time

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
        env={**os.environ, "WHISPER_LIVE_FAKE_TRANSCRIBER": "1"},
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
                    break
                msg = json.loads(raw)
                if msg.get("type") in ("status", "error", "warning"):
                    print("control:", msg)
                    if msg.get("type") == "error":
                        ok = False
                else:
                    replies += 1
                    print(f"transcript[{msg.get('is_final', True)}]: {msg.get('text')!r}")
            assert replies >= 1, "no transcript replies received"

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

