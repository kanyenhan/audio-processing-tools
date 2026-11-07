"""Smoke test for the relay.

Starts the mock LocalAI REST server + the FastAPI app, then simulates the
browser: connect to the app's /ws, send a start command, stream a few
binary PCM frames, and verify that transcript replies flow back to the
browser.

Run:  python tests/smoke_test.py
"""

import asyncio
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import websockets  # noqa: E402

APP_WS = "ws://127.0.0.1:8090/ws"
MOCK_URL = "http://127.0.0.1:1025"


async def wait_for(port, what):
    for _ in range(100):
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            return
        except OSError:
            await asyncio.sleep(0.2)
    raise RuntimeError(f"{what} did not come up on port {port}")


async def main():
    procs = []
    try:
        procs.append(
            subprocess.Popen(
                [sys.executable, "tests/mock_whisper_server.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        )
        procs.append(
            subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app.main:app",
                 "--host", "127.0.0.1", "--port", "8090", "--log-level", "warning"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        )
        await wait_for(1025, "mock local-ai server")
        await wait_for(8090, "app")

        ok = True
        async with websockets.connect(APP_WS) as ws:
            # 1. start session
            await ws.send(json.dumps(
                {"type": "start", "whisper_url": MOCK_URL, "model": "whisper-large", "language": "en"}
            ))
            status = json.loads(await asyncio.wait_for(ws.recv(), 10))
            print("status after start:", status)
            assert status.get("type") == "status" and status.get("status") == "connected", \
                "expected connected status"

            # 2. stream enough audio chunks to trigger transcriptions
            # Each frame is ~25.6ms of 16kHz PCM16 (8192 bytes)
            # MIN_BYTES = 16000 (0.5s), so after 2 frames we have enough
            frame = b"\x00\x01" * 4096  # fake 25.6ms of 16kHz PCM16
            replies = 0
            deadline = time.time() + 20
            for i in range(20):
                await ws.send(frame)
                while time.time() < deadline:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), 0.5)
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
                if replies >= 3:
                    break
            assert replies >= 1, "no transcript replies received"

            # 3. stop session
            await ws.send(json.dumps({"type": "stop"}))
            status = json.loads(await asyncio.wait_for(ws.recv(), 10))
            print("status after stop:", status)
            assert status.get("status") == "idle", "expected idle status"

            # 4. error path: bad local-ai url
            await ws.send(json.dumps(
                {"type": "start", "whisper_url": "http://127.0.0.1:59999", "language": "auto"}
            ))
            err = json.loads(await asyncio.wait_for(ws.recv(), 15))
            print("bad url reply:", err)
            assert err.get("type") == "error", "expected error for bad url"

        print("\nSMOKE TEST PASSED" if ok else "\nSMOKE TEST FAILED (see errors)")
        sys.exit(0 if ok else 1)
    finally:
        for p in procs:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    asyncio.run(main())
