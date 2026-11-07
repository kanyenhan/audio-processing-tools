"""Mock LocalAI REST server for local testing.

Mimics the LocalAI transcription API so the web app can be exercised
without the real model.

Endpoints:
  GET  /v1/models                 -> list of available models
  POST /v1/audio/transcriptions   -> accept WAV, return {"text": "..."}

Usage:  python tests/mock_whisper_server.py
        (listens on http://127.0.0.1:1025)
"""

import json

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

PORT = 1025

app = FastAPI(title="Mock LocalAI")

WORDS = ["this", "is", "a", "mock", "transcription", "from", "the", "test", "server"]
_request_count = 0


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "whisper-large", "object": "model"}],
    }


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form("whisper-large"),
    language: str = Form(None),
):
    global _request_count
    _request_count += 1
    content = await file.read()
    # Echo back a mock transcription based on the audio length
    word_idx = _request_count % len(WORDS)
    word = WORDS[word_idx]
    print(
        f"transcription request #{_request_count}: model={model}, "
        f"file={len(content)} bytes, language={language}"
    )
    return JSONResponse({"text": f"{word} "})


if __name__ == "__main__":
    import uvicorn

    print(f"mock local-ai server listening on http://127.0.0.1:{PORT}")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
