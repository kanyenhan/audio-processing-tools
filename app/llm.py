"""LLM post-processing of the transcript (summarize / translate / custom
prompt) via any local OpenAI-compatible server (e.g. LM Studio at
127.0.0.1:1234).

The app proxies the calls (see app.main's /api/llm/* routes) so the browser
never makes cross-origin requests to the LLM server (no CORS concerns) and
the UI stays single-origin.
"""

from __future__ import annotations

import json
import re
import urllib.request

from fastapi import HTTPException

#: Target-language names for the LLM prompts.  Mirrors the codes of the
#: UI language dropdowns; zh maps to Traditional Chinese to match the
#: app opencc output convention.
LLM_LANGUAGES = {
    "en": "English",
    "zh": "Traditional Chinese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
    "uk": "Ukrainian",
    "ja": "Japanese",
    "ko": "Korean",
    "hi": "Hindi",
    "ar": "Arabic",
    "tr": "Turkish",
    "pl": "Polish",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "th": "Thai",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "cs": "Czech",
    "el": "Greek",
    "he": "Hebrew",
    "ro": "Romanian",
    "hu": "Hungarian",
    "sk": "Slovak",
}


def llm_prompt(action: str, language: str, custom_prompt: str = "") -> str:
    """Build the system prompt for an LLM post-processing action.
    language is a UI language code or auto: summarize writes in the
    chosen language (auto = the language of the transcription), translate
    targets the chosen language (auto = English, the historical
    default). custom uses the user's prompt verbatim (they fully control
    the instruction, including the language)."""
    if action == "custom":
        return custom_prompt
    if language == "auto":
        target = "the same language as the transcription"
        if action == "translate":
            target = "English"
    else:
        target = LLM_LANGUAGES[language]
    if action == "summarize":
        return (
            "You are a helpful assistant. Write a concise summary of the "
            "transcription provided by the user, preserving the key facts "
            f"and decisions. Write the summary in {target}. "
            "Reply with the summary only - no preamble or commentary."
        )
    return (
        "You are a professional translator. Translate the transcription "
        f"provided by the user into {target}. Keep the original meaning, "
        "tone and punctuation as far as possible. Reply with the "
        "translation only - no preamble or commentary."
    )


def normalize_llm_base_url(url: str) -> str:
    """Validate/normalize the OpenAI-compatible API base URL (e.g. .../v1)."""
    base = (url or "").strip().rstrip("/")
    if not re.match(r"^https?://", base, re.IGNORECASE):
        raise HTTPException(
            status_code=400, detail="LLM URL must start with http:// or https://."
        )
    return base


def http_json(
    method: str, url: str, payload: dict | None = None, timeout: float = 10.0
) -> dict:
    """Blocking JSON request (always run via asyncio.to_thread).  Stdlib
    urllib only - no extra dependency for talking to a local LLM server."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))
