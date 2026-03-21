# pyre-unsafe
"""Shared Hugging Face inference utilities (OpenAI-compatible chat API)."""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional

import aiohttp
import httpx

import config

_MAX_RETRIES = 3
_BACKOFF_BASE = 1.5


def _headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if config.HF_TOKEN:
        headers["Authorization"] = f"Bearer {config.HF_TOKEN}"
    return headers


def _chat_payload(
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": max(temperature, 0.01),
    }


def _parse_chat_response(data: Any) -> Optional[str]:
    if isinstance(data, dict):
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            return message.get("content", "")
    return None


def hf_generate_sync(
    model_id: str,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.2,
    do_sample: bool = False,
) -> str:
    payload = _chat_payload(model_id, prompt, max_new_tokens, temperature)
    url = config.HF_CHAT_API_URL
    for attempt in range(_MAX_RETRIES):
        try:
            resp = httpx.post(
                url,
                headers=_headers(),
                json=payload,
                timeout=config.REQUEST_TIMEOUT_SECS,
            )
            data = resp.json()
            if isinstance(data, dict) and data.get("error"):
                err_msg = data["error"]
                if isinstance(err_msg, dict):
                    err_msg = err_msg.get("message", str(err_msg))
                err_str = str(err_msg).lower()
                if "loading" in err_str or "overloaded" in err_str or resp.status_code in (503, 429):
                    wait = _BACKOFF_BASE ** attempt
                    if config.DEBUG:
                        print(f"[inference] model busy, retry {attempt+1} in {wait:.1f}s")
                    time.sleep(wait)
                    continue
                if config.DEBUG:
                    print(f"[inference] sync error: {err_msg}")
                return ""
            out = _parse_chat_response(data)
            return out or ""
        except Exception as exc:
            if config.DEBUG:
                print(f"[inference] sync failed (attempt {attempt+1}): {exc}")
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BACKOFF_BASE ** attempt)
    return ""


async def hf_generate_async(
    session: aiohttp.ClientSession,
    model_id: str,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.2,
    do_sample: bool = False,
) -> str:
    payload = _chat_payload(model_id, prompt, max_new_tokens, temperature)
    url = config.HF_CHAT_API_URL
    timeout = aiohttp.ClientTimeout(total=config.REQUEST_TIMEOUT_SECS)
    for attempt in range(_MAX_RETRIES):
        try:
            async with session.post(
                url,
                headers=_headers(),
                data=json.dumps(payload),
                timeout=timeout,
            ) as resp:
                data = await resp.json()
            if isinstance(data, dict) and data.get("error"):
                err_msg = data["error"]
                if isinstance(err_msg, dict):
                    err_msg = err_msg.get("message", str(err_msg))
                err_str = str(err_msg).lower()
                if "loading" in err_str or "overloaded" in err_str or resp.status in (503, 429):
                    wait = _BACKOFF_BASE ** attempt
                    if config.DEBUG:
                        print(f"[inference] model busy, retry {attempt+1} in {wait:.1f}s")
                    await asyncio.sleep(wait)
                    continue
                if config.DEBUG:
                    print(f"[inference] async error: {err_msg}")
                return ""
            out = _parse_chat_response(data)
            return out or ""
        except Exception as exc:
            if config.DEBUG:
                print(f"[inference] async failed (attempt {attempt+1}): {exc}")
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(_BACKOFF_BASE ** attempt)
    return ""


async def hf_feature_extract_async(
    session: aiohttp.ClientSession,
    model_id: str,
    text: str,
) -> Optional[Any]:
    payload = {"inputs": text, "options": {"wait_for_model": True}}
    url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
    timeout = aiohttp.ClientTimeout(total=config.REQUEST_TIMEOUT_SECS)
    for attempt in range(_MAX_RETRIES):
        try:
            async with session.post(
                url,
                headers=_headers(),
                data=json.dumps(payload),
                timeout=timeout,
            ) as resp:
                data = await resp.json()
            if isinstance(data, dict) and data.get("error"):
                if config.DEBUG:
                    print(f"[inference] feature extraction error: {data['error']}")
                return None
            if isinstance(data, list):
                return data
            return None
        except Exception as exc:
            if config.DEBUG:
                print(f"[inference] feature extraction failed (attempt {attempt+1}): {exc}")
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(_BACKOFF_BASE ** attempt)
    return None
