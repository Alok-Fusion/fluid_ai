import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    """Raised when all configured LLM providers fail to complete a request."""


@dataclass
class LLMResponse:
    """Wrapper holding model output text and the provider that produced it."""

    text: str
    provider: str


def extract_json(raw: str) -> dict[str, Any] | list[Any]:
    """Strip markdown code fences and parse the first JSON object or array."""
    text = raw.strip()

    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()

    first_brace = text.find("{")
    first_bracket = text.find("[")

    if first_brace == -1 and first_bracket == -1:
        raise ValueError("No JSON object or array found in text")

    if first_bracket == -1 or (first_brace != -1 and first_brace < first_bracket):
        last_brace = text.rfind("}")
        if last_brace == -1:
            raise ValueError("Unterminated JSON object")
        target = text[first_brace : last_brace + 1]
    else:
        last_bracket = text.rfind("]")
        if last_bracket == -1:
            raise ValueError("Unterminated JSON array")
        target = text[first_bracket : last_bracket + 1]

    return json.loads(target)


class LLMClient:
    """Manages LLM requests with exponential backoff and provider fallback."""

    def __init__(
        self,
        groq_api_key: str | None = None,
        gemini_api_key: str | None = None,
        force_fallback: bool | None = None,
        timeout_seconds: float = 45.0,
    ) -> None:
        self.groq_api_key = (
            groq_api_key if groq_api_key is not None else os.getenv("GROQ_API_KEY", "")
        )
        self.gemini_api_key = (
            gemini_api_key if gemini_api_key is not None else os.getenv("GEMINI_API_KEY", "")
        )

        if force_fallback is not None:
            self.force_fallback = force_fallback
        else:
            flag = os.getenv("FORCE_FALLBACK", "false").lower()
            self.force_fallback = flag in ("1", "true", "yes")

        self.timeout_seconds = timeout_seconds
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        gap_env = os.getenv("LLM_MIN_GAP_SECONDS", "1.5")
        try:
            self.min_gap_seconds = float(gap_env)
        except ValueError:
            self.min_gap_seconds = 1.5
        self._groq_semaphore = asyncio.Semaphore(1)
        self._last_groq_call_time: float = 0.0

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """Execute request via Groq with retries, falling back to Gemini if needed."""
        groq_error: Exception | None = None

        if not self.force_fallback and self.groq_api_key:
            try:
                text = await self._call_groq(
                    system_prompt, user_prompt, temperature
                )
                return LLMResponse(text=text, provider="groq")
            except (httpx.HTTPError, RuntimeError) as exc:
                groq_error = exc
                logger.warning(
                    "Groq execution failed, switching to Gemini fallback: %s", exc
                )
        elif self.force_fallback:
            logger.info("FORCE_FALLBACK is active; bypassing Groq call.")
        else:
            groq_error = ValueError("GROQ_API_KEY is not configured")
            logger.warning("GROQ_API_KEY missing, attempting Gemini fallback.")

        if not self.gemini_api_key:
            reason = (
                f"Groq failed ({groq_error}) and GEMINI_API_KEY is not configured"
                if groq_error
                else "GEMINI_API_KEY is not configured"
            )
            raise LLMProviderError(reason)

        try:
            text = await self._call_gemini(system_prompt, user_prompt, temperature)
            return LLMResponse(text=text, provider="gemini")
        except Exception as gemini_exc:
            logger.error("Gemini fallback also failed: %s", gemini_exc)
            raise LLMProviderError(
                f"Both LLM providers failed. Groq: {groq_error}; Gemini: {gemini_exc}"
            ) from gemini_exc

    async def _post_with_retry(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        max_retries: int = 3,
        base_delay: float = 3.0,
    ) -> dict[str, Any]:
        for attempt in range(max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    resp = await client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    return resp.json()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                should_retry = (status == 429 or status >= 500) and attempt < max_retries
                if not should_retry:
                    logger.warning(
                        "Provider request failed with status %s: %s",
                        status,
                        exc.response.text[:200],
                    )
                    raise

                retry_after = exc.response.headers.get("retry-after")
                if status == 429 and retry_after:
                    try:
                        delay = min(float(retry_after), 30.0)
                    except (ValueError, TypeError):
                        delay = min(base_delay * (2**attempt), 30.0)
                else:
                    delay = base_delay * (2**attempt)

                logger.info(
                    "HTTP %s on attempt %d; retrying in %.1fs",
                    status,
                    attempt + 1,
                    delay,
                )
                await asyncio.sleep(delay)
            except httpx.TimeoutException:
                if attempt >= max_retries:
                    logger.warning(
                        "Provider request timed out after %d attempts",
                        max_retries + 1,
                    )
                    raise
                delay = base_delay * (2**attempt)
                logger.info(
                    "Timeout on attempt %d; retrying in %.1fs",
                    attempt + 1,
                    delay,
                )
                await asyncio.sleep(delay)

        raise RuntimeError("Retry loop exited unexpectedly")

    async def _call_groq(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.groq_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        async with self._groq_semaphore:
            now = time.monotonic()
            elapsed = now - self._last_groq_call_time
            if elapsed < self.min_gap_seconds:
                await asyncio.sleep(self.min_gap_seconds - elapsed)
            try:
                data = await self._post_with_retry(self.groq_url, headers, payload)
                return data["choices"][0]["message"]["content"]
            finally:
                self._last_groq_call_time = time.monotonic()

    async def _call_gemini(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.gemini_model}:generateContent"
        )
        headers = {
            "x-goog-api-key": self.gemini_api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        data = await self._post_with_retry(url, headers, payload)
        return data["candidates"][0]["content"]["parts"][0]["text"]
