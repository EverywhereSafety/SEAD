"""Official asynchronous google-genai adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .config import JudgeConfig


class GeminiAdapterError(RuntimeError):
    pass


@dataclass(frozen=True)
class BackendResponse:
    raw_text: str
    token_usage: Mapping[str, Any]


class GeminiBackend:
    def __init__(self, *, config: JudgeConfig, api_key: str):
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - installation dependent
            raise GeminiAdapterError(
                "google-genai is unavailable; install sead[judge]"
            ) from exc
        self._config = config
        self._client = genai.Client(api_key=api_key)

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: Mapping[str, Any],
    ) -> BackendResponse:
        try:
            from google.genai import types

            response = await self._client.aio.models.generate_content(
                model=self._config.model.removeprefix("gemini/"),
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=self._config.temperature,
                    max_output_tokens=self._config.max_output_tokens,
                    response_mime_type="application/json",
                    response_json_schema=dict(response_schema),
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=False,
                        thinking_level=types.ThinkingLevel.MINIMAL,
                    ),
                ),
            )
        except Exception as exc:
            raise GeminiAdapterError(str(exc)) from exc
        raw = getattr(response, "text", None)
        if not isinstance(raw, str) or not raw.strip():
            raise GeminiAdapterError("Gemini returned no response text")
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            usage_dict: dict[str, Any] = {}
        elif hasattr(usage, "model_dump"):
            usage_dict = usage.model_dump(mode="json", exclude_none=True)
        else:
            usage_dict = {"value": str(usage)}
        return BackendResponse(raw, usage_dict)
