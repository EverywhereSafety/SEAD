"""Shared defender configuration, clients and factories for SAGE inference and training callers."""

from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any


from .sage import SAGEDefenseTool, SAGEDefender
from .base import BaseToolDefender

ONLINE_TOOL_DEFENDER_TYPES = frozenset({"sage"})
# Backwards-compatible name used by the tool-only version of this module.
ONLINE_DEFENDER_TYPES = ONLINE_TOOL_DEFENDER_TYPES
DEFAULT_BLOCKED_OBSERVATION = (
    "This tool action was blocked by the safety classifier and was not executed. "
    "Choose a safe alternative action or explain that the request cannot be completed."
)


class OnlineToolDefenseError(RuntimeError):
    """A defender failure that must not be counted as a safety block."""


class OpenAICompatibleBatchLLMClient:
    """Small stdlib client for an OpenAI-compatible defender endpoint."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 300,
        max_workers: int = 8,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        response_format: Mapping[str, Any] | None = None,
        allowed_token_ids: Sequence[int] | None = None,
        api_mode: str = "chat_completions",
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_workers = max_workers
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.response_format = dict(response_format or {})
        self.allowed_token_ids = (
            list(allowed_token_ids) if allowed_token_ids is not None else None
        )
        if api_mode not in {"chat_completions", "responses"}:
            raise ValueError("api_mode must be 'chat_completions' or 'responses'")
        self.api_mode = api_mode

    @property
    def completions_url(self) -> str:
        if self.api_mode == "responses":
            if self.endpoint.endswith("/responses"):
                return self.endpoint
            return self.endpoint + "/responses"
        if self.endpoint.endswith("/chat/completions"):
            return self.endpoint
        return self.endpoint + "/chat/completions"

    def _generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None,
        temperature: float,
        top_p: float | None = None,
    ) -> str:
        if self.api_mode == "responses":
            payload: dict[str, Any] = {
                "model": self.model,
                "input": messages,
            }
        else:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }
            if top_p is not None:
                payload["top_p"] = top_p
        if max_tokens is not None:
            payload[
                "max_output_tokens" if self.api_mode == "responses" else "max_tokens"
            ] = max_tokens
        if self.chat_template_kwargs and self.api_mode == "chat_completions":
            payload["chat_template_kwargs"] = self.chat_template_kwargs
        if self.response_format and self.api_mode == "chat_completions":
            payload["response_format"] = self.response_format
        if self.allowed_token_ids is not None and self.api_mode == "chat_completions":
            payload["allowed_token_ids"] = self.allowed_token_ids
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            # Azure OpenAI v1 accepts an API key through this header. Sending
            # both also preserves compatibility with ordinary OpenAI servers.
            headers["api-key"] = self.api_key
        request = urllib.request.Request(
            self.completions_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                value = json.load(response)
                response_audit = getattr(self, "response_audit", None)
                if callable(response_audit):
                    response_audit(value)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OnlineToolDefenseError(
                f"defender endpoint returned HTTP {exc.code}: {detail[:1000]}"
            ) from exc
        if self.api_mode == "responses" and isinstance(value, Mapping):
            output_text = value.get("output_text")
            if isinstance(output_text, str):
                return output_text
            parts: list[str] = []
            for item in value.get("output", []):
                if not isinstance(item, Mapping):
                    continue
                for content_item in item.get("content", []):
                    if not isinstance(content_item, Mapping):
                        continue
                    text = content_item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            if parts:
                return "".join(parts)
            raise OnlineToolDefenseError(
                "defender Responses API result has no output text"
            )
        choices = value.get("choices") if isinstance(value, Mapping) else None
        first = choices[0] if isinstance(choices, list) and choices else None
        message = first.get("message") if isinstance(first, Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str):
            raise OnlineToolDefenseError(
                "defender endpoint response has no assistant content"
            )
        return content

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int | None = None,
        temperature: float = 0.0,
        top_p: float | None = None,
    ) -> Sequence[str]:
        if not prompts:
            return []

        def generate(prompt: str) -> str:
            return self._generate(
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        workers = min(self.max_workers, len(prompts))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(generate, prompts))

    def generate_batch_with_messages(
        self,
        messages_list: list[list[dict[str, str]]],
        max_tokens: int | None = None,
        temperature: float = 0.0,
        top_p: float | None = None,
    ) -> Sequence[str]:
        if not messages_list:
            return []

        def generate(messages: list[dict[str, str]]) -> str:
            return self._generate(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        workers = min(self.max_workers, len(messages_list))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(generate, messages_list))


def _validate_model_gate_config(
    value: Mapping[str, Any] | None,
    *,
    label: str,
    defender_types: frozenset[str],
    require_credentials: bool = False,
) -> dict[str, Any]:
    if value is not None and not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    config = dict(value or {})
    enabled = config.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError(f"{label}.enabled must be boolean")
    config["enabled"] = enabled
    if not enabled:
        return config

    defender_type = str(config.get("type") or "")
    if defender_type not in defender_types:
        raise ValueError(
            f"{label}.type must be one of: " + ", ".join(sorted(defender_types))
        )
    if label == "defense.tool":
        investigation = config.get("environment_investigation", {"enabled": True})
        if not isinstance(investigation, Mapping):
            raise TypeError("defense.tool.environment_investigation must be an object")
        investigation = {"enabled": True, **investigation}
        if investigation["enabled"] is not True:
            raise ValueError("SAGE requires environment investigation")
        config["environment_investigation"] = investigation
        roots = investigation.get("allowed_roots")
        if roots is not None and (
            not isinstance(roots, (list, tuple)) or not roots
            or any(not isinstance(root, str) or not root.startswith("/") for root in roots)
        ):
            raise ValueError("environment investigation roots must be absolute paths")
    if label == "defense.tool" and defender_type == "sage":
        max_steps = int(config.get("max_steps", 6))
        max_tokens_per_step = int(config.get("max_tokens_per_step", 1024))
        max_tool_output_chars = int(config.get("max_tool_output_chars", 8_000))
        max_memory_entries = int(config.get("max_memory_entries", 32))
        min_block_confidence = float(config.get("min_block_confidence", 0.0))
        initial_history_max_chars = config.get("initial_history_max_chars")
        if initial_history_max_chars is not None:
            initial_history_max_chars = int(initial_history_max_chars)
        temperature = float(config.get("temperature", 0.0))
        if max_steps < 1:
            raise ValueError(f"{label}.max_steps must be positive")
        if max_tokens_per_step < 1:
            raise ValueError(f"{label}.max_tokens_per_step must be positive")
        if max_tool_output_chars < 256:
            raise ValueError(f"{label}.max_tool_output_chars must be at least 256")
        if max_memory_entries < 1:
            raise ValueError(f"{label}.max_memory_entries must be positive")
        if not math.isfinite(min_block_confidence) or not (
            0.0 <= min_block_confidence <= 1.0
        ):
            raise ValueError(f"{label}.min_block_confidence must be between 0 and 1")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError(f"{label}.temperature must be finite and non-negative")
        if initial_history_max_chars is not None and initial_history_max_chars < 1_000:
            raise ValueError(
                f"{label}.initial_history_max_chars must be at least 1000 or null"
            )
        system_prompt = config.get("system_prompt")
        if system_prompt is not None and not str(system_prompt).strip():
            raise ValueError(f"{label}.system_prompt must be non-empty")
        config.update(
            {
                "max_steps": max_steps,
                "max_tokens_per_step": max_tokens_per_step,
                "max_tool_output_chars": max_tool_output_chars,
                "max_memory_entries": max_memory_entries,
                "min_block_confidence": min_block_confidence,
                "temperature": temperature,
                "initial_history_max_chars": initial_history_max_chars,
            }
        )
    endpoint = str(config.get("endpoint") or "").rstrip("/")
    if not re.match(r"^https?://[^/\s]+(?:/.*)?$", endpoint):
        raise ValueError(f"{label}.endpoint must be an HTTP(S) URL")
    if not str(config.get("model") or "").strip():
        raise ValueError(f"{label}.model is required")
    api_key_env = config.get("api_key_env")
    if api_key_env is not None:
        api_key_env = str(api_key_env)
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", api_key_env):
            raise ValueError(
                f"{label}.api_key_env must be an environment variable name"
            )
        if require_credentials and not os.environ.get(api_key_env):
            raise ValueError(
                f"missing Defender credential environment variable: {api_key_env}"
            )
        config["api_key_env"] = api_key_env
    timeout_seconds = float(config.get("timeout_seconds", 300))
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError(f"{label}.timeout_seconds must be positive")
    max_workers = int(config.get("max_workers", 8))
    if max_workers < 1:
        raise ValueError(f"{label}.max_workers must be positive")
    history_event_limit = config.get("history_event_limit")
    if history_event_limit is not None and int(history_event_limit) < 0:
        raise ValueError(f"{label}.history_event_limit must be non-negative or null")
    history_max_chars = config.get("history_max_chars")
    if history_max_chars is not None and int(history_max_chars) < 1_000:
        raise ValueError(f"{label}.history_max_chars must be at least 1000 or null")
    chat_template_kwargs = config.get("chat_template_kwargs")
    response_format = config.get("response_format")
    allowed_token_ids = config.get("allowed_token_ids")
    api_mode = str(config.get("api_mode", "chat_completions"))
    if api_mode not in {"chat_completions", "responses"}:
        raise ValueError(
            f"{label}.api_mode must be 'chat_completions' or 'responses'"
        )
    if allowed_token_ids is not None and (
        not isinstance(allowed_token_ids, list)
        or not allowed_token_ids
        or any(type(token) is not int or token < 0 for token in allowed_token_ids)
        or len(allowed_token_ids) != len(set(allowed_token_ids))
    ):
        raise ValueError(
            f"{label}.allowed_token_ids must be distinct nonnegative integers"
        )
    if chat_template_kwargs is not None and not isinstance(
        chat_template_kwargs, Mapping
    ):
        raise TypeError(f"{label}.chat_template_kwargs must be an object")
    if response_format is not None and not isinstance(response_format, Mapping):
        raise TypeError(f"{label}.response_format must be an object")
    preserve_latest_user_message = config.get("preserve_latest_user_message", False)
    if not isinstance(preserve_latest_user_message, bool):
        raise TypeError(f"{label}.preserve_latest_user_message must be boolean")
    config.update(
        {
            "endpoint": endpoint,
            "timeout_seconds": timeout_seconds,
            "max_workers": max_workers,
            "history_event_limit": (
                None if history_event_limit is None else int(history_event_limit)
            ),
            "history_max_chars": (
                None if history_max_chars is None else int(history_max_chars)
            ),
            "chat_template_kwargs": dict(chat_template_kwargs or {}),
            "response_format": dict(response_format or {}),
            "preserve_latest_user_message": preserve_latest_user_message,
            "api_mode": api_mode,
        }
    )
    return config


def validate_online_tool_defense_config(
    value: Mapping[str, Any] | None,
    *,
    require_credentials: bool = False,
) -> dict[str, Any]:
    """Validate one optional pre-execution tool gate."""

    return _validate_model_gate_config(
        value,
        label="defense.tool",
        defender_types=ONLINE_TOOL_DEFENDER_TYPES,
        require_credentials=require_credentials,
    )




def validate_online_defense_config(
    value: Mapping[str, Any] | None,
    *,
    require_credentials: bool = False,
) -> dict[str, dict[str, Any]]:
    """Normalize the SAGE tool gate; accept a direct tool configuration."""
    if value is not None and not isinstance(value, Mapping):
        raise TypeError("defense must be an object")
    config = dict(value or {})
    if "output" in config:
        raise ValueError("SAGE supports only defense.tool")
    if "tool" in config:
        if set(config) != {"tool"}:
            raise ValueError("defense supports only the tool section")
        tool_value = config["tool"]
    else:
        tool_value = config
    return {
        "tool": validate_online_tool_defense_config(
            tool_value, require_credentials=require_credentials,
        ),
    }


def _configured_client(config: Mapping[str, Any]) -> OpenAICompatibleBatchLLMClient:
    api_key_env = config.get("api_key_env")
    return OpenAICompatibleBatchLLMClient(
        endpoint=str(config["endpoint"]),
        model=str(config["model"]),
        api_key=(os.environ.get(str(api_key_env)) if api_key_env else None),
        timeout_seconds=float(config["timeout_seconds"]),
        max_workers=int(config["max_workers"]),
        chat_template_kwargs=config.get("chat_template_kwargs"),
        response_format=config.get("response_format"),
        allowed_token_ids=config.get("allowed_token_ids"),
        api_mode=str(config.get("api_mode", "chat_completions")),
    )


def build_tool_defender(
    value: Mapping[str, Any],
    *,
    extra_tools: Sequence[SAGEDefenseTool] = (),
) -> BaseToolDefender:
    """Construct SAGE without coupling the model client to OpenHands."""

    config = validate_online_tool_defense_config(value, require_credentials=True)
    if not config["enabled"]:
        raise ValueError("cannot build a disabled online tool defender")
    client = _configured_client(config)
    defender_type = str(config["type"])
    tool_descriptions = config.get("tool_descriptions")
    if tool_descriptions is not None and not isinstance(tool_descriptions, Mapping):
        raise TypeError("defense.tool.tool_descriptions must be an object")
    if defender_type == "sage":
        return SAGEDefender(
            client,
            max_steps=int(config["max_steps"]),
            max_tokens_per_step=int(config["max_tokens_per_step"]),
            temperature=float(config["temperature"]),
            max_tool_output_chars=int(config["max_tool_output_chars"]),
            max_memory_entries=int(config["max_memory_entries"]),
            min_block_confidence=float(config["min_block_confidence"]),
            initial_history_max_chars=config["initial_history_max_chars"],
            extra_tools=extra_tools,
            system_prompt=(
                str(config["system_prompt"])
                if config.get("system_prompt") is not None
                else None
            ),
        )
    raise AssertionError(f"unhandled defender type: {defender_type}")
