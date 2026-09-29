"""Shared configuration for the semantic judge."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "posthoc-llm-judge-v1"


class PosthocJudgeConfigError(ValueError):
    pass


@dataclass(frozen=True)
class JudgeConfig:
    provider: str = "gemini"
    model: str = "gemini/gemini-3-flash-preview"
    api_key_env: str = "GEMINI_API_KEY"
    temperature: float = 0.0
    max_output_tokens: int = 2048
    num_retries: int = 2
    request_timeout_seconds: float = 120.0
    max_concurrency: int = 8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PosthocJudgeConfig:
    path: Path
    schema_version: str
    judge: JudgeConfig

    def fingerprint_payload(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "judge": self.judge.to_dict()}


_JUDGE_KEYS = set(JudgeConfig.__dataclass_fields__)


def _integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PosthocJudgeConfigError(f"judge.{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str, *, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PosthocJudgeConfigError(f"judge.{name} must be a number")
    result = float(value)
    if result < minimum:
        raise PosthocJudgeConfigError(f"judge.{name} must be >= {minimum}")
    return result


def load_config(path: Path | str, *, value: Mapping[str, Any] | None = None) -> PosthocJudgeConfig:
    path = Path(path).resolve()
    try:
        raw = dict(value) if value is not None else yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PosthocJudgeConfigError(f"cannot read Judge config {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise PosthocJudgeConfigError("Judge config must be an object")
    if set(raw) != {"schema_version", "judge"}:
        raise PosthocJudgeConfigError(
            "Judge config must contain only schema_version and judge"
        )
    if raw["schema_version"] != SCHEMA_VERSION:
        raise PosthocJudgeConfigError(f"expected schema_version={SCHEMA_VERSION}")
    section = raw["judge"]
    if not isinstance(section, Mapping):
        raise PosthocJudgeConfigError("judge must be an object")
    unknown = set(section) - _JUDGE_KEYS
    if unknown:
        raise PosthocJudgeConfigError(f"unknown judge fields: {sorted(unknown)}")

    defaults = JudgeConfig()
    provider = section.get("provider", defaults.provider)
    model = section.get("model", defaults.model)
    api_key_env = section.get("api_key_env", defaults.api_key_env)
    if provider != "gemini":
        raise PosthocJudgeConfigError("judge.provider must be gemini")
    if not isinstance(model, str) or not model.startswith("gemini/") or not model[7:]:
        raise PosthocJudgeConfigError("judge.model must start with gemini/")
    if api_key_env != "GEMINI_API_KEY" or not re.fullmatch(
        r"[A-Z_][A-Z0-9_]*", str(api_key_env)
    ):
        raise PosthocJudgeConfigError("judge.api_key_env must be GEMINI_API_KEY")
    temperature = _number(
        section.get("temperature", defaults.temperature), "temperature", minimum=0
    )
    if temperature != 0:
        raise PosthocJudgeConfigError("judge.temperature must be 0")
    judge = JudgeConfig(
        provider=provider,
        model=model,
        api_key_env=api_key_env,
        temperature=temperature,
        max_output_tokens=_integer(
            section.get("max_output_tokens", defaults.max_output_tokens),
            "max_output_tokens",
            minimum=1,
        ),
        num_retries=_integer(
            section.get("num_retries", defaults.num_retries),
            "num_retries",
            minimum=0,
        ),
        request_timeout_seconds=_number(
            section.get(
                "request_timeout_seconds", defaults.request_timeout_seconds
            ),
            "request_timeout_seconds",
            minimum=0.001,
        ),
        max_concurrency=_integer(
            section.get("max_concurrency", defaults.max_concurrency),
            "max_concurrency",
            minimum=1,
        ),
    )
    return PosthocJudgeConfig(path, SCHEMA_VERSION, judge)
