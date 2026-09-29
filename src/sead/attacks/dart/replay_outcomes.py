"""Shared tool-outcome equivalence rules for hard replay workers."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class ToolOutcome:
    succeeded: bool
    failure_type: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "succeeded": self.succeeded,
            "failure_type": self.failure_type,
        }


@dataclass(frozen=True)
class ToolOutcomeComparison:
    equivalent: bool
    recorded: ToolOutcome
    actual: ToolOutcome
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "equivalent": self.equivalent,
            "recorded": self.recorded.to_dict(),
            "actual": self.actual.to_dict(),
            "reason": self.reason,
        }


def _json_mapping(value: object) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _message(value: object) -> str:
    if isinstance(value, Mapping):
        parts = [
            str(value.get(name) or "")
            for name in ("type", "message", "content", "error")
        ]
        return " ".join(part for part in parts if part)
    return str(value or "")


def _failure_type(message: str, *, declared_type: str | None = None) -> str:
    lowered = message.casefold()
    patterns = (
        (r"\benoent\b|no such file|not found", "not_found"),
        (r"\beacces\b|permission denied|not permitted", "permission_denied"),
        (r"timed? ?out|timeout", "timeout"),
        (r"connection (?:closed|refused|reset)|unreachable", "connection"),
        (r"invalid (?:argument|input)|validation|parsing|parse error", "invalid_input"),
    )
    for pattern, name in patterns:
        if re.search(pattern, lowered):
            return name
    if declared_type:
        return f"declared:{declared_type.casefold()}"
    return "tool_error"


def classify_tool_outcome(
    *,
    error: object = None,
    content: object = "",
    observation: object = None,
    exit_code: object = None,
) -> ToolOutcome:
    """Normalize Inspect and OpenHands results into comparable outcomes."""

    if error not in (None, {}, ""):
        declared = str(error.get("type") or "") if isinstance(error, Mapping) else ""
        return ToolOutcome(
            False,
            _failure_type(_message(error), declared_type=declared or None),
        )
    if exit_code not in (None, 0, "0"):
        return ToolOutcome(False, f"exit_code:{exit_code}")
    observation_value = (
        observation.value if isinstance(observation, Enum) else observation
    )
    observation_name = str(observation_value or "").casefold()
    if observation_name in {"error", "user_reject"}:
        return ToolOutcome(False, f"observation:{observation_name}")
    payload = _json_mapping(content)
    if payload is not None and payload.get("isError") is True:
        text = _message(payload.get("content") or payload)
        return ToolOutcome(False, _failure_type(text))
    return ToolOutcome(True)


def classify_native_observation(value: Mapping[str, Any]) -> ToolOutcome:
    extras = value.get("extras")
    metadata = extras.get("metadata") if isinstance(extras, Mapping) else None
    exit_code = (extras.get("exit_code") if isinstance(extras, Mapping) else None)
    if exit_code is None and isinstance(metadata, Mapping):
        exit_code = metadata.get("exit_code")
    return classify_tool_outcome(
        error=value.get("error"),
        content=value.get("content") or value.get("message") or "",
        observation=value.get("observation"),
        exit_code=exit_code,
    )


def observation_failed(value: Mapping[str, Any]) -> bool:
    """Return the legacy OpenHands failure predicate used during replay import."""

    if str(value.get("observation") or "").casefold() in {"error", "user_reject"}:
        return True
    extras = value.get("extras")
    if isinstance(extras, Mapping):
        metadata = extras.get("metadata")
        exit_code = extras.get("exit_code")
        if exit_code is None and isinstance(metadata, Mapping):
            exit_code = metadata.get("exit_code")
        if exit_code not in {None, 0}:
            return True
    content = str(value.get("content") or "")
    return '"isError": true' in content or '"isError":true' in content


def compare_tool_outcomes(
    recorded: ToolOutcome, actual: ToolOutcome
) -> ToolOutcomeComparison:
    if recorded.succeeded and actual.succeeded:
        return ToolOutcomeComparison(True, recorded, actual, "both_succeeded")
    if recorded.succeeded != actual.succeeded:
        return ToolOutcomeComparison(
            False,
            recorded,
            actual,
            "success_failure_mismatch",
        )
    if recorded.failure_type == actual.failure_type:
        return ToolOutcomeComparison(True, recorded, actual, "same_failure_type")
    return ToolOutcomeComparison(
        False,
        recorded,
        actual,
        "failure_type_mismatch",
    )
