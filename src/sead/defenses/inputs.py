"""Versioned normalization of the history visible to the online tool gate.

Normalize only the explicit pre-action prefix; never consult future events to
construct a model input. IDs retain their source values, represented as strings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from enum import Enum
import json
from typing import Any

CONTEXT_POLICY_ID = "openhands-v2-full-user-clean-v2"
NORMALIZER_VERSION = "openhands_tool_events_v2_clean_framework_v1"
FRAMEWORK_FINISH_MESSAGE = "All done! What's next on the agenda?"


class InvalidTrajectory(ValueError):
    """Invalid history or candidate action at the online model boundary."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason


def _require(condition: bool, reason: str, detail: str) -> None:
    if not condition:
        raise InvalidTrajectory(reason, detail)


_NON_TOOL_ACTIONS = {
    "",
    "change_agent_state",
    "finish",
    "message",
    "reject",
    "system",
    "task_tracking",
    "think",
}


def _value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _explicit_tool_name(row: Mapping[str, Any]) -> str:
    metadata = row.get("tool_call_metadata")
    name = row.get("tool_name") or (
        metadata.get("function_name") if isinstance(metadata, Mapping) else None
    )
    return str(_value(name)) if name else ""


def compact_event(row: Mapping[str, Any]) -> dict[str, Any]:
    """Project one event using the existing v2 data contract, without filtering.

    Explicit empty output is evidence. Only absent/null content falls back to
    the UI message; an empty tool result must not become an execution summary.
    """
    event = {
        key: _value(row[key])
        for key in ("id", "source", "action", "observation", "cause")
        if row.get(key) is not None
    }
    for key in ("id", "cause"):
        if key in event:
            _require(
                type(event[key]) in (int, str) and bool(str(event[key]).strip()),
                "invalid_event_id",
                f"invalid {key}: {event[key]!r}",
            )
            event[key] = str(event[key])
    if isinstance(row.get("args"), Mapping):
        event["args"] = deepcopy(dict(row["args"]))
    if row.get("content") is not None:
        event["content"] = deepcopy(row["content"])
    elif row.get("message") is not None:
        event["content"] = deepcopy(row["message"])
    if name := _explicit_tool_name(row):
        event["tool_name"] = name
    return event


def tool_action_for_model(action: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the canonical candidate fields used by every model adapter.

    Native events must first go through tool_action_from_event. Do not confuse
    their ``action`` field with either ``action_type`` or the agentic protocol.
    """
    action_id = action.get("action_id")
    action_type = _value(action.get("action_type"))
    if type(action_id) not in (int, str) or not str(action_id).strip():
        raise ValueError("tool action has missing or invalid action_id")
    if not isinstance(action_type, str) or not action_type.strip():
        raise ValueError("tool action is missing action_type")
    arguments = action.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError("tool action arguments must be an object")
    return {
        "action_id": str(action_id),
        "action_type": action_type,
        "tool_name": _explicit_tool_name(action) or action_type,
        "arguments": deepcopy(dict(arguments)),
    }


def tool_action_from_event(row: Mapping[str, Any]) -> dict[str, Any] | None:
    if str(_value(row.get("source")) or "").casefold() != "agent":
        return None
    action_type = str(_value(row.get("action")) or "").casefold()
    if action_type in _NON_TOOL_ACTIONS:
        return None
    if row.get("id") is None:
        raise ValueError("tool action event is missing id")
    arguments = row.get("args")
    return tool_action_for_model(
        {
            "action_id": row["id"],
            "action_type": action_type,
            "tool_name": _explicit_tool_name(row) or action_type,
            "arguments": dict(arguments) if isinstance(arguments, Mapping) else {},
        }
    )


def normalize_history(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize the explicit pre-action prefix, preserving literal /exit requests."""
    rows = []
    for index, event in enumerate(events):
        _require(
            isinstance(event, Mapping),
            "invalid_events",
            f"event {index} is not an object",
        )
        row = dict(event)
        for key in ("source", "action", "observation"):
            if key in row:
                row[key] = _value(row[key])
        rows.append(row)
    seen: set[str] = set()
    compact = []
    referenced_ids = {
        str(event["cause"])
        for event in rows
        if isinstance(event, dict) and event.get("cause") is not None
    }
    for index, event in enumerate(rows):
        _require(
            isinstance(event, dict), "invalid_events", f"event {index} is not an object"
        )
        native_id = event.get("id")
        _require(
            native_id is None
            or (type(native_id) in (int, str) and str(native_id) != ""),
            "invalid_event_id",
            f"invalid ID at event {index}",
        )
        event_id = None if native_id is None else str(native_id)
        _require(
            event_id is None or event_id not in seen,
            "invalid_event_id",
            f"duplicate event ID {event_id}",
        )
        if event.get("args") is not None:
            _require(
                isinstance(event["args"], dict),
                "invalid_events",
                "args must be an object",
            )
        try:
            tool_action_from_event(event)
        except ValueError as exc:
            raise InvalidTrajectory("invalid_event_id", str(exc)) from exc
        normalized = compact_event(event)
        if event_id is not None:
            seen.add(event_id)
        # AgentFinishAction.message is framework UI text, not a Target utterance.
        # Match the typed event and whole message, never text inside tool arguments,
        # observations, user messages, or genuine answers quoting this sentence.
        if (
            event.get("source") == "agent"
            and event.get("action") == "finish"
            and str(event.get("message") or event.get("content") or "").strip()
            == FRAMEWORK_FINISH_MESSAGE
        ):
            if str(normalized.get("content") or "").strip() == FRAMEWORK_FINISH_MESSAGE:
                normalized.pop("content", None)
            if (
                not normalized.get("content")
                and not any(normalized.get("args", {}).values())
                and event_id not in referenced_ids
            ):
                continue
        compact.append(normalized)
    return compact


def normalize_tool_defense_input(
    history: Sequence[Mapping[str, Any]], tool_action: Mapping[str, Any]
) -> dict[str, Any]:
    """Idempotent model boundary for native or already normalized evidence."""
    action = tool_action_for_model(tool_action)
    _require(
        all(
            str(event.get("id")) != action["action_id"]
            and str(event.get("cause")) != action["action_id"]
            for event in history
        ),
        "candidate_in_history",
        "current candidate or its result must not appear in prior history",
    )
    return {"history": normalize_history(history), "tool_action": action}


def limit_history(
    history: Sequence[Mapping[str, Any]],
    *,
    history_event_limit: int | None,
    preserve_latest_user_message: bool,
) -> list[dict[str, Any]]:
    if history_event_limit is not None and history_event_limit < 0:
        raise ValueError("history_event_limit must be non-negative or None")
    values = [dict(event) for event in history]
    if history_event_limit is None:
        return values
    if history_event_limit == 0:
        return []
    limited = values[-history_event_limit:]
    if not preserve_latest_user_message or any(
        event.get("source") == "user" and event.get("action") == "message"
        for event in limited
    ):
        return limited
    latest_user = next(
        (
            event
            for event in reversed(values[:-history_event_limit])
            if event.get("source") == "user" and event.get("action") == "message"
        ),
        None,
    )
    return limited if latest_user is None else [latest_user, *limited[1:]]


def limit_history_chars(
    history: Sequence[Mapping[str, Any]],
    *,
    max_chars: int | None,
    preserve_latest_user_message: bool,
) -> list[dict[str, Any]]:
    """Retain the largest recent event suffix within a serialized-char budget."""
    values = [dict(event) for event in history]
    if max_chars is None:
        return values
    if max_chars < 1_000:
        raise ValueError("max_chars must be at least 1000 or None")
    latest_user = next(
        (
            event for event in reversed(values)
            if event.get("source") == "user" and event.get("action") == "message"
        ),
        None,
    ) if preserve_latest_user_message else None
    selected: list[dict[str, Any]] = []
    used = 0
    for event in reversed(values):
        if event is latest_user:
            continue
        size = len(json.dumps(event, ensure_ascii=False, sort_keys=True, default=str))
        if selected and used + size > max_chars:
            break
        if not selected and size > max_chars:
            continue
        selected.append(event)
        used += size
    selected.reverse()
    if latest_user is not None and latest_user not in selected:
        selected.insert(0, latest_user)
    return selected


def build_tool_defense_input(
    prior_events: Sequence[Mapping[str, Any]],
    action_event: Mapping[str, Any],
    *,
    history_event_limit: int | None = None,
    history_max_chars: int | None = None,
    preserve_latest_user_message: bool = False,
) -> dict[str, Any] | None:
    """Build a single case from an explicit pre-action prefix and native candidate."""
    action = tool_action_from_event(action_event)
    if action is None:
        return None
    sample = normalize_tool_defense_input(prior_events, action)
    sample["history"] = limit_history(
        sample["history"],
        history_event_limit=history_event_limit,
        preserve_latest_user_message=preserve_latest_user_message,
    )
    # Apply the shared context policy: event limit first,
    # then the collaborator's serialized-character window (not a token limit).
    sample["history"] = limit_history_chars(
        sample["history"],
        max_chars=history_max_chars,
        preserve_latest_user_message=preserve_latest_user_message,
    )
    return sample
