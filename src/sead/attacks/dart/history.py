"""Deterministic, controller-only projection of MTAR OpenHands replays."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .models import ReplayToolCall, ReplayTurn

MAX_TOOL_RESULT_CHARS = 12_000
TOOL_RESULT_HEAD_CHARS = 8_000
TOOL_RESULT_TAIL_CHARS = 4_000

_INTERNAL_ACTIONS = {
    "agent_state",
    "agent-state",
    "agentstate",
    "system",
    "think",
}


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _mcp_result(content: Any) -> tuple[str, bool | None]:
    """Extract only result text and the error flag from an MCP envelope."""

    envelope = content
    if isinstance(envelope, str):
        try:
            envelope = json.loads(envelope)
        except json.JSONDecodeError:
            return envelope, None
    if not isinstance(envelope, Mapping):
        return _text(envelope), None

    blocks = envelope.get("content")
    texts: list[str] = []
    if isinstance(blocks, Sequence) and not isinstance(blocks, (str, bytes)):
        for block in blocks:
            if isinstance(block, Mapping) and block.get("type") == "text":
                text = block.get("text")
                if text is not None:
                    texts.append(str(text))
    if texts:
        result = "\n".join(texts)
    else:
        structured = envelope.get("structuredContent")
        if isinstance(structured, Mapping) and "content" in structured:
            result = _text(structured["content"])
        else:
            result = ""
    is_error = envelope.get("isError")
    return result, is_error if isinstance(is_error, bool) else None


def _truncate_result(result: str) -> dict[str, Any]:
    original_length = len(result)
    if original_length <= MAX_TOOL_RESULT_CHARS:
        return {"result": result, "result_truncated": False}
    return {
        "result": (
            result[:TOOL_RESULT_HEAD_CHARS]
            + result[-TOOL_RESULT_TAIL_CHARS:]
        ),
        "result_truncated": True,
        "original_result_length": original_length,
    }


def _project_tool_call(call: ReplayToolCall) -> dict[str, Any] | None:
    function = call.function
    arguments = dict(call.arguments)
    if function == "call_tool_mcp":
        tool = str(arguments.get("name") or "").strip()
        raw_arguments = arguments.get("arguments")
        actual_arguments = (
            dict(raw_arguments) if isinstance(raw_arguments, Mapping) else {}
        )
        result, envelope_error = _mcp_result(call.result.content)
    else:
        tool = function
        actual_arguments = {
            key: value for key, value in arguments.items() if key != "thought"
        }
        result = _text(call.result.content)
        envelope_error = None

    if tool.casefold() in _INTERNAL_ACTIONS:
        return None
    error = call.result.error
    failed = error not in (None, "", False) or envelope_error is True
    if not result and error not in (None, "", False):
        result = _text(error)
    return {
        "type": "tool",
        "tool": tool,
        "arguments": actual_arguments,
        "status": "failed" if failed else "success",
        **_truncate_result(result),
    }


def project_controller_transcript(
    replay_turns: Sequence[ReplayTurn],
) -> tuple[Mapping[str, Any], ...]:
    """Project full replay records into the minimal MTAR Controller history.

    Call IDs are used only to suppress duplicate replay records and are never
    emitted. User instructions remain in ``ControllerContext.instructions``.
    """

    projected_turns: list[Mapping[str, Any]] = []
    for turn_index, turn in enumerate(replay_turns, 1):
        events: list[Mapping[str, Any]] = []
        seen_call_ids: set[str] = set()
        for message in turn.assistant_messages:
            action = str(message.native_action.get("action") or "").casefold()
            if action in _INTERNAL_ACTIONS:
                continue
            if message.tool_calls:
                for call in message.tool_calls:
                    if call.id in seen_call_ids:
                        continue
                    seen_call_ids.add(call.id)
                    projected = _project_tool_call(call)
                    if projected is not None:
                        events.append(projected)
                # OpenHands repeats tool intent in message.content. The actual
                # function and arguments above are the authoritative record.
                continue
            if message.content:
                events.append({"type": "target_message", "text": message.content})
        projected_turns.append({"turn": turn_index, "events": events})
    return tuple(projected_turns)
