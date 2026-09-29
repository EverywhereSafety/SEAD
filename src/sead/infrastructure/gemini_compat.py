"""Preserve native Gemini call IDs in LiteLLM 1.80's signed-part conversion."""

from __future__ import annotations

import copy
import json
from functools import wraps


def raise_on_signature_error(error: object) -> None:
    text = str(error)
    if any(marker in text.lower() for marker in ("thought_signature", "thoughtsignature", "thought signature")):
        raise RuntimeError("Target thought-signature protocol failure: " + text)


def _native_calls(message):
    """Map LiteLLM's generated tool IDs to the original signed function calls."""
    tools = list(message.get("tool_calls") or [])
    result = {}
    for block in message.get("thinking_blocks") or []:
        if block.get("type") != "thinking" or not block.get("signature"):
            continue
        try:
            part = json.loads(block.get("thinking", ""))
        except (ValueError, TypeError):
            continue
        if not isinstance(part, dict):
            continue
        native = part.get("functionCall", part.get("function_call"))
        if not isinstance(native, dict) or not native.get("id"):
            continue
        for index, tool in enumerate(tools):
            function = tool.get("function", {})
            if function.get("name") != native.get("name"):
                continue
            try:
                arguments = json.loads(function.get("arguments", "{}"))
            except (ValueError, TypeError):
                continue
            if arguments == native.get("args", {}):
                result[tool["id"]] = native
                tools.pop(index)
                break
    return result


def install_gemini_call_id_compat() -> None:
    """Keep signed parts intact and prevent an extra unsigned copy of each call.

    LiteLLM generates its own OpenAI tool ID, drops Gemini's native ID when
    rebuilding a call, then fails to deduplicate it against the signed part.
    Restore the native call before that comparison and use its ID for results.
    """
    from litellm.llms.vertex_ai.gemini import transformation

    invoke = transformation.convert_to_gemini_tool_call_invoke
    if getattr(invoke, "_sead_native_call_ids", False):
        return
    result = transformation.convert_to_gemini_tool_call_result

    @wraps(invoke)
    def signed_invoke(message):
        parts = invoke(message)
        native_calls = _native_calls(message)
        for tool, part in zip(message.get("tool_calls") or [], parts):
            native = native_calls.get(tool.get("id"))
            if native is not None:
                key = "functionCall" if "functionCall" in part else "function_call"
                part[key] = copy.deepcopy(native)
        return parts

    @wraps(result)
    def signed_result(message, last_message_with_tool_calls):
        part = result(message, last_message_with_tool_calls)
        native = _native_calls(last_message_with_tool_calls or {}).get(message.get("tool_call_id"))
        if native is not None:
            key = "functionResponse" if "functionResponse" in part else "function_response"
            part[key]["id"] = native["id"]
        return part

    signed_invoke._sead_native_call_ids = True
    transformation.convert_to_gemini_tool_call_invoke = signed_invoke
    transformation.convert_to_gemini_tool_call_result = signed_result
