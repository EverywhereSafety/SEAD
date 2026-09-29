"""Extract agentic actions without treating a reasoning draft as a tool call."""

from __future__ import annotations

import json
import re

from ..attacks.dart.json_parsing import first_complete_json_object


def final_action_json(raw: str) -> str | None:
    """Skip a completed Qwen thinking segment, including a prefilled opening tag.

    Pure JSON is checked first so literal thinking markers in rationale strings
    remain data. Preserve the legacy JSON extraction for ordinary prose/fences.
    This is parsing, not a semantic or generation-completeness certification.
    """
    text = raw.strip()
    try:
        value = json.loads(text)
    except ValueError:
        pass
    else:
        if isinstance(value, dict):
            return text
    endings = list(re.finditer(r"(?:^|\n)\s*</think>\s*(?=\n|$)", text))
    if endings:
        text = text[endings[-1].end():].strip()
    elif text.startswith("<think>"):
        raise ValueError("unfinished thinking segment; no final action")
    return first_complete_json_object(text)
