"""Check decision/rationale consistency in prepared SFT records."""

from __future__ import annotations

import re


_RATIONALE_CONCLUSION_PATTERNS = (
    re.compile(
        r"(?i)(?:^|[.!?]\s+)(?:(?:therefore|thus|hence|so)\s*[:,]?\s*)?"
        r"(?:this\s+(?:is|should(?:\s+be)?)|it\s+should(?:\s+be)?|"
        r"(?:the\s+)?(?:decision|verdict)\s+should\s+be)\s+(pass|block)\b"
        r"(?!\s+(?:if|unless|when|only\s+if)\b)"
    ),
    re.compile(
        r"(?i)\b(?:correct|final|appropriate)\s+(?:decision|verdict)\s*"
        r"(?:is|:|=)\s*(pass|block)\b"
    ),
    re.compile(
        r"(?i)\b(?:decision|verdict)\s*(?:is|:|=)\s*(pass|block)\b"
    ),
    re.compile(
        r"(?i)\b(?:hence|therefore|thus)\s*[:,]?\s*(?:the\s+)?"
        r"(?:decision\s+(?:is|should\s+be)\s*)?(pass|block)\b"
    ),
    re.compile(r"(?i)\bI\s+(?:should|would)\s+(pass|block)\b"),
    re.compile(r"(?i)(?:^|[.!?]\s+)(pass|block)[.!]?\s*$"),
)


def _explicit_rationale_conclusion(rationale: str) -> str | None:
    """Return the last explicit PASS/BLOCK conclusion, if one is present.

    This deliberately recognizes only high-precision conclusion language. Merely
    discussing PASS or BLOCK in a counterfactual is not enough to reject a trace.
    """

    matches: list[tuple[int, str]] = []
    for pattern in _RATIONALE_CONCLUSION_PATTERNS:
        matches.extend(
            (match.start(1), match.group(1).lower())
            for match in pattern.finditer(rationale)
        )
    return max(matches)[1] if matches else None


def rationale_contradicts_decision(rationale: str, decision: str) -> bool:
    conclusion = _explicit_rationale_conclusion(rationale)
    return conclusion is not None and conclusion != decision
