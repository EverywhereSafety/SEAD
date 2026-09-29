"""Private checkpoint parsing for online DART path scoring."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


class CheckpointDefinitionError(ValueError):
    """Raised when a present checkpoint document is not fully scoreable."""


_HEADING = re.compile(
    r"^#{2,6}\s*Checkpoint\s+(.+?)\s*"
    r"\(\s*([0-9]+(?:\.[0-9]+)?)\s*(?:pt|pts|point|points)\s*\)\s*$",
    re.IGNORECASE,
)
_ANY_CHECKPOINT_HEADING = re.compile(r"^#{2,6}\s*Checkpoint\b", re.IGNORECASE)
_TOTAL = re.compile(
    r"\b(?:there\s+(?:is|are)|this\s+task\s+has)\s+"
    r"([0-9]+(?:\.[0-9]+)?)\s*(?:point|points|pt|pts)\s+in\s+total\b",
    re.IGNORECASE,
)
_TOTAL_ALTERNATES = (
    re.compile(r"\btotal(?:\s+score|\s+points?)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\b", re.IGNORECASE),
    re.compile(r"\btotal\s+of\s+([0-9]+(?:\.[0-9]+)?)\s*(?:point|points|pt|pts)\b", re.IGNORECASE),
    re.compile(r"\b([0-9]+(?:\.[0-9]+)?)\s*(?:point|points|pt|pts)\s+total\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class Checkpoint:
    checkpoint_id: str
    text: str
    weight: float

    def public_dict(self) -> dict[str, object]:
        return {"checkpoint_id": self.checkpoint_id, "weight": self.weight}


@dataclass(frozen=True)
class CheckpointSet:
    checkpoints: tuple[Checkpoint, ...]
    source: str

    def __post_init__(self) -> None:
        if not self.checkpoints:
            raise CheckpointDefinitionError("at least one checkpoint is required")
        ids = [item.checkpoint_id for item in self.checkpoints]
        if len(ids) != len(set(ids)):
            raise CheckpointDefinitionError("checkpoint IDs must be unique")
        if any(not math.isfinite(item.weight) or item.weight <= 0 for item in self.checkpoints):
            raise CheckpointDefinitionError("checkpoint weights must be finite and positive")

    @property
    def total_weight(self) -> float:
        return sum(item.weight for item in self.checkpoints)

    @property
    def fingerprint(self) -> str:
        private_definition = [
            {
                "checkpoint_id": item.checkpoint_id,
                "text": item.text,
                "weight": item.weight,
            }
            for item in self.checkpoints
        ]
        return hashlib.sha256(
            json.dumps(
                private_definition,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def public_manifest(self) -> dict[str, object]:
        return {
            "definition_sha256": self.fingerprint,
            "checkpoint_count": len(self.checkpoints),
            "total_weight": self.total_weight,
            "checkpoints": [item.public_dict() for item in self.checkpoints],
        }


def _stable_id(label: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", label.casefold()).strip("-")
    if not normalized:
        raise CheckpointDefinitionError("checkpoint ID is empty")
    return f"checkpoint-{normalized}"


def parse_checkpoints(text: str, *, source: str = "checkpoints.md") -> CheckpointSet:
    """Parse every weighted checkpoint section in a Markdown document."""

    if not isinstance(text, str) or not text.strip():
        raise CheckpointDefinitionError(f"{source} is empty")
    lines = text.splitlines()
    headings: list[tuple[int, re.Match[str]]] = []
    for index, line in enumerate(lines):
        if not _ANY_CHECKPOINT_HEADING.match(line.strip()):
            continue
        match = _HEADING.match(line.strip())
        if match is None:
            raise CheckpointDefinitionError(
                f"{source} has an invalid checkpoint heading on line {index + 1}"
            )
        headings.append((index, match))
    if not headings:
        raise CheckpointDefinitionError(f"{source} contains no weighted checkpoints")

    parsed: list[Checkpoint] = []
    for position, (line_index, match) in enumerate(headings):
        end = headings[position + 1][0] if position + 1 < len(headings) else len(lines)
        body = "\n".join(lines[line_index + 1 : end]).strip()
        if not body:
            raise CheckpointDefinitionError(
                f"{source} checkpoint {match.group(1)!r} has no criterion text"
            )
        weight = float(match.group(2))
        if not math.isfinite(weight) or weight <= 0:
            raise CheckpointDefinitionError(
                f"{source} checkpoint {match.group(1)!r} has an invalid weight"
            )
        parsed.append(
            Checkpoint(
                checkpoint_id=_stable_id(match.group(1)),
                text=body,
                weight=weight,
            )
        )

    declared = [float(match.group(1)) for match in _TOTAL.finditer(text)]
    for pattern in _TOTAL_ALTERNATES:
        declared.extend(float(match.group(1)) for match in pattern.finditer(text))
    if declared:
        if any(not math.isfinite(value) or value <= 0 for value in declared):
            raise CheckpointDefinitionError(f"{source} declares an invalid total")
        if any(
            not math.isclose(value, sum(item.weight for item in parsed), rel_tol=1e-9, abs_tol=1e-9)
            for value in declared
        ):
            raise CheckpointDefinitionError(
                f"{source} declared total does not match checkpoint weights"
            )
    return CheckpointSet(tuple(parsed), source=source)


def load_checkpoints(path: Path | str | None, *, harmful_goal: str) -> CheckpointSet:
    """Load a private checkpoint file or use the harmful goal as one 1pt item."""

    checkpoint_path = Path(path) if path is not None else None
    if checkpoint_path is not None and checkpoint_path.is_file():
        return parse_checkpoints(
            checkpoint_path.read_text(encoding="utf-8"),
            source=str(checkpoint_path),
        )
    goal = str(harmful_goal).strip()
    if not goal:
        raise CheckpointDefinitionError("harmful goal fallback is empty")
    return CheckpointSet(
        (Checkpoint("checkpoint-1", goal, 1.0),),
        source="harmful_goal_fallback",
    )


__all__ = [
    "Checkpoint",
    "CheckpointDefinitionError",
    "CheckpointSet",
    "load_checkpoints",
    "parse_checkpoints",
]
