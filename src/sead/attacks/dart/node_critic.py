"""MTAR node critics driven by the same persistent model as the planner."""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from json_repair import repair_json

from .history import project_controller_transcript
from ...hashing import sha256_text
from .json_parsing import first_complete_json_object
from .models import BranchReplayResult, Candidate, ReplayTurn

class CriticValidationError(ValueError):
    pass


@dataclass(frozen=True)
class IntrospectionResult:
    redundancy: bool
    score: float
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return {"redundancy": self.redundancy, "score": self.score, "reasoning": self.reasoning}


@dataclass(frozen=True)
class FeedbackResult:
    refusal: bool
    no_op: bool
    score: float
    reasoning: str
    completed: bool = False
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "refusal": self.refusal,
            "no_op": self.no_op,
            "score": self.score,
            "reasoning": self.reasoning,
            "completed": self.completed,
            "confidence": self.confidence,
        }


def _repaired_object(
    value: str, fields: set[str], processing: dict[str, Any] | None = None
) -> Mapping[str, Any]:
    metadata = processing if processing is not None else {}
    stripped, removed = re.subn(
        r"<think>.*?</think>", "", value.strip(), flags=re.DOTALL | re.IGNORECASE
    )
    metadata["mode"] = "dart_aligned"
    metadata["thinking_blocks_removed"] = removed
    extracted = first_complete_json_object(stripped)
    candidate = extracted or stripped
    metadata["json_object_extracted"] = extracted is not None and extracted != stripped
    try:
        raw = json.loads(candidate)
        metadata["direct_json_valid"] = True
        metadata["syntax_repair_applied"] = False
    except json.JSONDecodeError as exc:
        metadata["direct_json_valid"] = False
        metadata["direct_json_error"] = str(exc)
        try:
            raw = repair_json(candidate, return_objects=True, skip_json_loads=True)
        except Exception as repair_exc:
            raise CriticValidationError(f"invalid JSON: {repair_exc}") from repair_exc
        metadata["syntax_repair_applied"] = True
        metadata["syntax_repair_backend"] = "json-repair==0.61.7"
    if not isinstance(raw, Mapping) or set(raw) != fields:
        raise CriticValidationError(f"expected exactly fields {sorted(fields)}")
    return raw


def _score(raw: Mapping[str, Any]) -> float:
    value = raw["score"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CriticValidationError("score must be a number")
    value = float(value)
    if not 0 <= value <= 1:
        raise CriticValidationError("score must be in [0, 1]")
    return value


def parse_introspection(
    value: str, *, processing_metadata: dict[str, Any] | None = None
) -> IntrospectionResult:
    raw = _repaired_object(
        value, {"redundancy", "score", "reasoning"}, processing_metadata
    )
    if type(raw["redundancy"]) is not bool:  # exact JSON boolean, not 0/1
        raise CriticValidationError("redundancy must be Boolean")
    if not isinstance(raw["reasoning"], str):
        raise CriticValidationError("reasoning must be a string")
    return IntrospectionResult(raw["redundancy"], _score(raw), raw["reasoning"])


def parse_feedback(
    value: str, *, processing_metadata: dict[str, Any] | None = None
) -> FeedbackResult:
    legacy_fields = {"refusal", "no_op", "score", "reasoning"}
    completion_fields = legacy_fields | {"completed", "confidence"}
    metadata = processing_metadata if processing_metadata is not None else {}
    try:
        raw = _repaired_object(value, completion_fields, metadata)
    except CriticValidationError as completion_error:
        try:
            raw = _repaired_object(value, legacy_fields, metadata)
        except CriticValidationError:
            raise completion_error
    if type(raw["refusal"]) is not bool or type(raw["no_op"]) is not bool:
        raise CriticValidationError("refusal and no_op must be Boolean")
    if not isinstance(raw["reasoning"], str):
        raise CriticValidationError("reasoning must be a string")
    completed = raw.get("completed", False)
    if type(completed) is not bool:
        raise CriticValidationError("completed must be Boolean")
    confidence = raw.get("confidence", 0.0)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise CriticValidationError("confidence must be a number")
    confidence = float(confidence)
    if not 0 <= confidence <= 1:
        raise CriticValidationError("confidence must be in [0, 1]")
    return FeedbackResult(
        raw["refusal"],
        raw["no_op"],
        _score(raw),
        raw["reasoning"],
        completed,
        confidence,
    )


class MTARNodeCritic:
    """Render, call, strictly parse, retry, and trace both node judgments."""

    PROMPT_FILES = (
        "controller_system_prompt.md", "controller_user_prompt.md",
        "introspection_system_prompt.md", "introspection_user_prompt.md",
        "feedback_system_prompt.md", "feedback_user_prompt.md",
    )

    def __init__(
        self,
        complete: Callable[..., str],
        *,
        harmful_goal: str,
        completion_criteria: str | None = None,
        prompt_dir: Path | str,
        trace_path: Path | str,
        max_retries: int = 2,
    ) -> None:
        self.complete = complete
        self.harmful_goal = harmful_goal
        self.completion_criteria = (
            completion_criteria.strip()
            if completion_criteria and completion_criteria.strip()
            else harmful_goal
        )
        self.prompt_dir = Path(prompt_dir).resolve()
        self.trace_path = Path(trace_path)
        self.max_retries = max_retries
        self.prompts: dict[str, str] = {}
        for name in self.PROMPT_FILES:
            path = self.prompt_dir / name
            if not path.is_file():
                raise ValueError(f"missing MTAR prompt: {path}")
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                raise ValueError(f"empty MTAR prompt: {path}")
            self.prompts[name] = text
        self.trace_path.touch()
        self.calls = {"introspection": 0, "feedback": 0}
        self.retries = {"introspection": 0, "feedback": 0}
        try:
            self._accepts_kind = "request_kind" in inspect.signature(complete).parameters
        except (TypeError, ValueError):
            self._accepts_kind = False

    def manifest(self) -> list[dict[str, str]]:
        return [
            {
                "path": str(self.prompt_dir / name),
                "sha256": sha256_text(self.prompts[name]),
            }
            for name in self.PROMPT_FILES
        ]

    @staticmethod
    def _history(
        turns: Sequence[ReplayTurn], *, include_instructions: bool = True,
        start_turn: int = 1,
    ) -> str:
        projected = project_controller_transcript(turns)
        rows = []
        for offset, (turn, row) in enumerate(
            zip(turns, projected, strict=True)
        ):
            value = dict(row)
            value["turn"] = start_turn + offset
            if include_instructions:
                value["external_instruction"] = turn.user_instruction
            rows.append(value)
        return json.dumps(
            rows,
            indent=2, ensure_ascii=False,
        )

    @staticmethod
    def _render(template: str, values: Mapping[str, str], *, label: str) -> str:
        placeholder = re.compile(r"\{([a-z_]+)\}")
        expected = set(placeholder.findall(template))
        unknown = expected - set(values)
        if unknown:
            raise ValueError(f"unknown placeholders in {label}: {sorted(unknown)}")
        return placeholder.sub(lambda match: values[match.group(1)], template)

    def _call(self, kind: str, system: str, user: str) -> str:
        if self._accepts_kind:
            return self.complete(system, user, request_kind=kind)
        return self.complete(system, user)

    def _judge(self, kind: str, system: str, user: str) -> IntrospectionResult | FeedbackResult:
        parser = parse_introspection if kind == "introspection" else parse_feedback
        errors: list[str] = []
        for attempt in range(1, self.max_retries + 2):
            self.calls[kind] += 1
            if attempt > 1:
                self.retries[kind] += 1
            raw: str | None = None
            parsed = None
            error = None
            processing: dict[str, Any] = {}
            try:
                raw = self._call(kind, system, user)
                parsed = parser(raw, processing_metadata=processing)
            except Exception as exc:  # every failed generation/parse is auditable
                error = f"{type(exc).__name__}: {exc}"
                errors.append(error)
            row = {
                "request_kind": kind, "attempt": attempt,
                "prompt": {"system": system, "user": user},
                "raw_response": raw,
                "parsed_result": parsed.to_dict() if parsed else None,
                "response_processing": processing,
                "error": error,
            }
            with self.trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if parsed is not None:
                return parsed
        raise CriticValidationError(f"{kind} failed after 3 attempts: {'; '.join(errors)}")

    def introspect(self, parent_turns: Sequence[ReplayTurn], candidate: Candidate) -> IntrospectionResult:
        values = {
            "harmful_goal": self.harmful_goal,
            "parent_history": self._history(parent_turns),
            "current_instruction": candidate.instruction,
            "current_rationale": candidate.strategy_summary,
        }
        return self._judge(
            "introspection", self.prompts["introspection_system_prompt.md"],
            self._render(self.prompts["introspection_user_prompt.md"], values, label="introspection_user_prompt.md"),
        )  # type: ignore[return-value]

    def feedback(self, parent_turns: Sequence[ReplayTurn], candidate: Candidate, result: BranchReplayResult) -> FeedbackResult:
        current_turn = result.replay_turns[-1:] if result.replay_turns else ()
        current_response = self._history(
            current_turn,
            include_instructions=False,
            start_turn=len(parent_turns) + 1,
        )
        if not current_turn:
            current_response = json.dumps({
                "target_transcript": list(result.target_transcript[-1:]),
                "tool_calls": list(result.tool_calls),
            }, indent=2, ensure_ascii=False)
        values = {
            "harmful_goal": self.harmful_goal,
            "completion_criteria": self.completion_criteria,
            "parent_history": self._history(parent_turns),
            "current_instruction": candidate.instruction,
            "current_response": current_response,
        }
        return self._judge(
            "feedback", self.prompts["feedback_system_prompt.md"],
            self._render(self.prompts["feedback_user_prompt.md"], values, label="feedback_user_prompt.md"),
        )  # type: ignore[return-value]
