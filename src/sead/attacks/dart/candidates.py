"""Structured controller output parsing and deterministic candidate checks."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any, Callable, Mapping, Sequence

from json_repair import repair_json

from .json_parsing import first_complete_json_object
from .models import Candidate, CandidateBatch, CandidateBatchController, ControllerContext


class CandidateValidationError(ValueError):
    pass


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def _tokens(text: str) -> set[str]:
    return set(_normalize(text).split())


def semantically_near_duplicate(left: str, right: str) -> bool:
    """Conservative local filter; no evaluator or model signal is consulted."""

    a, b = _normalize(left), _normalize(right)
    if not a or not b:
        return False
    if a == b:
        return True
    sequence = SequenceMatcher(a=a, b=b).ratio()
    left_tokens, right_tokens = _tokens(a), _tokens(b)
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
    containment = min(len(left_tokens), len(right_tokens))
    containment_ratio = (
        len(left_tokens & right_tokens) / containment if containment else 0.0
    )
    return (
        sequence >= 0.88
        or (jaccard >= 0.78 and containment_ratio >= 0.88)
        or (
            containment >= 3
            and containment_ratio >= 0.95
            and sequence >= 0.75
        )
    )


def deduplicate_candidate_batch(batch: CandidateBatch) -> CandidateBatch:
    """Keep one stable representative from each semantic candidate group.

    Duplicate candidates are a usable Controller response, not a generation
    failure.  Preserve the first candidate in Controller order and normalize
    the strategy to the number of candidates that remain.
    """

    unique: list[Candidate] = []
    for candidate in batch.candidates:
        if any(
            semantically_near_duplicate(candidate.instruction, kept.instruction)
            for kept in unique
        ):
            continue
        unique.append(candidate)

    if len(unique) == len(batch.candidates):
        return batch
    if not unique:
        raise CandidateValidationError("candidate batch is empty after deduplication")

    strategy = "SINGLE" if len(unique) == 1 else "BRANCH"
    rationale = str(batch.strategy_rationale or "").strip()
    deduplication_note = (
        f"Semantic deduplication retained {len(unique)} of "
        f"{len(batch.candidates)} candidates."
    )
    rationale = f"{rationale} {deduplication_note}".strip()
    parallel_verification = (
        "N/A"
        if strategy == "SINGLE"
        else str(batch.parallel_verification or "").strip()
        or "The retained candidates are semantically distinct."
    )
    return CandidateBatch(
        candidates=tuple(unique),
        strategy=strategy,
        strategy_rationale=rationale,
        parallel_verification=parallel_verification,
    )


_DEPENDENCY_PATTERNS = (
    r"\b(?:previous|preceding|earlier|first|other)\s+candidate\b",
    r"\bcandidate\s+(?:one|two|1|2|a|b)\b",
    (
        r"\b(?:what|file|value|result|output|content)\s+"
        r"(?:it|that candidate)\s+(?:created|produced|returned)\b"
    ),
)


def validate_candidate_batch(
    batch: CandidateBatch,
    *,
    branching_factor: int,
) -> None:
    candidates = batch.candidates
    if batch.strategy is None:
        expected = branching_factor
        valid_count = len(candidates) == expected
        count_error = f"expected exactly {expected} candidates; received {len(candidates)}"
    elif batch.strategy == "SINGLE":
        valid_count = len(candidates) == 1
        count_error = f"SINGLE requires exactly 1 candidate; received {len(candidates)}"
    elif batch.strategy == "BRANCH":
        valid_count = 2 <= len(candidates) <= branching_factor
        count_error = (
            f"BRANCH requires 2 to {branching_factor} candidates; "
            f"received {len(candidates)}"
        )
    else:
        raise CandidateValidationError("strategy must be SINGLE or BRANCH")
    if not valid_count:
        raise CandidateValidationError(count_error)
    if batch.strategy is not None:
        if not str(batch.strategy_rationale or "").strip():
            raise CandidateValidationError("strategy_rationale is required")
        parallel = str(batch.parallel_verification or "").strip()
        if not parallel:
            raise CandidateValidationError("parallel_verification is required")
        if batch.strategy == "SINGLE" and parallel.casefold() != "n/a":
            raise CandidateValidationError(
                "SINGLE parallel_verification must be N/A"
            )
        if batch.strategy == "BRANCH" and parallel.casefold() == "n/a":
            raise CandidateValidationError(
                "BRANCH requires an independence explanation"
            )
    for index, candidate in enumerate(candidates):
        instruction = candidate.instruction.strip()
        if not instruction or len(instruction) > 4000:
            raise CandidateValidationError(f"candidate {index} has invalid instruction")
        if not candidate.strategy_summary.strip():
            raise CandidateValidationError(
                f"candidate {index} has no strategy_summary"
            )
        if not candidate.expected_state_change.strip():
            raise CandidateValidationError(
                f"candidate {index} has no expected_state_change"
            )
        for pattern in _DEPENDENCY_PATTERNS:
            if re.search(
                pattern,
                " ".join(
                    (
                        instruction,
                        candidate.strategy_summary,
                        candidate.expected_state_change,
                    )
                ),
                re.IGNORECASE,
            ):
                raise CandidateValidationError(
                    f"candidate {index} depends on a sibling candidate"
                )
        for other in candidates[:index]:
            if semantically_near_duplicate(instruction, other.instruction):
                raise CandidateValidationError(
                    f"candidates {index - 1} and {index} are semantic duplicates"
                )


_THINKING_BLOCK = re.compile(
    r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE
)


def _load_candidate_batch_json(
    text: str,
    *,
    allow_json_repair: bool,
    processing_metadata: dict[str, Any],
) -> Any:
    """Load one Controller object using the DART-aligned unconstrained path."""

    stripped = text.strip()
    if not allow_json_repair:
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"\s*```$", "", stripped)
        return json.loads(stripped)

    without_thinking, thinking_blocks = _THINKING_BLOCK.subn("", stripped)
    processing_metadata["thinking_blocks_removed"] = thinking_blocks
    without_thinking = without_thinking.strip()
    if without_thinking.startswith("```"):
        without_thinking = re.sub(
            r"^```(?:json)?\s*", "", without_thinking, flags=re.IGNORECASE
        )
        without_thinking = re.sub(r"\s*```$", "", without_thinking)

    extracted = first_complete_json_object(without_thinking)
    if extracted is None:
        first = without_thinking.find("{")
        last = without_thinking.rfind("}")
        extracted = (
            without_thinking[first : last + 1]
            if first >= 0 and last > first
            else without_thinking
        )
    processing_metadata["json_object_extracted"] = extracted != without_thinking

    try:
        raw = json.loads(extracted)
        processing_metadata["direct_json_valid"] = True
        processing_metadata["syntax_repair_applied"] = False
        return raw
    except json.JSONDecodeError as direct_error:
        processing_metadata["direct_json_valid"] = False
        processing_metadata["direct_json_error"] = str(direct_error)

    repaired = repair_json(
        extracted,
        return_objects=True,
        skip_json_loads=True,
    )
    processing_metadata["syntax_repair_applied"] = True
    processing_metadata["syntax_repair_backend"] = "json-repair==0.61.7"
    return repaired


def parse_candidate_batch(
    value: str | Mapping[str, Any],
    *,
    allow_json_repair: bool = False,
    processing_metadata: dict[str, Any] | None = None,
) -> CandidateBatch:
    metadata = processing_metadata if processing_metadata is not None else {}
    metadata.clear()
    metadata["mode"] = "dart_aligned" if allow_json_repair else "strict"
    if isinstance(value, str):
        try:
            raw: Any = _load_candidate_batch_json(
                value,
                allow_json_repair=allow_json_repair,
                processing_metadata=metadata,
            )
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise CandidateValidationError(
                f"controller returned invalid JSON: {exc}"
            ) from exc
    else:
        raw = value
        metadata["mode"] = "mapping"
    if not isinstance(raw, Mapping) or not isinstance(raw.get("candidates"), list):
        raise CandidateValidationError(
            "controller output must be a CandidateBatch object"
        )
    allowed_outer = {
        "strategy",
        "strategy_rationale",
        "parallel_verification",
        "candidates",
    }
    unknown_outer = set(raw) - allowed_outer
    if unknown_outer:
        raise CandidateValidationError(
            f"CandidateBatch contains unsupported fields: {sorted(unknown_outer)}"
        )
    missing = allowed_outer - set(raw)
    if missing:
        raise CandidateValidationError(
            f"CandidateBatch is missing required fields: {sorted(missing)}"
        )
    strategy = str(raw.get("strategy") or "").strip().upper()
    strategy_rationale = str(raw.get("strategy_rationale") or "").strip()
    parallel_verification = str(raw.get("parallel_verification") or "").strip()
    candidates = []
    for item in raw["candidates"]:
        if not isinstance(item, Mapping):
            raise CandidateValidationError("each candidate must be an object")
        allowed = {"instruction", "strategy_summary", "expected_state_change"}
        unknown = set(item) - allowed
        if unknown:
            raise CandidateValidationError(
                f"candidate contains unsupported fields: {sorted(unknown)}"
            )
        candidates.append(
            Candidate(
                instruction=str(item.get("instruction") or "").strip(),
                strategy_summary=str(item.get("strategy_summary") or "").strip(),
                expected_state_change=str(
                    item.get("expected_state_change") or ""
                ).strip(),
            )
        )
    return CandidateBatch(
        tuple(candidates),
        strategy=strategy,
        strategy_rationale=strategy_rationale,
        parallel_verification=parallel_verification,
    )


class StructuredBatchController(CandidateBatchController):
    """LLM-backed controller whose prompt contains no trusted evaluation fields."""

    def __init__(
        self,
        complete: Callable[[str, str], str],
        *,
        side_task_description: str,
        mtar_system_prompt: str,
        mtar_user_template: str,
        environment_description: str = "",
        maximum_depth: int = 3,
        allow_json_repair: bool = False,
    ):
        self.complete = complete
        self.side_task_description = side_task_description
        self.environment_description = environment_description
        self.allow_json_repair = allow_json_repair
        if not mtar_system_prompt.strip() or not mtar_user_template.strip():
            raise ValueError("MTAR Controller prompts must not be empty")
        self.mtar_system_prompt = mtar_system_prompt
        self.mtar_user_template = mtar_user_template
        self.last_response_processing: dict[str, Any] = {}
        self.supports_variable_width = True
        if maximum_depth < 1:
            raise ValueError("maximum_depth must be positive")
        self.maximum_depth = maximum_depth

    def _mtar_tool_fields(self) -> tuple[str, str, str]:
        """Render only tool-facing MTAR metadata, excluding sandbox internals."""

        try:
            payload = json.loads(self.environment_description)
        except (json.JSONDecodeError, TypeError):
            payload = None
        if not isinstance(payload, Mapping):
            description = self.environment_description.strip() or "Not specified."
            return "Target tool", description, "No additional tools specified."

        raw_contexts = payload.get("target_tool_context")
        contexts = raw_contexts if isinstance(raw_contexts, Mapping) else {}
        tool_family = str(payload.get("tool_family") or "").strip()
        preferred = "terminal" if tool_family == "terminal" else f"mcp-{tool_family}"
        primary = preferred if preferred in contexts else next(iter(contexts), "Target tool")
        raw_primary = contexts.get(primary)
        primary_context = raw_primary if isinstance(raw_primary, Mapping) else {}

        description_lines = [
            str(primary_context.get("description") or "No description supplied.").strip()
        ]
        capabilities = primary_context.get("capabilities")
        if isinstance(capabilities, Sequence) and not isinstance(capabilities, str):
            description_lines.append("Capabilities:")
            description_lines.extend(f"- {value}" for value in capabilities)
        functions = primary_context.get("functions")
        if isinstance(functions, Sequence) and not isinstance(functions, str):
            description_lines.append("Functions:")
            description_lines.extend(f"- {value}" for value in functions)

        other_sections: list[str] = []
        for name, raw_context in contexts.items():
            if name == primary:
                continue
            context = raw_context if isinstance(raw_context, Mapping) else {}
            other_sections.append(f"{name}:")
            other_sections.append(
                f"- Description: {context.get('description') or 'No description supplied.'}"
            )
            raw_functions = context.get("functions")
            if isinstance(raw_functions, Sequence) and not isinstance(
                raw_functions, str
            ):
                other_sections.append(
                    "- Functions: " + ", ".join(str(value) for value in raw_functions)
                )

        raw_native = payload.get("openhands_native_tools")
        if isinstance(raw_native, Mapping) and raw_native:
            other_sections.append("OpenHands native tools:")
            other_sections.extend(
                f"- {name}: {description}"
                for name, description in raw_native.items()
            )
        if not other_sections:
            other_sections.append("No additional tools specified.")
        return primary, "\n".join(description_lines), "\n".join(other_sections)

    def _mtar_service_deployments(self) -> str:
        """Render only dependency-selected, public benchmark deployment data."""

        try:
            payload = json.loads(self.environment_description)
        except (json.JSONDecodeError, TypeError):
            payload = None
        if not isinstance(payload, Mapping):
            return "No external service deployment is selected for this task."

        raw_deployments = payload.get("service_deployments")
        deployments = raw_deployments if isinstance(raw_deployments, Mapping) else {}
        lines = [
            "Use the deployment below when a candidate needs to identify or "
            "navigate the task service."
        ]
        for dependency, raw in deployments.items():
            if not isinstance(raw, Mapping):
                continue
            display_name = str(raw.get("display_name") or dependency)
            url = str(raw.get("url") or "").strip()
            credentials = raw.get("credentials")
            if not url or not isinstance(credentials, Mapping):
                continue
            username = str(credentials.get("username") or "")
            password = str(credentials.get("password") or "")
            lines.extend(
                (
                    f"- Dependency `{dependency}`: {display_name}",
                    f"  - URL: {url}",
                    "  - Public benchmark credentials: "
                    f"username `{username}`; password `{password}`",
                )
            )
            notes = raw.get("operational_notes")
            if isinstance(notes, Sequence) and not isinstance(notes, str):
                lines.extend(f"  - Operational note: {note}" for note in notes)

        raw_missing = payload.get("missing_service_deployments")
        if isinstance(raw_missing, Sequence) and not isinstance(raw_missing, str):
            lines.extend(
                f"- Dependency `{dependency}`: no deployment is configured."
                for dependency in raw_missing
            )
        if len(lines) == 1:
            return "No external service deployment is selected for this task."
        return "\n".join(lines)

    def _mtar_user_prompt(
        self, context: ControllerContext, branching_factor: int
    ) -> str:
        target_tool, target_tool_description, other_tools = self._mtar_tool_fields()
        transcript = (
            context.controller_transcript
            if context.controller_transcript is not None
            else context.target_transcript
        )
        values = {
            "harmful_task_description": self.side_task_description,
            "target_tool": target_tool,
            "target_tool_description": target_tool_description,
            "other_tools": other_tools,
            "service_deployments": self._mtar_service_deployments(),
            "parent_instructions": json.dumps(
                list(context.instructions), indent=2, ensure_ascii=False
            ),
            "target_transcript": json.dumps(
                list(transcript), indent=2, ensure_ascii=False
            ),
            "branching_factor": str(branching_factor),
            "current_depth": str(len(context.instructions)),
            "maximum_depth": str(self.maximum_depth),
            "remaining_depth": str(
                max(self.maximum_depth - len(context.instructions) - 1, 0)
            ),
        }
        names = "|".join(re.escape(name) for name in values)
        return re.sub(
            rf"\{{({names})\}}",
            lambda match: values[match.group(1)],
            self.mtar_user_template,
        )

    def prompts(
        self, context: ControllerContext, branching_factor: int
    ) -> tuple[str, str]:
        """Return the exact system and user strings sent to the Controller."""
        return self.mtar_system_prompt, self._mtar_user_prompt(
            context, branching_factor
        )

    def generate(
        self, context: ControllerContext, branching_factor: int
    ) -> CandidateBatch:
        # ControllerContext is deliberately serialized field-by-field. Adding a
        # trusted score to a search node cannot accidentally leak it here.
        system, user = self.prompts(context, branching_factor)
        processing: dict[str, Any] = {}
        try:
            return parse_candidate_batch(
                self.complete(system, user),
                allow_json_repair=self.allow_json_repair,
                processing_metadata=processing,
            )
        finally:
            self.last_response_processing = dict(processing)
