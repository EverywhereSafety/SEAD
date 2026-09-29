"""Contracts for deterministic MTAR DART tree search.

The controller-facing types intentionally contain no trusted evaluation data.
Replay and deterministic progress records live on the other side of that boundary.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

SCHEMA_VERSION = "dart-tree-search-v4"


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            item.name: _plain(getattr(value, item.name))
            for item in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class Candidate:
    instruction: str
    strategy_summary: str
    expected_state_change: str

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)


@dataclass(frozen=True)
class CandidateBatch:
    candidates: tuple[Candidate, ...]
    strategy: str | None = None
    strategy_rationale: str | None = None
    parallel_verification: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"candidates": _plain(self.candidates)}
        if self.strategy is not None:
            value = {
                "strategy": self.strategy,
                "strategy_rationale": self.strategy_rationale,
                "parallel_verification": self.parallel_verification,
                **value,
            }
        return value


@dataclass(frozen=True)
class ControllerContext:
    """The complete and only dynamic information exposed to the controller."""

    node_id: str
    instructions: tuple[str, ...]
    target_transcript: tuple[Mapping[str, Any], ...]
    controller_transcript: tuple[Mapping[str, Any], ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        if self.controller_transcript is not None:
            return {
                "node_id": self.node_id,
                "instructions": _plain(self.instructions),
                "controller_transcript": _plain(self.controller_transcript),
            }
        return {
            "node_id": self.node_id,
            "instructions": _plain(self.instructions),
            "target_transcript": _plain(self.target_transcript),
        }


@runtime_checkable
class CandidateBatchController(Protocol):
    def generate(
        self, context: ControllerContext, branching_factor: int
    ) -> CandidateBatch:
        ...


@dataclass(frozen=True)
class ReplayToolResult:
    """A tool message paired with the call that caused it."""

    tool_call_id: str
    function: str
    content: Any
    error: Any = None

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)


@dataclass(frozen=True)
class ReplayToolCall:
    """A recorded call, including the stable Inspect call id."""

    id: str
    function: str
    arguments: Mapping[str, Any]
    result: ReplayToolResult
    native_action: Mapping[str, Any] = field(default_factory=dict)
    native_observation: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", dict(self.arguments))
        object.__setattr__(self, "native_action", dict(self.native_action))
        object.__setattr__(self, "native_observation", dict(self.native_observation))
        if self.result.tool_call_id != self.id:
            raise ValueError("tool result id must match its tool call id")

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)


@dataclass(frozen=True)
class ReplayAssistantMessage:
    """One assistant-message boundary in a replay turn."""

    content: str
    tool_calls: tuple[ReplayToolCall, ...] = ()
    native_action: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        object.__setattr__(self, "native_action", dict(self.native_action))

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)


@dataclass(frozen=True)
class ReplayTurn:
    """The complete secret-safe record required to hard-restore one turn."""

    user_instruction: str
    assistant_messages: tuple[ReplayAssistantMessage, ...]
    origin: str = "target"

    def __post_init__(self) -> None:
        if not self.user_instruction.strip():
            raise ValueError("ReplayTurn.user_instruction is required")
        if self.origin not in {"target"}:
            raise ValueError("unknown replay turn origin")
        object.__setattr__(self, "assistant_messages", tuple(self.assistant_messages))

    def to_dict(self) -> dict[str, Any]:
        value = _plain(self)
        if self.origin == "target":
            value.pop("origin")
        return value

    def transcript_row(self, turn: int) -> dict[str, Any]:
        calls = [
            call.to_dict()
            for message in self.assistant_messages
            for call in message.tool_calls
        ]
        return {
            "turn": turn,
            "external_instruction": self.user_instruction,
            "target_text": "\n".join(
                message.content for message in self.assistant_messages if message.content
            ),
            "assistant_messages": [message.to_dict() for message in self.assistant_messages],
            "tool_calls": calls,
            "tool_results": [call.result.to_dict() for call in (
                call for message in self.assistant_messages for call in message.tool_calls
            )],
        }


@dataclass(frozen=True)
class BranchReplayRequest:
    run_id: str
    node_id: str
    parent_node_id: str
    benchmark_kind: str
    task_id: str
    parent_replay_turns: tuple[ReplayTurn, ...] = ()
    new_instruction: str | None = None
    final_replay: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "parent_replay_turns", tuple(self.parent_replay_turns))
        if self.final_replay and self.new_instruction is not None:
            raise ValueError("final replay must not contain a new instruction")
        if not self.final_replay and not str(self.new_instruction or "").strip():
            raise ValueError("branch replay requires exactly one new instruction")

    @property
    def instructions(self) -> tuple[str, ...]:
        suffix = () if self.new_instruction is None else (self.new_instruction,)
        return tuple(turn.user_instruction for turn in self.parent_replay_turns) + suffix

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)


@dataclass(frozen=True)
class GoalProgress:
    attack_progress: float
    benign_task_score: float
    terminal_success: bool
    official_main_result: Mapping[str, Any] = field(default_factory=dict)
    official_side_result: Mapping[str, Any] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.attack_progress) <= 1.0:
            raise ValueError("attack_progress must be in [0, 1]")
        if not 0.0 <= float(self.benign_task_score) <= 1.0:
            raise ValueError("benign_task_score must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)


@dataclass(frozen=True)
class BranchReplayResult:
    replay_id: str
    environment_id: str
    instructions: tuple[str, ...]
    target_transcript: tuple[Mapping[str, Any], ...]
    tool_calls: tuple[Mapping[str, Any], ...]
    official_main_result: Mapping[str, Any] = field(default_factory=dict)
    official_side_result: Mapping[str, Any] = field(default_factory=dict)
    attack_progress: float = 0.0
    benign_task_score: float = 0.0
    cleanup_succeeded: bool = True
    cleanup_details: Mapping[str, Any] = field(default_factory=dict)
    technical_error: str | None = None
    infrastructure_status: str | None = None
    artifact_locations: Mapping[str, str] = field(default_factory=dict)
    events: tuple[Mapping[str, Any], ...] = ()
    replay_turns: tuple[ReplayTurn, ...] = ()
    restore_audit: Mapping[str, Any] = field(default_factory=dict)
    controller_transcript: tuple[Mapping[str, Any], ...] | None = None
    phase_timings: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "instructions", tuple(self.instructions))
        object.__setattr__(self, "target_transcript", tuple(self.target_transcript))
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        if self.controller_transcript is not None:
            object.__setattr__(
                self, "controller_transcript", tuple(self.controller_transcript)
            )
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "replay_turns", tuple(self.replay_turns))
        object.__setattr__(self, "restore_audit", dict(self.restore_audit))
        object.__setattr__(
            self,
            "phase_timings",
            {str(name): float(seconds) for name, seconds in self.phase_timings.items()},
        )

    @property
    def technically_valid(self) -> bool:
        return self.technical_error is None and self.cleanup_succeeded

    def with_evaluation(self, progress: GoalProgress) -> BranchReplayResult:
        return dataclasses.replace(
            self,
            official_main_result=dict(progress.official_main_result),
            official_side_result=dict(progress.official_side_result),
            attack_progress=float(progress.attack_progress),
            benign_task_score=float(progress.benign_task_score),
        )

    def to_dict(self) -> dict[str, Any]:
        value = _plain(self)
        if self.controller_transcript is None:
            value.pop("controller_transcript", None)
        return value


@runtime_checkable
class BranchReplayBackend(Protocol):
    benchmark_kind: str
    task_id: str

    def execute_path(self, request: BranchReplayRequest) -> BranchReplayResult:
        ...


@runtime_checkable
class GoalProgressEvaluator(Protocol):
    def evaluate(self, result: BranchReplayResult) -> GoalProgress:
        ...


@dataclass(frozen=True)
class TreeSearchConfig:
    max_depth: int = 3
    branching_factor: int = 2
    max_executed_nodes: int = 8
    max_controller_calls: int = 4
    exploration_weight: float = 1.414
    replay_retries: int = 1
    controller_max_retries: int = 1
    semantic_completion: bool = False
    semantic_completion_min_confidence: float = 0.9

    def __post_init__(self) -> None:
        if self.max_depth < 1:
            raise ValueError("max_depth must be positive")
        if self.branching_factor < 1:
            raise ValueError("branching_factor must be positive")
        if self.max_executed_nodes < 1:
            raise ValueError("max_executed_nodes must be positive")
        if self.max_controller_calls < 1:
            raise ValueError("max_controller_calls must be positive")
        if self.exploration_weight < 0:
            raise ValueError("exploration_weight must be non-negative")
        if self.controller_max_retries not in {0, 1}:
            raise ValueError("controller_max_retries must be 0 or 1")
        if self.replay_retries not in {0, 1}:
            raise ValueError("replay_retries must be 0 or 1")
        if type(self.semantic_completion) is not bool:
            raise ValueError("semantic_completion must be Boolean")
        if not 0.0 <= float(self.semantic_completion_min_confidence) <= 1.0:
            raise ValueError(
                "semantic_completion_min_confidence must be in [0, 1]"
            )

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)
