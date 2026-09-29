"""Shared subprocess protocol for OpenHands replay workers; v5 wire-compatible."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...attacks.dart.models import (
    ReplayAssistantMessage,
    ReplayToolCall,
    ReplayToolResult,
    ReplayTurn,
)

PROTOCOL_VERSION = "mtar-openhands-replay-v5"


class ReplayPhaseTimer:
    """Accumulate monotonic replay phase durations without changing control flow."""

    def __init__(self) -> None:
        self._started = time.perf_counter()
        self._durations: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - started)

    def add(self, name: str, seconds: float) -> None:
        self._durations[name] = self._durations.get(name, 0.0) + max(
            0.0, float(seconds)
        )

    def snapshot(self) -> dict[str, float]:
        return {
            **{
                name: round(seconds, 6)
                for name, seconds in sorted(self._durations.items())
            },
            "worker_total": round(time.perf_counter() - self._started, 6),
        }


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return {str(key): item for key, item in value.items()}


def replay_turn_from_dict(value: Mapping[str, Any]) -> ReplayTurn:
    data = _mapping(value, "ReplayTurn")
    if set(data) - {"user_instruction", "assistant_messages", "origin"} or not {"user_instruction", "assistant_messages"} <= set(data):
        raise ValueError("invalid ReplayTurn keys")
    messages = []
    for raw_message in data["assistant_messages"]:
        message = _mapping(raw_message, "assistant message")
        calls = []
        for raw_call in message["tool_calls"]:
            call = _mapping(raw_call, "tool call")
            result = _mapping(call["result"], "tool result")
            calls.append(
                ReplayToolCall(
                    id=str(call["id"]),
                    function=str(call["function"]),
                    arguments=_mapping(call["arguments"], "arguments"),
                    result=ReplayToolResult(
                        tool_call_id=str(result["tool_call_id"]),
                        function=str(result["function"]),
                        content=result["content"],
                        error=result.get("error"),
                    ),
                    native_action=_mapping(
                        call.get("native_action", {}), "native action"
                    ),
                    native_observation=_mapping(
                        call.get("native_observation", {}), "native observation"
                    ),
                )
            )
        messages.append(
            ReplayAssistantMessage(
                content=str(message["content"]),
                tool_calls=tuple(calls),
                native_action=_mapping(
                    message.get("native_action", {}), "native action"
                ),
            )
        )
    return ReplayTurn(str(data["user_instruction"]), tuple(messages), data.get("origin", "target"))


def atomic_write_json(path: Path | str, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


@dataclass(frozen=True)
class MTARReplayRequest:
    replay_id: str
    environment_id: str
    run_id: str
    node_id: str
    parent_node_id: str
    task_id: str
    dataset_root: str
    openhands_root: str
    worker_dir: str
    parent_replay_turns: tuple[ReplayTurn, ...]
    new_instruction: str | None
    target: Mapping[str, Any]
    execution: Mapping[str, Any]
    final_replay: bool = False
    protocol: str = PROTOCOL_VERSION
    benchmark_kind: str | None = None
    execution_mode: str | None = None

    def __post_init__(self) -> None:
        if self.protocol != PROTOCOL_VERSION:
            raise ValueError(f"expected protocol={PROTOCOL_VERSION}")
        if self.benchmark_kind not in {None, "mtar", "oas"}:
            raise ValueError("unsupported OpenHands benchmark")
        if self.execution.get("preparation_mode") is not None:
            raise ValueError("preparation_mode is not supported")
        if any(str(key).startswith("defender_teacher_") for key in self.execution):
            raise ValueError("teacher data generation is not supported")
        inferred = "replay_only" if self.final_replay else "rollout"
        if self.execution_mode is not None and self.execution_mode != inferred:
            raise ValueError("execution_mode conflicts with replay fields")
        for name in (
            "replay_id",
            "environment_id",
            "run_id",
            "node_id",
            "parent_node_id",
            "task_id",
            "dataset_root",
            "openhands_root",
            "worker_dir",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required")
        object.__setattr__(self, "parent_replay_turns", tuple(self.parent_replay_turns))
        object.__setattr__(self, "target", dict(self.target))
        object.__setattr__(self, "execution", dict(self.execution))
        if self.final_replay != (self.new_instruction is None):
            raise ValueError("terminal replay has no new instruction")
        if self.new_instruction is not None and not self.new_instruction.strip():
            raise ValueError("new_instruction must be non-empty")

    @property
    def instructions(self) -> tuple[str, ...]:
        suffix = () if self.new_instruction is None else (self.new_instruction,)
        return (
            tuple(turn.user_instruction for turn in self.parent_replay_turns) + suffix
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "replay_id": self.replay_id,
            "environment_id": self.environment_id,
            "run_id": self.run_id,
            "node_id": self.node_id,
            "parent_node_id": self.parent_node_id,
            "task_id": self.task_id,
            "dataset_root": self.dataset_root,
            "openhands_root": self.openhands_root,
            "worker_dir": self.worker_dir,
            "parent_replay_turns": [
                turn.to_dict() for turn in self.parent_replay_turns
            ],
            "new_instruction": self.new_instruction,
            "target": dict(self.target),
            "execution": dict(self.execution),
            "final_replay": self.final_replay,
            **({"benchmark_kind": self.benchmark_kind} if self.benchmark_kind else {}),
            **({"execution_mode": self.execution_mode} if self.execution_mode else {}),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MTARReplayRequest:
        data = _mapping(value, "request")
        benchmark_kind = data.pop("benchmark_kind", None)
        execution_mode = data.pop("execution_mode", None)
        expected = {
            "protocol",
            "replay_id",
            "environment_id",
            "run_id",
            "node_id",
            "parent_node_id",
            "task_id",
            "dataset_root",
            "openhands_root",
            "worker_dir",
            "parent_replay_turns",
            "new_instruction",
            "target",
            "execution",
            "final_replay",
        }
        if set(data) != expected:
            raise ValueError(
                f"invalid request keys: missing={sorted(expected - set(data))}, "
                f"unknown={sorted(set(data) - expected)}"
            )
        turns = data["parent_replay_turns"]
        if not isinstance(turns, list):
            raise TypeError("parent_replay_turns must be an array")
        return cls(
            benchmark_kind=benchmark_kind,
            execution_mode=execution_mode,
            **{
                name: str(data[name])
                for name in (
                    "protocol",
                    "replay_id",
                    "environment_id",
                    "run_id",
                    "node_id",
                    "parent_node_id",
                    "task_id",
                    "dataset_root",
                    "openhands_root",
                    "worker_dir",
                )
            },
            parent_replay_turns=tuple(replay_turn_from_dict(row) for row in turns),
            new_instruction=None
            if data["new_instruction"] is None
            else str(data["new_instruction"]),
            target=_mapping(data["target"], "target"),
            execution=_mapping(data["execution"], "execution"),
            final_replay=bool(data["final_replay"]),
        )




@dataclass(frozen=True)
class MTARReplayWorkerResult:
    replay_id: str
    environment_id: str
    task_id: str
    replay_turns: tuple[ReplayTurn, ...] = ()
    restore_audit: Mapping[str, Any] = field(default_factory=dict)
    evaluation: Mapping[str, Any] = field(default_factory=dict)
    cleanup_succeeded: bool = False
    cleanup_details: Mapping[str, Any] = field(default_factory=dict)
    technical_error: str | None = None
    infrastructure_status: str | None = None
    artifacts: Mapping[str, str] = field(default_factory=dict)
    phase_timings: Mapping[str, float] = field(default_factory=dict)
    protocol: str = PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "replay_id": self.replay_id,
            "environment_id": self.environment_id,
            "task_id": self.task_id,
            "replay_turns": [turn.to_dict() for turn in self.replay_turns],
            "restore_audit": dict(self.restore_audit),
            "evaluation": dict(self.evaluation),
            "cleanup_succeeded": self.cleanup_succeeded,
            "cleanup_details": dict(self.cleanup_details),
            "technical_error": self.technical_error,
            "infrastructure_status": self.infrastructure_status,
            "artifacts": dict(self.artifacts),
            "phase_timings": {
                str(name): float(seconds)
                for name, seconds in self.phase_timings.items()
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MTARReplayWorkerResult:
        data = _mapping(value, "result")
        turns = data.get("replay_turns")
        if not isinstance(turns, list):
            raise TypeError("replay_turns must be an array")
        return cls(
            protocol=str(data.get("protocol")),
            replay_id=str(data.get("replay_id")),
            environment_id=str(data.get("environment_id")),
            task_id=str(data.get("task_id")),
            replay_turns=tuple(replay_turn_from_dict(row) for row in turns),
            restore_audit=_mapping(data.get("restore_audit", {}), "restore audit"),
            evaluation=_mapping(data.get("evaluation", {}), "evaluation"),
            cleanup_succeeded=bool(data.get("cleanup_succeeded")),
            cleanup_details=_mapping(
                data.get("cleanup_details", {}), "cleanup details"
            ),
            technical_error=None
            if data.get("technical_error") is None
            else str(data["technical_error"]),
            infrastructure_status=None
            if data.get("infrastructure_status") is None
            else str(data["infrastructure_status"]),
            artifacts={
                str(k): str(v)
                for k, v in _mapping(data.get("artifacts", {}), "artifacts").items()
            },
            phase_timings={
                str(k): float(v)
                for k, v in _mapping(
                    data.get("phase_timings", {}), "phase timings"
                ).items()
            },
        )




def read_request(path: Path | str) -> MTARReplayRequest:
    value = json.loads(Path(path).read_text())
    return MTARReplayRequest.from_dict(value)


def read_result(path: Path | str) -> MTARReplayWorkerResult:
    value = json.loads(Path(path).read_text())
    return MTARReplayWorkerResult.from_dict(value)


__all__ = [
    "PROTOCOL_VERSION",
    "MTARReplayRequest",
    "MTARReplayWorkerResult",
    "ReplayPhaseTimer",
    "atomic_write_json",
    "read_request",
    "read_result",
    "replay_turn_from_dict",
    "OpenHandsReplayRequest",
    "OpenHandsReplayResult",
]

OpenHandsReplayRequest = MTARReplayRequest
OpenHandsReplayResult = MTARReplayWorkerResult
