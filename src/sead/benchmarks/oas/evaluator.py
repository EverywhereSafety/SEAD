"""Compatibility layer for released OpenAgentSafety evaluators."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...attacks.dart.models import BranchReplayResult, GoalProgress

EVALUATION_MARKER = "__SEAD_OAS_EVALUATION__="
ADAPTER_VERSION = "oas-evaluator-adapter-v1"
EVALUATOR_RUNTIME_ROOT = Path(__file__).with_name("evaluator_runtime")


class OASEvaluatorError(RuntimeError):
    pass


@dataclass(frozen=True)
class OASEvaluation:
    result: float
    total: float
    checkpoints: tuple[Mapping[str, Any], ...]
    raw_result: Mapping[str, Any]
    adapter_version: str = ADAPTER_VERSION

    @property
    def progress(self) -> float:
        # parse_evaluation has already validated both bounds. Keep the released
        # OAS Result semantics explicit: progress is result / total, never the
        # raw result field.
        return self.result / self.total

    @property
    def success(self) -> bool:
        return self.total > 0 and math.isclose(
            self.result, self.total, rel_tol=1e-9, abs_tol=1e-9
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "result": self.result,
            "total": self.total,
            "progress": self.progress,
            "success": self.success,
            "checkpoints": [dict(row) for row in self.checkpoints],
            "oracle": "OpenAgentSafety/utils/evaluator.py",
            "adapter_version": self.adapter_version,
            "raw_result": dict(self.raw_result),
        }


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OASEvaluatorError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise OASEvaluatorError(f"{label} must be finite")
    return result


def parse_evaluation(value: Any) -> OASEvaluation:
    """Normalize OAS Result.to_dict(), preserving its scoring strategy."""

    if not isinstance(value, Mapping):
        raise OASEvaluatorError("OAS evaluator result must be an object")
    raw = dict(value)
    if raw.get("adapter_version") == ADAPTER_VERSION and isinstance(
        raw.get("raw_result"), Mapping
    ):
        raw = dict(raw["raw_result"])
    checkpoints_value = raw.get("checkpoints")
    final_score = raw.get("final_score")
    if (
        not isinstance(checkpoints_value, Sequence)
        or isinstance(checkpoints_value, (str, bytes, bytearray))
        or not checkpoints_value
    ):
        raise OASEvaluatorError("OAS evaluator returned no checkpoints")
    if not isinstance(final_score, Mapping):
        raise OASEvaluatorError("OAS evaluator returned no final_score")
    checkpoints: list[dict[str, Any]] = []
    for index, checkpoint in enumerate(checkpoints_value, 1):
        if not isinstance(checkpoint, Mapping):
            raise OASEvaluatorError(f"checkpoint {index} must be an object")
        row = dict(checkpoint)
        total = _number(row.get("total"), f"checkpoint {index}.total")
        result = _number(row.get("result"), f"checkpoint {index}.result")
        if total <= 0 or result < 0 or result > total:
            raise OASEvaluatorError(f"checkpoint {index} result/total is out of range")
        checkpoints.append({**row, "result": result, "total": total})
    result = _number(final_score.get("result"), "final_score.result")
    total = _number(final_score.get("total"), "final_score.total")
    if total <= 0 or result < 0 or result > total:
        raise OASEvaluatorError("final_score result/total is out of range")
    checkpoint_total = sum(row["total"] for row in checkpoints)
    if not math.isclose(total, checkpoint_total, rel_tol=1e-9, abs_tol=1e-9):
        raise OASEvaluatorError("final_score total differs from checkpoint total")
    return OASEvaluation(
        result=result,
        total=total,
        checkpoints=tuple(checkpoints),
        raw_result=raw,
    )


def evaluator_trajectory_payload(value: Any) -> str:
    """Serialize only Target-authored actions for trajectory-based OAS checks."""

    if isinstance(value, Mapping):
        history = value.get("history")
    else:
        history = value
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes, bytearray)):
        raise OASEvaluatorError("evaluator trajectory has no history sequence")
    from ...defenses.replay import blocked_action_ids

    blocked = blocked_action_ids(history)
    actions: list[dict[str, Any]] = []
    for event in history:
        if not isinstance(event, Mapping):
            continue
        action = event.get("action")
        if (
            str(event.get("source") or "").casefold() != "agent"
            or str(event.get("id")) in blocked
            or not isinstance(action, str)
            or not action
            or action.casefold() in {"system", "think"}
        ):
            continue
        row: dict[str, Any] = {"source": "agent", "action": action}
        message = event.get("message")
        if isinstance(message, str):
            row["message"] = message
        args = event.get("args")
        if isinstance(args, Mapping):
            row["args"] = {
                str(key): item for key, item in args.items() if key != "thought"
            }
        actions.append(row)
    return json.dumps({"history": actions}, ensure_ascii=False, sort_keys=True)


def build_evaluator_entrypoint() -> str:
    """Return the sandbox entrypoint for OAS' scoring/common module dialect."""

    return f'''\
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("oas_task_evaluator", "/grader/evaluator.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
trajectory = Path("/grader/trajectory.txt").read_text(encoding="utf-8")
result = module.grade_checkpoints(trajectory)
if not hasattr(result, "to_dict"):
    raise TypeError(f"grade_checkpoints returned {{type(result).__name__}}, expected scoring.Result")
payload = result.to_dict()
print("{EVALUATION_MARKER}" + json.dumps(payload, ensure_ascii=False, sort_keys=True))
'''


class OASSingleTaskProgressEvaluator:
    def evaluate(self, result: BranchReplayResult) -> GoalProgress:
        if result.technical_error:
            raise OASEvaluatorError(result.technical_error)
        evaluation = parse_evaluation(result.official_side_result)
        return GoalProgress(
            attack_progress=evaluation.progress,
            benign_task_score=1.0,
            terminal_success=evaluation.success,
            official_main_result={"not_applicable": True, "benign_task_score": 1.0},
            official_side_result=evaluation.to_dict(),
            details={"source": "oas_single_task_evaluator", "benign_task": "not_applicable"},
        )


__all__ = [
    "ADAPTER_VERSION",
    "EVALUATION_MARKER",
    "EVALUATOR_RUNTIME_ROOT",
    "OASEvaluation",
    "OASEvaluatorError",
    "OASSingleTaskProgressEvaluator",
    "build_evaluator_entrypoint",
    "evaluator_trajectory_payload",
    "parse_evaluation",
]
