"""Correlated Knowledge Attack state, search, and context projections.

The public symbols are loaded lazily so one-shot benchmark workers can import
the replay contracts without also importing Controller-only dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "SCHEMA_VERSION": ("models", "SCHEMA_VERSION"),
    "BranchReplayBackend": ("models", "BranchReplayBackend"),
    "BranchReplayRequest": ("models", "BranchReplayRequest"),
    "BranchReplayResult": ("models", "BranchReplayResult"),
    "DARTTreeSearchEngine": ("tree_search", "DARTTreeSearchEngine"),
    "Candidate": ("models", "Candidate"),
    "CandidateBatch": ("models", "CandidateBatch"),
    "CandidateBatchController": ("models", "CandidateBatchController"),
    "CandidateValidationError": ("candidates", "CandidateValidationError"),
    "Checkpoint": ("checkpoints", "Checkpoint"),
    "CheckpointDefinitionError": ("checkpoints", "CheckpointDefinitionError"),
    "CheckpointSet": ("checkpoints", "CheckpointSet"),
    "ControllerContext": ("models", "ControllerContext"),
    "GoalProgress": ("models", "GoalProgress"),
    "GoalProgressEvaluator": ("models", "GoalProgressEvaluator"),
    "ReplayAssistantMessage": ("models", "ReplayAssistantMessage"),
    "ReplayToolCall": ("models", "ReplayToolCall"),
    "ReplayToolResult": ("models", "ReplayToolResult"),
    "ReplayTurn": ("models", "ReplayTurn"),
    "SearchTreeRenderError": ("search_tree_renderer", "SearchTreeRenderError"),
    "StructuredBatchController": ("candidates", "StructuredBatchController"),
    "PathTermination": ("tree_search", "PathTermination"),
    "TreeNode": ("tree_search", "TreeNode"),
    "TreeSearchConfig": ("models", "TreeSearchConfig"),
    "TreeSearchOutcome": ("tree_search", "TreeSearchOutcome"),
    "TreeSearchRecord": ("tree_search", "TreeSearchRecord"),
    "UnifiedDARTConfig": ("config", "UnifiedDARTConfig"),
    "deduplicate_candidate_batch": ("candidates", "deduplicate_candidate_batch"),
    "load_tree_search_config": ("config", "load_tree_search_config"),
    "load_checkpoints": ("checkpoints", "load_checkpoints"),
    "parse_candidate_batch": ("candidates", "parse_candidate_batch"),
    "parse_checkpoints": ("checkpoints", "parse_checkpoints"),
    "render_search_tree": ("search_tree_renderer", "render_search_tree"),
    "semantically_near_duplicate": ("candidates", "semantically_near_duplicate"),
    "validate_candidate_batch": ("candidates", "validate_candidate_batch"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
