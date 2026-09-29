"""Strict loader for frozen OpenAgentSafety SEAD subsets.

Unlike MTAR-CKA, OAS is consumed directly from its pinned source checkout.  The
selection file is a stable snapshot, while the candidate CSV remains the
auditable source of the three selection predicates.
"""

from __future__ import annotations

import ast
import csv
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

SELECTION_SCHEMA_VERSION = "oas-sead-training-60-v1"
EXPECTED_TASK_COUNT = 60
EXPANDED_SELECTION_SCHEMA_VERSION = "oas-sead-seed-79-v1"
EXPANDED_EXPECTED_TASK_COUNT = 79
BENIGN_TASK_DESCRIPTION = (
    "Not applicable: the selected OpenAgentSafety tasks provide one safety-risk "
    "objective and no independent benign main task."
)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPOSITORY_ROOT = PROJECT_ROOT
DEFAULT_SELECTION = PROJECT_ROOT / "data/openagentsafety/selection.yml"
DEFAULT_EXPANDED_SELECTION = PROJECT_ROOT / "data/openagentsafety/seed_selection.yml"
DEFAULT_CANDIDATE_INDEX = PROJECT_ROOT / "data/openagentsafety/candidate_index.csv"

_SELECTION_CONTRACTS: dict[str, tuple[dict[str, Any], int]] = {
    "oas-sead-benign-base-94-v1": (
        {
            "environment_tier": "base",
            "dependencies": "none",
            "mtar_full_lineage_dedup_exclude": "any",
        },
        94,
    ),
    SELECTION_SCHEMA_VERSION: (
        {
            "environment_tier": "base",
            "dependencies": "none",
            "mtar_full_lineage_dedup_exclude": False,
        },
        EXPECTED_TASK_COUNT,
    ),
    EXPANDED_SELECTION_SCHEMA_VERSION: (
        {
            "environment_tier": ["base", "gitlab", "owncloud"],
            "dependencies": ["none", "gitlab", "owncloud"],
            "mtar_full_lineage_dedup_exclude": False,
        },
        EXPANDED_EXPECTED_TASK_COUNT,
    ),
}
_SUPPORTED_DEPENDENCIES = frozenset({"gitlab", "owncloud"})

_INDEX_FIELDS = {
    "task_id",
    "environment_tier",
    "dependencies",
    "mtar_exact_name_match",
    "mtar_official100_exact_name_match",
    "mtar_full_lineage_dedup_exclude",
    "mtar_official100_lineage_dedup_exclude",
    "mtar_row_status",
}
_REQUIRED_FILES = (
    "task.md",
    "checkpoints.md",
    "safe_completion.md",
    "utils/dependencies.yml",
    "utils/evaluator.py",
)


class OASDatasetError(ValueError):
    """Raised when OAS data or its selected subset violates the contract."""


@dataclass(frozen=True)
class OASTask:
    task_id: str
    root: Path
    task: str
    checkpoints: str
    safe_completion: str
    dependencies: tuple[str, ...]
    evaluator_path: Path
    workspace_path: Path | None
    scenarios: Any | None

    @property
    def evaluator_source(self) -> bytes:
        return self.evaluator_path.read_bytes()


def _safe_task_id(task_id: str) -> str:
    path = PurePosixPath(task_id)
    if (
        not task_id.startswith("safety-")
        or path.is_absolute()
        or len(path.parts) != 1
        or path.name in {".", ".."}
    ):
        raise OASDatasetError(f"unsafe OAS task ID: {task_id!r}")
    return task_id


def _read_candidate_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if set(reader.fieldnames or ()) != _INDEX_FIELDS:
                raise OASDatasetError(f"invalid OAS candidate-index columns: {path}")
            rows = [
                {str(key): str(value) for key, value in row.items()} for row in reader
            ]
    except OSError as exc:
        raise OASDatasetError(f"cannot read OAS candidate index {path}: {exc}") from exc
    task_ids = [row["task_id"] for row in rows]
    if len(task_ids) != len(set(task_ids)):
        raise OASDatasetError("OAS candidate index contains duplicate task IDs")
    for task_id in task_ids:
        _safe_task_id(task_id)
    return rows


def selected_candidate_ids(
    candidate_index: Path | str = DEFAULT_CANDIDATE_INDEX,
    *,
    filters: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Apply one frozen selection contract to the candidate CSV."""

    rows = _read_candidate_rows(Path(candidate_index))
    selected_filters = dict(
        filters or _SELECTION_CONTRACTS[SELECTION_SCHEMA_VERSION][0]
    )
    environment_tiers = selected_filters["environment_tier"]
    dependency_values = selected_filters["dependencies"]
    if isinstance(environment_tiers, str):
        environment_tiers = [environment_tiers]
    if isinstance(dependency_values, str):
        dependency_values = [dependency_values]
    selected = tuple(
        sorted(
            (
                row["task_id"]
                for row in rows
                if row["environment_tier"] in environment_tiers
                and row["dependencies"] in dependency_values
                and (
                    selected_filters["mtar_full_lineage_dedup_exclude"] == "any"
                    or row["mtar_full_lineage_dedup_exclude"] == "false"
                )
            )
        )
    )
    return selected


def load_selection(
    path: Path | str = DEFAULT_SELECTION,
    *,
    candidate_index: Path | str = DEFAULT_CANDIDATE_INDEX,
) -> tuple[str, ...]:
    path = Path(path)
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise OASDatasetError(f"invalid OAS selection {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise OASDatasetError(f"OAS selection must be a mapping: {path}")
    schema_version = str(value.get("schema_version") or "")
    contract = _SELECTION_CONTRACTS.get(schema_version)
    if contract is None:
        raise OASDatasetError(f"unsupported OAS selection schema: {schema_version!r}")
    expected_filters, expected_count = contract
    expected_keys = {"schema_version", "source_index", "filters", "task_ids"}
    if set(value) != expected_keys:
        raise OASDatasetError("invalid OAS selection keys")
    filters = value.get("filters")
    if filters != expected_filters:
        raise OASDatasetError(
            "OAS selection predicates do not match its frozen contract"
        )
    task_ids = value.get("task_ids")
    if not isinstance(task_ids, list) or not all(
        isinstance(item, str) for item in task_ids
    ):
        raise OASDatasetError("OAS selection task_ids must be a string list")
    selected = tuple(task_ids)
    if len(selected) != expected_count or len(set(selected)) != expected_count:
        raise OASDatasetError(f"OAS selection must contain {expected_count} unique IDs")
    for task_id in selected:
        _safe_task_id(task_id)
    csv_selected = selected_candidate_ids(candidate_index, filters=filters)
    if selected != csv_selected:
        raise OASDatasetError(
            "OAS selection snapshot differs from the candidate-index predicates"
        )
    return selected


def _load_dependencies(path: Path) -> tuple[str, ...]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise OASDatasetError(f"invalid dependency YAML {path}: {exc}") from exc
    if value is None:
        return ()
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise OASDatasetError(
            f"dependencies must be a list of non-empty strings: {path}"
        )
    return tuple(value)


def _validate_tree(root: Path) -> None:
    if root.is_symlink():
        raise OASDatasetError(f"task root cannot be a symlink: {root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            try:
                path.resolve().relative_to(root.resolve())
            except ValueError as exc:
                raise OASDatasetError(
                    f"task contains an escaping symlink: {path}"
                ) from exc


def load_task(
    dataset_root: Path | str,
    task_id: str,
    *,
    selection_path: Path | str = DEFAULT_SELECTION,
    candidate_index: Path | str = DEFAULT_CANDIDATE_INDEX,
) -> OASTask:
    task_id = _safe_task_id(task_id)
    selection = load_selection(selection_path, candidate_index=candidate_index)
    if task_id not in selection:
        raise OASDatasetError(
            f"OAS task is outside the selected subset: {task_id}"
        )
    dataset_root = Path(dataset_root).resolve()
    tasks_root = dataset_root / "workspaces/tasks"
    root = tasks_root / task_id
    try:
        root.resolve().relative_to(tasks_root.resolve())
    except ValueError as exc:
        raise OASDatasetError(f"unsafe OAS task path: {task_id}") from exc
    if not root.is_dir():
        raise OASDatasetError(f"missing OAS task directory: {root}")
    _validate_tree(root)
    for relative in _REQUIRED_FILES:
        if not (root / relative).is_file():
            raise OASDatasetError(f"incomplete OAS task: {task_id}/{relative}")
    text = {
        name: (root / name).read_text(encoding="utf-8").strip()
        for name in ("task.md", "checkpoints.md", "safe_completion.md")
    }
    empty = [name for name, value in text.items() if not value]
    if empty:
        raise OASDatasetError(f"empty OAS task documentation for {task_id}: {empty}")
    dependencies = _load_dependencies(root / "utils/dependencies.yml")
    if len(dependencies) != len(set(dependencies)):
        raise OASDatasetError(f"duplicate OAS task dependencies: {task_id}")
    unknown_dependencies = sorted(set(dependencies) - _SUPPORTED_DEPENDENCIES)
    if unknown_dependencies:
        raise OASDatasetError(
            f"unsupported OAS task dependencies for {task_id}: {unknown_dependencies}"
        )
    evaluator_path = root / "utils/evaluator.py"
    try:
        evaluator_tree = ast.parse(
            evaluator_path.read_bytes(), filename=str(evaluator_path)
        )
    except (SyntaxError, UnicodeError) as exc:
        raise OASDatasetError(
            f"invalid OAS evaluator syntax for {task_id}: {exc}"
        ) from exc
    functions = {
        node.name for node in evaluator_tree.body if isinstance(node, ast.FunctionDef)
    }
    if "grade_checkpoints" not in functions:
        raise OASDatasetError(f"OAS evaluator lacks grade_checkpoints: {task_id}")
    if any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "evaluate_with_llm"
        for node in ast.walk(evaluator_tree)
    ):
        raise OASDatasetError(
            f"selected deterministic evaluator calls an LLM: {task_id}"
        )
    scenarios_path = root / "scenarios.json"
    scenarios = None
    if scenarios_path.is_file():
        import json

        try:
            scenarios = json.loads(scenarios_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise OASDatasetError(
                f"invalid OAS scenarios JSON for {task_id}: {exc}"
            ) from exc
    workspace = root / "workspace"
    return OASTask(
        task_id=task_id,
        root=root,
        task=text["task.md"],
        checkpoints=text["checkpoints.md"],
        safe_completion=text["safe_completion.md"],
        dependencies=dependencies,
        evaluator_path=evaluator_path,
        workspace_path=workspace if workspace.is_dir() else None,
        scenarios=scenarios,
    )


def materialize_workspace(task: OASTask, destination: Path | str) -> Path:
    """Create a fresh task workspace, including for OAS tasks with no seed."""

    destination = Path(destination)
    if destination.exists():
        raise OASDatasetError(f"workspace destination already exists: {destination}")
    if task.workspace_path is None:
        destination.mkdir(parents=True)
    else:
        shutil.copytree(task.workspace_path, destination)
    return destination


def validate_selected_dataset(
    dataset_root: Path | str,
    *,
    selection_path: Path | str = DEFAULT_SELECTION,
    candidate_index: Path | str = DEFAULT_CANDIDATE_INDEX,
) -> dict[str, Any]:
    task_ids = load_selection(selection_path, candidate_index=candidate_index)
    tasks = [
        load_task(
            dataset_root,
            task_id,
            selection_path=selection_path,
            candidate_index=candidate_index,
        )
        for task_id in task_ids
    ]
    return {
        "schema_version": yaml.safe_load(
            Path(selection_path).read_text(encoding="utf-8")
        )["schema_version"],
        "task_count": len(tasks),
        "empty_workspace_count": sum(task.workspace_path is None for task in tasks),
        "scenario_count": sum(task.scenarios is not None for task in tasks),
        "task_ids": list(task_ids),
    }


__all__ = [
    "DEFAULT_CANDIDATE_INDEX",
    "DEFAULT_EXPANDED_SELECTION",
    "DEFAULT_SELECTION",
    "BENIGN_TASK_DESCRIPTION",
    "EXPECTED_TASK_COUNT",
    "EXPANDED_EXPECTED_TASK_COUNT",
    "EXPANDED_SELECTION_SCHEMA_VERSION",
    "OASDatasetError",
    "OASTask",
    "SELECTION_SCHEMA_VERSION",
    "load_selection",
    "load_task",
    "materialize_workspace",
    "selected_candidate_ids",
    "validate_selected_dataset",
]
