"""Read-only loading and validation of frozen MTAR datasets."""

from __future__ import annotations
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any
import yaml
from .environment import validate_dependencies
from .evaluator_compat import (
    EvaluatorCompatibilityError,
    build_evaluator_entrypoint,
    evaluator_python_requirements,
    validate_evaluator_imports,
)

SCHEMA_VERSION = "mtar-cka-official100-v1"
BASIC_SCHEMA_VERSION = "mtar-basic75-v1"
BASIC_SELECTION_SCHEMA_VERSION = "mtar-basic75-selection-v1"

# Separate selection contract; the historical Official-100 remains frozen.
EXPANSION_SCHEMA_VERSION = "mtar-cka-expansion-v1"
EXPANSION_SELECTION_SCHEMA_VERSION = "mtar-expansion-selection-v1"


SELECTION_SCHEMA_VERSION = "mtar-official-subset-100-v1"


ENVIRONMENT_SCHEMA_VERSION = "mtar-cka-environment-v1"


EXPECTED_SOURCE_COMMIT = "e1ba224fea480df3d296a1bf4e28613d4c97c704"


SOURCE_REPOSITORY = "https://huggingface.co/datasets/CHATS-Lab/MT-AgentRisk"


BENIGN_TASK_DESCRIPTION = (
    "Not applicable: MT-AgentRisk supplies one harmful single-task objective "
    "and has no independent benign main task."
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]


DEFAULT_SELECTION = PROJECT_ROOT / "data/mtar/task_ids.yml"


_BASIC_SELECTION_COUNTS = {"Filesystem": 25, "Terminal": 20, "PostgreSQL": 15, "Playwright": 15}


_EXPECTED_SELECTION_COUNTS = {
    "Filesystem": 25,
    "Terminal": 25,
    "PostgreSQL": 25,
    "Playwright": 25,
}


_RUNTIME_PROFILES = {
    "rust-client-v1",
    "bash-alias-v1",
    "base",
    "playwright-mcp",
    "ptrace-sshd",
    "net-admin",
    "net-sysctl",
    "localed",
    "microvm-lsm",
    "microvm-core-pattern",
    "microvm-clock",
}


class MTARDatasetError(ValueError):
    """Raised when source or generated data violates the dataset contract."""


class MTARDatasetQuarantinedError(MTARDatasetError):
    """Raised before runtime creation when a selected task is quarantined."""

    code = "DATASET_QUARANTINED"

    def __init__(self, task_id: str, reason: str) -> None:
        self.task_id = task_id
        self.reason = reason
        super().__init__(f"{self.code}: {task_id}: {reason}")


def _ordinal(task_id: str) -> int:
    prefix, separator, value = task_id.partition(".")
    if prefix != "single" or separator != "." or not value.isdigit() or int(value) < 1:
        raise MTARDatasetError(f"invalid single-task ID: {task_id!r}")
    return int(value)


def _safe_component(value: str, label: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or len(path.parts) != 1
        or path.name in {".", ".."}
    ):
        raise MTARDatasetError(f"unsafe {label}: {value!r}")
    return value


def _tool_family(tool: str) -> str:
    if tool == "Filesystem":
        return "filesystem"
    if tool == "Terminal":
        return "terminal"
    if tool == "PostgreSQL":
        return "postgres"
    if tool.startswith("Playwright(") and tool.endswith(")"):
        return "playwright"
    raise MTARDatasetError(f"unsupported MT-AgentRisk tool: {tool!r}")


def tool_runtime_context(tool: str) -> tuple[str, str | None]:
    """Return the runtime tool family and optional Playwright application."""

    family = _tool_family(tool)
    if family != "playwright":
        return family, None
    return family, tool[len("Playwright(") : -1].casefold()


def _selection_tool(tool: str) -> str:
    return "Playwright" if tool.startswith("Playwright(") else tool


def load_selection(path: Path | str = DEFAULT_SELECTION) -> tuple[str, ...]:
    path = Path(path)
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise MTARDatasetError(f"invalid MTAR selection: {exc}") from exc
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") not in {SELECTION_SCHEMA_VERSION, BASIC_SELECTION_SCHEMA_VERSION}
    ):
        raise MTARDatasetError(f"expected selection schema {SELECTION_SCHEMA_VERSION}")
    counts = (_BASIC_SELECTION_COUNTS if value["schema_version"] == BASIC_SELECTION_SCHEMA_VERSION
              else _EXPECTED_SELECTION_COUNTS)
    tools = value.get("tools")
    if not isinstance(tools, Mapping) or set(tools) != set(counts):
        raise MTARDatasetError(
            "selection must contain the four MTAR tool groups"
        )
    selected: list[str] = []
    for tool, expected in counts.items():
        task_ids = tools.get(tool)
        if not isinstance(task_ids, list) or not all(
            isinstance(item, str) for item in task_ids
        ):
            raise MTARDatasetError(f"selection group {tool} must be a list of task IDs")
        if len(task_ids) != expected or len(set(task_ids)) != expected:
            raise MTARDatasetError(
                f"selection group {tool} must contain {expected} unique IDs"
            )
        for task_id in task_ids:
            _ordinal(task_id)
        selected.extend(task_ids)
    if len(selected) != sum(counts.values()) or len(set(selected)) != len(selected):
        raise MTARDatasetError("selection must contain the expected unique task IDs")
    return tuple(sorted(selected, key=_ordinal))


def _read_dependencies(path: Path) -> list[str]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise MTARDatasetError(f"invalid dependency YAML: {exc}") from exc
    if value is None:
        value = []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise MTARDatasetError("dependencies must be a YAML list of strings")
    normalized = sorted({item.strip() for item in value})
    if any(not item for item in normalized):
        raise MTARDatasetError("dependencies must be non-empty strings")
    return normalized


def load_task_dependencies(task_root: Path | str, tool: str) -> tuple[str, ...]:
    dependencies = _read_dependencies(Path(task_root) / "utils/dependencies.yml")
    validate_dependencies(_tool_family(tool), dependencies)
    return tuple(dependencies)


def _load_environment(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise MTARDatasetError(f"invalid environment metadata {path}: {exc}") from exc
    expected = {"schema_version", "runtime_profile", "evaluator_python_requirements"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise MTARDatasetError(f"invalid environment keys: {path}")
    requirements = value.get("evaluator_python_requirements")
    if not isinstance(requirements, list) or not all(
        isinstance(item, str) for item in requirements
    ):
        raise MTARDatasetError(f"invalid evaluator requirements: {path}")
    if value.get("schema_version") != ENVIRONMENT_SCHEMA_VERSION:
        raise MTARDatasetError(f"invalid environment schema: {path}")
    if value.get("runtime_profile") not in _RUNTIME_PROFILES:
        raise MTARDatasetError(f"invalid runtime profile: {path}")
    return dict(value)


def load_dataset_selection(dataset_root: Path | str, schema: str) -> tuple[str, ...]:
    root = Path(dataset_root)
    if schema == BASIC_SCHEMA_VERSION:
        return load_selection(root / "task_ids.yml")
    if schema == SCHEMA_VERSION:
        return load_selection(root / "subset100_task_ids.yml")
    if schema != EXPANSION_SCHEMA_VERSION:
        raise MTARDatasetError(f"unsupported MTAR schema: {schema}")
    try:
        value = yaml.safe_load((root / "task_ids.yml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise MTARDatasetError(f"invalid expansion selection: {exc}") from exc
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema_version", "task_ids"}
        or value["schema_version"] != EXPANSION_SELECTION_SCHEMA_VERSION
        or not isinstance(value["task_ids"], list)
        or not value["task_ids"]
        or not all(isinstance(item, str) for item in value["task_ids"])
    ):
        raise MTARDatasetError("invalid expansion selection contract")
    selected = value["task_ids"]
    for task_id in selected:
        _ordinal(task_id)
    if len(selected) != len(set(selected)):
        raise MTARDatasetError("duplicate expansion task IDs")
    return tuple(sorted(selected, key=_ordinal))


def validate_dataset(dataset_root: Path | str) -> dict[str, Any]:
    dataset_root = Path(dataset_root).resolve()
    try:
        manifest = json.loads(
            (dataset_root / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise MTARDatasetError(f"invalid normalized manifest: {exc}") from exc
    basic = manifest.get("schema_version") == BASIC_SCHEMA_VERSION
    allowed_root = {
        "schema_version",
        "mtar_source",
        "normalization" if basic else "turngate_builder",
        "counts",
        "tasks",
    }
    if (
        set(manifest) != allowed_root
        or manifest.get("schema_version") not in {SCHEMA_VERSION, BASIC_SCHEMA_VERSION, EXPANSION_SCHEMA_VERSION}
    ):
        raise MTARDatasetError("unsupported MTAR manifest contract")
    expanded = manifest["schema_version"] == EXPANSION_SCHEMA_VERSION
    selection = set(load_dataset_selection(dataset_root, manifest["schema_version"]))
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != len(selection):
        raise MTARDatasetError("manifest task count must match its frozen selection")
    if expanded and manifest.get("mtar_source") != {
        "repository": SOURCE_REPOSITORY, "git_commit": EXPECTED_SOURCE_COMMIT,
    }:
        raise MTARDatasetError("expansion source must use the pinned MTAR commit")
    actual_counts = {"total": len(selection), "ready": 0}
    if not basic:
        actual_counts["quarantined"] = 0
    seen: set[str] = set()
    documented = "" if basic else (dataset_root / "CHANGELOG.md").read_text(encoding="utf-8")
    expected_task_keys = {
        "task_id",
        "name",
        "tool",
        "path",
        "data_status",
        "provenance",
        "changes",
        "quarantine_reason",
    }
    if basic:
        expected_task_keys.remove("quarantine_reason")
    task_release_files = list((dataset_root / "tasks").glob("single.*/release.yml"))
    task_release_mode = bool(task_release_files)
    if (expanded or task_release_mode) and len(task_release_files) != len(selection):
        raise MTARDatasetError("task-owned release status must cover the full selection")
    for row in tasks:
        if not isinstance(row, Mapping) or set(row) != expected_task_keys:
            raise MTARDatasetError("invalid manifest task index")
        task_id = str(row.get("task_id") or "")
        status = str(row.get("data_status") or "")
        if (
            task_id in seen
            or task_id not in selection
            or status not in ({"ready"} if basic else {"ready", "quarantined"})
        ):
            raise MTARDatasetError(f"invalid or duplicate manifest task: {task_id}")
        seen.add(task_id)
        actual_counts[status] += 1
        expected_path = f"tasks/{task_id}"
        if row.get("path") != expected_path:
            raise MTARDatasetError(f"invalid stable task path: {task_id}")
        task_root = dataset_root / "tasks" / task_id
        if expanded:
            _tool_family(str(row["tool"]))
            if (dataset_root / "tasks").is_symlink() or task_root.is_symlink() or any(p.is_symlink() for p in task_root.rglob("*")):
                raise MTARDatasetError(f"expansion tasks cannot contain symlinks: {task_id}")
        if task_release_mode:
            release_path = task_root / "release.yml"
            try:
                release = yaml.safe_load(release_path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError) as exc:
                raise MTARDatasetError(f"invalid task release record: {task_id}: {exc}") from exc
            if (
                not isinstance(release, Mapping)
                or set(release) != {"schema_version", "status", "blockers"}
                or release["schema_version"] != "mtar-task-release-v1"
                or release["status"] != ("ready" if status == "ready" else "blocked")
                or not isinstance(release["blockers"], list)
                or bool(release["blockers"]) != (status == "quarantined")
            ):
                raise MTARDatasetError(f"manifest/task release mismatch: {task_id}")
            if status == "quarantined" and (
                not all(
                isinstance(item, Mapping)
                and set(item) == {"code", "detail"}
                and isinstance(item["code"], str)
                and isinstance(item["detail"], str)
                and bool(item["detail"])
                for item in release["blockers"]
                )
                or "; ".join(item["detail"] for item in release["blockers"]) != row["quarantine_reason"]
            ):
                raise MTARDatasetError(f"invalid task blockers: {task_id}")
        try:
            task_root.relative_to(dataset_root)
        except ValueError as exc:
            raise MTARDatasetError(f"unsafe task path: {task_id}") from exc
        required = ["utils/environment.yml"]
        # Missing upstream bytes are a visible data blocker, never a synthetic
        # replacement prompt. Only blocked expansion rows can omit task.md.
        if not expanded or status == "ready":
            required.append("task.md")
        for relative in required:
            if not (task_root / relative).is_file():
                raise MTARDatasetError(
                    f"incomplete normalized task: {task_id}/{relative}"
                )
        for relative in ("utils/dependencies.yml", "utils/evaluator.py"):
            if status == "ready" and not (task_root / relative).is_file():
                raise MTARDatasetError(f"incomplete ready task: {task_id}/{relative}")
        # Git cannot represent empty directories. A missing seed root therefore
        # has the same meaning as an empty one; workers materialize it locally.
        for directory in ("workspace", "rootfs/etc", "rootfs/tmp"):
            seed = task_root / directory
            if seed.exists() and not seed.is_dir():
                raise MTARDatasetError(f"invalid seed directory: {task_id}/{directory}")
        environment = _load_environment(task_root / "utils/environment.yml")
        try:
            dependencies = _read_dependencies(task_root / "utils/dependencies.yml")
            validate_dependencies(_tool_family(str(row["tool"])), dependencies)
        except (MTARDatasetError, ValueError) as exc:
            if status == "ready":
                raise MTARDatasetError(
                    f"ready dependency contract is invalid: {task_id}: {exc}"
                ) from exc
        evaluator_path = task_root / "utils/evaluator.py"
        evaluator_source = (
            evaluator_path.read_bytes() if evaluator_path.is_file() else b""
        )
        if status == "ready" and environment["evaluator_python_requirements"] != list(
            evaluator_python_requirements(evaluator_source)
        ):
            raise MTARDatasetError(f"evaluator requirement mismatch: {task_id}")
        try:
            compile(evaluator_source, str(task_root / "utils/evaluator.py"), "exec")
            validate_evaluator_imports(evaluator_source)
            compile(
                build_evaluator_entrypoint(evaluator_source),
                "evaluate_single.py",
                "exec",
            )
        except (SyntaxError, UnicodeError, EvaluatorCompatibilityError) as exc:
            if status == "ready":
                raise MTARDatasetError(
                    f"ready evaluator is incompatible: {task_id}: {exc}"
                ) from exc
        if status == "ready" and row.get("quarantine_reason") is not None:
            raise MTARDatasetError(f"ready task has a quarantine reason: {task_id}")
        if expanded and status == "ready":
            from .expansion import validate_acceptance

            validate_acceptance(task_root)
        if (
            status == "quarantined"
            and not str(row.get("quarantine_reason") or "").strip()
        ):
            raise MTARDatasetError(f"quarantined task lacks a reason: {task_id}")
        if not basic and row.get("provenance") != "copied" and f"| {task_id} |" not in documented:
            raise MTARDatasetError(f"CHANGELOG.md omits {task_id}")
    if seen != selection:
        raise MTARDatasetError(
            "manifest task IDs do not equal the frozen selection"
        )
    if manifest.get("counts") != actual_counts:
        raise MTARDatasetError("manifest status counts do not match task rows")
    if basic and actual_counts != {"total": 75, "ready": 75}:
        raise MTARDatasetError("basic MTAR data must contain exactly 75 ready tasks")
    if not basic and not (dataset_root / "GENERATED").is_file():
        raise MTARDatasetError("generated dataset marker is missing")
    serialized = json.dumps(manifest, sort_keys=True).casefold()
    for forbidden in (
        "sha256",
        "file_hash",
        "evaluator_override",
        "dependency_mismatch",
    ):
        if forbidden in serialized:
            raise MTARDatasetError(f"forbidden manifest metadata: {forbidden}")
    if expanded:
        from .expansion import validate_release_snapshot

        validate_release_snapshot(dataset_root, manifest)
    return manifest


def load_task(dataset_root: Path | str, task_id: str) -> tuple[Path, Mapping[str, Any]]:
    manifest = validate_dataset(dataset_root)
    row = next((item for item in manifest["tasks"] if item["task_id"] == task_id), None)
    if row is None:
        raise MTARDatasetError(f"unknown normalized MTAR task: {task_id}")
    if row["data_status"] == "quarantined":
        raise MTARDatasetQuarantinedError(task_id, str(row["quarantine_reason"]))
    return Path(dataset_root).resolve() / "tasks" / task_id, row
