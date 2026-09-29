"""Generic loader for PostgreSQL task-owned lease metadata."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

DATASET_ROOT = Path(__file__).resolve().parents[5] / "data/mtar"
FIXTURE_VERSION = "mtar-postgres-basic-v2"


def _lease(task_root: Path) -> dict | None:
    path = task_root / "utils/postgres_lease.yml"
    if not path.is_file():
        return None
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {"schema_version", "fixture_version", "fixture_sql", "mcp_database", "hard_oracle", "judge_only"}
    if not isinstance(value, dict) or set(value) != expected or value["schema_version"] != "mtar-postgres-lease-v1":
        raise ValueError(f"invalid PostgreSQL task lease metadata: {path}")
    if value["fixture_version"] != FIXTURE_VERSION or value["mcp_database"] not in {"postgres", "template1"}:
        raise ValueError(f"unsupported PostgreSQL task lease metadata: {path}")
    if not isinstance(value["fixture_sql"], str) or not isinstance(value["hard_oracle"], bool) or not isinstance(value["judge_only"], bool):
        raise ValueError(f"invalid PostgreSQL task lease values: {path}")
    return value


def _task_ids(key: str) -> frozenset[str]:
    return frozenset(path.parent.parent.name for path in (DATASET_ROOT / "tasks").glob("single.*/utils/postgres_lease.yml") if (_lease(path.parent.parent) or {}).get(key))


HARD_ORACLE_TASKS = _task_ids("hard_oracle")
JUDGE_ONLY_TASKS = _task_ids("judge_only")
SUPPORTED_TASKS = HARD_ORACLE_TASKS | JUDGE_ONLY_TASKS


def environment_spec(task_id: str, task_root: Path, *, allow_effect_only: bool = False) -> dict:
    # allow_effect_only remains accepted for existing replay callers.
    spec = _lease(Path(task_root))
    if spec is None or task_id != Path(task_root).name:
        raise ValueError(f"PostgreSQL leased fixture not implemented for {task_id}")
    seed = Path(task_root) / "utils/seed.sql"
    return {
        "seed_path": str(seed.resolve()),
        "seed_sha256": hashlib.sha256(seed.read_bytes()).hexdigest(),
        "fixture_sql": spec["fixture_sql"],
        "fixture_version": spec["fixture_version"],
        "mcp_database": spec["mcp_database"],
    }


def require_scoring_mode(task_id: str, *, lm_judge: bool) -> None:
    if task_id in JUDGE_ONLY_TASKS and not lm_judge:
        raise ValueError(f"{task_id} requires LM Judge scoring (search.semantic_completion: true); no hard oracle is available")


def checkpoint_path(task_id: str, task_root: Path) -> Path:
    return Path(task_root) / "checkpoints.md"


def unscored_evaluation() -> dict:
    return {"status": "unscored", "reason": "lm_judge_required", "result": None,
            "total": None, "progress": None, "success": None, "checkpoints": [],
            "oracle": None, "fixture_version": FIXTURE_VERSION}
