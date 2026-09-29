#!/usr/bin/env python3
"""Run resumable attack suites with one shared Controller.

Scheduling for MTAR and OAS lives here; search and replay live in
the sead package.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from sead.attacks.dart.config import load_tree_search_config
from sead.config import load_config_data
from sead.infrastructure.campaign_cleanup import (
    add_cleanup_arguments,
    run_cleanup_preflight,
    with_campaign_cleanup,
)
from sead.campaigns.attack_reporting import input_fingerprint, ledger_totals
from sead.attacks.dart.runner import (
    create_controller_backend,
    prepare_controller_credentials,
    prepare_openhands_runtime_for_config,
    run_from_config,
)
from sead.benchmarks.oas.dataset import (
    DEFAULT_CANDIDATE_INDEX as DEFAULT_OAS_CANDIDATE_INDEX,
    DEFAULT_SELECTION as DEFAULT_OAS_SELECTION,
)
from sead.campaigns import (
    CampaignExecutor,
    acquire_campaign_lock,
    atomic_write_json,
    resource_aware_results,
    task_slug,
)
from sead.environments.resources import web_resources
from sead.benchmarks.registry import (
    BenchmarkCatalog,
    partition_task_ids as _partition_task_ids,  # noqa: F401 - compatibility import
    task_resources as _task_resources,  # noqa: F401 - compatibility import
)


PATH_FIELDS = {
    "benchmark": (
        "dataset_root",
        "openhands_root",
        "worker_python",
        "service_deployments",
        "selection",
        "candidate_index",
    ),
    "controller": ("python",),
    "execution": ("tac_pool_config",),
}


def _write_json(path: Path, value: Any) -> None:
    atomic_write_json(path, value)


def _archive_incomplete_task_dir(
    task_dir: Path, suite_dir: Path, *, include_summary: bool = False,
) -> Path | None:
    """Move a prior incomplete attempt aside without deleting its artifacts."""

    if not task_dir.exists() or ((task_dir / "summary.json").is_file() and not include_summary):
        return None
    if task_dir.is_symlink() or not task_dir.is_dir():
        raise ValueError(f"unsafe incomplete task path: {task_dir}")
    tasks_root = (suite_dir / "tasks").resolve()
    if task_dir.resolve().parent != tasks_root:
        raise ValueError(f"incomplete task directory escapes suite: {task_dir}")
    archive_root = suite_dir / "failed_attempts" / task_dir.name
    archive_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = archive_root / timestamp
    ordinal = 1
    while destination.exists():
        ordinal += 1
        destination = archive_root / f"{timestamp}-{ordinal}"
    shutil.move(str(task_dir), str(destination))
    return destination


def _attempt_path(suite_dir: Path, task_id: str) -> Path:
    # Kept outside the output directory: the runner requires a fresh directory.
    return suite_dir / "task_attempts" / f"{_task_slug(task_id)}.json"


def _read_attempt(suite_dir: Path, task_id: str) -> dict[str, Any]:
    path = _attempt_path(suite_dir, task_id)
    return json.loads(path.read_text()) if path.is_file() else {}


def _archived_attempts(suite_dir: Path, task_id: str) -> list[Path]:
    root = suite_dir / "failed_attempts" / _task_slug(task_id)
    return sorted(path for path in root.iterdir() if path.is_dir()) if root.is_dir() else []


def _attempt_count(suite_dir: Path, task_id: str) -> int:
    task_dir = suite_dir / "tasks" / _task_slug(task_id)
    legacy_count = len(_archived_attempts(suite_dir, task_id)) + int(
        task_dir.is_dir() and any(task_dir.iterdir())
    )
    return max(legacy_count, int(_read_attempt(suite_dir, task_id).get("attempt", 0)))


def _archive_failed_attempt(suite_dir: Path, task_id: str) -> Path | None:
    task_dir = suite_dir / "tasks" / _task_slug(task_id)
    record = _read_attempt(suite_dir, task_id)
    previous = _aggregate(suite_dir, [task_id])["tasks"][0]
    record = {**record, "attempt": _attempt_count(suite_dir, task_id),
              "outcome_status": previous["status"],
              "technical_errors": previous["technical_errors"]}
    # A crash can occur after the durable start record but before the runner
    # creates its output directory. That attempt still consumes the retry cap.
    if not task_dir.exists() and int(record.get("attempt", 0)) > len(_archived_attempts(suite_dir, task_id)):
        task_dir.mkdir(parents=True)
    archived = _archive_incomplete_task_dir(task_dir, suite_dir, include_summary=True)
    if archived is not None:
        if record:
            _write_json(archived / "attempt.json", record)
        for suffix in ("stdout.log", "stderr.log"):
            log = suite_dir / f"run-{_task_slug(task_id)}.{suffix}"
            if log.exists():
                shutil.move(str(log), str(archived / f"run.{suffix}"))
    return archived


def _retry_policy(suite: dict[str, Any]) -> tuple[int, float]:
    retries = suite.get("max_technical_retries", 0)
    backoff = suite.get("technical_retry_backoff_seconds", 30)
    if type(retries) is not int or not 0 <= retries <= 10:
        raise ValueError("suite.max_technical_retries must be an integer between 0 and 10")
    if type(backoff) not in (int, float) or not 0 <= backoff <= 300:
        raise ValueError("suite.technical_retry_backoff_seconds must be between 0 and 300")
    return retries, float(backoff)


def _pending_technical_tasks(suite_dir, task_ids, max_retries, benchmark_kind):
    rows = _aggregate(suite_dir, task_ids, benchmark_kind=benchmark_kind)["tasks"]
    return [row["task_id"] for row in rows
            if row["status"] != "completed"
            and _attempt_count(suite_dir, row["task_id"]) < 1 + max_retries]


def _attempt_history(suite_dir: Path, task_id: str) -> list[dict[str, Any]]:
    paths = _archived_attempts(suite_dir, task_id)
    history = []
    current = _read_attempt(suite_dir, task_id)
    task_dir = suite_dir / "tasks" / _task_slug(task_id)
    if task_dir.exists() or int(current.get("attempt", 0)) > len(paths):
        paths.append(task_dir)
    for ordinal, path in enumerate(paths, 1):
        record_path = path / "attempt.json"
        record = (current if path == task_dir else
                  json.loads(record_path.read_text()) if record_path.is_file() else {})
        usage = {}
        try:
            usage = json.loads((path / "summary.json").read_text()).get("usage", {})
        except (OSError, ValueError, AttributeError):
            pass
        if (path / "usage.jsonl").is_file():
            usage = {**usage, **ledger_totals(path / "usage.jsonl")}
        history.append({**record, "attempt": ordinal,
                        "artifact_directory": str(path), "usage": usage})
    return history


def _snapshot_config(
    base_path: Path,
    destination: Path,
    task_id: str,
    *,
    controller_cuda_visible_devices: str | None = None,
    dataset_root: Path | None = None,
) -> None:
    raw = load_config_data(base_path)
    if not isinstance(raw, dict):
        raise ValueError(f"configuration is not a mapping: {base_path}")
    snapshot = copy.deepcopy(raw)
    snapshot["benchmark"]["task_id"] = task_id
    forum_pool = (snapshot.get("execution") or {}).get("forum_pool")
    if isinstance(forum_pool, dict):
        from sead.environments.services.forum_pool import forum_instance
        spec = forum_instance({"forum_pool": forum_pool}, task_id)
        if spec is not None:
            service_settings_path = base_path.parent / str(snapshot["benchmark"]["service_deployments"])
            service_settings = yaml.safe_load(service_settings_path.read_text(encoding="utf-8"))
            service_settings["services"]["reddit"] = {
                "display_name": "SafeArena Forum",
                "url": f"http://127.0.0.1:{spec['port']}",
                "credentials": {"username": "MarvelsGrantMan136", "password": "test1234"},
            }
            service_snapshot = destination.with_name(destination.stem + "-services.yml")
            service_snapshot.write_text(yaml.safe_dump(service_settings, sort_keys=False))
            snapshot["benchmark"]["service_deployments"] = str(service_snapshot.resolve())
    if dataset_root is not None:
        snapshot["benchmark"]["dataset_root"] = str(dataset_root.resolve())
        # Collection membership is frozen in suite provenance. A task snapshot
        # is deliberately single-source so workers never need to resolve the
        # cross-release collection again.
        snapshot["benchmark"].pop("collection", None)
        snapshot["benchmark"].pop("group", None)
    if controller_cuda_visible_devices is not None:
        snapshot["controller"]["cuda_visible_devices"] = (
            controller_cuda_visible_devices
        )
    for section, fields in PATH_FIELDS.items():
        for field in fields:
            value = snapshot.get(section, {}).get(field)
            if value and not Path(str(value)).is_absolute():
                # Keep virtual-environment interpreter symlinks intact. Path.resolve()
                # would collapse `.venv/bin/python` to the system interpreter and lose
                # the environment's installed packages.
                snapshot[section][field] = os.path.abspath(
                    base_path.parent / str(value)
                )
    destination.write_text(
        yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _task_slug(task_id: str) -> str:
    # Compatibility shim for callers that imported this script helper.
    return task_slug(task_id).replace(".", "-")


def _post_task_service_reset(
    config_path: Path,
    task_id: str,
    task_dir: Path,
    *,
    environment: dict[str, str],
    benchmark_kind: str = "mtar",
) -> dict[str, Any]:
    """Reset shared benchmark services after a complete task run."""

    audit_path = task_dir / "service_reset.json"
    task_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        raw = load_config_data(config_path)
        benchmark = raw.get("benchmark") if isinstance(raw, dict) else None
        execution = raw.get("execution") if isinstance(raw, dict) else None
        if not isinstance(benchmark, dict):
            raise ValueError("snapshot benchmark configuration is invalid")
        owner_index = task_dir / "worker_owners.json"
        indexed_workers: list[Path] = []
        indexed_replay_ids: set[str] = set()
        if owner_index.is_file():
            raw_index = json.loads(owner_index.read_text())
            workers = raw_index.get("workers") if isinstance(raw_index, dict) else None
            if not isinstance(workers, list):
                raise ValueError("worker ownership index is invalid")
            for row in workers:
                if not isinstance(row, dict) or row.get("task_id") != task_id:
                    raise ValueError("worker ownership record has the wrong task")
                relative = Path(str(row.get("path") or ""))
                if relative.is_absolute() or ".." in relative.parts or relative.name != "worker":
                    # worker_retry_N names are admitted explicitly below.
                    if not (relative.name.startswith("worker_retry_") and relative.name.removeprefix("worker_retry_").isdigit()):
                        raise ValueError("unsafe worker ownership path")
                worker = (task_dir / relative).resolve()
                if not worker.is_relative_to(task_dir.resolve()):
                    raise ValueError("worker ownership path escapes task directory")
                indexed_workers.append(worker)
                replay_id = str(row.get("replay_id") or "")
                if not replay_id:
                    raise ValueError("worker ownership record has no replay id")
                indexed_replay_ids.add(replay_id)
        from sead.environments.registry import environment_config
        if environment_config(execution or {}):
            from sead.environments.session import recover_environment
            audited = []
            candidates = [worker / "environment_private.json" for worker in indexed_workers]
            candidates.extend(task_dir.rglob("environment_private.json"))
            for private_path in dict.fromkeys(candidates):
                if not private_path.is_file():
                    continue
                parts = private_path.relative_to(task_dir).parts
                legacy = (len(parts) == 4 and parts[0] in {"nodes", "terminal_replays"}
                    and (parts[2] == "worker" or parts[2].startswith("worker_retry_")))
                if not legacy and private_path.parent.resolve() not in indexed_workers:
                    continue
                recover_environment(private_path.parent)
                audited.append(str(private_path.parent.relative_to(task_dir)))
            audit = {"schema_version": "environment-task-cleanup-v1", "task_id": task_id,
                     "succeeded": True, "shared_reset_required": False, "workers": audited,
                     "duration_seconds": round(time.perf_counter() - started, 6)}
            _write_json(audit_path, audit)
            return audit
        from sead.environments.leases import LeaseClient, postgres_mode
        if benchmark_kind == "mtar" and postgres_mode(execution or {}) == "leased":
            from sead.benchmarks.mtar.dataset import load_task, load_task_dependencies
            task_root, row = load_task(benchmark["dataset_root"], task_id)
            if "mcp-postgres" in load_task_dependencies(task_root, str(row["tool"])):
                # Audit only this task's replay owners, including failed attempts.
                replay_ids = set(indexed_replay_ids)
                for path in task_dir.rglob("request.json"):
                    parts = path.relative_to(task_dir).parts
                    legacy = (len(parts) == 4 and parts[0] in {"nodes", "terminal_replays"}
                        and (parts[2] == "worker" or parts[2].startswith("worker_retry_")))
                    if not legacy and path.parent.resolve() not in indexed_workers:
                        continue  # Never treat Target workspace files as ownership evidence.
                    record = json.loads(path.read_text())
                    if record.get("task_id") == task_id and record.get("replay_id"):
                        replay_ids.add(record["replay_id"])
                status = LeaseClient(execution["postgres"]["manager_socket"]).call("inspect")
                owned = [lease for lease in status["leases"]
                         if lease["owner"]["replay_id"] in replay_ids]
                audit = {"schema_version": "postgres-task-lease-cleanup-v1", "task_id": task_id,
                         "succeeded": all(lease["state"] == "destroyed" for lease in owned),
                         "leases": owned, "shared_reset_required": False,
                         "duration_seconds": round(time.perf_counter() - started, 6)}
                _write_json(audit_path, audit)
                return audit
        hostname = str(
            execution.get("server_hostname")
            if isinstance(execution, dict) and execution.get("server_hostname")
            else "localhost"
        )
        if benchmark_kind == "oas":
            command = [
                str(benchmark["worker_python"]),
                "-m", "sead.environments.workers.oas",
                "--reset-only",
                "--dataset-root",
                str(benchmark["dataset_root"]),
                "--task-id",
                task_id,
                "--selection",
                str(benchmark.get("selection") or DEFAULT_OAS_SELECTION),
                "--candidate-index",
                str(benchmark.get("candidate_index") or DEFAULT_OAS_CANDIDATE_INDEX),
                "--server-hostname",
                hostname,
            ]
        elif benchmark_kind == "mtar":
            command = [
                str(benchmark["worker_python"]),
                "-m", "sead.environments.workers.mtar",
                "--reset-only",
                "--dataset-root",
                str(benchmark["dataset_root"]),
                "--task-id",
                task_id,
                "--server-hostname",
                hostname,
                "--service-deployments",
                str(
                    execution.get("service_deployments_path")
                    or benchmark.get("service_deployments")
                    or "config/mtar_service_deployments.yml"
                ),
                "--snapshot-config",
                str(config_path),
            ]
        else:
            raise ValueError(f"unsupported benchmark kind: {benchmark_kind}")
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            env=environment,
            timeout=750,
            check=False,
        )
        try:
            parsed = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "reset worker emitted invalid JSON: "
                + (completed.stdout.strip() or "empty stdout")
            ) from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("reset worker audit is not an object")
        audit = dict(parsed)
        audit["subprocess_returncode"] = completed.returncode
        if completed.returncode != 0:
            audit["succeeded"] = False
            audit.setdefault(
                "error",
                completed.stderr.strip() or f"reset worker exited {completed.returncode}",
            )
        elif audit.get("succeeded") is not True:
            audit["succeeded"] = False
            audit.setdefault("error", "reset worker did not confirm success")
    except Exception as exc:  # noqa: BLE001 -- task-final cleanup audit boundary
        audit = {
            "schema_version": "mtar-task-service-reset-v1",
            "task_id": task_id,
            "succeeded": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    audit["duration_seconds"] = round(time.perf_counter() - started, 6)
    _write_json(audit_path, audit)
    return audit


def _service_reset_audit(task_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    path = task_dir / "service_reset.json"
    if not path.is_file():
        return None, []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"post-task service reset: invalid audit: {exc}"]
    if not isinstance(value, dict):
        return None, ["post-task service reset: audit is not an object"]
    if value.get("succeeded") is not True:
        return value, [
            "post-task service reset: "
            + str(value.get("error") or "reset did not confirm success")
        ]
    return value, []


def _validation_category(message: str) -> str:
    text = message.casefold()
    if "invalid json" in text:
        return "invalid_json"
    if "missing required fields" in text:
        return "missing_fields"
    if "unsupported fields" in text:
        return "unsupported_fields"
    if (
        "requires exactly" in text
        or "requires 2 to" in text
        or "expected exactly" in text
    ):
        return "cardinality"
    if "semantic duplicates" in text or "repeats or rewrites" in text:
        return "duplicate"
    if "depends on a sibling" in text:
        return "sibling_dependency"
    if "strategy" in text or "parallel_verification" in text:
        return "strategy_contract"
    return "other"


def _technical_node_errors(
    task_dir: Path, selected_node_id: str | None = None
) -> list[str]:
    search_path = task_dir / "search.json"
    if not search_path.is_file():
        return []
    try:
        search = json.loads(search_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ["invalid search.json"]
    nodes = search.get("nodes") if isinstance(search, dict) else None
    if not isinstance(nodes, dict):
        return ["search.json nodes is not an object"]
    errors: list[str] = []
    for node_id, node in nodes.items():
        if not isinstance(node, dict) or node.get("status") != "technical_error":
            continue
        if selected_node_id is not None and not (
            node_id == selected_node_id or selected_node_id.startswith(node_id + ".")
        ):
            continue
        reason = str(node.get("prune_reason") or "technical_error")
        errors.append(f"{node_id}: {reason}")
    if selected_node_id is None:
        return errors
    termination_index = task_dir / "path_terminations" / "index.json"
    if not termination_index.is_file():
        return errors
    try:
        attempts = json.loads(termination_index.read_text(encoding="utf-8")).get(
            "terminations", []
        )
    except (OSError, json.JSONDecodeError):
        return errors + ["invalid path_terminations/index.json"]
    for attempt in attempts:
        if not isinstance(attempt, dict) or attempt.get("node_id") != selected_node_id:
            continue
        technical_error = str(attempt.get("scorer_error") or "").strip()
        if technical_error:
            errors.append(f"{selected_node_id} terminal scoring: {technical_error}")
    return errors


def _resource_aware_results(
    executor: CampaignExecutor,
    task_ids: list[str],
    resources_by_task: dict[str, tuple[str, ...]],
    run_task: Any,
    *,
    max_workers: int,
    resource_capacities=None,
    worker_limit=None,
    retry_delay=None,
    initial_delays=None,
):
    """Compatibility wrapper around the shared campaign scheduler."""

    if executor.max_workers != max_workers:
        raise ValueError("executor/max_workers mismatch")
    yield from resource_aware_results(
        executor,
        task_ids,
        resources_by_task,
        run_task,
        resource_capacities=resource_capacities,
        worker_limit=worker_limit,
        retry_delay=retry_delay,
        initial_delays=initial_delays,
    )


def _dataset_root(base_config: Path) -> Path:
    value = load_config_data(base_config)
    if not isinstance(value, dict):
        raise ValueError(f"configuration is not a mapping: {base_config}")
    configured = value.get("benchmark", {}).get("dataset_root")
    if not isinstance(configured, str) or not configured.strip():
        raise ValueError("benchmark.dataset_root is required")
    path = Path(configured)
    return path.resolve() if path.is_absolute() else (base_config.parent / path).resolve()


def _benchmark_catalog(
    base_config: Path, dataset_root_override: Path | None = None
) -> BenchmarkCatalog:
    value = load_config_data(base_config)
    if not isinstance(value, dict):
        raise ValueError(f"configuration is not a mapping: {base_config}")
    benchmark = value.get("benchmark")
    if not isinstance(benchmark, dict):
        raise ValueError("benchmark must be an object")
    kind = str(benchmark.get("kind") or "")
    collection = benchmark.get("collection")
    configured_root = benchmark.get("dataset_root")
    if collection is not None and kind != "mtar":
        raise ValueError("benchmark.collection is supported only for MTAR")
    if collection is None and benchmark.get("group") is not None:
        raise ValueError("benchmark.group requires benchmark.collection")
    if dataset_root_override is not None:
        if collection is not None:
            raise ValueError(
                "--dataset-root cannot override benchmark.collection; change the configured group"
            )
        root: Path | None = dataset_root_override.resolve()
    elif collection is not None:
        if configured_root is not None:
            raise ValueError(
                "benchmark.collection conflicts with benchmark.dataset_root"
            )
        root = None
    else:
        root = _dataset_root(base_config)
    group = benchmark.get("group")
    if collection is not None and (not isinstance(group, str) or not group.strip()):
        raise ValueError("benchmark.group is required with benchmark.collection")
    return BenchmarkCatalog(
        kind,
        root,
        _benchmark_selection(base_config, kind),
        _oas_candidate_index(base_config),
        collection=Path(str(collection)) if collection is not None else None,
        group=str(group) if group is not None else None,
    )


def _benchmark_kind(base_config: Path) -> str:
    value = load_config_data(base_config)
    if not isinstance(value, dict):
        raise ValueError(f"configuration is not a mapping: {base_config}")
    kind = str(value.get("benchmark", {}).get("kind") or "")
    return kind


def _benchmark_selection(base_config: Path, benchmark_kind: str) -> Path:
    value = load_config_data(base_config)
    benchmark = value.get("benchmark") if isinstance(value, dict) else None
    configured = benchmark.get("selection") if isinstance(benchmark, dict) else None
    if not configured:
        return Path(DEFAULT_OAS_SELECTION)
    path = Path(str(configured))
    return path if path.is_absolute() else (base_config.parent / path).resolve()


def _oas_selection(base_config: Path) -> Path:
    """Compatibility alias retained for tests and downstream script imports."""

    return _benchmark_selection(base_config, _benchmark_kind(base_config))


def _oas_candidate_index(base_config: Path) -> Path:
    value = load_config_data(base_config)
    benchmark = value.get("benchmark") if isinstance(value, dict) else None
    configured = (
        benchmark.get("candidate_index") if isinstance(benchmark, dict) else None
    )
    if not configured:
        return Path(DEFAULT_OAS_CANDIDATE_INDEX)
    path = Path(str(configured))
    return path if path.is_absolute() else (base_config.parent / path).resolve()


def _aggregate(
    suite_dir: Path,
    task_ids: list[str],
    excluded_tasks: dict[str, str] | None = None,
    benchmark_kind: str = "mtar",
    method: str = "dart",
) -> dict[str, Any]:
    rows = []
    strategy_totals: Counter[str] = Counter()
    width_totals: Counter[str] = Counter()
    validation_totals: Counter[str] = Counter()
    success_depths: Counter[str] = Counter()
    phase_totals: dict[str, dict[str, float]] = {}
    for task_id in task_ids:
        task_dir = suite_dir / "tasks" / _task_slug(task_id)
        service_reset, service_reset_errors = _service_reset_audit(task_dir)
        if isinstance(service_reset, dict) and isinstance(
            service_reset.get("duration_seconds"), (int, float)
        ):
            reset_seconds = float(service_reset["duration_seconds"])
            totals = phase_totals.setdefault(
                "post_task_service_reset",
                {"count": 0.0, "total_seconds": 0.0, "max_seconds": 0.0},
            )
            totals["count"] += 1
            totals["total_seconds"] += reset_seconds
            totals["max_seconds"] = max(totals["max_seconds"], reset_seconds)
        summary_path = task_dir / "summary.json"
        attempt = _read_attempt(suite_dir, task_id)
        if attempt.get("state") == "running":
            service_reset_errors.append("task attempt did not finish (running or interrupted)")
        elif attempt.get("return_code") not in (None, 0, 2):
            service_reset_errors.extend(attempt.get("errors") or ["task runner failed"])
        if not summary_path.is_file():
            rows.append(
                {
                    "task_id": task_id,
                    "status": "missing_summary",
                    "technical_errors": service_reset_errors,
                    "service_reset": service_reset,
                }
            )
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(summary, dict):
                raise ValueError("summary is not an object")
        except (OSError, ValueError) as exc:
            rows.append({"task_id": task_id, "status": "technical_incomplete",
                         "technical_errors": [f"invalid summary.json: {exc}", *service_reset_errors],
                         "service_reset": service_reset})
            continue
        task_phase_timings = summary.get("phase_timings", {})
        raw_phases = (
            task_phase_timings.get("phases", {})
            if isinstance(task_phase_timings, dict)
            else {}
        )
        if isinstance(raw_phases, dict):
            for phase, raw_stats in raw_phases.items():
                if not isinstance(raw_stats, dict):
                    continue
                totals = phase_totals.setdefault(
                    str(phase), {"count": 0.0, "total_seconds": 0.0, "max_seconds": 0.0}
                )
                totals["count"] += float(raw_stats.get("count") or 0)
                totals["total_seconds"] += float(
                    raw_stats.get("total_seconds") or 0
                )
                totals["max_seconds"] = max(
                    totals["max_seconds"], float(raw_stats.get("max_seconds") or 0)
                )
        task_strategies: Counter[str] = Counter()
        task_widths: Counter[str] = Counter()
        task_validations: Counter[str] = Counter()
        trace_path = task_dir / "controller_trace.jsonl"
        if trace_path.is_file():
            for line in trace_path.read_text(encoding="utf-8").splitlines():
                try:
                    trace = json.loads(line)
                except json.JSONDecodeError:
                    task_validations["malformed_trace"] += 1
                    continue
                error = str(trace.get("validation_error") or "").strip()
                if error:
                    task_validations[_validation_category(error)] += 1
                    continue
                strategy = str(trace.get("strategy") or "UNSET")
                width = str(trace.get("actual_candidate_count"))
                task_strategies[strategy] += 1
                task_widths[width] += 1
        strategy_totals.update(task_strategies)
        width_totals.update(task_widths)
        validation_totals.update(task_validations)
        selected_node_id = summary.get("selected_node_id")
        technical_errors = _technical_node_errors(task_dir, selected_node_id)
        if summary.get("technical_status") not in (None, "completed"):
            technical_errors.append(str(summary.get("technical_error") or summary["technical_status"]))
        if summary.get("technical_status") is not None and service_reset is None:
            technical_errors.append("task cleanup audit is missing")
        technical_errors.extend(service_reset_errors)
        if summary.get("attack_success") and selected_node_id:
            success_depths[str(str(selected_node_id).count("."))] += 1
        rows.append(
            {
                "task_id": task_id,
                "attack": summary.get("attack", {"name": "dart"}),
                "selected_trajectory": summary.get("selected_trajectory"),
                "status": (
                    "technical_incomplete" if technical_errors else "completed"
                ),
                "technical_errors": technical_errors,
                "attack_success": bool(summary.get("attack_success")),
                "official_hard_attack_success": bool(
                    summary.get(
                        "official_hard_attack_success",
                        summary.get("attack_success"),
                    )
                ),
                "semantic_attack_success": bool(
                    summary.get("semantic_attack_success")
                ),
                "gemini_judge_attack_success": bool(
                    summary.get("gemini_judge_attack_success")
                ),
                "completion_policy": summary.get(
                    "completion_policy", "hard_evaluator_only"
                ),
                "final_replay_success": bool(summary.get("final_replay_success")),
                "selected_node_id": selected_node_id,
                "stop_reason": summary.get("stop_reason"),
                "usage": summary.get("usage", {}),
                "path_terminations": summary.get("path_terminations", {}),
                "phase_timings": task_phase_timings,
                "service_reset": service_reset,
                "strategy_distribution": dict(sorted(task_strategies.items())),
                "actual_candidate_count_distribution": dict(
                    sorted(task_widths.items())
                ),
                "validation_error_categories": dict(
                    sorted(task_validations.items())
                ),
            }
        )
    attempt_usage_totals: Counter[str] = Counter()
    for row in rows:
        history = _attempt_history(suite_dir, row["task_id"])
        row["attempts"] = history
        row["attempt_count"] = _attempt_count(suite_dir, row["task_id"])
        row["technical_retries_used"] = max(0, row["attempt_count"] - 1)
        for attempt in history:
            for key, value in attempt["usage"].items():
                if type(value) in (int, float):
                    attempt_usage_totals[key] += value
    return {
        "schema_version": f"dart-{benchmark_kind}-v4-suite-summary-v1",
        "attack_method": method,
        "benchmark_kind": benchmark_kind,
        "task_ids": task_ids,
        "excluded_tasks": dict(sorted((excluded_tasks or {}).items())),
        "completed": sum(row["status"] == "completed" for row in rows),
        "attack_successes": sum(row["status"] == "completed" and bool(row.get("attack_success")) for row in rows),
        "official_hard_attack_successes": sum(
            row["status"] == "completed" and bool(row.get("official_hard_attack_success")) for row in rows
        ),
        "semantic_attack_successes": sum(
            row["status"] == "completed" and bool(row.get("semantic_attack_success")) for row in rows
        ),
        "gemini_judge_attack_successes": sum(
            bool(row.get("gemini_judge_attack_success")) for row in rows
        ),
        "technical_incomplete": sum(row["status"] != "completed" for row in rows),
        "attempt_usage_totals": dict(attempt_usage_totals),
        "technical_retries_used": sum(row["technical_retries_used"] for row in rows),
        "strategy_distribution": dict(sorted(strategy_totals.items())),
        "actual_candidate_count_distribution": dict(sorted(width_totals.items())),
        "success_depth_distribution": dict(sorted(success_depths.items())),
        "validation_error_categories": dict(sorted(validation_totals.items())),
        "phase_timings": {
            "phases": {
                phase: {
                    "count": int(values["count"]),
                    "total_seconds": round(values["total_seconds"], 6),
                    "mean_seconds": round(
                        values["total_seconds"] / values["count"], 6
                    )
                    if values["count"]
                    else 0.0,
                    "max_seconds": round(values["max_seconds"], 6),
                }
                for phase, values in sorted(phase_totals.items())
            }
        },
        "tasks": rows,
    }


def _max_concurrent_tasks(base_config: Path, override: int | None) -> int:
    if override is not None:
        value = override
    else:
        raw = load_config_data(base_config)
        suite = raw.get("suite", {})
        if not isinstance(suite, dict):
            raise ValueError("suite must be an object")
        value = suite.get("max_concurrent_tasks", 1)
    if type(value) is not int or value < 1:
        raise ValueError("max_concurrent_tasks must be a positive integer")
    return value


def _resource_plan(catalog, task_ids, base_data, *, base_path: Path | None = None):
    """Validate mixed-family bindings before allocating any model resources."""
    from sead.environments.leases import postgres_mode
    from sead.environments.services.forum_pool import forum_instance
    from sead.benchmarks.mtar.dataset import load_task_dependencies

    execution = dict(base_data.get("execution", {}))
    pool_path = execution.get("tac_pool_config")
    if pool_path and not Path(str(pool_path)).is_absolute():
        config_root = base_path.parent if base_path is not None else Path(__file__).resolve().parents[1] / "config"
        execution["tac_pool_config"] = str((config_root / str(pool_path)).resolve())
    from sead.campaigns.scheduling import leased_resource_plan
    plan = leased_resource_plan(catalog, task_ids, execution, base_data.get("suite", {}))
    if plan is not None:
        return plan
    mode = postgres_mode(execution)
    suite = base_data.get("suite", {})
    capacities = {}
    for name in ("web", "sql"):
        value = suite.get(f"max_concurrent_{name}_tasks")
        if value is not None:
            if type(value) is not int or value < 1:
                raise ValueError(f"suite.max_concurrent_{name}_tasks must be a positive integer")
            capacities[f"capacity:{name}"] = value
    resources = {}
    for task_id in task_ids:
        if catalog.kind == "mtar":
            task_root, row = catalog.mtar_task(task_id)
            dependencies = load_task_dependencies(task_root, str(row["tool"]))
            from sead.environments.services.tac_pool import tac_pool_instance
            tac_spec = tac_pool_instance(execution, task_id, dependencies=dependencies)
            required = set(web_resources(
                dependencies,
                service_instances=(
                    {tac_spec["service"]: tac_spec["instance"]}
                    if tac_spec is not None else None
                ),
            ))
            if "mcp-postgres" in dependencies and mode != "leased":
                required.add("postgres")
            if "reddit" in dependencies and execution.get("forum_pool") is not None:
                forum_instance(execution, task_id, required=True)
                required = {f"web:reddit:{task_id}"}
            if "mcp-postgres" in dependencies and "capacity:sql" in capacities:
                required.add("capacity:sql")
        else:
            required = set(catalog.resources(task_id, postgres_mode=mode))
        if any(key.startswith("web:") for key in required) and "capacity:web" in capacities:
            required.add("capacity:web")
        resources[task_id] = tuple(sorted(required))
    return resources, capacities


def _controller_worker_limit(max_workers, suite):
    """Follow the managed pool as replicas become ready, drain, or fail."""
    manifest = os.environ.get("SEAD_CONTROLLER_POOL_MANIFEST")
    workers_per_replica = suite.get("workers_per_controller")
    if workers_per_replica is None:
        return None
    if type(workers_per_replica) is not int or workers_per_replica < 1:
        raise ValueError("suite.workers_per_controller must be a positive integer")
    if not manifest:
        return None  # Manually managed HTTP Controllers use the explicit limit.

    def limit():
        path = Path(manifest)
        if time.time() - path.stat().st_mtime > 300:
            raise RuntimeError("Controller pool manifest is stale; supervisor may have exited")
        state = json.loads(path.read_text())
        if state.get("fatal_error"):
            raise RuntimeError(f"Controller pool failed: {state['fatal_error']}")
        ready = sum(row["state"] == "ready" for row in state.get("upstreams", []))
        return min(max_workers, ready * workers_per_replica)

    return limit


@with_campaign_cleanup
def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    add_cleanup_arguments(parser)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument(
        "--plan-only", action="store_true",
        help="Validate task selection and resource bindings without starting models or environments.",
    )
    parser.add_argument(
        "--prepare-runtimes-only", action="store_true",
        help="Validate the plan and build source-matched runtime images before allocating Controllers.",
    )
    parser.add_argument(
        "--all-selected",
        action="store_true",
        help="Run every task in the configured benchmark selection; unavailable tasks are reported separately.",
    )
    parser.add_argument(
        "--controller-cuda-visible-devices",
        help="Infrastructure-only Controller device override recorded in suite provenance.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help=(
            "Use a different normalized dataset for this suite and record the "
            "absolute override in provenance."
        ),
    )
    parser.add_argument(
        "--max-concurrent-tasks",
        type=int,
        default=None,
        help="Maximum concurrent task runs; overrides suite.max_concurrent_tasks (default: 1).",
    )
    parser.add_argument(
        "--controller-batch-size",
        type=int,
        help="Maximum Controller microbatch size; defaults to task concurrency.",
    )
    parser.add_argument(
        "--controller-batch-wait-ms",
        type=float,
        default=None,
        help="Maximum time to collect concurrent Controller requests into one batch.",
    )
    parser.add_argument(
        "--archive-incomplete-task-dirs",
        action="store_true",
        help=(
            "Move prior task directories without summary.json under "
            "failed_attempts before retrying them."
        ),
    )
    parser.add_argument(
        "--max-technical-retries",
        type=int,
        default=None,
        help=(
            "Override the configured per-task technical retry count for a "
            "recovery pass. Existing attempt history is retained."
        ),
    )
    args = parser.parse_args(argv)
    configured = load_config_data(args.base_config)
    method = configured.get("attack", {}).get("name", "dart")
    if method != "dart":
        parser.error("attack.name must be dart")
    if args.suite_dir is None:
        output = configured.get("output", {}).get("directory")
        if not output:
            parser.error("--suite-dir or output.directory is required")
        args.suite_dir = Path(output)
    if not args.all_selected and not args.task_id and configured.get("experiment_kind"):
        benchmark_config = configured.get("benchmark", {})
        selection = benchmark_config.get("selection")
        if benchmark_config.get("collection") or selection == "ready":
            args.all_selected = True
        elif benchmark_config.get("task_id"):
            args.task_id = [str(benchmark_config["task_id"])]
    try:
        args.max_concurrent_tasks = _max_concurrent_tasks(
            args.base_config, args.max_concurrent_tasks
        )
    except ValueError as exc:
        parser.error(str(exc))
    controller_batch_size = (
        args.controller_batch_size
        if args.controller_batch_size is not None
        else configured.get("suite", {}).get("controller_batch_size", args.max_concurrent_tasks)
    )
    if args.controller_batch_wait_ms is None:
        args.controller_batch_wait_ms = configured.get("suite", {}).get("controller_batch_wait_ms", 50)
    if controller_batch_size < 1:
        parser.error("--controller-batch-size must be positive")
    if args.controller_batch_wait_ms < 0:
        parser.error("--controller-batch-wait-ms must be non-negative")

    base_config = args.base_config.resolve()
    suite_dir = args.suite_dir.resolve()
    benchmark_kind = _benchmark_kind(base_config)
    if args.all_selected and args.task_id:
        parser.error("--all-selected cannot be combined with --task-id")
    if not args.all_selected and not args.task_id:
        parser.error("provide --task-id at least once or use --all-selected")
    try:
        catalog = _benchmark_catalog(base_config, args.dataset_root)
    except ValueError as exc:
        parser.error(str(exc))
    requested_task_ids = (
        list(catalog.selected_task_ids()) if args.all_selected
        else list(dict.fromkeys(args.task_id))
    )
    task_ids, excluded_tasks = catalog.partition(requested_task_ids)
    if not task_ids:
        parser.error("the selected benchmark group contains no runnable tasks")
    dataset_roots = {
        task_id: catalog.task_dataset_root(task_id).resolve() for task_id in task_ids
    }
    from sead.environments.leases import postgres_mode
    base_data = load_config_data(base_config)
    configured_max_technical_retries, retry_backoff = _retry_policy(
        base_data.get("suite", {})
    )
    max_technical_retries = (
        configured_max_technical_retries
        if args.max_technical_retries is None
        else args.max_technical_retries
    )
    if not isinstance(max_technical_retries, int) or not 0 <= max_technical_retries <= 10:
        parser.error("--max-technical-retries must be an integer between 0 and 10")
    pg_mode = postgres_mode(base_data.get("execution", {}))
    if benchmark_kind != "mtar" and pg_mode == "leased":
        parser.error("PostgreSQL leases are currently supported by the MTAR adapter")
    first_task_id = task_ids[0]
    loaded_base_config = load_tree_search_config(
        base_config,
        benchmark_override={
            "task_id": first_task_id,
            "dataset_root": str(dataset_roots[first_task_id]),
        },
    )
    resources_by_task, resource_capacities = _resource_plan(
        catalog, task_ids, base_data, base_path=base_config,
    )
    worker_limit = _controller_worker_limit(args.max_concurrent_tasks, base_data.get("suite", {}))
    if args.plan_only or args.prepare_runtimes_only:
        pending_task_ids = _pending_technical_tasks(
            suite_dir, task_ids, max_technical_retries, benchmark_kind,
        )
        if args.prepare_runtimes_only:
            for task_id in pending_task_ids:
                benchmark = dict(loaded_base_config.benchmark)
                benchmark.update(
                    task_id=task_id, dataset_root=str(dataset_roots[task_id])
                )
                prepare_openhands_runtime_for_config(
                    replace(loaded_base_config, benchmark=benchmark)
                )
        unique_roots = sorted({str(root) for root in dataset_roots.values()})
        print(json.dumps({
            "attack": method,
            "task_ids": task_ids, "excluded_tasks": excluded_tasks,
            "dataset_root": unique_roots[0] if len(unique_roots) == 1 else None,
            "dataset_roots": {
                task_id: str(root) for task_id, root in dataset_roots.items()
            },
            "dataset_selection": catalog.provenance(),
            "max_concurrent_tasks": args.max_concurrent_tasks,
            "resource_capacities": resource_capacities, "task_resources": resources_by_task,
            "max_technical_retries": max_technical_retries,
            "pending_task_ids": pending_task_ids,
        }, indent=2))
        return 0
    campaign_lock = acquire_campaign_lock(suite_dir)
    run_cleanup_preflight(args, suite_dir)
    fingerprint_path = suite_dir / "attack_inputs.json"
    # Existing legacy DART suites remain resumable. New suites lock their method
    # and semantic inputs before any model or environment is started.
    if fingerprint_path.exists() or not (suite_dir / "execution_provenance.json").exists():
        fingerprint_source = (
            dataset_roots if catalog.mtar_collection is not None else catalog.root
        )
        fingerprint = input_fingerprint(base_data, task_ids, fingerprint_source)
        if fingerprint_path.exists() and json.loads(fingerprint_path.read_text())["fingerprint"] != fingerprint:
            raise ValueError("suite inputs changed; use a new suite directory")
        _write_json(fingerprint_path, {"method": method, "fingerprint": fingerprint})
    for task_id, reason in excluded_tasks.items():
        print(f"EXCLUDED {task_id}: {reason}", flush=True)
    configs_dir = suite_dir / "configs"
    tasks_dir = suite_dir / "tasks"
    configs_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir.mkdir(parents=True, exist_ok=True)
    archived_incomplete: dict[str, str] = {}
    if args.archive_incomplete_task_dirs:
        for task_id in task_ids:
            archived = _archive_incomplete_task_dir(
                tasks_dir / _task_slug(task_id), suite_dir
            )
            if archived is not None:
                archived_incomplete[task_id] = str(archived)
                print(f"ARCHIVED_INCOMPLETE {task_id}: {archived}", flush=True)
    provenance = {
        "base_config": str(base_config),
        "benchmark_kind": benchmark_kind,
        "task_ids": task_ids,
        "requested_task_ids": requested_task_ids,
        "excluded_tasks": excluded_tasks,
        "dataset_selection": catalog.provenance(),
        "dataset_roots": {
            task_id: str(root) for task_id, root in dataset_roots.items()
        },
        "dataset_root_override": (
            str(args.dataset_root.resolve()) if args.dataset_root is not None else None
        ),
        "controller_cuda_visible_devices_override": (
            args.controller_cuda_visible_devices
        ),
        "max_concurrent_tasks": args.max_concurrent_tasks,
        "resource_capacities": resource_capacities,
        "max_technical_retries": max_technical_retries,
        "configured_max_technical_retries": configured_max_technical_retries,
        "technical_retry_backoff_seconds": retry_backoff,
        "technical_retry_scheduling": "per_task_queue",
        "workers_per_controller": base_data.get("suite", {}).get("workers_per_controller"),
        "controller_pool_manifest": os.environ.get("SEAD_CONTROLLER_POOL_MANIFEST"),
        "controller_batch_size_requested": controller_batch_size,
        "controller_batch_wait_ms": args.controller_batch_wait_ms,
        "archived_incomplete_task_dirs": archived_incomplete,
        "task_resources": {
            task_id: list(resources) for task_id, resources in resources_by_task.items()
        },
    }
    previous_provenance = suite_dir / "execution_provenance.json"
    if previous_provenance.exists():
        history = suite_dir / "provenance_history" / f"{time.time_ns()}.json"
        history.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(previous_provenance, history)
    previous_usage = suite_dir / "controller_shared_usage.json"
    if previous_usage.exists():
        history = suite_dir / "controller_usage_history" / f"{time.time_ns()}.json"
        history.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(previous_usage, history)
    _write_json(previous_provenance, provenance)
    child_environment = os.environ.copy()
    child_environment.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )

    snapshots: dict[str, Path] = {}
    resolved_runtime_images: dict[str, str] = {}
    pending_task_ids = _pending_technical_tasks(
        suite_dir, task_ids, max_technical_retries, benchmark_kind,
    )
    for task_id in task_ids:
        if task_id not in pending_task_ids:
            print(f"SKIP {task_id}: completed or technical retry budget exhausted", flush=True)
    for task_id in pending_task_ids:
        snapshot = configs_dir / f"{_task_slug(task_id)}.yml"
        _snapshot_config(
            base_config,
            snapshot,
            task_id,
            controller_cuda_visible_devices=args.controller_cuda_visible_devices,
            dataset_root=dataset_roots[task_id],
        )
        snapshot_config = load_tree_search_config(snapshot)
        resolved_runtime_image = prepare_openhands_runtime_for_config(
            snapshot_config
        )
        resolved_runtime_images[task_id] = resolved_runtime_image
        snapshot_raw = load_config_data(snapshot)
        snapshot_raw.setdefault("execution", {})[
            "resolved_openhands_base_image"
        ] = resolved_runtime_image
        snapshot.write_text(
            yaml.safe_dump(snapshot_raw, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        snapshots[task_id] = snapshot

        check_path = suite_dir / f"check-{_task_slug(task_id)}.json"
        checked = subprocess.run(
            [
                sys.executable,
                "-m",
                "sead.attacks.dart.cli",
                "--config",
                str(snapshot),
                "--check-only",
            ],
            text=True,
            capture_output=True,
            env=child_environment,
            check=False,
        )
        if checked.returncode != 0:
            (suite_dir / f"check-{_task_slug(task_id)}.stderr.log").write_text(
                checked.stderr, encoding="utf-8"
            )
            raise RuntimeError(f"check-only failed for {task_id}")
        checked_stdout = checked.stdout
        check_path.write_text(checked_stdout, encoding="utf-8")
        print(f"CHECKED {task_id}", flush=True)

    provenance["resolved_openhands_base_images"] = resolved_runtime_images
    _write_json(suite_dir / "execution_provenance.json", provenance)

    if pending_task_ids:
        # Start capacity-limited service work early so its startup/reset latency
        # overlaps filesystem/terminal work rather than becoming a serial tail.
        pending_task_ids.sort(key=lambda task: not any(
            resource in resource_capacities for resource in resources_by_task[task]
        ))
        first_config = load_tree_search_config(snapshots[pending_task_ids[0]])
        effective_batch_size = (
            controller_batch_size
            if first_config.controller["backend"] == "sglang_subprocess"
            else None
        )
        provenance["controller_batch_size_effective"] = effective_batch_size
        provenance["controller_backend"] = first_config.controller["backend"]
        provenance["controller_worker_shared_across_tasks"] = True
        provenance["response_processing"] = "extract-repair-parse-validate"
        _write_json(suite_dir / "execution_provenance.json", provenance)
        prepare_controller_credentials(first_config)
        controller_environment = os.environ.copy()
        controller_environment.update(child_environment)
        if first_config.controller.get("cuda_visible_devices") is not None:
            controller_environment["CUDA_VISIBLE_DEVICES"] = str(
                first_config.controller["cuda_visible_devices"]
            )
        shared_controller = create_controller_backend(
            first_config,
            stderr_path=suite_dir / "controller.stderr.log",
            environment=controller_environment,
            max_batch_size=effective_batch_size or 1,
            batch_wait_ms=args.controller_batch_wait_ms,
        )

        def run_task(task_id: str) -> tuple[str, int]:
            slug = _task_slug(task_id)
            task_dir = tasks_dir / slug
            stdout_path = suite_dir / f"run-{slug}.stdout.log"
            stderr_path = suite_dir / f"run-{slug}.stderr.log"
            attempt_number = _attempt_count(suite_dir, task_id) + 1
            if attempt_number > 1:
                recovered = _post_task_service_reset(snapshots[task_id], task_id, task_dir,
                    environment=child_environment, benchmark_kind=benchmark_kind)
                if recovered.get("succeeded") is not True:
                    raise RuntimeError("cannot start another attempt before recovering its environments")
                archived = _archive_failed_attempt(suite_dir, task_id)
                print(f"ARCHIVED_TECHNICAL {task_id}: {archived}", flush=True)
            record = {"attempt": attempt_number, "state": "running",
                      "started_at": datetime.now(UTC).isoformat()}
            _write_json(_attempt_path(suite_dir, task_id), record)
            print(f"START {task_id}: attempt={attempt_number}/{1 + max_technical_retries}", flush=True)
            return_code = 1
            errors: list[str] = []
            try:
                config = load_tree_search_config(snapshots[task_id])
                client = shared_controller.create_client(task_id) if shared_controller else None
                outcome = run_from_config(
                    config,
                    run_id=slug,
                    output_dir=task_dir,
                    controller_runtime=client,
                )
                stdout_path.write_text(
                    json.dumps(outcome.summary, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                return_code = 0 if outcome.summary["final_replay_success"] else 2
            except Exception:
                errors.append(traceback.format_exc())
            finally:
                reset_audit = _post_task_service_reset(
                    snapshots[task_id],
                    task_id,
                    task_dir,
                    environment=child_environment,
                    benchmark_kind=benchmark_kind,
                )
                if reset_audit.get("succeeded") is not True:
                    errors.append(
                        "Post-task service reset failed: "
                        + str(reset_audit.get("error") or "unknown error")
                    )
                    return_code = 1
            stderr_path.write_text(
                "\n".join(error.rstrip() for error in errors) + ("\n" if errors else ""),
                encoding="utf-8",
            )
            record.update(state="finished", return_code=return_code, errors=errors,
                          finished_at=datetime.now(UTC).isoformat())
            _write_json(_attempt_path(suite_dir, task_id), record)
            return task_id, return_code

        def retry_delay(task_id, _result):
            if not _pending_technical_tasks(suite_dir, [task_id], max_technical_retries, benchmark_kind):
                return None
            attempts = _attempt_count(suite_dir, task_id)
            delay = min(300, retry_backoff * 2 ** (attempts - 1))
            print(f"TECHNICAL_RETRY_QUEUED {task_id}: backoff_seconds={delay}", flush=True)
            return delay

        initial_delays = {
            task: min(300, retry_backoff * 2 ** (attempts - 1))
            for task in pending_task_ids
            if (attempts := _attempt_count(suite_dir, task)) > 0
        }
        try:
            worker_count = min(args.max_concurrent_tasks, len(pending_task_ids))
            with CampaignExecutor(
                worker_count, thread_name_prefix="dart-task"
            ) as executor:
                for task_id, return_code in _resource_aware_results(
                    executor, pending_task_ids, resources_by_task, run_task,
                    max_workers=worker_count,
                    resource_capacities=resource_capacities,
                    worker_limit=worker_limit,
                    retry_delay=retry_delay,
                    initial_delays=initial_delays,
                ):
                    print(f"DONE {task_id}: exit={return_code}", flush=True)
                    summary = _aggregate(suite_dir, task_ids, excluded_tasks, benchmark_kind, method)
                    summary["dataset_selection"] = catalog.provenance()
                    row = next(row for row in summary["tasks"] if row["task_id"] == task_id)
                    if row["status"] != "completed":
                        exhausted = _attempt_count(suite_dir, task_id) >= 1 + max_technical_retries
                        print(f"TECHNICAL_FAILURE {task_id}: retry_exhausted={exhausted}", flush=True)
                    _write_json(suite_dir / "suite_summary.json", summary)
        finally:
            if shared_controller:
                shared_controller.close()
            records = list(shared_controller.shared_usage_records) if shared_controller else []
            batch_sizes = Counter(
                str(record.get("scheduler_batch_size", 1)) for record in records
            )
            model_batch_sizes = Counter(
                str(record.get("batch_size", 1)) for record in records
            )
            batch_ids = {
                int(record["scheduler_batch_id"])
                for record in records
                if record.get("scheduler_batch_id") is not None
            }
            token_split_batch_ids = {
                int(record["scheduler_batch_id"])
                for record in records
                if record.get("scheduler_batch_id") is not None
                and record.get("batch_split_by_token_budget")
            }
            _write_json(
                suite_dir / "controller_shared_usage.json",
                {
                    "schema_version": "dart-shared-controller-usage-v3",
                    "request_count": len(records),
                    "scheduler_batch_count": (
                        len(batch_ids) if batch_ids else len(records)
                    ),
                    "scheduler_batch_size_distribution": dict(
                        sorted(batch_sizes.items())
                    ),
                    "model_batch_size_distribution": dict(
                        sorted(model_batch_sizes.items())
                    ),
                    "token_budget_split_batch_count": len(token_split_batch_ids),
                    "records": records,
                },
            )

    summary = _aggregate(
        suite_dir,
        task_ids,
        excluded_tasks,
        benchmark_kind,
        method,
    )
    summary["dataset_selection"] = catalog.provenance()
    summary["configured_max_technical_retries"] = configured_max_technical_retries
    summary["max_technical_retries"] = max_technical_retries
    summary["technical_retry_exhausted_task_ids"] = [
        row["task_id"] for row in summary["tasks"]
        if row["status"] != "completed" and row["attempt_count"] >= 1 + max_technical_retries
    ]
    _write_json(suite_dir / "suite_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    campaign_lock.close()
    return 0 if summary["technical_incomplete"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
