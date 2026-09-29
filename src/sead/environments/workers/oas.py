#!/usr/bin/env python3
"""Execute one OAS instruction in a fresh OpenHands sandbox."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from sead.environments.workers import mtar as shared_worker

from sead.attacks.dart.config import validate_target_temperature
from sead.benchmarks.mtar.replay_protocol import (
    MTARReplayRequest,
    MTARReplayWorkerResult,
    ReplayPhaseTimer,
    atomic_write_json,
    read_request,
)
from sead.benchmarks.mtar.runtime_profiles import RuntimeProfile
from sead.benchmarks.mtar.environment import (
    DEFAULT_SERVICE_DEPLOYMENTS,
    TOOL_SCHEMAS,
    service_instruction_context,
    unavailable_runtime_services,
)
from sead.benchmarks.mtar.runtime_profiles import (
    UnsupportedRuntimeProfile,
    load_runtime_registry,
    openhands_base_image_alias,
    resolve_service_image,
)
from sead.benchmarks.oas.dataset import (
    DEFAULT_CANDIDATE_INDEX,
    DEFAULT_SELECTION,
    load_task,
    materialize_workspace,
)
from sead.benchmarks.oas.evaluator import (
    EVALUATION_MARKER,
    EVALUATOR_RUNTIME_ROOT,
    build_evaluator_entrypoint,
    evaluator_trajectory_payload,
    parse_evaluation,
)
from sead.defenses import (
    OnlineToolDefenseGate,
    build_online_tool_defender,
    configured_openhands_tool_defense,
    validate_online_defense_config,
)
from sead.environments.resources import playwright_port
from sead.environments.services.tac import (
    acquire_web_lock,
    release_web_lock,
    reset_services as reset_tac_services,
)


PROJECT = Path(__file__).resolve().parents[4]


def _selection_path(request: MTARReplayRequest) -> Path:
    return Path(str(request.execution.get("oas_selection_path") or DEFAULT_SELECTION))


def _service_deployments_path(request: MTARReplayRequest) -> Path:
    return Path(
        str(
            request.execution.get("service_deployments_path")
            or DEFAULT_SERVICE_DEPLOYMENTS
        )
    )


def _candidate_index_path(request: MTARReplayRequest) -> Path:
    return Path(
        str(
            request.execution.get("oas_candidate_index_path") or DEFAULT_CANDIDATE_INDEX
        )
    )


def _runtime_dependencies(dependencies: tuple[str, ...]) -> tuple[str, ...]:
    if not dependencies:
        return ()
    return tuple((*dependencies, "mcp-playwright"))


def _oas_profile(dependencies: tuple[str, ...]) -> RuntimeProfile:
    profile_id = "oas-web" if dependencies else "oas-base"
    profile = load_runtime_registry().get(profile_id)
    if profile is None:
        raise UnsupportedRuntimeProfile(profile_id, "profile is absent from registry")
    return profile.require_available()


def _validate_request(request: MTARReplayRequest) -> dict[str, Any]:
    task = load_task(
        request.dataset_root,
        request.task_id,
        selection_path=_selection_path(request),
        candidate_index=_candidate_index_path(request),
    )
    profile = _oas_profile(task.dependencies)
    pre_resolved_image = request.execution.get("resolved_openhands_base_image")
    if pre_resolved_image is None:
        base_image = shared_worker._prepare_openhands_base_image(profile)
    else:
        base_image = str(pre_resolved_image)
        if base_image != openhands_base_image_alias(profile):
            raise ValueError(
                "execution.resolved_openhands_base_image does not match runtime profile"
            )
    runtime_dependencies = _runtime_dependencies(task.dependencies)
    from sead.environments.registry import environment_config, EnvironmentRegistry
    leased_config = environment_config(request.execution)
    environment_plan = EnvironmentRegistry(leased_config["registry"]).plan(
        runtime_dependencies, request.task_id
    ) if leased_config else None
    unavailable = unavailable_runtime_services(
        runtime_dependencies,
        tool_family="playwright" if task.dependencies else "terminal",
    ) if not leased_config else ()
    if unavailable:
        raise UnsupportedRuntimeProfile(
            ",".join(unavailable), "required OAS service is not deployed"
        )
    for service_id in (task.dependencies if not leased_config else ()):
        resolve_service_image(service_id)
    if task.dependencies and not leased_config:
        resolve_service_image("playwright-mcp")
        service_instruction_context(
            task.dependencies,
            deployments_path=_service_deployments_path(request),
        )
    for name, path in {
        "OpenHands": Path(request.openhands_root),
        "worker directory": Path(request.worker_dir),
    }.items():
        if not path.exists():
            raise ValueError(f"{name} does not exist: {path}")
    if not (Path(request.openhands_root) / "openhands/core/main.py").is_file():
        raise ValueError("invalid OpenHands checkout")
    if str(request.target.get("agent") or "CodeActAgent") != "CodeActAgent":
        raise ValueError("OAS target.agent must be CodeActAgent")
    validate_target_temperature(request.target, label="OAS target")
    for mapping, names in (
        (request.target, ("max_output_tokens", "max_tool_calls_per_instruction")),
        (request.execution, ("max_steps", "sample_timeout_seconds")),
    ):
        for name in names:
            if int(mapping.get(name, 0)) < 1:
                raise ValueError(f"{name} must be positive")
    provider = str(request.target.get("provider") or "")
    variable = str(request.target.get("api_key_env") or "")
    if not request.final_replay and (not variable or not os.environ.get(variable)):
        raise ValueError(f"missing Target credential environment variable: {variable}")
    if provider == "gemini":
        if variable != "GEMINI_API_KEY":
            raise ValueError("Gemini target.api_key_env must be GEMINI_API_KEY")
        if not str(request.target.get("model") or "").startswith("gemini/"):
            raise ValueError("Gemini target.model must start with gemini/")
    elif provider == "azure_openai_v1":
        endpoint = str(request.target.get("endpoint") or "")
        if not endpoint.startswith("https://") or not endpoint.rstrip("/").endswith(
            "/openai/v1"
        ):
            raise ValueError("invalid Azure OpenAI v1 endpoint")
    else:
        raise ValueError("unsupported OAS Target provider")
    actual_litellm = importlib.metadata.version("litellm")
    expected_litellm = str(request.execution.get("litellm_version") or "")
    if expected_litellm and actual_litellm != expected_litellm:
        raise ValueError(
            f"OAS Target requires litellm {expected_litellm}; found {actual_litellm}"
        )
    defense = validate_online_defense_config(
        request.execution.get("defense"),
        require_credentials=not request.final_replay,
    )
    return {
        "status": "validated",
        "task_id": task.task_id,
        "task_root": str(task.root),
        "dependencies": list(task.dependencies),
        "runtime_dependencies": list(runtime_dependencies),
        "runtime_profile": profile.profile_id,
        "openhands_base_image": base_image,
        "evaluator_image": str(profile.image),
        "selection_path": str(_selection_path(request)),
        "candidate_index_path": str(_candidate_index_path(request)),
        "service_deployments_path": str(_service_deployments_path(request)),
        "litellm_version": actual_litellm,
        "agent": "CodeActAgent",
        "defense": defense,
        "environment_plan": environment_plan,
    }


def _evaluate(
    task: Any,
    trajectory: Path,
    worker_dir: Path,
    workspace: Path,
    *,
    evaluator_image: str,
    environment_binding: dict | None = None,
) -> dict[str, Any]:
    grader = worker_dir / "grader"
    grader.mkdir()
    entry = grader / "evaluate_oas.py"
    entry.write_text(build_evaluator_entrypoint(), encoding="utf-8")
    payload = evaluator_trajectory_payload(
        json.loads(trajectory.read_text(encoding="utf-8"))
    )
    (grader / "trajectory.txt").write_text(payload, encoding="utf-8")
    # Evaluate in a second, read-only-oracle container. Some OAS objectives
    # intentionally stop broad classes of processes and can kill OpenHands'
    # own action server. The host-mounted workspace survives that event and is
    # still the authoritative final state.
    network_arguments = (
        [
            "--network",
            "host",
            "--add-host",
            "the-agent-company.com:127.0.0.1",
            "-e",
            "SERVER_HOSTNAME=the-agent-company.com",
        ]
        if task.dependencies
        else ["--network", "none"]
    )
    if environment_binding:
        network_arguments = ["--network", environment_binding["network"],
                             "-e", f"SERVER_HOSTNAME={environment_binding['server_hostname']}"]
        for service, port in environment_binding["service_ports"].items():
            network_arguments += ["-e", f"{service.upper()}_PORT={port}"]
        for key, value in environment_binding["labels"].items():
            network_arguments += ["--label", f"{key}={value}"]
    completed = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            *network_arguments,
            "-v",
            f"{workspace}:/workspace:ro",
            "-v",
            f"{task.evaluator_path}:/grader/evaluator.py:ro",
            "-v",
            f"{EVALUATOR_RUNTIME_ROOT / 'common.py'}:/grader/common.py:ro",
            "-v",
            f"{EVALUATOR_RUNTIME_ROOT / 'scoring.py'}:/grader/scoring.py:ro",
            "-v",
            f"{grader / 'trajectory.txt'}:/grader/trajectory.txt:ro",
            "-v",
            f"{entry}:/grader/evaluate_oas.py:ro",
            evaluator_image,
            "bash",
            "-lc",
            "cd /grader && PYTHONPATH=/grader python evaluate_oas.py",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"OAS evaluator failed: {detail}")
    line = next(
        (
            row
            for row in reversed(completed.stdout.splitlines())
            if row.startswith(EVALUATION_MARKER)
        ),
        None,
    )
    if line is None:
        raise RuntimeError("OAS evaluator emitted no compatibility envelope")
    return parse_evaluation(json.loads(line[len(EVALUATION_MARKER) :])).to_dict()


def execute(request: MTARReplayRequest, *, phase_timer: ReplayPhaseTimer | None = None) -> MTARReplayWorkerResult:
    """Compatibility entry into the shared OpenHands replay engine."""
    from sead.environments.workers.openhands_engine import execute as run
    from sead.benchmarks.oas.worker_adapter import OASWorkerAdapter
    return run(request, OASWorkerAdapter(), phase_timer=phase_timer)


def reset_task_services(
    dataset_root: Path | str,
    task_id: str,
    *,
    selection_path: Path | str = DEFAULT_SELECTION,
    candidate_index: Path | str = DEFAULT_CANDIDATE_INDEX,
    hostname: str = "localhost",
) -> dict[str, Any]:
    """Restore an OAS task's shared TAC services without starting OpenHands."""

    task = load_task(
        dataset_root,
        task_id,
        selection_path=selection_path,
        candidate_index=candidate_index,
    )
    lock = acquire_web_lock(task.dependencies)
    try:
        operations = reset_tac_services(task.dependencies, hostname=hostname)
    finally:
        release_web_lock(lock)
    return {
        "schema_version": "oas-task-service-reset-v1",
        "task_id": task.task_id,
        "dependencies": list(task.dependencies),
        "operations": list(operations),
        "succeeded": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--reset-only", action="store_true")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--task-id")
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--candidate-index", type=Path, default=DEFAULT_CANDIDATE_INDEX)
    parser.add_argument("--server-hostname", default="localhost")
    args = parser.parse_args()
    if args.reset_only:
        if args.dataset_root is None or args.task_id is None:
            parser.error("--reset-only requires --dataset-root and --task-id")
        try:
            audit = reset_task_services(
                args.dataset_root,
                str(args.task_id),
                selection_path=args.selection,
                candidate_index=args.candidate_index,
                hostname=str(args.server_hostname),
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            audit = {
                "schema_version": "oas-task-service-reset-v1",
                "task_id": str(args.task_id),
                "succeeded": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(json.dumps(audit, indent=2))
            return 1
        print(json.dumps(audit, indent=2))
        return 0
    if args.request is None:
        parser.error("--request is required unless --reset-only is used")
    request = read_request(args.request)
    if args.check_only:
        print(json.dumps(_validate_request(request), indent=2))
        return 0
    result_path = Path(request.worker_dir) / "result.json"
    phase_timings_path = Path(request.worker_dir) / "phase_timings.json"
    phase_timer = ReplayPhaseTimer()
    try:
        result = execute(request, phase_timer=phase_timer)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        defense_audit_path = Path(request.worker_dir) / "tool_defense.json"
        result = MTARReplayWorkerResult(
            replay_id=request.replay_id,
            environment_id=request.environment_id,
            task_id=request.task_id,
            replay_turns=request.parent_replay_turns,
            restore_audit={
                "target_model_calls_during_restore": 0,
                "succeeded": False,
            },
            cleanup_succeeded=False,
            cleanup_details={"error": f"{type(exc).__name__}: {exc}"},
            technical_error=f"{type(exc).__name__}: {exc}",
            artifacts={
                **(
                    {"tool_defense": str(defense_audit_path)}
                    if defense_audit_path.is_file()
                    else {}
                ),
            },
        )
    phase_timings = phase_timer.snapshot()
    atomic_write_json(
        phase_timings_path,
        {
            "schema_version": "sead-replay-phase-timings-v1",
            "replay_id": request.replay_id,
            "phases_seconds": phase_timings,
            "restore": {
                name: result.restore_audit.get(name, 0)
                for name in (
                    "restore_action_count",
                    "restore_action_seconds",
                    "restore_record_only_count",
                    "bash_session_rebuild_count",
                )
            },
        },
    )
    result = dataclasses.replace(
        result,
        phase_timings=phase_timings,
        artifacts={
            **dict(result.artifacts),
            "phase_timings": str(phase_timings_path),
        },
    )
    atomic_write_json(result_path, result.to_dict())
    return 0 if result.technical_error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
