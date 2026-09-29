"""One-clean-worker-per-path MTAR backend for DART tree search."""

from __future__ import annotations

from sead.environments.workers import worker_path

import json
import os
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ...attacks.dart.models import BranchReplayRequest, BranchReplayResult
from ...infrastructure.docker_cleanup import DockerReplayAuditor
from ...attacks.dart.history import project_controller_transcript
from .replay_protocol import (
    MTARReplayRequest,
    atomic_write_json,
    read_result,
)


class MTAROpenHandsWorkerBackend:
    benchmark_kind = "mtar"

    def __init__(
        self,
        *,
        task_id: str,
        dataset_root: Path | str,
        openhands_root: Path | str,
        worker_python: Path | str,
        target: Mapping[str, Any],
        execution: Mapping[str, Any],
        output_dir: Path | str,
        worker_script: Path | str | None = None,
        auditor: DockerReplayAuditor | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.task_id = str(task_id)
        self.dataset_root = Path(dataset_root).resolve()
        self.openhands_root = Path(openhands_root).resolve()
        # Preserve the virtualenv launcher. Path.resolve() follows its symlink to
        # the system interpreter and silently drops the environment's packages.
        self.worker_python = Path(worker_python).absolute()
        self.target = dict(target)
        self.execution = dict(execution)
        self.output_dir = Path(output_dir).resolve()
        self.worker_script = (
            Path(worker_script).resolve()
            if worker_script
            else worker_path("mtar")
        )
        self.auditor = auditor or DockerReplayAuditor()
        self.runner = runner

    def _worker_dir(self, request: BranchReplayRequest) -> Path:
        if request.final_replay:
            base = (
                self.output_dir
                / "terminal_replays"
                / request.node_id.removeprefix("terminal-")
                / "worker"
            )
        else:
            base = self.output_dir / "nodes" / request.node_id / "worker"
        candidate = base
        attempt = 1
        while candidate.exists():
            attempt += 1
            candidate = base.with_name(f"worker_retry_{attempt}")
        return candidate

    def execute_path(self, request: BranchReplayRequest) -> BranchReplayResult:
        replay_started = time.perf_counter()
        replay_id = f"{self.benchmark_kind}-{uuid.uuid4().hex}"
        environment_id = f"dart-{self.benchmark_kind}-{uuid.uuid4().hex}"
        worker_dir = self._worker_dir(request)
        worker_dir.mkdir(parents=True)
        request_path = worker_dir / "request.json"
        result_path = worker_dir / "result.json"
        stdout_path = worker_dir / "stdout.log"
        stderr_path = worker_dir / "stderr.log"
        worker_request = MTARReplayRequest(
            benchmark_kind=self.benchmark_kind,
            replay_id=replay_id,
            environment_id=environment_id,
            run_id=request.run_id,
            node_id=request.node_id,
            parent_node_id=request.parent_node_id,
            task_id=self.task_id,
            dataset_root=str(self.dataset_root),
            openhands_root=str(self.openhands_root),
            worker_dir=str(worker_dir),
            parent_replay_turns=request.parent_replay_turns,
            new_instruction=request.new_instruction,
            target=self.target,
            execution=self.execution,
            final_replay=request.final_replay,
        )
        atomic_write_json(request_path, worker_request.to_dict())
        artifacts = {
            "request": str(request_path),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "result": str(result_path),
        }
        completed: subprocess.CompletedProcess[str] | None = None
        launch_error: str | None = None
        worker_started = time.perf_counter()
        try:
            from sead.environments.session import worker_timeout
            from sead.infrastructure.replay_worker import run_replay_worker, ReplayCleanupError
            if self.runner is subprocess.run:
                completed = run_replay_worker(
                    [str(self.worker_python), str(self.worker_script), "--request", str(request_path)],
                    environment_id=environment_id, worker_dir=worker_dir,
                    cwd=Path(__file__).resolve().parents[4],
                    timeout=worker_timeout(self.execution), env=dict(os.environ))
            else:
                with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
                    completed = self.runner(
                        [
                            str(self.worker_python),
                            str(self.worker_script),
                            "--request",
                            str(request_path),
                        ],
                        text=True,
                        stdout=stdout,
                        stderr=stderr,
                        check=False,
                        timeout=int(self.execution.get("sample_timeout_seconds", 3600)),
                    )
        except (OSError, subprocess.SubprocessError, ReplayCleanupError) as exc:
            launch_error = f"{type(exc).__name__}: {exc}"
        worker_subprocess_seconds = time.perf_counter() - worker_started
        cleanup_started = time.perf_counter()
        cleanup = self.auditor.cleanup(environment_id)
        parent_cleanup_seconds = time.perf_counter() - cleanup_started
        if launch_error or completed is None or not result_path.is_file():
            error = launch_error or (
                f"{self.benchmark_kind.upper()} worker exited {completed.returncode}"
                if completed is not None
                else "MTAR worker did not start"
            )
            return BranchReplayResult(
                replay_id=replay_id,
                environment_id=environment_id,
                instructions=request.instructions,
                target_transcript=(),
                tool_calls=(),
                controller_transcript=project_controller_transcript(
                    request.parent_replay_turns
                ),
                cleanup_succeeded=bool(cleanup.get("succeeded")),
                cleanup_details=cleanup,
                technical_error=error,
                infrastructure_status="WORKER_LAUNCH_FAILED",
                artifact_locations=artifacts,
                replay_turns=request.parent_replay_turns,
                phase_timings={
                    "worker_subprocess": round(worker_subprocess_seconds, 6),
                    "parent_cleanup": round(parent_cleanup_seconds, 6),
                    "replay_total": round(time.perf_counter() - replay_started, 6),
                },
            )
        try:
            result = read_result(result_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return BranchReplayResult(
                replay_id=replay_id,
                environment_id=environment_id,
                instructions=request.instructions,
                target_transcript=(),
                tool_calls=(),
                controller_transcript=project_controller_transcript(
                    request.parent_replay_turns
                ),
                cleanup_succeeded=bool(cleanup.get("succeeded")),
                cleanup_details=cleanup,
                technical_error=f"invalid {self.benchmark_kind.upper()} worker result: {exc}",
                infrastructure_status="INVALID_WORKER_RESULT",
                artifact_locations=artifacts,
                replay_turns=request.parent_replay_turns,
                phase_timings={
                    "worker_subprocess": round(worker_subprocess_seconds, 6),
                    "parent_cleanup": round(parent_cleanup_seconds, 6),
                    "replay_total": round(time.perf_counter() - replay_started, 6),
                },
            )
        cleanup_succeeded = result.cleanup_succeeded and bool(cleanup.get("succeeded"))
        cleanup_details = {
            "worker": dict(result.cleanup_details),
            "parent_audit": cleanup,
        }
        evaluation = dict(result.evaluation)
        phase_timings = {
            **dict(result.phase_timings),
            "worker_subprocess": round(worker_subprocess_seconds, 6),
            "parent_cleanup": round(parent_cleanup_seconds, 6),
            "replay_total": round(time.perf_counter() - replay_started, 6),
        }
        progress = float(evaluation.get("progress") or 0.0) if evaluation else 0.0
        events = tuple(
            {
                "turn": index,
                "external_instruction": turn.user_instruction,
                "target_text": "\n".join(
                    message.content
                    for message in turn.assistant_messages
                    if message.content
                ),
                "target_tool_calls": [
                    call.to_dict()
                    for message in turn.assistant_messages
                    for call in message.tool_calls
                ],
            }
            for index, turn in enumerate(result.replay_turns, 1)
        )
        return BranchReplayResult(
            replay_id=result.replay_id,
            environment_id=result.environment_id,
            instructions=tuple(turn.user_instruction for turn in result.replay_turns),
            target_transcript=tuple(
                turn.transcript_row(index)
                for index, turn in enumerate(result.replay_turns, 1)
            ),
            tool_calls=tuple(
                call.to_dict()
                for turn in result.replay_turns
                for message in turn.assistant_messages
                for call in message.tool_calls
            ),
            controller_transcript=project_controller_transcript(
                result.replay_turns
            ),
            official_main_result={
                "not_applicable": True,
                "benign_task_score": 1.0,
            },
            official_side_result=evaluation,
            attack_progress=max(0.0, min(1.0, progress)),
            benign_task_score=1.0,
            cleanup_succeeded=cleanup_succeeded,
            cleanup_details=cleanup_details,
            technical_error=result.technical_error
            if cleanup_succeeded
            else (result.technical_error or "environment cleanup failed"),
            infrastructure_status=result.infrastructure_status,
            artifact_locations={**artifacts, **dict(result.artifacts)},
            events=events,
            replay_turns=result.replay_turns,
            restore_audit=result.restore_audit,
            phase_timings=phase_timings,
        )
