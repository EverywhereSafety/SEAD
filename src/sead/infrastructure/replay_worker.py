"""Run an isolated replay subprocess with parent-owned Docker cleanup."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
from pathlib import Path
from collections.abc import Mapping, Sequence

from sead.infrastructure.docker_cleanup import DockerReplayAuditor
from sead.infrastructure.campaign_cleanup import (
    check_worker_shutdown,
    wait_for_worker,
)


class ReplayCleanupError(RuntimeError):
    """Owned replay resources could not be cleaned and verified."""


def run_replay_worker(
    command: Sequence[str],
    *,
    environment_id: str,
    worker_dir: Path,
    cwd: Path,
    timeout: float,
    env: Mapping[str, str],
) -> subprocess.CompletedProcess:
    """Reap the worker before cleanup, including timeout and launch failures.

    Every caller must supply a unique environment id. Cleanup failures abort
    the caller rather than silently allowing subsequent jobs to leak resources.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", environment_id):
        raise ValueError("invalid Docker environment id")
    try:
        check_worker_shutdown()
        with (
            (worker_dir / "stdout.log").open("w") as stdout,
            (worker_dir / "stderr.log").open("w") as stderr,
        ):
            with subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            ) as process:
                try:
                    process_stat = Path(f"/proc/{process.pid}/stat")
                    if process_stat.exists():
                        (worker_dir / "process.json").write_text(
                            json.dumps(
                                {
                                    "pid": process.pid,
                                    "start_ticks": process_stat.read_text().rsplit(")", 1)[1].split()[19],
                                    "environment_id": environment_id,
                                }
                            )
                            + "\n"
                        )
                    returncode = wait_for_worker(process, timeout)
                finally:
                    # Also terminate descendants left behind by a worker that
                    # exited normally; they must not recreate cleaned resources.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            return subprocess.CompletedProcess(command, returncode)
    finally:
        from sead.environments.session import recover_environment
        recovery_error = None
        try:
            recover_environment(worker_dir)
        except Exception as exc:
            recovery_error = exc
        cleanup = DockerReplayAuditor().cleanup(environment_id)
        if recovery_error is not None:
            cleanup["succeeded"] = False
            cleanup["lease_error"] = str(recovery_error)
        (worker_dir / "cleanup.json").write_text(
            json.dumps(cleanup, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if not cleanup["succeeded"]:
            raise ReplayCleanupError(
                f"Replay Docker cleanup failed; see {worker_dir / 'cleanup.json'}"
            )
