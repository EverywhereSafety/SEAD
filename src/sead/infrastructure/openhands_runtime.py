"""Prepare source-matched OpenHands images before dispatching task workers."""

from __future__ import annotations

import fcntl
import os
import subprocess
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
_BUILD_SCRIPT = """
import sys
sys.path.insert(0, sys.argv[1])
import docker
from openhands.runtime.builder.docker import DockerRuntimeBuilder
from openhands.runtime.utils.runtime_build import build_runtime_image
image = build_runtime_image(
    sys.argv[2], DockerRuntimeBuilder(docker.from_env()), enable_browser=False,
)
print('RUNTIME_READY ' + image, flush=True)
"""


@lru_cache(maxsize=None)
def prepare_runtime_image(base_image: str, openhands_root: Path, worker_python: Path) -> None:
    """Build once per profile in the configured checkout and worker interpreter.

    Buildx defaults to the user's home directory, which can be an unavailable
    network mount on cluster hosts. Keep its state on local project storage;
    preserve explicit Buildx overrides and Docker's credentials/context.
    """
    environment = os.environ.copy()
    environment.setdefault("BUILDX_CONFIG", str(PROJECT_ROOT / ".runtime/docker-buildx"))
    state_dir = Path(environment["BUILDX_CONFIG"])
    state_dir.mkdir(parents=True, exist_ok=True)
    # Multiple suites can prepare the same source image concurrently.
    with (state_dir / "sead-runtime.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        subprocess.run(
            [str(worker_python), "-c", _BUILD_SCRIPT, str(openhands_root), base_image],
            env=environment,
            check=True,
        )
