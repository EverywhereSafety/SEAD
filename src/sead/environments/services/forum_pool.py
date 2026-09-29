"""Task-dedicated SafeArena forum replicas for concurrent MTAR replay.

Each selected task owns one immutable-image container and one host port. A
replay recreates its container while holding that task's host-wide lock; other
tasks keep their own website state. This provider is local-host only.
"""

from __future__ import annotations

import re
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any


def forum_instance(
    execution: Mapping[str, Any], task_id: str, *, required: bool = False
) -> dict[str, Any] | None:
    pool = execution.get("forum_pool")
    if pool is None:
        if required:
            raise ValueError(f"no SafeArena forum pool for {task_id}")
        return None
    if not isinstance(pool, dict) or not isinstance(pool.get("instances"), dict):
        raise ValueError("execution.forum_pool must contain instances")
    raw = pool["instances"].get(task_id)
    if task_id not in pool["instances"] and not required:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"no SafeArena forum instance for {task_id}")
    if set(raw) != {"port", "browser_port"}:
        raise ValueError(f"invalid SafeArena forum instance for {task_id}")
    for field in ("port", "browser_port"):
        value = raw[field]
        if isinstance(value, bool) or not isinstance(value, int) or not 1024 <= value <= 65535:
            raise ValueError(f"invalid SafeArena forum {field} for {task_id}")
    image = str(pool.get("image") or "")
    if not image.startswith("sha256:") or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("SafeArena forum image must be pinned by image ID")
    ports = [item.get("port") for item in pool["instances"].values() if isinstance(item, dict)]
    browsers = [item.get("browser_port") for item in pool["instances"].values() if isinstance(item, dict)]
    if len(ports + browsers) != len(set(ports + browsers)):
        raise ValueError("SafeArena forum instance ports must be unique")
    return {"task_id": task_id, "port": raw["port"], "browser_port": raw["browser_port"], "image": image}


def reset_forum_instance(spec: Mapping[str, Any], *, timeout_seconds: float = 240) -> str:
    """Recreate one forum from the published image and wait for HTTP readiness."""

    task_id = str(spec["task_id"])
    name = "sead-forum-" + task_id.replace(".", "-")
    remove = subprocess.run(
        ["docker", "rm", "-f", "-v", name], text=True, capture_output=True
    )
    if remove.returncode and "No such container" not in remove.stderr:
        raise RuntimeError(f"cannot remove {name}: {remove.stderr.strip()}")
    launch = subprocess.run(
        [
            "docker", "run", "-d", "--name", name,
            "--label", "org.sead.service=safearena-forum",
            "--label", f"org.sead.task-id={task_id}",
            "-p", f"127.0.0.1:{spec['port']}:80", str(spec["image"]),
        ],
        text=True, capture_output=True,
    )
    if launch.returncode:
        raise RuntimeError(f"cannot start {name}: {launch.stderr.strip()}")
    url = f"http://127.0.0.1:{spec['port']}/forums/all"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status < 500:
                    return url
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                return url
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(2)
    raise RuntimeError(f"SafeArena forum {name} did not become ready at {url}")


def stop_forum_instance(spec: Mapping[str, Any]) -> None:
    """Release a task's idle website after its final reset, under its web lock."""
    name = "sead-forum-" + str(spec["task_id"]).replace(".", "-")
    result = subprocess.run(
        ["docker", "stop", "--time", "10", name], capture_output=True, text=True,
        timeout=30,
    )
    if result.returncode:
        raise RuntimeError(f"cannot stop {name}: {result.stderr.strip()}")
