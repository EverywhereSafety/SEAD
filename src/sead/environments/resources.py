"""Capacity-one TAC resources shared by schedulers and replay workers.

This is not a replica pool. A service still has one mutable deployment. Browser
ports follow the same exclusion boundary, so unrelated TAC services can run in
parallel without introducing a general Playwright lease protocol.
"""

from __future__ import annotations

import fcntl
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

TAC_WEB_SERVICES = frozenset({"gitlab", "owncloud", "plane"})
TAC_BROWSER_PORTS = {"gitlab": 19092, "owncloud": 29092, "plane": 39092}


def web_resources(
    dependencies: Sequence[str], *, instance_key: str | None = None,
    service_instances: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    if instance_key is not None and "reddit" in dependencies:
        return (f"web:reddit:{instance_key}",)
    services = set(dependencies) & TAC_WEB_SERVICES
    if service_instances:
        return tuple(
            f"web:{service}:{service_instances[service]}"
            for service in sorted(services)
            if service in service_instances
        )
    if services:
        return tuple(f"web:{service}" for service in sorted(services))
    if "mcp-playwright" in dependencies:
        # Undeployed/non-TAC browser tasks retain conservative global exclusion.
        return tuple(f"web:{service}" for service in sorted(TAC_WEB_SERVICES))
    return ()


def playwright_port(dependencies: Sequence[str]) -> int:
    services = sorted(set(dependencies) & TAC_WEB_SERVICES)
    # Multi-service tasks hold every relevant lock, including this port's owner.
    return TAC_BROWSER_PORTS[services[0]] if services else 9092


@dataclass
class WebResourceLock:
    handles: list[IO[str]]

    def close(self) -> None:
        while self.handles:
            handle = self.handles.pop()
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    def __enter__(self) -> WebResourceLock:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def acquire_web_lock(
    dependencies: Sequence[str], *, lock_directory: Path | None = None,
    instance_key: str | None = None,
    service_instances: Mapping[str, str] | None = None,
) -> WebResourceLock | None:
    """Hold service locks across reset, execution, evaluation, and cleanup.

    The legacy lock is shared for TAC and exclusive for unknown browser tasks.
    An older worker holding the original exclusive lock still excludes *all*
    new workers. Sorted service acquisition prevents multi-service deadlocks.
    Kernel-owned locks are released when a worker dies; the next owner must
    reset before using the service. Never use this as a cross-host lease.
    """
    if not web_resources(
        dependencies, instance_key=instance_key, service_instances=service_instances
    ):
        return None
    directory = lock_directory or Path(tempfile.gettempdir())
    services = sorted(set(dependencies) & TAC_WEB_SERVICES)
    if service_instances:
        services = [
            f"{service}-{service_instances[service]}"
            for service in services if service in service_instances
        ]
    if instance_key is not None and "reddit" in dependencies:
        if not instance_key or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789.-_" for character in instance_key.lower()):
            raise ValueError("invalid forum instance lock key")
        services = [f"reddit-{instance_key}"]
    result = WebResourceLock([])
    paths = [
        (
            directory / "sead-mtar-web.lock",
            fcntl.LOCK_SH if services else fcntl.LOCK_EX,
        )
    ]
    paths.extend(
        (directory / f"sead-mtar-web-{service}.lock", fcntl.LOCK_EX)
        for service in services
    )
    try:
        for path, mode in paths:
            handle = path.open("a+")
            result.handles.append(handle)
            fcntl.flock(handle.fileno(), mode)
        return result
    except BaseException:
        result.close()
        raise


def release_web_lock(handle: WebResourceLock | None) -> None:
    if handle is not None:
        handle.close()
