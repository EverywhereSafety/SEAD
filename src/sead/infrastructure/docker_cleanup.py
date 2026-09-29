"""Label-scoped Docker cleanup used by isolated benchmark workers."""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar


def install_openhands_scoped_cleanup(environment_id: str) -> None:
    """Scope OpenHands close/shutdown to this isolated worker's one sandbox.

    OpenHands 0.54 lists and inspects every container when closing a runtime,
    and its shutdown listener stops every runtime with the global prefix.
    Each SEAD worker owns exactly one runtime; an exact lookup avoids
    both unrelated container stops and host-wide inspection timeouts. Remove
    the owned container as well, including containers already stopped.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", environment_id):
        raise ValueError("invalid Docker environment id")
    import docker
    from openhands.runtime.impl.docker import docker_runtime

    name = docker_runtime.CONTAINER_NAME_PREFIX + environment_id

    def stop_owned_container(prefix: str) -> None:
        if not prefix or not name.startswith(prefix):
            raise ValueError("OpenHands cleanup prefix does not match this worker")
        client = docker.from_env()
        try:
            try:
                container = client.containers.get(name)
                # ``v=True`` removes anonymous volumes declared by the image;
                # Docker otherwise preserves them after the container is gone.
                container.remove(force=True, v=True)
            except docker.errors.NotFound:
                pass  # Parent cleanup or an earlier close already removed it.
        finally:
            client.close()

    # The shutdown listener resolves this module global when it runs. Keep
    # the scoped implementation installed through worker process shutdown.
    docker_runtime.stop_all_containers = stop_owned_container


def cleanup_dangling_anonymous_volumes(
    created_since: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
    """Remove anonymous volumes created since an explicit timestamp.

    Docker's dangling filter ensures that no container currently references a
    candidate.  The timestamp keeps campaign cleanup from touching older data
    on a shared daemon, and Docker rechecks attachment when removing each
    batch to close the enumeration/removal race.
    """

    cutoff = datetime.fromisoformat(created_since.replace("Z", "+00:00"))
    if cutoff.tzinfo is None:
        raise ValueError("created_since must include a timezone")

    def run(command: Sequence[str], *, timeout: int = 120):
        return runner(
            list(command),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )

    listed = run(["docker", "volume", "ls", "-q", "--filter", "dangling=true"])
    if listed.returncode:
        raise RuntimeError(listed.stderr.strip() or "cannot list dangling volumes")
    names = listed.stdout.split()
    candidates: list[str] = []
    errors: list[str] = []
    for offset in range(0, len(names), 100):
        inspected = run(["docker", "volume", "inspect", *names[offset : offset + 100]])
        if inspected.returncode:
            raise RuntimeError(
                inspected.stderr.strip() or "cannot inspect dangling volumes"
            )
        for volume in json.loads(inspected.stdout):
            labels = volume.get("Labels") or {}
            created = datetime.fromisoformat(
                str(volume["CreatedAt"]).replace("Z", "+00:00")
            )
            if (
                "com.docker.volume.anonymous" in labels
                and created >= cutoff
            ):
                candidates.append(str(volume["Name"]))

    removed: list[str] = []
    for offset in range(0, len(candidates), 50):
        batch = candidates[offset : offset + 50]
        result = run(["docker", "volume", "rm", *batch], timeout=3600)
        removed.extend(result.stdout.split())
        if result.returncode:
            errors.append(result.stderr.strip() or "docker volume rm failed")
    return {
        "created_since": created_since,
        "dangling_scanned": len(names),
        "candidates": candidates,
        "removed": removed,
        "errors": errors,
    }


def _compose_project(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_-]+", "-", value.casefold()).strip("-_")
    return (normalized or "replay")[:63]


@dataclass(frozen=True)
class DockerObject:
    kind: str
    object_id: str
    name: str
    labels: Mapping[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "id": self.object_id,
            "name": self.name,
            "labels": dict(self.labels),
        }


class DockerReplayAuditor:
    """Remove only Compose objects whose labels contain one environment id."""

    _KINDS: ClassVar[dict[str, str]] = {
        "container": "container",
        "network": "network",
        "volume": "volume",
    }

    def __init__(
        self,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.runner = runner

    def _run(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return self.runner(
            list(command), text=True, capture_output=True, check=False, timeout=120
        )

    @staticmethod
    def _labels(value: object) -> dict[str, str]:
        if isinstance(value, Mapping):
            return {str(key): str(item) for key, item in value.items()}
        labels = {}
        for part in str(value or "").split(","):
            key, separator, item = part.partition("=")
            if separator:
                labels[key] = item
        return labels

    def objects(self, environment_id: str) -> tuple[list[DockerObject], str | None]:
        project = _compose_project(environment_id)
        objects: list[DockerObject] = []
        seen: set[tuple[str, str]] = set()
        try:
            for kind, noun in self._KINDS.items():
                if kind == "container":
                    commands = [
                        [
                            "docker",
                            noun,
                            "ls",
                            "-a",
                            "--filter",
                            f"name={environment_id}",
                        ],
                        [
                            "docker",
                            noun,
                            "ls",
                            "-a",
                            "--filter",
                            f"label=com.docker.compose.project={project}",
                        ],
                    ]
                else:
                    commands = [
                        [
                            "docker",
                            noun,
                            "ls",
                            "--filter",
                            f"label=com.docker.compose.project={project}",
                        ]
                    ]
                commands.append(
                    [
                        "docker",
                        noun,
                        "ls",
                        *(["-a"] if kind == "container" else []),
                        "--filter",
                        f"label=org.sead.environment-id={environment_id}",
                    ]
                )
                for command in commands:
                    command.extend(["--format", "{{json .}}"])
                    completed = self._run(command)
                    if completed.returncode:
                        return (
                            objects,
                            completed.stderr.strip() or f"docker {noun} audit failed",
                        )
                    for line in completed.stdout.splitlines():
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        labels = self._labels(row.get("Labels") or row.get("labels"))
                        label_text = json.dumps(labels, sort_keys=True)
                        object_name = str(row.get("Names") or row.get("Name") or "")
                        named_runtime = (
                            kind == "container"
                            and object_name == f"openhands-runtime-{environment_id}"
                        )
                        if (
                            not named_runtime
                            and environment_id not in label_text
                            and labels.get("com.docker.compose.project") != project
                        ):
                            continue
                        identifier = str(
                            row.get("ID") or row.get("Id") or row.get("Name") or ""
                        )
                        key = (kind, identifier)
                        if key in seen:
                            continue
                        seen.add(key)
                        objects.append(
                            DockerObject(
                                kind,
                                identifier,
                                object_name or identifier,
                                labels,
                            )
                        )
            return objects, None
        except (
            OSError,
            subprocess.SubprocessError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            return objects, f"{type(exc).__name__}: {exc}"

    def cleanup(self, environment_id: str) -> dict[str, object]:
        started = time.perf_counter()
        before, audit_error = self.objects(environment_id)
        errors = [audit_error] if audit_error else []
        removed: list[dict[str, object]] = []
        for kind in ("container", "network", "volume"):
            items = [item for item in before if item.kind == kind]
            if not items:
                continue
            command = ["docker", self._KINDS[kind], "rm"]
            if kind == "container":
                # Anonymous volumes do not carry the Compose/SEAD labels
                # used by the following volume cleanup pass.
                command.extend(("-f", "-v"))
            command.extend(item.object_id for item in items)
            try:
                completed = self._run(command)
            except (OSError, subprocess.SubprocessError) as exc:
                errors.append(f"{kind}: {exc}")
                continue
            if completed.returncode:
                errors.append(
                    completed.stderr.strip() or f"docker {kind} remove failed"
                )
            else:
                removed.extend(item.to_dict() for item in items)
        after, after_error = self.objects(environment_id)
        if after_error:
            errors.append(after_error)
        if after:
            errors.append("owned Docker objects remain after cleanup")
        return {
            "succeeded": not errors and not after,
            "environment_id": environment_id,
            "owned_before_cleanup": [item.to_dict() for item in before],
            "removed": removed,
            "owned_after_cleanup": [item.to_dict() for item in after],
            "errors": errors,
            "cleanup_seconds": round(time.perf_counter() - started, 6),
        }
