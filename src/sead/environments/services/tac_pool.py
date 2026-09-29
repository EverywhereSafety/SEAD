"""Fixed task-to-instance bindings for parallel mutable TAC services."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml


SCHEMA_VERSION = "mtar-tac-pool-v1"
SERVICES = {"gitlab", "owncloud"}
CANONICAL_PORTS = {"gitlab": 8929, "owncloud": 8092}
_INSTANCE_ID = re.compile(r"^(?:gitlab|owncloud)-[a-z0-9][a-z0-9-]*$")


def load_tac_pool(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "controller_url", "instances", "task_bindings",
    }:
        raise ValueError(f"invalid TAC pool keys: {path}")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"expected {SCHEMA_VERSION}: {path}")
    controller_url = str(value["controller_url"]).rstrip("/")
    parsed = urlsplit(controller_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("invalid TAC pool controller_url")
    raw_instances = value["instances"]
    bindings = value["task_bindings"]
    if not isinstance(raw_instances, Mapping) or not isinstance(bindings, Mapping):
        raise ValueError("TAC pool instances and task_bindings must be mappings")
    instances: dict[str, dict[str, Any]] = {}
    all_ports: list[int] = []
    for instance_id, raw in raw_instances.items():
        if not isinstance(instance_id, str) or not _INSTANCE_ID.fullmatch(instance_id):
            raise ValueError(f"invalid TAC instance ID: {instance_id!r}")
        if not isinstance(raw, Mapping) or set(raw) != {"service", "port", "browser_port"}:
            raise ValueError(f"invalid TAC instance: {instance_id}")
        service = str(raw["service"])
        if service not in SERVICES or not instance_id.startswith(service + "-"):
            raise ValueError(f"TAC instance/service mismatch: {instance_id}")
        ports = []
        for field in ("port", "browser_port"):
            port = raw[field]
            if isinstance(port, bool) or not isinstance(port, int) or not 1024 <= port <= 65535:
                raise ValueError(f"invalid TAC {field}: {instance_id}")
            ports.append(port)
        all_ports.extend(ports)
        instances[instance_id] = {
            "instance": instance_id, "service": service,
            "port": ports[0], "browser_port": ports[1],
            "controller_url": controller_url,
        }
    if len(all_ports) != len(set(all_ports)):
        raise ValueError("TAC service and browser ports must be globally unique")
    normalized_bindings: dict[str, str] = {}
    for task_id, instance_id in bindings.items():
        if not isinstance(task_id, str) or not task_id.startswith("single."):
            raise ValueError(f"invalid TAC task binding: {task_id!r}")
        if instance_id not in instances:
            raise ValueError(f"unknown TAC instance for {task_id}: {instance_id}")
        normalized_bindings[task_id] = str(instance_id)
    return {
        "path": str(path.resolve()), "controller_url": controller_url,
        "instances": instances, "task_bindings": normalized_bindings,
    }


def tac_pool_instance(
    execution: Mapping[str, Any], task_id: str,
    *, dependencies: Sequence[str] = (), required: bool = False,
) -> dict[str, Any] | None:
    raw_path = execution.get("tac_pool_config")
    selected_services = set(dependencies) & SERVICES
    if raw_path is None:
        if required and selected_services:
            raise ValueError(f"no TAC pool configured for {task_id}")
        return None
    pool = load_tac_pool(str(raw_path))
    instance_id = pool["task_bindings"].get(task_id)
    if instance_id is None:
        if required and selected_services:
            raise ValueError(f"no TAC pool instance for {task_id}")
        return None
    spec = dict(pool["instances"][instance_id])
    if selected_services and selected_services != {spec["service"]}:
        raise ValueError(
            f"TAC binding for {task_id} is {spec['service']}, "
            f"dependencies are {sorted(selected_services)}"
        )
    spec["task_id"] = task_id
    spec["config_path"] = pool["path"]
    return spec


def rewrite_service_urls(text: str, spec: Mapping[str, Any] | None) -> str:
    if spec is None:
        return text
    service = str(spec["service"])
    canonical = CANONICAL_PORTS[service]
    replacement = f"http://the-agent-company.com:{int(spec['port'])}"
    for host in ("the-agent-company.com", "localhost", "127.0.0.1"):
        text = text.replace(f"http://{host}:{canonical}", replacement)
    return text


def reset_tac_pool_instance(
    spec: Mapping[str, Any], *, timeout_seconds: float = 600,
) -> None:
    base = str(spec["controller_url"]).rstrip("/")
    instance = str(spec["instance"])
    request = urllib.request.Request(
        f"{base}/api/instances/{instance}/reset", method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status not in {200, 202}:
                raise RuntimeError(f"TAC pool reset returned {response.status}")
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(f"TAC pool reset failed for {instance}: {exc}") from exc
    import time
    deadline = time.monotonic() + timeout_seconds
    health = f"{base}/api/instances/{instance}/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=10) as response:
                body = json.loads(response.read().decode("utf-8"))
                if response.status == 200 and body.get("healthy") is True:
                    return
        except (OSError, ValueError, urllib.error.URLError):
            pass
        time.sleep(2)
    raise RuntimeError(f"TAC pool instance did not become ready: {instance}")
