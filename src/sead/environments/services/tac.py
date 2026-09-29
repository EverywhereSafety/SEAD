"""Shared TheAgentCompany service deployment and lifecycle helpers.

Both MTAR and OpenAgentSafety use the same mutable GitLab/ownCloud deployment.
This module is the single synchronization and reset boundary for those services.
"""

from __future__ import annotations

import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SERVICE_DEPLOYMENTS = PROJECT_ROOT / "config/mtar_service_deployments.yml"
SERVICE_DEPLOYMENTS_SCHEMA_VERSION = "mtar-service-deployments-v1"

SERVICE_DEPENDENCIES = {
    "gitlab",
    "owncloud",
    # Plane is part of the upstream TAC contract but intentionally remains
    # unavailable until its mutable reset inputs are digest-pinned.
    "plane",
    "reddit",
    "shopping",
    "shopping_admin",
    "history-injection",
    "owncloud-injection",
    "url-injection",
}

from sead.environments.resources import (
    TAC_WEB_SERVICES,
    acquire_web_lock,
    release_web_lock,
)

SERVICE_RESET_ENDPOINTS = {
    "gitlab": "reset-gitlab",
    "owncloud": "reset-owncloud",
    "plane": "reset-plane",
}


def load_service_deployments(
    path: Path | str = DEFAULT_SERVICE_DEPLOYMENTS,
) -> dict[str, dict[str, Any]]:
    """Load Controller/Target-visible benchmark endpoints and credentials."""

    path = Path(path)
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid service deployment settings {path}: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {"schema_version", "services"}:
        raise ValueError(f"invalid service deployment settings keys: {path}")
    if value.get("schema_version") != SERVICE_DEPLOYMENTS_SCHEMA_VERSION:
        raise ValueError(
            f"expected service deployment schema {SERVICE_DEPLOYMENTS_SCHEMA_VERSION}: {path}"
        )
    raw_services = value.get("services")
    if not isinstance(raw_services, dict):
        raise ValueError(f"service deployments must be a mapping: {path}")

    deployments: dict[str, dict[str, Any]] = {}
    required = {"display_name", "url", "credentials"}
    optional = {"operational_notes", "reset_url", "status_url", "health_url"}
    for dependency, raw in raw_services.items():
        if dependency not in SERVICE_DEPENDENCIES or not isinstance(raw, dict):
            raise ValueError(f"invalid service deployment: {dependency!r}")
        if not required.issubset(raw) or not set(raw).issubset(required | optional):
            raise ValueError(f"invalid service deployment fields: {dependency}")
        display_name = str(raw.get("display_name") or "").strip()
        url = str(raw.get("url") or "").strip().rstrip("/")
        parsed = urlsplit(url)
        if (
            not display_name
            or parsed.scheme not in {"http", "https"}
            or not parsed.netloc
        ):
            raise ValueError(f"invalid service deployment URL/name: {dependency}")
        credentials = raw.get("credentials")
        if not isinstance(credentials, dict) or set(credentials) != {
            "username",
            "password",
        }:
            raise ValueError(f"invalid benchmark credentials: {dependency}")
        username = credentials.get("username")
        password = credentials.get("password")
        if (
            not isinstance(username, str)
            or not username
            or not isinstance(password, str)
            or not password
        ):
            raise ValueError(f"empty benchmark credentials: {dependency}")
        operational_notes = raw.get("operational_notes", [])
        if not isinstance(operational_notes, list) or any(
            not isinstance(note, str) or not note.strip() for note in operational_notes
        ):
            raise ValueError(
                f"invalid service deployment operational notes: {dependency}"
            )
        deployments[dependency] = {
            "display_name": display_name,
            "url": url,
            "credentials": {"username": username, "password": password},
            "operational_notes": [note.strip() for note in operational_notes],
        }
        for field in ("reset_url", "status_url", "health_url"):
            if field in raw:
                endpoint = str(raw[field]).strip()
                parts = urlsplit(endpoint)
                if parts.scheme not in {"http", "https"} or not parts.netloc:
                    raise ValueError(f"invalid {field} for {dependency}")
                deployments[dependency][field] = endpoint
    return deployments


def service_instruction_context(
    dependencies: Sequence[str],
    *,
    deployments_path: Path | str = DEFAULT_SERVICE_DEPLOYMENTS,
) -> str:
    """Render only public credentials selected by task dependencies."""

    deployments = load_service_deployments(deployments_path)
    lines = []
    for dependency in dependencies:
        deployment = deployments.get(dependency)
        if deployment is None:
            continue
        credentials = deployment["credentials"]
        lines.append(
            f"{deployment['display_name']} is available at {deployment['url']}. "
            f"Sign in with username {credentials['username']!r} and "
            f"password {credentials['password']!r}."
            + "".join(f" {note}" for note in deployment.get("operational_notes", []))
        )
    if not lines:
        return ""
    return "\n\nBenchmark service access:\n" + "\n".join(f"- {line}" for line in lines)


def _refresh_gitlab_evaluator_token() -> None:
    """Extend the published root-token expiry without changing its value."""

    refreshed = subprocess.run(
        [
            "docker",
            "exec",
            "gitlab",
            "gitlab-psql",
            "-d",
            "gitlabhq_production",
            "-Atc",
            (
                "UPDATE personal_access_tokens SET expires_at='2099-12-31', "
                "updated_at=NOW() WHERE name='root-token' AND revoked=false "
                "AND user_id=(SELECT id FROM users WHERE username='root');"
            ),
        ],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if refreshed.returncode or "UPDATE 1" not in refreshed.stdout:
        raise RuntimeError(
            "failed to refresh TheAgentCompany GitLab evaluator token: "
            + (refreshed.stderr.strip() or refreshed.stdout.strip())
        )


def reset_services(
    dependencies: Sequence[str],
    *,
    hostname: str,
    timeout_seconds: float = 600,
    deployments_path: Path | str = DEFAULT_SERVICE_DEPLOYMENTS,
) -> tuple[str, ...]:
    """Reset selected TAC services through the official controller and wait."""

    operations: list[str] = []
    safe_dependencies = set(dependencies) & {"reddit", "shopping", "shopping_admin"}
    if safe_dependencies:
        deployments = load_service_deployments(deployments_path)
        selected = [deployments.get(name) for name in sorted(safe_dependencies)]
        if any(item is None for item in selected):
            raise RuntimeError("SafeArena deployment is not configured")
        reset_urls = {item.get("reset_url") for item in selected}
        status_urls = {item.get("status_url") for item in selected}
        if None in reset_urls or None in status_urls or len(reset_urls) != 1 or len(status_urls) != 1:
            raise RuntimeError("SafeArena services must share one configured instance/reset server")
        reset_url = reset_urls.pop()
        status_url = status_urls.pop()
        try:
            with urllib.request.urlopen(reset_url, timeout=30) as response:
                if response.status != 200:
                    raise RuntimeError(f"SafeArena reset returned {response.status}")
        except (OSError, urllib.error.URLError) as exc:
            raise RuntimeError(f"SafeArena reset failed: {exc}") from exc
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(status_url, timeout=10) as response:
                    body = response.read(4096).decode("utf-8", "replace")
                    if response.status == 200 and "Ready for duty!" in body:
                        break
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(2)
        else:
            raise RuntimeError("SafeArena reset did not become ready")
        for name, item in zip(sorted(safe_dependencies), selected):
            health_url = item.get("health_url", item["url"])
            try:
                with urllib.request.urlopen(health_url, timeout=20) as response:
                    if response.status >= 400:
                        raise RuntimeError(f"SafeArena {name} returned {response.status}")
            except (OSError, urllib.error.URLError) as exc:
                raise RuntimeError(f"SafeArena {name} is not reachable: {exc}") from exc
            operations.append(f"service:{name}")
    for dependency in dependencies:
        endpoint = SERVICE_RESET_ENDPOINTS.get(dependency)
        if endpoint is None:
            continue
        request = urllib.request.Request(
            f"http://{hostname}:2999/api/{endpoint}", method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status not in {200, 202}:
                    raise RuntimeError(
                        f"service reset for {dependency} returned {response.status}"
                    )
        except (OSError, urllib.error.URLError) as exc:
            raise RuntimeError(f"service reset for {dependency} failed: {exc}") from exc

        deadline = time.monotonic() + timeout_seconds
        health_url = f"http://{hostname}:2999/api/healthcheck/{dependency}"
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(health_url, timeout=10) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(2)
        else:
            raise RuntimeError(f"service reset for {dependency} did not become ready")

        if dependency == "gitlab":
            _refresh_gitlab_evaluator_token()
        operations.append(f"service:{dependency}")
    return tuple(operations)


__all__ = [
    "DEFAULT_SERVICE_DEPLOYMENTS",
    "SERVICE_DEPENDENCIES",
    "SERVICE_DEPLOYMENTS_SCHEMA_VERSION",
    "SERVICE_RESET_ENDPOINTS",
    "TAC_WEB_SERVICES",
    "acquire_web_lock",
    "load_service_deployments",
    "release_web_lock",
    "reset_services",
    "service_instruction_context",
]
