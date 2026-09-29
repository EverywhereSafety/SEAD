"""Controlled runtime-profile registry for normalized MTAR tasks."""

from __future__ import annotations

import json
import re
import subprocess
from functools import lru_cache
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .dataset import ENVIRONMENT_SCHEMA_VERSION, MTARDatasetError

REGISTRY_SCHEMA_VERSION = "mtar-runtime-profile-registry-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REGISTRY = PROJECT_ROOT / "data/mtar/runtime_profiles.yml"
_PROFILE_KEYS = {
    "backend",
    "available",
    "unavailable_reason",
    "image",
    "template",
    "network",
    "cap_add",
    "pid_namespace",
    "seccomp_profile",
}
# Registry-published images use ``repository@sha256:...``; a locally built,
# pre-run image may use its immutable Docker image ID directly.
_DIGEST_IMAGE = re.compile(r"^(?:[^\s@]+@)?sha256:[0-9a-f]{64}$")
_PID_NAMESPACES = {"isolated"}
_SECCOMP_PROFILES = {"default", "ptrace-sshd"}


class UnsupportedRuntimeProfile(RuntimeError):
    code = "UNSUPPORTED_RUNTIME_PROFILE"

    def __init__(self, profile_id: str, reason: str) -> None:
        self.profile_id = profile_id
        self.reason = reason
        super().__init__(f"{self.code}: {profile_id}: {reason}")


@dataclass(frozen=True)
class RuntimeProfile:
    profile_id: str
    backend: str
    available: bool
    unavailable_reason: str | None
    image: str | None
    template: str | None
    network: str
    cap_add: tuple[str, ...]
    pid_namespace: str
    seccomp_profile: str
    evaluator_wheelhouse: str
    evaluator_wheelhouse_available: bool
    evaluator_wheelhouse_unavailable_reason: str

    def require_available(self) -> RuntimeProfile:
        if not self.available:
            raise UnsupportedRuntimeProfile(
                self.profile_id,
                self.unavailable_reason or "profile is not deployed",
            )
        if self.backend != "docker":
            raise UnsupportedRuntimeProfile(
                self.profile_id,
                f"backend {self.backend} is not connected to the OpenHands worker",
            )
        return self

    def docker_runtime_kwargs(self) -> dict[str, Any]:
        if self.backend != "docker":
            raise UnsupportedRuntimeProfile(self.profile_id, "profile is not a Docker backend")
        values: dict[str, Any] = {}
        if self.cap_add:
            values["cap_add"] = list(self.cap_add)
        if self.pid_namespace != "isolated":
            values["pid_mode"] = self.pid_namespace
        if self.seccomp_profile != "default":
            profile_path = PROJECT_ROOT / "resources/seccomp" / f"{self.seccomp_profile}.json"
            try:
                raw_profile = json.loads(profile_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise UnsupportedRuntimeProfile(
                    self.profile_id,
                    f"invalid seccomp profile: {exc}",
                ) from exc
            if not isinstance(raw_profile, Mapping):
                raise UnsupportedRuntimeProfile(
                    self.profile_id,
                    "seccomp profile is not a JSON object",
                )
            # docker-py talks directly to the Engine API; unlike `docker run`,
            # it does not expand a host path supplied as `seccomp=/path`.
            values["security_opt"] = [
                "seccomp=" + json.dumps(raw_profile, separators=(",", ":"))
            ]
        return values

    def require_evaluator_requirements(
        self, requirements: tuple[str, ...]
    ) -> RuntimeProfile:
        if requirements and not self.evaluator_wheelhouse_available:
            raise UnsupportedRuntimeProfile(
                self.profile_id,
                self.evaluator_wheelhouse_unavailable_reason,
            )
        return self


def _load_yaml(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise MTARDatasetError(f"invalid {label} YAML {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise MTARDatasetError(f"{label} must be a mapping: {path}")
    return value


def load_runtime_registry(path: Path | str = DEFAULT_REGISTRY) -> dict[str, RuntimeProfile]:
    path = Path(path)
    value = _load_yaml(path, "runtime profile registry")
    if set(value) != {
        "schema_version",
        "evaluator_wheelhouse",
        "evaluator_wheelhouse_available",
        "evaluator_wheelhouse_unavailable_reason",
        "services",
        "profiles",
    }:
        raise MTARDatasetError("invalid runtime profile registry keys")
    if value.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise MTARDatasetError(f"expected registry schema {REGISTRY_SCHEMA_VERSION}")
    wheelhouse = str(value.get("evaluator_wheelhouse") or "")
    if not wheelhouse.startswith("/") or ".." in Path(wheelhouse).parts:
        raise MTARDatasetError("evaluator wheelhouse must be a fixed absolute sandbox path")
    wheelhouse_available = value["evaluator_wheelhouse_available"]
    wheelhouse_reason = value["evaluator_wheelhouse_unavailable_reason"]
    if not isinstance(wheelhouse_available, bool) or not isinstance(wheelhouse_reason, str):
        raise MTARDatasetError("invalid evaluator wheelhouse availability contract")
    raw_profiles = value.get("profiles")
    if not isinstance(raw_profiles, Mapping):
        raise MTARDatasetError("runtime registry profiles must be a mapping")
    services = value.get("services")
    if not isinstance(services, Mapping):
        raise MTARDatasetError("runtime registry services must be a mapping")
    for service_id, service in services.items():
        expected = {"available", "unavailable_reason", "image"}
        if not isinstance(service_id, str) or not isinstance(service, Mapping) or set(service) != expected:
            raise MTARDatasetError(f"invalid controlled service entry: {service_id!r}")
        service_available = service["available"]
        service_reason = service["unavailable_reason"]
        service_image = service["image"]
        if not isinstance(service_available, bool):
            raise MTARDatasetError(f"invalid service availability: {service_id}")
        if service_available:
            if service_reason is not None:
                raise MTARDatasetError(f"available service has an unavailable reason: {service_id}")
            if not isinstance(service_image, str) or not _DIGEST_IMAGE.fullmatch(service_image):
                raise MTARDatasetError(f"available service needs a digest-pinned image: {service_id}")
        elif not isinstance(service_reason, str) or service_image is not None:
            raise MTARDatasetError(f"invalid unavailable service contract: {service_id}")
    profiles: dict[str, RuntimeProfile] = {}
    for profile_id, raw in raw_profiles.items():
        if not isinstance(profile_id, str) or not isinstance(raw, Mapping) or set(raw) != _PROFILE_KEYS:
            raise MTARDatasetError(f"invalid runtime profile entry: {profile_id!r}")
        backend = str(raw["backend"])
        available = raw["available"]
        image = raw["image"]
        template = raw["template"]
        network = str(raw["network"])
        cap_add = raw["cap_add"]
        pid_namespace = str(raw["pid_namespace"])
        seccomp_profile = str(raw["seccomp_profile"])
        if backend not in {"docker", "microvm"} or not isinstance(available, bool):
            raise MTARDatasetError(f"invalid backend/availability for {profile_id}")
        if network not in {"host", "isolated"}:
            raise MTARDatasetError(f"invalid network policy for {profile_id}")
        if pid_namespace not in _PID_NAMESPACES:
            raise MTARDatasetError(f"invalid PID namespace policy for {profile_id}")
        if seccomp_profile not in _SECCOMP_PROFILES:
            raise MTARDatasetError(f"invalid seccomp profile for {profile_id}")
        if seccomp_profile != "default" and not (
            PROJECT_ROOT / "resources/seccomp" / f"{seccomp_profile}.json"
        ).is_file():
            raise MTARDatasetError(f"missing seccomp policy for {profile_id}")
        if not isinstance(cap_add, list) or not all(isinstance(item, str) for item in cap_add):
            raise MTARDatasetError(f"invalid capability list for {profile_id}")
        if any(item not in {"SYS_PTRACE", "NET_ADMIN"} for item in cap_add):
            raise MTARDatasetError(f"unsafe Docker capability for {profile_id}")
        if cap_add and network == "host":
            raise MTARDatasetError(f"capability profile cannot use host networking: {profile_id}")
        if backend == "docker":
            if available and (not isinstance(image, str) or not _DIGEST_IMAGE.fullmatch(image)):
                raise MTARDatasetError(f"available Docker profile needs a digest-pinned image: {profile_id}")
            if template is not None:
                raise MTARDatasetError(f"Docker profile cannot name a microVM template: {profile_id}")
        else:
            if image is not None or not isinstance(template, str) or not template:
                raise MTARDatasetError(f"microVM profile needs only a template ID: {profile_id}")
            if cap_add:
                raise MTARDatasetError(f"microVM profile cannot pass Docker capabilities: {profile_id}")
        reason = raw["unavailable_reason"]
        if available and reason is not None:
            raise MTARDatasetError(f"available profile has an unavailable reason: {profile_id}")
        if not available and not isinstance(reason, str):
            raise MTARDatasetError(f"unavailable profile needs a reason: {profile_id}")
        profiles[profile_id] = RuntimeProfile(
            profile_id=profile_id,
            backend=backend,
            available=available,
            unavailable_reason=reason,
            image=image,
            template=template,
            network=network,
            cap_add=tuple(cap_add),
            pid_namespace=pid_namespace,
            seccomp_profile=seccomp_profile,
            evaluator_wheelhouse=wheelhouse,
            evaluator_wheelhouse_available=wheelhouse_available,
            evaluator_wheelhouse_unavailable_reason=wheelhouse_reason,
        )
    return profiles


def resolve_service_image(
    service_id: str, path: Path | str = DEFAULT_REGISTRY
) -> str:
    value = _load_yaml(Path(path), "runtime profile registry")
    services = value.get("services")
    if not isinstance(services, Mapping):
        raise MTARDatasetError("runtime registry services must be a mapping")
    service = services.get(service_id)
    expected = {"available", "unavailable_reason", "image"}
    if not isinstance(service, Mapping) or set(service) != expected:
        raise MTARDatasetError(f"invalid controlled service entry: {service_id}")
    if not service["available"]:
        raise UnsupportedRuntimeProfile(
            service_id,
            str(service["unavailable_reason"] or "service image is not deployed"),
        )
    image = service["image"]
    if not isinstance(image, str) or not _DIGEST_IMAGE.fullmatch(image):
        raise MTARDatasetError(f"available service needs a digest-pinned image: {service_id}")
    return image


def load_task_environment(task_root: Path | str) -> dict[str, Any]:
    path = Path(task_root) / "utils/environment.yml"
    value = _load_yaml(path, "task environment")
    expected = {"schema_version", "runtime_profile", "evaluator_python_requirements"}
    if set(value) != expected or value.get("schema_version") != ENVIRONMENT_SCHEMA_VERSION:
        raise MTARDatasetError(f"invalid task environment contract: {path}")
    requirements = value["evaluator_python_requirements"]
    if not isinstance(requirements, list) or not all(isinstance(item, str) for item in requirements):
        raise MTARDatasetError(f"invalid evaluator requirements: {path}")
    return dict(value)


def resolve_task_profile(task_root: Path | str) -> tuple[RuntimeProfile, tuple[str, ...]]:
    environment = load_task_environment(task_root)
    profiles = load_runtime_registry()
    profile_id = str(environment["runtime_profile"])
    profile = profiles.get(profile_id)
    # Existing profile IDs keep their global deployment binding. A new release
    # may explicitly name an additive profile held in its own pinned registry;
    # do not silently override `base` for existing or running experiments.
    if profile is None:
        task = Path(task_root).resolve()
        registry = task.parent.parent / "runtime_profiles.yml"
        if task.parent.name == "tasks" and registry.is_file() and not registry.is_symlink():
            profile = load_runtime_registry(registry).get(profile_id)
    if profile is None:
        raise MTARDatasetError(f"task references unknown runtime profile: {profile_id}")
    return profile, tuple(environment["evaluator_python_requirements"])


def openhands_base_image_alias(profile: RuntimeProfile) -> str:
    """Return the deterministic local alias required by OpenHands' builder."""

    source = str(profile.image or "")
    digest_match = re.search(r"sha256:([0-9a-f]{64})$", source)
    if digest_match is None:
        raise UnsupportedRuntimeProfile(
            profile.profile_id,
            "controlled Docker image is not content-addressed",
        )
    return (
        f"sead/mtar-{profile.profile_id}:"
        f"base-{digest_match.group(1)[:16]}"
    )


@lru_cache(maxsize=None)
def prepare_openhands_base_image(profile: RuntimeProfile) -> str:
    """Verify and tag a controlled image once in the parent rollout process."""

    source = str(profile.image or "")
    alias = openhands_base_image_alias(profile)
    inspected = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", source],
        text=True,
        capture_output=True,
        check=False,
    )
    image_id = inspected.stdout.strip()
    if (
        inspected.returncode != 0
        or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None
    ):
        raise UnsupportedRuntimeProfile(
            profile.profile_id,
            "controlled base image is not present locally",
        )
    tagged = subprocess.run(
        ["docker", "image", "tag", source, alias],
        text=True,
        capture_output=True,
        check=False,
    )
    if tagged.returncode != 0:
        raise UnsupportedRuntimeProfile(
            profile.profile_id,
            "could not create the verified OpenHands base-image alias",
        )
    return alias


__all__ = [
    "DEFAULT_REGISTRY",
    "REGISTRY_SCHEMA_VERSION",
    "RuntimeProfile",
    "UnsupportedRuntimeProfile",
    "load_runtime_registry",
    "load_task_environment",
    "openhands_base_image_alias",
    "prepare_openhands_base_image",
    "resolve_task_profile",
    "resolve_service_image",
]
