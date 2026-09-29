"""One replay's environment binding, shared by every evaluation entry point."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

from .leases import LeaseClient
from .registry import EnvironmentRegistry, digest, environment_config, required_service


def task_digest(root):
    checksum = hashlib.sha256()
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            checksum.update(str(path.relative_to(root)).encode() + b"\0")
            checksum.update(hashlib.sha256(path.read_bytes()).digest())
    return checksum.hexdigest()


def write_private(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class EnvironmentSession:
    def __init__(self, request, dependencies, task_root, workspace):
        config = environment_config(request.execution)
        if config is None:
            raise ValueError("EnvironmentSession requires execution.environments")
        self.request = request
        self.config = config
        self.service = required_service(dependencies)
        self.lease = None
        self.binding = {}
        self.profile = {}
        self.pool = None
        self.closed = False
        self.relays = []
        self.mapping_audit = []
        if self.service is None:
            return
        registry = EnvironmentRegistry(config["registry"])
        if config.get("registry_digest", registry.digest) != registry.digest:
            raise ValueError("environment registry differs from frozen configuration")
        self.pool, self.profile_id, self.profile = registry.resolve(
            self.service, request.task_id
        )
        self.client = LeaseClient(self.pool["manager_socket"], timeout=float(config.get("rpc_timeout_seconds", 30)))
        if self.service == "postgres":
            from sead.benchmarks.mtar.postgres import environment_spec
            provider_spec = environment_spec(request.task_id, Path(task_root))
        else:
            provider_spec = {"workspace": str(Path(workspace).resolve())}
        self.spec = {"profile_id": self.profile_id, "profile_digest": digest(self.profile),
                     "provider_spec": provider_spec, "task_digest": task_digest(task_root),
                     "environment_id": request.environment_id}
        self.owner = {key: getattr(request, key) for key in ("run_id", "task_id", "node_id", "replay_id")}

    def __enter__(self):
        if self.service is None:
            return self
        status = self.client.call("inspect")
        protocol = self.pool.get("protocol", "environment-lease-v2")
        if status["capacity"] != self.pool["capacity"]:
            raise ValueError("environment pool capacity differs from configured deployment")
        if protocol != "postgres-v1" and status.get("config_digest") != self.pool["config_digest"]:
            raise ValueError("environment manager configuration digest mismatch")
        recovery = {"protocol": protocol, "socket": self.client.path,
                    "request_id": self.request.replay_id, "owner": self.owner}
        private = Path(self.request.worker_dir) / "environment_private.json"
        write_private(private, recovery)
        started = time.monotonic()
        if protocol == "postgres-v1":
            self.lease = self.client.acquire(self.spec["provider_spec"], self.owner,
                request_id=self.request.replay_id, wait_seconds=float(self.config.get("acquire_timeout_seconds", 300)))
        else:
            self.lease = self.client.acquire_v2(self.spec, self.owner,
                request_id=self.request.replay_id, wait_seconds=float(self.config.get("acquire_timeout_seconds", 300)),
                startup_seconds=float(self.pool.get("startup_timeout_seconds", 900)),
                poll_seconds=float(self.pool.get("poll_interval_seconds", 0.25)))
        self.binding = self.lease.binding
        self.lease.client = LeaseClient(self.client.path, timeout=max(self.client.timeout,
            float(self.pool.get("cleanup_timeout_seconds", 180))))
        write_private(private, {**recovery, "identity": self.lease.identity})
        self.write_audit(acquire_seconds=time.monotonic() - started)
        return self

    def write_audit(self, **extra):
        from sead.campaigns.infrastructure import atomic_write_json
        audit = self.lease.audit() if self.lease else {}
        atomic_write_json(Path(self.request.worker_dir) / "environment_lease.json", {
            "schema_version": "sead-environment-audit-v1", **audit,
            "task_digest": self.spec["task_digest"], "config_digest": self.pool["config_digest"],
            "url_mapping": self.mapping_audit, **extra,
        })

    def check(self):
        if self.closed:
            raise RuntimeError("environment session closed")
        if self.lease:
            self.lease.check()

    def install_runtime_network(self):
        if self.lease:
            from .services.docker_relay import install_openhands_lease_network
            self.relays = install_openhands_lease_network(self.binding["network"], self.binding["labels"], check=self.check)

    def transform(self, value):
        if isinstance(value, dict):
            return {key: self.transform(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.transform(item) for item in value]
        if not isinstance(value, str) or "public_url" not in self.profile:
            return value
        for source in sorted(self.profile.get("url_aliases", []), key=len, reverse=True):
            # Match an endpoint boundary; :8092 must not rewrite :80920.
            import re
            value, count = re.subn(re.escape(source.rstrip("/")) + r"(?=$|[/\s?#\"'<>])",
                                   lambda _: self.profile["public_url"].rstrip("/"), value)
            if count:
                entry = {"from": source, "to": self.profile["public_url"]}
                if entry not in self.mapping_audit:
                    self.mapping_audit.append(entry)
        return value

    def attach_runtime(self, runtime):
        self.check()
        runtime.sead_environment_session = self
        if self.service == "postgres":
            runtime.sead_postgres_lease = self.lease
        original = runtime.run_action

        def guarded(action, *args, **kwargs):
            self.check()
            derived = copy.deepcopy(action)
            for field in ("arguments", "command", "content"):
                if hasattr(derived, field):
                    setattr(derived, field, self.transform(getattr(derived, field)))
            result = original(derived, *args, **kwargs)
            self.check()
            return result

        runtime.run_action = guarded
        if hasattr(runtime, "call_tool_mcp"):
            dispatch = runtime.call_tool_mcp

            async def guarded_mcp(action, *args, **kwargs):
                self.check()
                derived = copy.deepcopy(action)
                derived.arguments = self.transform(action.arguments)
                result = await dispatch(derived, *args, **kwargs)
                self.check()
                return result

            runtime.call_tool_mcp = guarded_mcp

    def service_context(self):
        if not self.service or self.service == "postgres":
            return ""
        account = self.profile["credentials"]
        return (f"\n\nTask service: {self.service}\nURL: {self.profile['public_url']}\n"
                f"Username: {account['username']}\nPassword: {account['password']}\n"
                + "\n".join(self.profile.get("operational_notes", [])))

    def evaluator_settings(self):
        if not self.lease or self.service == "postgres":
            return {}
        url = urlsplit(self.profile["public_url"])
        return {"server_hostname": url.hostname, "service_ports": {self.service: url.port or 80},
                "network": self.binding["network"], "labels": self.binding["labels"]}

    def close(self):
        if self.closed:
            return
        errors = []
        for relay in self.relays:
            try:
                relay.close()
            except Exception as exc:
                errors.append(exc)
        if self.lease:
            try:
                self.lease.close()
            except Exception as exc:
                errors.append(exc)
            self.write_audit(cleanup_succeeded=not errors)
        self.closed = True
        if errors:
            raise RuntimeError("CLEANUP_FAILED: environment session cleanup incomplete") from errors[0]

    def __exit__(self, *_args):
        self.close()


def recover_environment(worker_dir):
    """Only call after the parent has confirmed the old worker has exited."""
    path = Path(worker_dir) / "environment_private.json"
    if not path.exists():
        return
    process_path = Path(worker_dir) / "process.json"
    if process_path.exists():
        process = json.loads(process_path.read_text())
        stat = Path(f"/proc/{int(process['pid'])}/stat")
        try:
            live_ticks = stat.read_text().rsplit(")", 1)[1].split()[19]
        except FileNotFoundError:
            live_ticks = None
        if live_ticks == str(process["start_ticks"]):
            raise RuntimeError("refusing environment recovery while its worker is alive")
    private = json.loads(path.read_text())
    request = json.loads((Path(worker_dir) / "request.json").read_text())
    request_owner = {
        **request,
        "replay_id": request.get("replay_id", request.get("request_id")),
    }
    if any(private["owner"][key] != request_owner.get(key) for key in private["owner"]):
        raise ValueError("environment recovery owner mismatch")
    client = LeaseClient(private["socket"])
    if private["protocol"] != "postgres-v1":
        client.call("cancel", request_id=private["request_id"])
    if private.get("identity"):
        client.call("release", **private["identity"])
    deadline = time.monotonic() + float(request["execution"].get("cleanup_timeout_seconds", 180))
    while True:
        records = client.call("inspect", owner=private["owner"])["leases"]
        if all(row["state"] == "destroyed" for row in records):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("CLEANUP_FAILED: manager has unreleased environment resources")
        time.sleep(0.25)


def worker_timeout(execution):
    """Parent deadline includes environment admission, preparation and cleanup."""
    replay = float(execution.get("sample_timeout_seconds", 900))
    config = environment_config(execution)
    if not config:
        return replay + 120
    registry = EnvironmentRegistry(config["registry"])
    startup = max((float(pool.get("startup_timeout_seconds", 900)) for pool in registry.pools.values()), default=0)
    return replay + float(config.get("acquire_timeout_seconds", 300)) + startup + float(execution.get("cleanup_timeout_seconds", 180))
