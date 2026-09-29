"""Cancellable providers for task-independent, isolated environment pools."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from .postgres import PostgresProvider, IMAGE
from .docker_relay import DockerTCPRelay
from ..registry import digest


class CancellableDocker:
    def destroy(self, lease_id):
        self.creation.context = (threading.Event(), time.monotonic() + float(self.pool.get("cleanup_timeout_seconds", 180)))
        try:
            return super().destroy(lease_id)
        finally:
            del self.creation.context

    def create_cancellable(self, lease_id, spec, *, cancel, deadline):
        self.creation.context = (cancel, deadline)
        try:
            return self.create(lease_id, spec)
        finally:
            del self.creation.context

    def docker(self, *args, input=None, timeout=90):
        context = getattr(self.creation, "context", None)
        if context:
            cancel, deadline = context
            if cancel.is_set():
                raise InterruptedError("environment creation cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("environment preparation deadline exceeded")
            timeout = min(timeout, remaining)
        # Docker calls are bounded. Cancellation never runs destroy concurrently
        # with a still-running creation command; manager waits for this to exit.
        return super().docker(*args, input=input, timeout=timeout)


class PooledPostgresProvider(CancellableDocker, PostgresProvider):
    def __init__(self, config, manager_id):
        self.creation = threading.local()
        self.pool = config
        super().__init__(config["provider_config"], manager_id)

    def validate_spec(self, spec):
        if "profile_id" not in spec:  # Internal PG create() revalidates its wire spec.
            return super().validate_spec(spec)
        validate_profile(self.pool, spec)
        super().validate_spec(spec["provider_spec"])
        return dict(spec)

    def create(self, lease_id, spec):
        self.validate_spec(spec)
        binding = super().create(lease_id, spec["provider_spec"])
        return {**binding, **public_metadata(self.pool, spec)}


def validate_profile(pool, spec):
    if set(spec) != {"profile_id", "profile_digest", "provider_spec", "task_digest", "environment_id"}:
        raise ValueError("invalid environment specification")
    profile = pool["profile_definitions"].get(spec["profile_id"])
    if profile is None or digest(profile) != spec["profile_digest"]:
        raise ValueError("PROFILE_MISMATCH")
    return profile


def public_metadata(pool, spec):
    return {"pool_id": pool["pool_id"], "service": pool["service"],
            "profile_id": spec["profile_id"], "profile_digest": spec["profile_digest"],
            "environment_version": "environment-lease-v2",
            "runtime_resources": pool["profile_definitions"][spec["profile_id"]].get("runtime_resources", {})}


class WebProvider(CancellableDocker, PostgresProvider):
    """Reuse label-owned Docker cleanup, without PostgreSQL operations.

    Websites and browsers are created by the manager on one isolated network.
    Only inbound Docker-exec relays are exposed to the trusted host.
    """
    def __init__(self, config, manager_id):
        self.creation = threading.local()
        self.pool = config
        self.config = config
        self.manager_id = manager_id
        self.relays = {}
        self.extra_relays = {}
        self.roots = [Path(p).resolve() for p in config["provider_config"]["workspace_roots"]]
        for profile in config["profile_definitions"].values():
            for image in (profile["image"], profile["browser"]["image"]):
                if not IMAGE.fullmatch(image):
                    raise ValueError("environment images must be immutable")
            if profile.get("network") != {"mode": "lease_isolated", "egress": "deny"}:
                raise ValueError("Web leases require isolated networks with egress denied")
            url = urlsplit(profile["public_url"])
            if url.scheme != "http" or url.hostname in {None, "localhost", "127.0.0.1"}:
                raise ValueError("profile public_url requires a network hostname")

    def validate_spec(self, spec):
        validate_profile(self.pool, spec)
        provider_spec = spec["provider_spec"]
        if set(provider_spec) != {"workspace"}:
            raise ValueError("Web provider requires a workspace")
        workspace = Path(provider_spec["workspace"]).resolve()
        if not workspace.is_dir() or not any(workspace.is_relative_to(root) for root in self.roots):
            raise ValueError("workspace outside pool roots")
        return dict(spec)

    def labels(self, lease_id):
        return {**super().labels(lease_id), "sead.pool": self.pool["pool_id"]}

    def _wait_browser_mcp(self, browser: str, browser_port: int) -> None:
        # The website can be healthy before the browser MCP finishes opening
        # its own port. A single probe intermittently fails under a busy pool.
        browser_deadline = time.monotonic() + 30
        while True:
            try:
                self.docker("exec", browser, "node", "-e",
                            "const s=require('net').connect(Number(process.argv[1]),'127.0.0.1',()=>s.end());s.on('error',()=>process.exit(1))",
                            str(browser_port), timeout=10)
                return
            except (RuntimeError, subprocess.TimeoutExpired):
                if time.monotonic() >= browser_deadline:
                    raise
                time.sleep(float(self.pool.get("poll_interval_seconds", 0.25)))

    def create(self, lease_id, spec):
        self.validate_spec(spec)
        profile = self.pool["profile_definitions"][spec["profile_id"]]
        service = self.pool["service"]
        name = f"sg-{service}-{lease_id}"
        browser = name + "-browser"
        labels = self._label_args(lease_id)
        labels += ["--label", f"org.sead.environment-id={spec['environment_id']}"]
        url = urlsplit(profile["public_url"])
        port = url.port or 80
        for image in (profile["image"], profile["browser"]["image"]):
            self.docker("image", "inspect", image)
        self.docker("network", "create", "--internal", "--opt",
                    "com.docker.network.bridge.gateway_mode_ipv4=isolated", *labels, name)
        network = json.loads(self.docker("network", "inspect", name))[0]
        if not network["Internal"] or network["Options"].get("com.docker.network.bridge.gateway_mode_ipv4") != "isolated":
            raise RuntimeError("Docker does not support isolated environment networks")
        resources = profile["resources"]
        limits = ["--cpus", str(resources["cpus"]), "--memory", resources["memory"],
                  "--pids-limit", str(resources["pids_limit"])]
        command = ["run", "-d", "--pull=never", "--name", name, *labels, *limits,
                   "--network", name, "--network-alias", url.hostname, "--hostname", url.hostname]
        if service == "gitlab":
            tuning = profile.get("gitlab", {})
            workers = int(tuning.get("puma_workers", 2))
            concurrency = int(tuning.get("sidekiq_concurrency", 10))
            command += ["--shm-size", resources["shm_size"], "-e",
                        f"GITLAB_OMNIBUS_CONFIG=external_url '{profile['public_url']}'; gitlab_rails['gitlab_shell_ssh_port'] = 2424; "
                        f"puma['worker_processes'] = {workers}; sidekiq['concurrency'] = {concurrency};"]
        elif service == "owncloud":
            if port != 80:
                raise ValueError("ownCloud profile must use internal port 80; use URL aliases for legacy ports")
            credentials = profile["credentials"]
            command += ["-e", f"OWNCLOUD_DOMAIN={url.netloc}", "-e", f"OWNCLOUD_TRUSTED_DOMAINS={url.netloc}",
                        "-e", f"OWNCLOUD_ADMIN_USERNAME={credentials['username']}",
                        "-e", f"OWNCLOUD_ADMIN_PASSWORD={credentials['password']}"]
        elif service != "reddit":
            raise ValueError("unsupported Web provider")
        self.docker(*command, profile["image"])
        browser_port = int(profile["browser"]["internal_port"])
        self.docker("run", "-d", "--pull=never", "--name", browser, *labels,
                    "--network", name, "--cap-drop=NET_RAW", "--security-opt=no-new-privileges",
                    "--cpus", str(profile["browser"].get("cpus", 1)),
                    "--memory", profile["browser"].get("memory", "1g"),
                    "--pids-limit", str(profile["browser"].get("pids_limit", 512)),
                    "-e", f"MCP_PLAYWRIGHT_PORT={browser_port}",
                    "-v", f"{spec['provider_spec']['workspace']}:/workspace", profile["browser"]["image"])
        health = profile["public_url"].rstrip("/") + profile["health_path"]
        restarted = False
        while True:
            try:
                self.docker("exec", browser, "node", "-e",
                            "fetch(process.argv[1]).then(async r=>{if(!r.ok)process.exit(1);"
                            "const b=await r.text();if(process.argv[2]==='owncloud'){"
                            "const v=JSON.parse(b);if(!v.installed||v.maintenance)process.exit(1)}}).catch(()=>process.exit(1))",
                            health, service, timeout=10)
                break
            except (RuntimeError, subprocess.TimeoutExpired):
                if service == "gitlab" and not restarted:
                    status = self.docker("inspect", "--format", "{{.State.Status}}", name)
                    if status == "exited":
                        self.docker("start", name)
                        restarted = True
                time.sleep(float(self.pool.get("poll_interval_seconds", 0.25)))
        if service == "gitlab":
            refreshed = self.docker("exec", name, "gitlab-psql", "-d", "gitlabhq_production", "-Atc",
                "UPDATE personal_access_tokens SET expires_at='2099-12-31', updated_at=NOW() "
                "WHERE name='root-token' AND revoked=false AND user_id=(SELECT id FROM users WHERE username='root');")
            if "UPDATE 1" not in refreshed:
                raise RuntimeError("GitLab evaluator token preparation failed")
        self._wait_browser_mcp(browser, browser_port)
        relay = DockerTCPRelay(browser, browser_port, python="node")
        self.relays[lease_id] = relay
        return {**public_metadata(self.pool, spec), "network": name, "labels": {
                    **self.labels(lease_id), "org.sead.environment-id": spec["environment_id"]},
                "public_url": profile["public_url"], "image": profile["image"], "browser_container": browser,
                "host_mcp_url": f"http://127.0.0.1:{relay.port}/sse",
                "target_mcp_url": f"http://{browser}:{browser_port}/sse"}

    def sql(self, *args, **kwargs):
        raise ValueError("SQL capability is unavailable for Web leases")

    evidence = authenticate = sql


def build_provider(config, manager_id):
    if config["service"] == "postgres":
        return PooledPostgresProvider(config, manager_id)
    return WebProvider(config, manager_id)
