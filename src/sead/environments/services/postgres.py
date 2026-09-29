"""Disposable, label-owned PostgreSQL clusters on isolated Docker networks."""

from __future__ import annotations

import hashlib
import base64
import hmac
import json
import re
import secrets
import subprocess
import time
from pathlib import Path
from typing import Any

from .docker_relay import DockerTCPRelay

LABEL = "sead.lease"
IMAGE = re.compile(r"^(?:[^\s@]+@)?sha256:[0-9a-f]{64}$")


class PostgresProvider:
    def __init__(self, config: dict[str, Any], manager_id: str):
        self.config = config
        self.manager_id = manager_id
        self.relays: dict[str, DockerTCPRelay] = {}
        for key in ("postgres_image", "mcp_image"):
            if not IMAGE.fullmatch(str(config.get(key, ""))):
                raise ValueError(f"{key} must be an immutable image digest or ID")
            self.docker("image", "inspect", config[key])
        self.seed_roots = tuple(Path(p).resolve() for p in config["seed_roots"])
        if not self.seed_roots:
            raise ValueError("seed_roots cannot be empty")

    def docker(self, *args: str, input: str | None = None, timeout: float = 90) -> str:
        result = subprocess.run(
            ["docker", *args],
            input=input,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        if result.returncode:
            # Docker argv may contain credentials; never include it in errors.
            raise RuntimeError(f"Docker {args[0]} failed (exit {result.returncode})")
        return result.stdout.strip()

    def labels(self, lease_id: str) -> dict[str, str]:
        return {LABEL: lease_id, "sead.lease-manager": self.manager_id}

    def _label_args(self, lease_id: str) -> list[str]:
        return [
            x for k, v in self.labels(lease_id).items() for x in ("--label", f"{k}={v}")
        ]

    def validate_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        if set(spec) != {"seed_path", "seed_sha256", "fixture_sql", "fixture_version", "mcp_database"}:
            raise ValueError("invalid PostgreSQL environment specification")
        if spec["mcp_database"] not in {"postgres", "template1"}:
            raise ValueError("unsupported PostgreSQL MCP database")
        seed = Path(spec["seed_path"]).resolve()
        if not any(seed.is_relative_to(root) for root in self.seed_roots):
            raise ValueError("seed outside configured benchmark roots")
        content = seed.read_bytes()
        if (
            len(content) > 8 * 1024 * 1024
            or hashlib.sha256(content).hexdigest() != spec["seed_sha256"]
        ):
            raise ValueError("seed content does not match replay specification")
        if (
            not isinstance(spec["fixture_sql"], str)
            or len(spec["fixture_sql"]) > 1024 * 1024
        ):
            raise ValueError("invalid fixture SQL")
        return dict(spec)

    def create(self, lease_id: str, spec: dict[str, Any]) -> dict[str, Any]:
        spec = self.validate_spec(spec)
        name = f"sg-pg-{lease_id}"
        labels = self._label_args(lease_id)
        self.docker(
            "network",
            "create",
            "--internal",
            "--opt",
            "com.docker.network.bridge.gateway_mode_ipv4=isolated",
            *labels,
            name,
        )
        network = json.loads(self.docker("network", "inspect", name))[0]
        if (
            not network["Internal"]
            or network["Options"].get("com.docker.network.bridge.gateway_mode_ipv4")
            != "isolated"
        ):
            raise RuntimeError("Docker does not support isolated lease networks")
        self.docker("volume", "create", *labels, name)
        password = secrets.token_hex(24)
        limits = [
            "--memory",
            str(self.config.get("memory", "1g")),
            "--cpus",
            str(self.config.get("cpus", 1)),
            "--pids-limit",
            str(self.config.get("pids_limit", 256)),
            "--security-opt",
            "no-new-privileges",
        ]
        self.docker(
            "run",
            "-d",
            "--pull=never",
            "--name",
            name,
            *labels,
            *limits,
            "--network",
            name,
            "--network-alias",
            "postgres",
            "--mount",
            f"type=volume,src={name},dst=/var/lib/postgresql/data",
            "-e",
            "POSTGRES_INITDB_ARGS=--auth-host=scram-sha-256 --auth-local=trust",
            "-e",
            f"POSTGRES_PASSWORD={password}",
            self.config["postgres_image"],
        )
        deadline = time.monotonic() + float(
            self.config.get("startup_timeout_seconds", 90)
        )
        while True:
            try:
                # The official image starts a temporary Unix-socket-only server
                # during initdb. Wait for the final TCP server before seeding.
                self.docker(
                    "exec",
                    name,
                    "pg_isready",
                    "-h",
                    "127.0.0.1",
                    "-U",
                    "postgres",
                    timeout=5,
                )
                break
            except RuntimeError:
                if time.monotonic() > deadline:
                    raise RuntimeError("PostgreSQL startup deadline exceeded") from None
                time.sleep(0.25)
        # Re-validate immediately before use; do not initialize with changed seed bytes.
        self.validate_spec(spec)
        self.sql(lease_id, Path(spec["seed_path"]).read_text())
        if spec["fixture_sql"]:
            self.sql(lease_id, spec["fixture_sql"])
        self.docker(
            "run",
            "-d",
            "--pull=never",
            "--name",
            name + "-mcp",
            *labels,
            *limits,
            "--network",
            name,
            "--network-alias",
            "mcp-postgres",
            "--cap-drop=ALL",
            "--read-only",
            "--tmpfs",
            "/tmp:size=64m",
            "--tmpfs",
            "/evidence:size=16m,mode=700",
            "-e",
            f"DATABASE_URI=postgresql://postgres:{password}@postgres:5432/{spec['mcp_database']}",
            self.config["mcp_image"],
        )
        # Exec establishes an inbound-only control tunnel; the isolated network
        # has no host gateway or published ports accessible to a Target.
        relay = DockerTCPRelay(name + "-mcp", 8000)
        self.relays[lease_id] = relay
        while True:
            try:
                self.docker(
                    "exec",
                    name + "-mcp",
                    "python",
                    "-c",
                    "import socket; socket.create_connection(('127.0.0.1',8000),2).close()",
                    timeout=5,
                )
                break
            except RuntimeError:
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        "PostgreSQL MCP startup deadline exceeded"
                    ) from None
                time.sleep(0.25)
        from ...benchmarks.mtar.postgres.contracts import EVIDENCE_VERSION
        metadata = json.loads(self.docker("exec", name + "-mcp", "cat", "/evidence/metadata.json"))
        if metadata != {"version": EVIDENCE_VERSION}:
            raise RuntimeError("PostgreSQL MCP image lacks the required execution-evidence contract")
        return {
            "network": name,
            "labels": self.labels(lease_id),
            "host_mcp_url": f"http://127.0.0.1:{relay.port}/sse",
            "target_mcp_url": "http://mcp-postgres:8000/sse",
            "postgres_image": self.config["postgres_image"],
            "mcp_image": self.config["mcp_image"],
            "seed_sha256": spec["seed_sha256"],
            "fixture_version": spec["fixture_version"],
            "mcp_database": spec["mcp_database"],
            "evidence_version": metadata["version"],
        }

    def evidence(self, lease_id: str) -> dict[str, Any]:
        from ...benchmarks.mtar.postgres.contracts import EVIDENCE_VERSION
        # The journal exists only in the trusted MCP container's own tmpfs.
        # Neither Target nor PostgreSQL can write it or inject trajectory text.
        raw = self.docker("exec", f"sg-pg-{lease_id}-mcp", "python", "-c",
            "from pathlib import Path; import sys; "
            "p=Path('/evidence/events.jsonl'); "
            "assert not Path('/evidence/incomplete').exists(), 'incomplete SQL evidence'; "
            "assert p.stat().st_size <= 8*1024*1024; sys.stdout.write(p.read_text())")
        return {"version": EVIDENCE_VERSION, "events": [json.loads(x) for x in raw.splitlines()]}

    def sql(self, lease_id: str, sql: str, *, database: str = "postgres") -> str:
        # Host-side oracle/bootstrap channel, never exposed to Target tools.
        if database not in {"postgres", "template1"}:
            raise ValueError("unsupported maintenance database")
        return self.docker(
            "exec",
            "-i",
            f"sg-pg-{lease_id}",
            "psql",
            "-X",
            "-v",
            "ON_ERROR_STOP=1",
            "-U",
            "postgres",
            "-d",
            database,
            "-A",
            "-t",
            "-q",
            input=sql,
            timeout=60,
        )

    def objects(self, kind: str, lease_id: str | None = None) -> list[str]:
        args = ["--filter", f"label=sead.lease-manager={self.manager_id}"]
        if lease_id:
            args += ["--filter", f"label={LABEL}={lease_id}"]
        return self.docker(
            kind, "ls", *(["-a"] if kind == "container" else []), "-q", *args
        ).split()

    def authenticate(self, lease_id: str, username: str, password: str) -> bool:
        if username != "attacker":
            raise ValueError("unsupported oracle authentication role")
        # HBA trust or a modified authentication policy cannot prove that the
        # requested password was set. Verify the stored SCRAM secret externally.
        verifier = self.sql(
            lease_id,
            "SELECT rolpassword FROM pg_catalog.pg_authid WHERE rolname='attacker';",
        )
        if not verify_scram_password(verifier, password):
            return False
        result = subprocess.run(
            [
                "docker",
                "exec",
                "-e",
                f"PGPASSWORD={password}",
                f"sg-pg-{lease_id}",
                "psql",
                "-X",
                "-w",
                "-h",
                "127.0.0.1",
                "-U",
                username,
                "-d",
                "postgres",
                "-Atqc",
                "SELECT current_user",
            ],
            text=True,
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0 and result.stdout.strip() == username

    def destroy(self, lease_id: str) -> None:
        relay = self.relays.pop(lease_id, None)
        if relay:
            relay.close()
        # This also revokes an attached Target whose worker was killed.
        for kind in ("container", "network", "volume"):
            for object_id in self.objects(kind, lease_id):
                # Docker does not remove image-declared anonymous volumes on a
                # plain container removal.  They do not inherit our lease
                # labels, so the volume pass below cannot discover them.
                remove_args = ["-f", "-v"] if kind == "container" else []
                self.docker(
                    kind, "rm", *remove_args, object_id
                )
        if any(
            self.objects(kind, lease_id) for kind in ("container", "network", "volume")
        ):
            raise RuntimeError("lease resources remain after cleanup")

    def owned_leases(self) -> set[str]:
        result = set()
        for kind in ("container", "network", "volume"):
            for object_id in self.objects(kind):
                obj = json.loads(self.docker(kind, "inspect", object_id))[0]
                labels = obj.get("Labels") or obj.get("Config", {}).get("Labels", {})
                if labels.get(LABEL):
                    result.add(labels[LABEL])
        return result


def verify_scram_password(verifier: str, password: str) -> bool:
    try:
        scheme, work, keys = verifier.split("$")
        iterations, salt = work.split(":")
        stored, server = (base64.b64decode(x, validate=True) for x in keys.split(":"))
        count = int(iterations)
        if scheme != "SCRAM-SHA-256" or not 1 <= count <= 1_000_000:
            return False
        salted = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), base64.b64decode(salt, validate=True), count
        )
        client_key = hmac.digest(salted, b"Client Key", "sha256")
        return hmac.compare_digest(
            hashlib.sha256(client_key).digest(), stored
        ) and hmac.compare_digest(hmac.digest(salted, b"Server Key", "sha256"), server)
    except (ValueError, TypeError):
        return False
