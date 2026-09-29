"""Explicit per-replay lease context; never changes global environment variables."""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from typing import Any, Mapping


def postgres_mode(execution: Mapping[str, Any]) -> str:
    config = execution.get("postgres", {})
    if not isinstance(config, Mapping):
        raise ValueError("execution.postgres must be a mapping")
    mode = config.get("mode", "legacy_shared")
    if mode not in {"legacy_shared", "leased"}:
        raise ValueError("unknown PostgreSQL environment mode")
    if (
        mode == "leased"
        and not Path(str(config.get("manager_socket", ""))).is_absolute()
    ):
        raise ValueError("leased PostgreSQL requires an absolute manager_socket")
    return str(mode)


class LeaseClient:
    def __init__(self, socket_path: str | Path, *, timeout: float = 240):
        self.path = str(socket_path)
        self.timeout = timeout

    def call(self, operation: str, **arguments):
        # A burst of replay workers can temporarily fill the Unix socket's
        # accept queue.  Linux reports that condition as EAGAIN; retrying for a
        # short bounded interval keeps a transient admission spike from
        # consuming a task-level technical retry.
        deadline = time.monotonic() + min(self.timeout, 5.0)
        while True:
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.settimeout(self.timeout)
            try:
                connection.connect(self.path)
                break
            except BlockingIOError:
                connection.close()
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
        with connection:
            connection.sendall(
                json.dumps({"operation": operation, "arguments": arguments}).encode()
                + b"\n"
            )
            with connection.makefile("rb") as stream:
                raw = stream.readline(16 * 1024 * 1024 + 1)
            if len(raw) > 16 * 1024 * 1024:
                raise RuntimeError("lease response too large")
            response = json.loads(raw)
            if not response["ok"]:
                raise RuntimeError(response["error"])
            return response["value"]

    def acquire(
        self, spec: dict, owner: dict, *, request_id: str, wait_seconds: float = 300
    ) -> LeaseHandle:
        deadline = time.monotonic() + wait_seconds
        while True:
            try:
                record = self.call(
                    "acquire", spec=spec, owner=owner, request_id=request_id
                )
                return LeaseHandle(self, record)
            except RuntimeError as exc:
                if str(exc) not in {"CAPACITY", "PENDING"}:
                    raise
                if time.monotonic() >= deadline:
                    self.call("cancel_wait", request_id=request_id)
                    raise TimeoutError(
                        "PostgreSQL lease acquisition deadline exceeded"
                    ) from None
                time.sleep(min(0.25, max(0, deadline - time.monotonic())))

    def acquire_v2(self, spec, owner, *, request_id, wait_seconds=300,
                   startup_seconds=900, poll_seconds=0.25):
        queued_at = time.monotonic()
        preparing_at = None
        try:
            while True:
                value = self.call("request", spec=spec, owner=owner, request_id=request_id)
                if value["state"] == "leased":
                    return LeaseHandle(self, value["record"])
                now = time.monotonic()
                if value["state"] == "preparing":
                    if preparing_at is None:
                        preparing_at = now
                    if now - preparing_at >= startup_seconds:
                        raise TimeoutError("PREPARE_TIMEOUT")
                elif now - queued_at >= wait_seconds:
                    raise TimeoutError("ACQUIRE_TIMEOUT")
                time.sleep(poll_seconds)
        except BaseException:
            try:
                self.call("cancel", request_id=request_id)
            except Exception:
                pass  # Parent recovery retries from the durable request audit.
            raise


class LeaseHandle:
    def __init__(self, client: LeaseClient, record: dict):
        self.client = client
        self.record = record
        self.binding = record["binding"]
        self.identity = {
            "lease_id": record["lease_id"],
            "generation": record["generation"],
        }
        self.stopping = threading.Event()
        self.lost: Exception | None = None
        self.released = False
        self.valid_until = time.monotonic() + max(0, record["expires"] - time.time())
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def _heartbeat(self):
        interval = min(self.record.get("heartbeat_interval_seconds", 10), self.record["ttl_seconds"] / 3)
        # A hung RPC cannot silently extend the local proof of lease ownership.
        heartbeat = LeaseClient(self.client.path, timeout=interval)
        while not self.stopping.wait(interval):
            try:
                started = time.monotonic()
                heartbeat.call("renew", **self.identity)
                self.valid_until = started + self.record["ttl_seconds"]
            except Exception as exc:
                self.lost = exc
                return

    def check(self):
        if self.lost is not None:
            raise RuntimeError("LEASE_LOST: environment lease lost") from self.lost
        if self.released:
            raise RuntimeError("LEASE_LOST: environment lease released")
        if time.monotonic() >= self.valid_until:
            raise RuntimeError("LEASE_LOST: local lease deadline exceeded")

    def sql(self, sql: str, *, database: str = "postgres") -> str:
        self.check()
        return self.client.call("sql", **self.identity, sql=sql, database=database)

    def audit(self) -> dict:
        return {
            "lease_id": self.identity["lease_id"],
            "owner": self.record["owner"],
            "environment_version": self.binding.get("environment_version", "postgres-leased-v1"),
            "binding": {
                k: v
                for k, v in self.binding.items()
                if k
                in {"seed_sha256", "fixture_version", "postgres_image", "mcp_image", "mcp_database", "evidence_version",
                    "service", "profile_id", "profile_digest", "image", "public_url", "pool_id"}
            },
        }

    def evidence(self) -> dict:
        self.check()
        return self.client.call("evidence", **self.identity)

    def close(self):
        if self.released:
            return
        self.stopping.set()
        self.thread.join(timeout=15)
        result = self.client.call("release", **self.identity)
        if result["state"] != "destroyed":
            raise RuntimeError("environment lease cleanup incomplete")
        self.released = True

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
