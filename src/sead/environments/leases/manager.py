"""Single-host lease authority. SQLite is local durable state, not a shared lock."""

from __future__ import annotations

import fcntl
import json
import os
import secrets
import socketserver
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any


class LeaseError(RuntimeError):
    pass


class LeaseManager:
    def __init__(
        self,
        directory: Path,
        provider_factory,
        *,
        capacity: int = 2,
        ttl_seconds: float = 180,
        max_parallel_creates: int | None = None,
        startup_timeout_seconds: float = 900,
        config_digest: str | None = None,
        heartbeat_interval_seconds: float = 10,
        request_timeout_seconds: float = 60,
    ):
        if capacity < 1 or ttl_seconds <= 0:
            raise ValueError("lease capacity and TTL must be positive")
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.directory, 0o700)
        self.lock_file = (self.directory / "manager.lock").open("a+")
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock_file.close()
            raise
        self.capacity = capacity
        self.ttl = ttl_seconds
        self.startup_timeout = startup_timeout_seconds
        self.config_digest = config_digest
        self.heartbeat_interval = min(heartbeat_interval_seconds, ttl_seconds / 3)
        self.request_timeout = request_timeout_seconds
        self.create_slots = threading.BoundedSemaphore(max_parallel_creates or capacity)
        self.requests: dict[str, dict] = {}
        self.closing = False
        self.lock = threading.RLock()
        self.operation_locks: dict[str, threading.Lock] = {}
        self.db = sqlite3.connect(
            self.directory / "leases.sqlite3", check_same_thread=False
        )
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS leases (
                id TEXT PRIMARY KEY, request_id TEXT UNIQUE NOT NULL,
                generation TEXT NOT NULL, state TEXT NOT NULL, expires REAL NOT NULL,
                owner TEXT NOT NULL, spec TEXT NOT NULL, binding TEXT NOT NULL,
                error TEXT, created REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS waiting (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT UNIQUE NOT NULL, spec TEXT NOT NULL,
                owner TEXT NOT NULL, expires REAL NOT NULL
            );
        """)
        self.db.execute("DELETE FROM waiting")
        self.db.execute(
            "INSERT OR IGNORE INTO metadata VALUES ('manager_id', ?)",
            (uuid.uuid4().hex,),
        )
        self.db.commit()
        self.manager_id = self.db.execute(
            "SELECT value FROM metadata WHERE key='manager_id'"
        ).fetchone()[0]
        self.provider = provider_factory(self.manager_id)
        # After manager restart, relay sessions and heartbeats cannot be trusted.
        # Revoke existing leases before serving requests; never resume dirty state.
        ids = {
            row[0]
            for row in self.db.execute(
                "SELECT id FROM leases WHERE state != 'destroyed'"
            )
        }
        ids.update(self.provider.owned_leases())
        for lease_id in ids:
            self._destroy(lease_id)

    def _row(self, lease_id: str):
        row = self.db.execute("SELECT * FROM leases WHERE id=?", (lease_id,)).fetchone()
        if row is None:
            raise LeaseError("unknown lease")
        return row

    def _record(self, row) -> dict[str, Any]:
        return {
            "lease_id": row["id"],
            "generation": row["generation"],
            "state": row["state"],
            "expires": row["expires"],
            "ttl_seconds": self.ttl,
            "heartbeat_interval_seconds": self.heartbeat_interval,
            "owner": json.loads(row["owner"]),
            "binding": json.loads(row["binding"]),
            "error": row["error"],
        }

    def request(self, spec: dict, owner: dict, request_id: str) -> dict:
        """V2 nonblocking acquisition; keep v1 acquire semantics intact."""
        spec = self.provider.validate_spec(spec)
        if set(owner) != {"run_id", "task_id", "node_id", "replay_id"} or not request_id:
            raise LeaseError("invalid lease ownership")
        with self.lock:
            if self.closing:
                raise LeaseError("POOL_UNAVAILABLE")
            job = self.requests.get(request_id)
            if job is not None:
                if job["spec"] != spec or job["owner"] != owner:
                    raise LeaseError("request_id reused with different owner or specification")
                job["last_poll"] = time.monotonic()
                if job.get("error"):
                    raise LeaseError(job["error"])
                if job.get("record"):
                    record = job["record"]
                    row = self._check(record["lease_id"], record["generation"])
                    return {"state": "leased", "record": self._record(row)}
                return {"state": job["state"]}
            job = {"spec": spec, "owner": owner, "state": "queued", "cancel": threading.Event(),
                   "last_poll": time.monotonic()}
            # Preserve RPC arrival order, independently of thread scheduling.
            self.db.execute(
                "INSERT OR IGNORE INTO waiting(request_id,spec,owner,expires) VALUES (?,?,?,?)",
                (request_id, json.dumps(spec, sort_keys=True), json.dumps(owner, sort_keys=True), time.time() + 5),
            )
            self.db.commit()
            self.requests[request_id] = job

            def prepare():
                try:
                    # Existing acquire implements durable FIFO admission. Do not
                    # reserve a create slot while waiting for pool capacity.
                    while not job["cancel"].is_set():
                        if time.monotonic() - job["last_poll"] > self.request_timeout:
                            raise LeaseError("ACQUIRE_ABANDONED")
                        try:
                            record = self.acquire(spec, owner, request_id)
                        except LeaseError as exc:
                            if str(exc) not in {"CAPACITY", "PENDING"}:
                                raise
                            job["cancel"].wait(0.1)
                            continue
                        with self.lock:
                            cancelled = job["cancel"].is_set()
                            if not cancelled:
                                job["record"] = record
                                job["state"] = "leased"
                        if cancelled:
                            self.release(record["lease_id"], record["generation"])
                            raise LeaseError("ACQUIRE_CANCELLED")
                        return
                    raise LeaseError("ACQUIRE_CANCELLED")
                except Exception as exc:
                    with self.lock:
                        if isinstance(exc, LeaseError):
                            job["error"] = str(exc)
                        else:
                            # Provider errors are already credential-sanitized. Keep
                            # their bounded diagnostic message so Docker create and
                            # health-check failures are not collapsed together.
                            detail = str(exc).strip()
                            job["error"] = f"PREPARE_FAILED: {type(exc).__name__}"
                            if detail:
                                job["error"] += f": {detail[:512]}"
                finally:
                    self.cancel_wait(request_id)

            thread = threading.Thread(target=prepare, name=f"prepare-{request_id}")
            job["thread"] = thread
            thread.start()
            return {"state": "queued"}

    def cancel(self, request_id: str) -> dict:
        """Cancel queue/preparation, or revoke a result whose delivery raced cancellation."""
        with self.lock:
            job = self.requests.get(request_id)
            if job is None:
                self.cancel_wait(request_id)
                return {"state": "cancelled"}
            job["cancel"].set()
            record = job.get("record")
        if record:
            self.release(record["lease_id"], record["generation"])
        return {"state": "cancelling" if job["thread"].is_alive() else "cancelled"}

    def acquire(self, spec: dict, owner: dict, request_id: str) -> dict:
        spec = self.provider.validate_spec(spec)
        if not request_id or set(owner) != {
            "run_id",
            "task_id",
            "node_id",
            "replay_id",
        }:
            raise LeaseError("invalid lease ownership")
        encoded_spec, encoded_owner = (
            json.dumps(spec, sort_keys=True),
            json.dumps(owner, sort_keys=True),
        )
        with self.lock:
            if request_id in self.requests and self.requests[request_id]["cancel"].is_set():
                raise LeaseError("ACQUIRE_CANCELLED")
            existing = self.db.execute(
                "SELECT * FROM leases WHERE request_id=?", (request_id,)
            ).fetchone()
            if existing:
                if (
                    existing["spec"] != encoded_spec
                    or existing["owner"] != encoded_owner
                ):
                    raise LeaseError(
                        "request_id reused with different owner or specification"
                    )
                if existing["state"] == "leased" and existing["expires"] > time.time():
                    return self._record(existing)
                if existing["state"] == "preparing":
                    raise LeaseError("PENDING")
                raise LeaseError("request already ended; start an explicit new replay")
            count = self.db.execute(
                "SELECT count(*) FROM leases WHERE state != 'destroyed'"
            ).fetchone()[0]
            self.db.execute("DELETE FROM waiting WHERE expires <= ?", (time.time(),))
            waiting = self.db.execute(
                "SELECT * FROM waiting WHERE request_id=?", (request_id,)
            ).fetchone()
            if waiting and (
                waiting["spec"] != encoded_spec or waiting["owner"] != encoded_owner
            ):
                raise LeaseError(
                    "request_id reused with different owner or specification"
                )
            self.db.execute(
                "INSERT INTO waiting(request_id,spec,owner,expires) VALUES (?,?,?,?) "
                "ON CONFLICT(request_id) DO UPDATE SET expires=excluded.expires",
                (request_id, encoded_spec, encoded_owner, time.time() + 5),
            )
            first = self.db.execute(
                "SELECT request_id FROM waiting ORDER BY sequence LIMIT 1"
            ).fetchone()[0]
            self.db.commit()
            if count >= self.capacity or first != request_id:
                raise LeaseError("CAPACITY")
            if not self.create_slots.acquire(blocking=False):
                raise LeaseError("CAPACITY")
            self.db.execute("DELETE FROM waiting WHERE request_id=?", (request_id,))
            lease_id = uuid.uuid4().hex
            generation = secrets.token_hex(24)
            operation_lock = self.operation_locks.setdefault(lease_id, threading.Lock())
            operation_lock.acquire()
            self.db.execute(
                "INSERT INTO leases VALUES (?, ?, ?, 'preparing', ?, ?, ?, '{}', NULL, ?)",
                (
                    lease_id,
                    request_id,
                    generation,
                    time.time() + self.ttl,
                    encoded_owner,
                    encoded_spec,
                    time.time(),
                ),
            )
            self.db.commit()
        try:
            job = self.requests.get(request_id)
            if job:
                job["state"] = "preparing"
            cancellable = getattr(self.provider, "create_cancellable", None)
            binding = (cancellable(lease_id, spec, cancel=job["cancel"] if job else threading.Event(),
                                   deadline=time.monotonic() + self.startup_timeout)
                       if cancellable else self.provider.create(lease_id, spec))
            if job and job["cancel"].is_set():
                raise LeaseError("ACQUIRE_CANCELLED")
            with self.lock:
                self.db.execute(
                    "UPDATE leases SET state='leased', expires=?, binding=? WHERE id=?",
                    (time.time() + self.ttl, json.dumps(binding), lease_id),
                )
                self.db.commit()
                return self._record(self._row(lease_id))
        except BaseException:
            self._destroy(lease_id)
            raise
        finally:
            operation_lock.release()
            self.create_slots.release()

    def _check(self, lease_id: str, generation: str, *, active: bool = True):
        row = self._row(lease_id)
        if not secrets.compare_digest(row["generation"], generation):
            raise LeaseError("stale lease generation")
        if active and (row["state"] != "leased" or row["expires"] <= time.time()):
            raise LeaseError("lease is no longer active")
        return row

    def renew(self, lease_id: str, generation: str) -> dict:
        with self.lock:
            self._check(lease_id, generation)
            self.db.execute(
                "UPDATE leases SET expires=? WHERE id=?",
                (time.time() + self.ttl, lease_id),
            )
            self.db.commit()
            return self._record(self._row(lease_id))

    def cancel_wait(self, request_id: str) -> None:
        with self.lock:
            self.db.execute("DELETE FROM waiting WHERE request_id=?", (request_id,))
            self.db.commit()

    def _destroy(self, lease_id: str) -> None:
        with self.lock:
            self.db.execute(
                "UPDATE leases SET state='releasing' WHERE id=?", (lease_id,)
            )
            self.db.commit()
        try:
            self.provider.destroy(lease_id)
        except Exception:
            with self.lock:
                self.db.execute(
                    "UPDATE leases SET state='quarantined', error='resource cleanup failed' WHERE id=?",
                    (lease_id,),
                )
                self.db.commit()
            raise
        with self.lock:
            self.db.execute(
                "UPDATE leases SET state='destroyed', binding='{}', error=NULL WHERE id=?",
                (lease_id,),
            )
            self.db.commit()

    def release(self, lease_id: str, generation: str) -> dict:
        with self.lock:
            row = self._check(lease_id, generation, active=False)
            if row["state"] == "destroyed":
                return self._record(row)
            operation_lock = self.operation_locks.setdefault(lease_id, threading.Lock())
        with operation_lock:
            self._destroy(lease_id)
        with self.lock:
            return self._record(self._row(lease_id))

    def sql(
        self, lease_id: str, generation: str, sql: str, database: str = "postgres"
    ) -> str:
        with self.lock:
            self._check(lease_id, generation)
            operation_lock = self.operation_locks.setdefault(lease_id, threading.Lock())
        with operation_lock:
            with self.lock:
                self._check(lease_id, generation)
            return self.provider.sql(lease_id, sql, database=database)

    def inspect(self, owner: dict | None = None) -> dict:
        with self.lock:
            rows = [
                self._record(row)
                for row in self.db.execute("SELECT * FROM leases ORDER BY created")
            ]
        # Inspection is audit-only: do not return renewal capabilities or endpoints.
        rows = [
            {k: v for k, v in row.items() if k not in {"generation", "binding"}}
            for row in rows
            if not owner or all(row["owner"].get(k) == v for k, v in owner.items())
        ]
        return {
            "manager_id": self.manager_id,
            "capacity": self.capacity,
            "config_digest": self.config_digest,
            "leases": rows,
        }

    def evidence(self, lease_id: str, generation: str) -> dict:
        with self.lock:
            self._check(lease_id, generation)
            operation_lock = self.operation_locks.setdefault(lease_id, threading.Lock())
        with operation_lock:
            with self.lock:
                self._check(lease_id, generation)
            return self.provider.evidence(lease_id)

    def authenticate(
        self, lease_id: str, generation: str, username: str, password: str
    ) -> bool:
        with self.lock:
            self._check(lease_id, generation)
            operation_lock = self.operation_locks.setdefault(lease_id, threading.Lock())
        with operation_lock:
            with self.lock:
                self._check(lease_id, generation)
            return self.provider.authenticate(lease_id, username, password)

    def reap(self) -> None:
        with self.lock:
            for job in self.requests.values():
                if not job.get("record") and time.monotonic() - job["last_poll"] > self.request_timeout:
                    job["cancel"].set()
            ids = [
                row[0]
                for row in self.db.execute(
                    "SELECT id FROM leases WHERE state='quarantined' OR (state='leased' AND expires <= ?)",
                    (time.time(),),
                )
            ]
        for lease_id in ids:
            with self.lock:
                operation_lock = self.operation_locks.setdefault(
                    lease_id, threading.Lock()
                )
            if not operation_lock.acquire(blocking=False):
                continue
            try:
                with self.lock:
                    row = self._row(lease_id)
                    if row["state"] == "leased" and row["expires"] > time.time():
                        continue
                    if row["state"] == "destroyed":
                        continue
                    self.db.execute(
                        "UPDATE leases SET state='releasing' WHERE id=?", (lease_id,)
                    )
                    self.db.commit()
                self._destroy(lease_id)
            except Exception:
                pass  # Quarantined slots stay occupied and visible in inspect.
            finally:
                operation_lock.release()

    def close(self):
        with self.lock:
            self.closing = True
            jobs = list(self.requests.values())
            for job in jobs:
                job["cancel"].set()
        for job in jobs:
            job["thread"].join()
        with self.lock:
            ids = [
                row[0]
                for row in self.db.execute(
                    "SELECT id FROM leases WHERE state != 'destroyed'"
                )
            ]
        try:
            for lease_id in ids:
                self._destroy(lease_id)
        finally:
            self.db.close()
            self.lock_file.close()


def serve(manager: LeaseManager) -> None:
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.connection.settimeout(300)
            try:
                raw = self.rfile.readline(12 * 1024 * 1024 + 1)
                if len(raw) > 12 * 1024 * 1024:
                    raise LeaseError("request too large")
                request = json.loads(raw)
                operation = request["operation"]
                if operation not in {
                    "acquire",
                    "renew",
                    "release",
                    "inspect",
                    "sql",
                    "authenticate",
                    "evidence",
                    "cancel_wait",
                    "request",
                    "cancel",
                }:
                    raise LeaseError("unsupported operation")
                value = getattr(manager, operation)(**request["arguments"])
                result = {"ok": True, "value": value}
            except LeaseError as exc:
                result = {"ok": False, "error": str(exc)}
            except Exception as exc:
                result = {
                    "ok": False,
                    "error": f"lease operation failed: {type(exc).__name__}",
                }
            try:
                self.wfile.write(json.dumps(result).encode() + b"\n")
            except OSError:
                pass  # Unclaimed allocations expire and are reclaimed.

    class Server(socketserver.ThreadingUnixStreamServer):
        daemon_threads = False
        # socketserver defaults to five pending connections, which is too
        # small when several evaluation campaigns release their workers at
        # once.  Handler threads still enforce all lease capacity limits.
        request_queue_size = 256

        def service_actions(self):
            # Reaping must not hold up accepts or heartbeat requests.
            if not getattr(self, "reaper", None) or not self.reaper.is_alive():
                self.reaper = threading.Thread(target=manager.reap, daemon=True)
                self.reaper.start()

    socket_path = manager.directory / "manager.sock"
    socket_path.unlink(missing_ok=True)
    try:
        with Server(str(socket_path), Handler) as server:
            os.chmod(socket_path, 0o600)
            try:
                server.serve_forever(poll_interval=1)
            finally:
                if getattr(server, "reaper", None):
                    server.reaper.join()
    finally:
        socket_path.unlink(missing_ok=True)
        manager.close()
