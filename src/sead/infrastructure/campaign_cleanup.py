"""Shared shutdown and optional orphan-volume cleanup for campaign entry points."""

from __future__ import annotations

import argparse
import signal
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path

from .docker_cleanup import cleanup_dangling_anonymous_volumes


class WorkerShutdown(BaseException):
    """Stop a replay without treating campaign shutdown as a retryable failure."""


@dataclass
class _Shutdown:
    requested: bool = False

    def check(self):
        if self.requested:
            raise WorkerShutdown("campaign is shutting down")


_active_shutdown: _Shutdown | None = None


def request_worker_shutdown() -> None:
    """Ask running replay threads to unwind through their cleanup finally blocks."""
    if _active_shutdown is not None:
        _active_shutdown.requested = True


def check_worker_shutdown() -> None:
    if _active_shutdown is not None:
        _active_shutdown.check()


def wait_for_worker(process, timeout: float) -> int:
    """Wait with a deadline while allowing the owning campaign to cancel workers."""
    shutdown = _active_shutdown
    if shutdown is None:
        return process.wait(timeout=timeout)
    deadline = time.monotonic() + timeout
    while True:
        shutdown.check()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout)
        try:
            return process.wait(timeout=min(0.25, remaining))
        except subprocess.TimeoutExpired:
            continue


@contextmanager
def campaign_cleanup():
    """Turn SIGINT/SIGTERM into cooperative worker teardown, then restore handlers.

    Worker subprocesses are reaped and Docker is cleaned by run_replay_worker.
    Repeated termination signals must not interrupt that teardown. This cannot
    intercept SIGKILL or survive a host failure.
    """
    global _active_shutdown
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("campaign cleanup must be installed by the main thread")
    if _active_shutdown is not None:
        yield
        return
    shutdown = _Shutdown()
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def interrupted(signum, _frame):
        if shutdown.requested:
            return
        shutdown.requested = True
        raise SystemExit(128 + signum)

    _active_shutdown = shutdown
    try:
        for sig in previous:
            signal.signal(sig, interrupted)
        yield
    finally:
        shutdown.requested = True
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        _active_shutdown = None


def with_campaign_cleanup(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        with campaign_cleanup():
            return function(*args, **kwargs)
    return guarded


def _timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("timestamp must include a timezone")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


def add_cleanup_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cleanup-dangling-anonymous-since", type=_timestamp,
        help="Explicitly remove daemon-wide dangling anonymous volumes created "
             "since this ISO timestamp; disabled by default. Requires a timezone.",
    )


def run_cleanup_preflight(args, output_dir: Path) -> dict | None:
    """Run only after validation/locking and before starting campaign resources."""
    since = getattr(args, "cleanup_dangling_anonymous_since", None)
    if not since:
        return None
    from sead.campaigns.infrastructure import atomic_write_json

    path = Path(output_dir) / "docker_cleanup_preflight.json"
    try:
        result = cleanup_dangling_anonymous_volumes(since)
    except Exception as exc:
        atomic_write_json(path, {
            "created_since": since, "succeeded": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
        })
        raise
    result = {**result, "succeeded": not result["errors"]}
    atomic_write_json(path, result)
    if not result["succeeded"]:
        raise RuntimeError(f"preflight Docker cleanup failed; see {path}")
    return result
