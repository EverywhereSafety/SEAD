"""Durable state, locking, and bounded concurrency shared by campaigns."""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Iterable, Mapping, TypeVar

_Result = TypeVar("_Result")
_TaskId = TypeVar("_TaskId", bound=str)


def utc_now() -> str:
    """Return an auditable UTC timestamp."""

    return datetime.now(UTC).isoformat()


def task_slug(task_id: str) -> str:
    """Map an external task ID to one safe, stable path component."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "-", task_id).strip("-.") or "task"


def atomic_write_json(path: Path | str, value: Any) -> None:
    """Atomically replace one JSON artifact on the destination filesystem."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def read_json_object(path: Path | str) -> dict[str, Any] | None:
    """Read an optional JSON object while rejecting ambiguous non-objects."""

    source = Path(path)
    if not source.is_file():
        return None
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact must be an object: {source}")
    return value


def acquire_campaign_lock(
    campaign_dir: Path | str,
    *,
    filename: str = "campaign.lock",
):
    """Acquire a non-blocking, process-scoped lock for one campaign directory."""

    root = Path(campaign_dir)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / filename
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"another campaign runner holds {lock_path}") from None
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} started_at={utc_now()}\n")
    handle.flush()
    return handle


class CampaignExecutor:
    """A bounded thread pool with a common first-completion interface."""

    def __init__(self, max_workers: int, *, thread_name_prefix: str) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.max_workers = int(max_workers)
        self.thread_name_prefix = str(thread_name_prefix)
        self._executor: ThreadPoolExecutor | None = None

    def __enter__(self) -> CampaignExecutor:
        if self._executor is not None:
            raise RuntimeError("campaign executor is already running")
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix=self.thread_name_prefix,
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        assert self._executor is not None
        if exc_type is not None:
            from sead.infrastructure.campaign_cleanup import request_worker_shutdown

            request_worker_shutdown()
        self._executor.shutdown(wait=True, cancel_futures=exc_type is not None)
        self._executor = None

    def submit(
        self,
        function: Callable[..., _Result],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Future[_Result]:
        if self._executor is None:
            raise RuntimeError("campaign executor is not running")
        return self._executor.submit(function, *args, **kwargs)

    @staticmethod
    def first_completed(futures: Iterable[Future[_Result]]) -> set[Future[_Result]]:
        pending = tuple(futures)
        if not pending:
            raise ValueError("at least one future is required")
        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
        return completed


def resource_aware_results(
    executor: CampaignExecutor,
    task_ids: Iterable[_TaskId],
    resources_by_task: Mapping[_TaskId, Iterable[str]],
    run_task: Callable[[_TaskId], _Result],
    *,
    resource_capacities: Mapping[str, int] | None = None,
    worker_limit: Callable[[], int] | None = None,
    retry_delay: Callable[[_TaskId, _Result], float | None] | None = None,
    initial_delays: Mapping[_TaskId, float] | None = None,
) -> Iterable[_Result]:
    """Admit tasks within resource capacities and optional live worker capacity.

    After the caller consumes each result, retry_delay may return a delay in
    seconds to append that task to the queue, or None to finish it. Delayed
    tasks hold neither a worker nor resources. Retry policy belongs to the caller.
    """

    capacities = dict(resource_capacities or {})
    if any(type(value) is not int or value < 1 for value in capacities.values()):
        raise ValueError("resource capacities must be positive integers")
    pending = list(task_ids)

    def ready_at(delay: float) -> float:
        if not isinstance(delay, (int, float)) or not math.isfinite(delay) or delay < 0:
            raise ValueError("task delay must be finite and non-negative")
        return time.monotonic() + delay

    eligible_at = {task: ready_at(delay) for task, delay in (initial_delays or {}).items()}
    running: dict[Future[_Result], tuple[_TaskId, frozenset[str]]] = {}
    resources_in_use: Counter[str] = Counter()
    while pending or running:
        limit = min(executor.max_workers, worker_limit()) if worker_limit else executor.max_workers
        if type(limit) is not int or limit < 0:
            raise ValueError("worker limit must be a non-negative integer")
        admission_time = time.monotonic()
        made_progress = True
        while pending and len(running) < limit and made_progress:
            made_progress = False
            for task_id in list(pending):
                if eligible_at.get(task_id, 0) > admission_time:
                    continue
                required = frozenset(resources_by_task.get(task_id, ()))
                if any(resources_in_use[key] >= capacities.get(key, 1) for key in required):
                    continue
                pending.remove(task_id)
                eligible_at.pop(task_id, None)
                resources_in_use.update(required)
                running[executor.submit(run_task, task_id)] = (task_id, required)
                made_progress = True
                if len(running) >= limit:
                    break
        now = time.monotonic()
        delays = [max(0, eligible_at[task] - now) for task in pending
                  if eligible_at.get(task, 0) > admission_time]
        timeout = min(delays) if delays else None
        if worker_limit is not None:
            timeout = min(timeout, 1) if timeout is not None else 1
        if not running:
            if limit == 0 and worker_limit is not None:
                # A rolling pool can temporarily have no healthy replicas.
                threading.Event().wait(1)
                continue
            if delays:
                threading.Event().wait(timeout)
                continue
            raise RuntimeError(
                "resource-aware scheduler cannot run remaining tasks: "
                + ", ".join(pending)
            )
        completed = (
            wait(running, timeout=timeout, return_when=FIRST_COMPLETED)[0]
            if timeout is not None else executor.first_completed(running)
        )
        for future in completed:
            task_id, required = running.pop(future)
            resources_in_use.subtract(required)
            result = future.result()
            yield result
            delay = retry_delay(task_id, result) if retry_delay is not None else None
            if delay is not None:
                eligible_at[task_id] = ready_at(delay)
                pending.append(task_id)
