"""Persistent subprocess bridge for SGLang SEAD controller models."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


class ControllerBackendError(RuntimeError):
    """Raised when the controller worker or its JSON-lines protocol fails."""


@dataclass(frozen=True)
class RequestOptions:
    """Per-request generation settings; never mutate a shared model backend."""

    temperature: float
    top_p: float
    max_output_tokens: int
    seed: int | None = None

    def __post_init__(self):
        if not 0 <= self.temperature <= 2 or not 0 < self.top_p <= 1:
            raise ValueError("invalid request sampling options")
        if type(self.max_output_tokens) is not int or self.max_output_tokens < 1:
            raise ValueError("request token cap must be a positive integer")
        if self.seed is not None and (type(self.seed) is not int or self.seed < 0):
            raise ValueError("request seed must be a non-negative integer")


@dataclass
class _PendingControllerRequest:
    request_id: int
    system: str
    user: str
    client_id: str | None = None
    request_kind: str = "candidate"
    event: threading.Event = field(default_factory=threading.Event)
    response: dict[str, Any] | None = None
    error: BaseException | None = None


class SGLangControllerClient:
    """Per-task usage/accounting view over one shared Controller worker."""

    def __init__(self, backend: Any, client_id: str):
        self._backend = backend
        self.client_id = client_id
        self.ready_info = dict(backend.ready_info)
        self.stderr_path = backend.stderr_path
        self.shared_across_tasks = True
        self.call_index = 0
        self.usage_records: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def complete(
        self,
        system: str,
        user: str,
        *,
        request_kind: str = "candidate",
        options: RequestOptions | None = None,
    ) -> str:
        response = self._backend._submit(
            system,
            user,
            client_id=self.client_id,
            request_kind=request_kind,
            **({"options": options} if options is not None else {}),
        )
        text = self._backend._response_text(response)
        metrics = response.get("metrics")
        with self._lock:
            self.call_index += 1
            call_index = self.call_index
            if isinstance(metrics, dict):
                self.usage_records.append(
                    {
                        "controller_call": call_index,
                        "shared_request_id": response["request_id"],
                        **(
                            {"request_kind": request_kind}
                            if request_kind != "candidate"
                            else {}
                        ),
                        **metrics,
                    }
                )
        return text


class SGLangSubprocessBackend:
    """Send Controller requests to one worker-owned persistent SGLang server."""

    def __init__(
        self,
        *,
        python_executable: Path | str,
        worker_script: Path | str,
        model: str,
        temperature: float,
        top_p: float,
        max_output_tokens: int,
        enable_thinking: bool,
        remove_thinking: bool,
        stderr_path: Path | str,
        context_length: int | None = None,
        environment: Mapping[str, str] | None = None,
        error_type: type[RuntimeError] = ControllerBackendError,
        max_batch_size: int = 1,
        batch_wait_ms: float = 0,
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
        mem_fraction_static: float = 0.82,
        max_running_requests: int = 16,
        max_inflight_batches: int = 4,
    ):
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if batch_wait_ms < 0:
            raise ValueError("batch_wait_ms must be non-negative")
        if max_inflight_batches < 1:
            raise ValueError("max_inflight_batches must be positive")
        self._error_type = error_type
        self.max_batch_size = int(max_batch_size)
        self.batch_wait_ms = float(batch_wait_ms)
        self.max_inflight_batches = int(max_inflight_batches)
        # Keep a virtualenv's bin/python path instead of resolving its symlink.
        self.python_executable = Path(python_executable).absolute()
        self.worker_script = Path(worker_script).resolve()
        self.stderr_path = Path(stderr_path).resolve()
        for required in (
            self.python_executable,
            self.worker_script,
        ):
            if not required.exists():
                self._fail(f"SGLang controller dependency missing: {required}")
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr = self.stderr_path.open("w")
        command = [
            str(self.python_executable),
            str(self.worker_script),
            "--model",
            model,
            "--temperature",
            str(temperature),
            "--top-p",
            str(top_p),
            "--max-output-tokens",
            str(max_output_tokens),
            "--enable-thinking",
            str(enable_thinking).lower(),
            "--remove-thinking",
            str(remove_thinking).lower(),
            "--tensor-parallel-size",
            str(tensor_parallel_size),
            "--data-parallel-size",
            str(data_parallel_size),
            "--mem-fraction-static",
            str(mem_fraction_static),
            "--max-running-requests",
            str(max_running_requests),
        ]
        if context_length is not None:
            if context_length < 1:
                raise ValueError("context_length must be positive")
            command.extend(("--context-length", str(context_length)))
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                text=True,
                bufsize=1,
                env=dict(os.environ if environment is None else environment),
            )
            ready = self._read_response()
        except BaseException:
            process = getattr(self, "process", None)
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            self._stderr.close()
            raise
        if ready.get("status") != "ready":
            self.close()
            self._fail(f"SGLang controller worker did not become ready: {ready}")
        self.ready_info = {
            **dict(ready),
            "max_batch_size": self.max_batch_size,
            "batch_wait_ms": self.batch_wait_ms,
            "max_inflight_batches": self.max_inflight_batches,
        }
        self.call_index = 0
        self.usage_records: list[dict[str, Any]] = []
        self.shared_usage_records: list[dict[str, Any]] = []
        self._state_lock = threading.RLock()
        self._next_request_id = 0
        self._scheduler_batch_index = 0
        self._closed = False
        self._fatal_error: BaseException | None = None
        self._inflight_by_request: dict[
            int, tuple[_PendingControllerRequest, int, int]
        ] = {}
        self._batch_remaining: dict[int, int] = {}
        self._inflight_slots = threading.BoundedSemaphore(self.max_inflight_batches)
        self._inflight_condition = threading.Condition(self._state_lock)
        self._request_queue: queue.Queue[_PendingControllerRequest | None] = (
            queue.Queue()
        )
        self._reader = threading.Thread(
            target=self._read_responses,
            name="dart-controller-reader",
            daemon=True,
        )
        self._dispatcher = threading.Thread(
            target=self._dispatch_requests,
            name="dart-controller-dispatcher",
            daemon=True,
        )
        self._reader.start()
        self._dispatcher.start()

    def _fail(self, message: str) -> None:
        raise self._error_type(message)

    def _read_response(self) -> dict[str, Any]:
        if self.process.stdout is None:
            self._fail("SGLang controller stdout is unavailable")
        line = self.process.stdout.readline()
        if not line:
            return_code = self.process.poll()
            self._fail(
                "SGLang controller worker exited without a response "
                f"(return_code={return_code}); see {self.stderr_path}"
            )
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise self._error_type(
                f"Invalid SGLang controller protocol response: {line[:500]}"
            ) from exc
        if not isinstance(value, dict):
            self._fail("SGLang controller response must be an object")
        return value

    def _write_request(self, value: Mapping[str, Any]) -> None:
        if self.process.stdin is None:
            self._fail("SGLang controller stdin is unavailable")
        try:
            self.process.stdin.write(json.dumps(value, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
        except BrokenPipeError as exc:
            raise self._error_type(
                f"SGLang controller worker pipe closed; see {self.stderr_path}"
            ) from exc

    def _response_text(self, response: Mapping[str, Any]) -> str:
        if response.get("error"):
            self._fail(f"SGLang controller generation failed: {response['error']}")
        value = response.get("text")
        if not isinstance(value, str):
            self._fail("SGLang controller returned no text")
        return value

    def _send_batch(self, pending: list[_PendingControllerRequest]) -> None:
        self._inflight_slots.acquire()
        with self._state_lock:
            if self._fatal_error is not None:
                self._inflight_slots.release()
                raise self._fatal_error
            self._scheduler_batch_index += 1
            scheduler_batch_id = self._scheduler_batch_index
            self._batch_remaining[scheduler_batch_id] = len(pending)
            for item in pending:
                self._inflight_by_request[item.request_id] = (
                    item,
                    scheduler_batch_id,
                    len(pending),
                )
        if len(pending) == 1:
            item = pending[0]
            self._write_request(
                {
                    "request_id": item.request_id,
                    "system": item.system,
                    "user": item.user,
                    "request_kind": item.request_kind,
                }
            )
        else:
            self._write_request(
                {
                    "requests": [
                        {
                            "request_id": item.request_id,
                            "system": item.system,
                            "user": item.user,
                            "request_kind": item.request_kind,
                        }
                        for item in pending
                    ]
                }
            )

    def _set_fatal_error(self, exc: BaseException) -> None:
        with self._inflight_condition:
            if self._fatal_error is None:
                self._fatal_error = exc
            pending = [value[0] for value in self._inflight_by_request.values()]
            batch_count = len(self._batch_remaining)
            self._inflight_by_request.clear()
            self._batch_remaining.clear()
            self._inflight_condition.notify_all()
        for _ in range(batch_count):
            self._inflight_slots.release()
        for item in pending:
            item.error = exc
            item.event.set()

    def _read_responses(self) -> None:
        while True:
            try:
                envelope = self._read_response()
            except BaseException as exc:
                with self._state_lock:
                    closing = self._closed and not self._inflight_by_request
                if not closing:
                    self._set_fatal_error(exc)
                return
            responses = envelope.get("responses")
            if responses is None:
                responses = [envelope]
            if not isinstance(responses, list) or not all(
                isinstance(response, dict) for response in responses
            ):
                self._set_fatal_error(
                    self._error_type(
                        "SGLang controller batch response must contain responses"
                    )
                )
                return
            for response in responses:
                request_id = response.get("request_id")
                with self._inflight_condition:
                    entry = self._inflight_by_request.pop(request_id, None)
                    if entry is None:
                        self._set_fatal_error(
                            self._error_type(
                                "SGLang controller protocol request/response mismatch"
                            )
                        )
                        return
                    item, scheduler_batch_id, scheduler_batch_size = entry
                    remaining = self._batch_remaining[scheduler_batch_id] - 1
                    if remaining:
                        self._batch_remaining[scheduler_batch_id] = remaining
                    else:
                        del self._batch_remaining[scheduler_batch_id]
                        self._inflight_slots.release()
                    metrics = response.get("metrics")
                    if isinstance(metrics, dict):
                        scheduler_metrics = {
                            "scheduler_batch_id": scheduler_batch_id,
                            "scheduler_batch_size": scheduler_batch_size,
                            **metrics,
                        }
                        response["metrics"] = scheduler_metrics
                        self.shared_usage_records.append(
                            {
                                "shared_request_id": item.request_id,
                                "client_id": item.client_id,
                                "request_kind": item.request_kind,
                                **scheduler_metrics,
                            }
                        )
                    item.response = response
                    if not self._inflight_by_request:
                        self._inflight_condition.notify_all()
                item.event.set()

    def _dispatch_requests(self) -> None:
        stop_after_batch = False
        while not stop_after_batch:
            first = self._request_queue.get()
            if first is None:
                return
            pending = [first]
            if self.max_batch_size > 1:
                deadline = time.monotonic() + (self.batch_wait_ms / 1000)
                while len(pending) < self.max_batch_size:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        item = self._request_queue.get(timeout=remaining)
                    except queue.Empty:
                        break
                    if item is None:
                        stop_after_batch = True
                        break
                    pending.append(item)
            try:
                self._send_batch(pending)
            except BaseException as exc:
                self._set_fatal_error(exc)
                for item in pending:
                    if not item.event.is_set():
                        item.error = exc
                        item.event.set()
            if stop_after_batch:
                return
            with self._state_lock:
                fatal_error = self._fatal_error
            if fatal_error is not None:
                while True:
                    try:
                        queued = self._request_queue.get_nowait()
                    except queue.Empty:
                        return
                    if queued is None:
                        return
                    queued.error = fatal_error
                    queued.event.set()

    def _submit(
        self,
        system: str,
        user: str,
        *,
        client_id: str | None = None,
        request_kind: str = "candidate",
    ) -> dict[str, Any]:
        with self._state_lock:
            if self._closed:
                self._fail("SGLang controller backend is closed")
            if self._fatal_error is not None:
                raise self._fatal_error
            self._next_request_id += 1
            request_id = self._next_request_id
        pending = _PendingControllerRequest(
            request_id,
            system,
            user,
            client_id,
            request_kind=request_kind,
        )
        self._request_queue.put(pending)
        pending.event.wait()
        if pending.error is not None:
            raise pending.error
        if pending.response is None:
            self._fail("SGLang controller worker returned no response")
        return pending.response

    def create_client(self, client_id: str) -> SGLangControllerClient:
        return SGLangControllerClient(self, client_id)

    def complete(
        self,
        system: str,
        user: str,
        *,
        request_kind: str = "candidate",
    ) -> str:
        response = self._submit(system, user, request_kind=request_kind)
        with self._state_lock:
            self.call_index += 1
            call_index = self.call_index
        if response.get("request_id") is None:
            self._fail("SGLang controller protocol request/response mismatch")
        text = self._response_text(response)
        metrics = response.get("metrics")
        if isinstance(metrics, dict):
            self.usage_records.append(
                {
                    "controller_call": call_index,
                    **(
                        {"request_kind": request_kind}
                        if request_kind != "candidate"
                        else {}
                    ),
                    **metrics,
                }
            )
        return text

    def close(self) -> None:
        dispatcher = getattr(self, "_dispatcher", None)
        reader = getattr(self, "_reader", None)
        state_lock = getattr(self, "_state_lock", None)
        if state_lock is not None:
            with state_lock:
                already_closed = self._closed
                self._closed = True
            if not already_closed:
                self._request_queue.put(None)
                dispatcher.join(timeout=30)
                condition = self._inflight_condition
                deadline = time.monotonic() + 600
                with condition:
                    while self._inflight_by_request and self._fatal_error is None:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            self._set_fatal_error(
                                self._error_type(
                                    "Timed out waiting for Controller requests to finish"
                                )
                            )
                            break
                        condition.wait(timeout=remaining)
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            if process.stdin is not None:
                try:
                    process.stdin.write('{"command":"close"}\n')
                    process.stdin.flush()
                    process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if reader is not None:
            reader.join(timeout=10)
        stderr = getattr(self, "_stderr", None)
        if stderr is not None and not stderr.closed:
            stderr.close()
        stdout = getattr(process, "stdout", None)
        if stdout is not None and not stdout.closed:
            stdout.close()

    def __enter__(self) -> "SGLangSubprocessBackend":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class SGLangHTTPBackend:
    """Concurrent client for a long-lived OpenAI-compatible SGLang endpoint."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        temperature: float,
        top_p: float,
        max_output_tokens: int,
        enable_thinking: bool,
        remove_thinking: bool,
        api_key: str | None = None,
        request_timeout_seconds: float = 600,
        max_inflight_requests: int = 16,
        error_type: type[RuntimeError] = ControllerBackendError,
    ) -> None:
        from urllib.parse import urlsplit

        endpoint = endpoint.strip().rstrip("/")
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("SGLang HTTP endpoint must be an HTTP(S) URL")
        if max_output_tokens < 1 or max_inflight_requests < 1:
            raise ValueError("Controller token and in-flight limits must be positive")
        if request_timeout_seconds <= 0:
            raise ValueError("Controller request timeout must be positive")
        self.endpoint = endpoint if endpoint.endswith("/v1") else endpoint + "/v1"
        self.model = model
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_output_tokens = int(max_output_tokens)
        self.enable_thinking = bool(enable_thinking)
        self.remove_thinking = bool(remove_thinking)
        self.api_key = api_key
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.max_inflight_requests = int(max_inflight_requests)
        self._error_type = error_type
        self._semaphore = threading.BoundedSemaphore(self.max_inflight_requests)
        self._lock = threading.Lock()
        self._closed = False
        self._next_request_id = 0
        self.call_index = 0
        self.usage_records: list[dict[str, Any]] = []
        self.shared_usage_records: list[dict[str, Any]] = []
        self.shared_across_tasks = True
        self.stderr_path = None
        self.ready_info = {
            "status": "ready",
            "backend": "sglang_http_external",
            "model": self.model,
            "server_url": self.endpoint,
            "max_output_tokens": self.max_output_tokens,
            "max_inflight_requests": self.max_inflight_requests,
            "owns_server": False,
        }

    def _fail(self, message: str) -> None:
        raise self._error_type(message)

    def _response_text(self, response: Mapping[str, Any]) -> str:
        if response.get("error"):
            self._fail(f"SGLang controller generation failed: {response['error']}")
        value = response.get("text")
        if not isinstance(value, str):
            self._fail("SGLang controller returned no text")
        return value

    def _post(
        self,
        payload: Mapping[str, Any],
        *,
        request_id: int,
        client_id: str | None,
        request_kind: str,
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        # A direct SGLang server ignores these headers.  A model gateway
        # uses them for task affinity and secret-free request provenance.
        headers["X-SEAD-Request-ID"] = str(request_id)
        headers["X-SEAD-Request-Kind"] = request_kind
        if client_id:
            headers["X-SEAD-Client-ID"] = client_id
        request = urllib.request.Request(
            self.endpoint + "/chat/completions",
            data=json.dumps(dict(payload)).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.request_timeout_seconds
            ) as response:
                value = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self._fail(f"SGLang HTTP {exc.code}: {detail[:1000]}")
        except (OSError, json.JSONDecodeError) as exc:
            self._fail(f"SGLang HTTP request failed: {type(exc).__name__}: {exc}")
        if not isinstance(value, dict):
            self._fail("SGLang HTTP response must be an object")
        return value

    @staticmethod
    def _choice_text(value: Mapping[str, Any]) -> tuple[str, str | None]:
        choices = value.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ControllerBackendError("SGLang response has no choices")
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, Mapping) else None
        text = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(text, str):
            raise ControllerBackendError("SGLang response has no assistant content")
        finish_reason = (
            choice.get("finish_reason") if isinstance(choice, Mapping) else None
        )
        return text, None if finish_reason is None else str(finish_reason)

    def _submit(
        self,
        system: str,
        user: str,
        *,
        client_id: str | None = None,
        request_kind: str = "candidate",
        options: RequestOptions | None = None,
    ) -> dict[str, Any]:
        queue_started = time.perf_counter()
        self._semaphore.acquire()
        queue_seconds = time.perf_counter() - queue_started
        try:
            with self._lock:
                if self._closed:
                    self._fail("SGLang controller backend is closed")
                self._next_request_id += 1
                request_id = self._next_request_id
            started = time.perf_counter()
            value = self._post(
                {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": options.max_output_tokens if options else self.max_output_tokens,
                    "temperature": options.temperature if options else self.temperature,
                    "top_p": options.top_p if options else self.top_p,
                    **({"seed": options.seed} if options and options.seed is not None else {}),
                    "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
                    **(
                        {"response_format": {"type": "json_schema", "json_schema": {
                            "name": "benign_user_message", "strict": True,
                            "schema": {"type": "object", "additionalProperties": False,
                                       "required": ["instruction"], "properties": {
                                           "instruction": {"type": "string", "minLength": 1}
                                       }}
                        }}}
                        if request_kind == "benign_user" else {}
                    ),
                },
                request_id=request_id,
                client_id=client_id,
                request_kind=request_kind,
            )
            generation_seconds = time.perf_counter() - started
            text_value, finish_reason = self._choice_text(value)
            if self.remove_thinking:
                import re

                text_value = re.sub(
                    r"<think>.*?</think>",
                    "",
                    text_value,
                    flags=re.DOTALL | re.IGNORECASE,
                ).strip()
            usage = (
                value.get("usage") if isinstance(value.get("usage"), Mapping) else {}
            )
            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")
            metrics = {
                "queue_seconds": queue_seconds,
                "generation_seconds": generation_seconds,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "tokens_per_second": (
                    output_tokens / generation_seconds
                    if isinstance(output_tokens, int) and generation_seconds > 0
                    else None
                ),
                "finish_reason": finish_reason,
                "backend": "sglang_http_external",
            }
            with self._lock:
                self.shared_usage_records.append(
                    {
                        "shared_request_id": request_id,
                        "client_id": client_id,
                        "request_kind": request_kind,
                        **metrics,
                    }
                )
            return {
                "request_id": request_id,
                "text": text_value,
                "error": None,
                "metrics": metrics,
            }
        finally:
            self._semaphore.release()

    def create_client(self, client_id: str) -> SGLangControllerClient:
        return SGLangControllerClient(self, client_id)

    def complete(
        self, system: str, user: str, *, request_kind: str = "candidate"
    ) -> str:
        response = self._submit(system, user, request_kind=request_kind)
        text_value = self._response_text(response)
        with self._lock:
            self.call_index += 1
            call_index = self.call_index
            metrics = response.get("metrics")
            if isinstance(metrics, dict):
                self.usage_records.append(
                    {
                        "controller_call": call_index,
                        "request_kind": request_kind,
                        **metrics,
                    }
                )
        return text_value

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def __enter__(self) -> "SGLangHTTPBackend":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class OpenAIControllerBackend:
    """OpenAI-v1 Controller client with the same accounting contract as SGLang.

    This backend is primarily useful for small integration experiments where a
    managed local Controller is unavailable. It is deliberately explicit in
    the run manifest so it cannot be mistaken for the standard DART Controller.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        model: str,
        temperature: float,
        top_p: float,
        max_output_tokens: int,
        reasoning_effort: str | None = None,
        request_timeout_seconds: float = 600,
        max_inflight_requests: int = 16,
        remove_thinking: bool = True,
        error_type: type[RuntimeError] = ControllerBackendError,
        client: Any | None = None,
    ) -> None:
        from urllib.parse import urlsplit

        endpoint = endpoint.strip().rstrip("/")
        parsed = urlsplit(endpoint)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("OpenAI Controller endpoint must be an HTTPS URL")
        if not api_key:
            raise ValueError("OpenAI Controller API key is required")
        if max_output_tokens < 1 or max_inflight_requests < 1:
            raise ValueError("Controller token and in-flight limits must be positive")
        if request_timeout_seconds <= 0:
            raise ValueError("Controller request timeout must be positive")
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=endpoint,
                timeout=request_timeout_seconds,
                max_retries=1,
            )
        self.client = client
        self.endpoint = endpoint
        self.model = model
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_output_tokens = int(max_output_tokens)
        self.reasoning_effort = reasoning_effort
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.max_inflight_requests = int(max_inflight_requests)
        self.remove_thinking = bool(remove_thinking)
        self._error_type = error_type
        self._semaphore = threading.BoundedSemaphore(self.max_inflight_requests)
        self._lock = threading.Lock()
        self._closed = False
        self._next_request_id = 0
        self.call_index = 0
        self.usage_records: list[dict[str, Any]] = []
        self.shared_usage_records: list[dict[str, Any]] = []
        self.shared_across_tasks = True
        self.stderr_path = None
        self.ready_info = {
            "status": "ready",
            "backend": "openai_v1",
            "model": self.model,
            "server_url": self.endpoint,
            "max_output_tokens": self.max_output_tokens,
            "max_inflight_requests": self.max_inflight_requests,
            "owns_server": False,
        }

    def _fail(self, message: str) -> None:
        raise self._error_type(message)

    def _response_text(self, response: Mapping[str, Any]) -> str:
        if response.get("error"):
            self._fail(f"OpenAI Controller generation failed: {response['error']}")
        value = response.get("text")
        if not isinstance(value, str):
            self._fail("OpenAI Controller returned no text")
        return value

    def _submit(
        self,
        system: str,
        user: str,
        *,
        client_id: str | None = None,
        request_kind: str = "candidate",
    ) -> dict[str, Any]:
        queue_started = time.perf_counter()
        self._semaphore.acquire()
        queue_seconds = time.perf_counter() - queue_started
        try:
            with self._lock:
                if self._closed:
                    self._fail("OpenAI Controller backend is closed")
                self._next_request_id += 1
                request_id = self._next_request_id
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_completion_tokens": self.max_output_tokens,
                "temperature": self.temperature,
            }
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
            started = time.perf_counter()
            try:
                value = self.client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 - provider SDK boundary
                self._fail(
                    f"OpenAI Controller request failed: {type(exc).__name__}: {exc}"
                )
            generation_seconds = time.perf_counter() - started
            choices = getattr(value, "choices", None)
            if not choices:
                self._fail("OpenAI Controller response has no choices")
            choice = choices[0]
            text_value = str(getattr(choice.message, "content", None) or "")
            if self.remove_thinking:
                import re

                text_value = re.sub(
                    r"<think>.*?</think>",
                    "",
                    text_value,
                    flags=re.DOTALL | re.IGNORECASE,
                ).strip()
            usage = getattr(value, "usage", None)
            if hasattr(usage, "model_dump"):
                usage = usage.model_dump(exclude_none=True)
            usage = dict(usage) if isinstance(usage, Mapping) else {}
            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")
            metrics = {
                "queue_seconds": queue_seconds,
                "generation_seconds": generation_seconds,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": usage.get("total_tokens"),
                "tokens_per_second": (
                    output_tokens / generation_seconds
                    if isinstance(output_tokens, int) and generation_seconds > 0
                    else None
                ),
                "finish_reason": getattr(choice, "finish_reason", None),
                "backend": "openai_v1",
            }
            with self._lock:
                self.shared_usage_records.append(
                    {
                        "shared_request_id": request_id,
                        "client_id": client_id,
                        "request_kind": request_kind,
                        **metrics,
                    }
                )
            return {
                "request_id": request_id,
                "text": text_value,
                "error": None,
                "metrics": metrics,
            }
        finally:
            self._semaphore.release()

    def create_client(self, client_id: str) -> SGLangControllerClient:
        return SGLangControllerClient(self, client_id)

    def complete(
        self, system: str, user: str, *, request_kind: str = "candidate"
    ) -> str:
        response = self._submit(system, user, request_kind=request_kind)
        text_value = self._response_text(response)
        with self._lock:
            self.call_index += 1
            call_index = self.call_index
            metrics = response.get("metrics")
            if isinstance(metrics, dict):
                self.usage_records.append(
                    {
                        "controller_call": call_index,
                        "request_kind": request_kind,
                        **metrics,
                    }
                )
        return text_value

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def __enter__(self) -> "OpenAIControllerBackend":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
