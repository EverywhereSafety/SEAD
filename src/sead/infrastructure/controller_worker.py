#!/usr/bin/env python3
"""JSON-lines bridge to one persistent SGLang Controller server.

The worker owns the SGLang server lifecycle and exposes the small synchronous
protocol used by SEAD.  Controller inference has no Transformers/HF
``generate`` fallback: initialization fails if SGLang cannot become ready.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_CONTEXT_LENGTH = 65536
DEFAULT_MEM_FRACTION_STATIC = 0.82
DEFAULT_MAX_RUNNING_REQUESTS = 16
SERVER_STARTUP_TIMEOUT_SECONDS = 600
EMIT_LOCK = threading.Lock()


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {value}")


def emit(value: dict[str, Any]) -> None:
    with EMIT_LOCK:
        sys.stdout.write(json.dumps(value, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def _free_local_port() -> int:
    # SGLang derives an internal gRPC port by adding 20,000 to the HTTP port,
    # so an arbitrary kernel ephemeral port can overflow 65,535.
    for _ in range(100):
        candidate = 20_000 + secrets.randbelow(25_000)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            try:
                listener.bind(("127.0.0.1", candidate))
            except OSError:
                continue
            return candidate
    raise RuntimeError("Could not reserve a safe local SGLang port")


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _http_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 300,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"SGLang HTTP {exc.code}: {detail[:1000]}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("SGLang response must be a JSON object")
    return value


def _wait_until_ready(process: subprocess.Popen[Any], base_url: str) -> None:
    deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT_SECONDS
    last_error = "server has not responded"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"SGLang server exited during startup (return_code={return_code})"
            )
        try:
            with urllib.request.urlopen(
                f"{base_url}/health_generate", timeout=10
            ) as response:
                if response.status == 200:
                    return
        except (OSError, RuntimeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.5)
    raise TimeoutError(
        f"SGLang server did not become ready within "
        f"{SERVER_STARTUP_TIMEOUT_SECONDS}s: {last_error}"
    )


def _stop_server(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _response_text(value: dict[str, Any]) -> str:
    choices = value.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("SGLang response has no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    text = message.get("content") if isinstance(message, dict) else None
    if not isinstance(text, str):
        raise RuntimeError("SGLang response has no assistant content")
    return text


def _strip_thinking(text: str) -> str:
    return re.sub(
        r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="huihui-ai/Huihui-Qwen3.8-27B-abliterated"
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS
    )
    parser.add_argument("--enable-thinking", type=parse_bool, default=False)
    parser.add_argument("--remove-thinking", type=parse_bool, default=True)
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument(
        "--mem-fraction-static", type=float, default=DEFAULT_MEM_FRACTION_STATIC
    )
    parser.add_argument(
        "--max-running-requests", type=int, default=DEFAULT_MAX_RUNNING_REQUESTS
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    server: subprocess.Popen[Any] | None = None
    try:
        if _installed_version("sglang") is None:
            raise RuntimeError(
                f"sglang is not installed in Controller interpreter {sys.executable}"
            )
        port = _free_local_port()
        base_url = f"http://127.0.0.1:{port}"
        environment = os.environ.copy()
        executable_bin = str(Path(sys.executable).absolute().parent)
        environment["PATH"] = executable_bin + os.pathsep + environment.get("PATH", "")
        if Path("/usr/local/cuda-12.9/bin").is_dir():
            environment["PATH"] = (
                "/usr/local/cuda-12.9/bin" + os.pathsep + environment["PATH"]
            )
            environment.setdefault("CUDA_HOME", "/usr/local/cuda-12.9")
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            args.model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--dtype",
            "bfloat16",
            "--context-length",
            str(args.context_length),
            "--mem-fraction-static",
            str(args.mem_fraction_static),
            "--max-running-requests",
            str(args.max_running_requests),
            "--tp-size",
            str(args.tensor_parallel_size),
            "--dp-size",
            str(args.data_parallel_size),
            "--log-level-http",
            "warning",
        ]
        server = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=environment,
        )
        _wait_until_ready(server, base_url)
    except BaseException as exc:
        logging.exception("Could not initialize SGLang controller")
        _stop_server(server)
        emit({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        return 1

    emit(
        {
            "status": "ready",
            "model": args.model,
            "backend": "sglang_http",
            "sglang_version": _installed_version("sglang"),
            "batch_protocol": "jsonl-concurrent-batch-v2",
            "supports_batch": True,
            "supports_multiple_inflight_batches": True,
            "max_output_tokens": args.max_output_tokens,
            "context_length": args.context_length,
            "max_running_requests": args.max_running_requests,
            "tensor_parallel_size": args.tensor_parallel_size,
            "data_parallel_size": args.data_parallel_size,
            "mem_fraction_static": args.mem_fraction_static,
            "server_url": base_url,
            "hf_generate_fallback": False,
        }
    )

    def complete(request: dict[str, Any]) -> dict[str, Any]:
        system = request.get("system")
        user = request.get("user")
        if not isinstance(system, str) or not isinstance(user, str):
            raise ValueError("system and user must be strings")
        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": args.max_output_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
        }
        started = time.perf_counter()
        value = _http_json(
            f"{base_url}/v1/chat/completions", payload=payload, timeout=600
        )
        seconds = time.perf_counter() - started
        text = _response_text(value)
        if args.remove_thinking:
            text = _strip_thinking(text)
        usage = value.get("usage") if isinstance(value.get("usage"), dict) else {}
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        return {
            "request_id": request.get("request_id"),
            "text": text,
            "error": None,
            "request_kind": request.get("request_kind", "candidate"),
            "metrics": {
                "generation_seconds": seconds,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "tokens_per_second": (
                    output_tokens / seconds
                    if isinstance(output_tokens, int) and seconds > 0
                    else None
                ),
                "finish_reason": value["choices"][0].get("finish_reason"),
                "backend": "sglang_http",
            },
        }

    completion_executor = ThreadPoolExecutor(
        max_workers=args.max_running_requests,
        thread_name_prefix="sglang-controller-request",
    )
    envelope_executor = ThreadPoolExecutor(
        max_workers=args.max_running_requests,
        thread_name_prefix="sglang-controller-envelope",
    )

    def error_response(request: Any, exc: BaseException) -> dict[str, Any]:
        return {
            "request_id": (
                request.get("request_id") if isinstance(request, dict) else None
            ),
            "text": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    def handle_envelope(request: dict[str, Any]) -> None:
        try:
            batch_requests = request.get("requests")
            if batch_requests is None:
                emit(completion_executor.submit(complete, request).result())
                return
            if not isinstance(batch_requests, list) or not batch_requests:
                raise ValueError("requests must be a non-empty list")
            if not all(isinstance(item, dict) for item in batch_requests):
                raise ValueError("each batch request must be an object")
            started = time.perf_counter()
            futures = [
                completion_executor.submit(complete, item)
                for item in batch_requests
            ]
            results = []
            for item, future in zip(batch_requests, futures, strict=True):
                try:
                    results.append(future.result())
                except BaseException as exc:
                    traceback.print_exc(file=sys.stderr)
                    results.append(error_response(item, exc))
            batch_seconds = time.perf_counter() - started
            batch_output_tokens = sum(
                int(item.get("metrics", {}).get("output_tokens") or 0)
                for item in results
            )
            for item in results:
                metrics = item.get("metrics")
                if not isinstance(metrics, dict):
                    continue
                metrics.update(
                    {
                        "batch_size": len(results),
                        "batch_output_tokens": batch_output_tokens,
                        "batch_generation_seconds": batch_seconds,
                        "batch_tokens_per_second": (
                            batch_output_tokens / batch_seconds
                            if batch_seconds > 0
                            else None
                        ),
                    }
                )
            emit({"responses": results})
        except BaseException as exc:
            traceback.print_exc(file=sys.stderr)
            batch_requests = request.get("requests")
            if isinstance(batch_requests, list) and batch_requests:
                emit(
                    {
                        "responses": [
                            error_response(item, exc) for item in batch_requests
                        ]
                    }
                )
            else:
                emit(error_response(request, exc))

    try:
        for line in sys.stdin:
            request: dict[str, Any] | None = None
            try:
                request = json.loads(line)
                if request.get("command") == "close":
                    break
                envelope_executor.submit(handle_envelope, request)
            except BaseException as exc:
                traceback.print_exc(file=sys.stderr)
                emit(error_response(request, exc))
    finally:
        envelope_executor.shutdown(wait=True)
        completion_executor.shutdown(wait=True)
        _stop_server(server)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
