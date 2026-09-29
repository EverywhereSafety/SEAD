"""OpenHands worker backend for OAS DART tree-search paths."""

from __future__ import annotations

from sead.environments.workers import worker_path

from pathlib import Path
from typing import Any

from ..mtar.openhands_backend import MTAROpenHandsWorkerBackend


class OASOpenHandsWorkerBackend(MTAROpenHandsWorkerBackend):
    benchmark_kind = "oas"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault(
            "worker_script",
            worker_path("oas"),
        )
        super().__init__(**kwargs)


__all__ = ["OASOpenHandsWorkerBackend"]
