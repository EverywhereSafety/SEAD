"""Packaged worker entry points shared by DART execution and SAGE defense."""
from pathlib import Path


def worker_path(kind: str) -> Path:
    if kind not in {"mtar", "oas"}:
        raise ValueError(f"unsupported worker kind: {kind}")
    return Path(__file__).with_name(f"{kind}.py")
