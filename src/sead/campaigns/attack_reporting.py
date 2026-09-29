"""Durable dispatch accounting and semantic input identities for attack methods."""
import json
import os
import time
import uuid
from collections import Counter
from pathlib import Path

from sead.hashing import sha256_file, sha256_json


_EPHEMERAL_RUNTIME_KEYS = frozenset({
    "api_key", "api_key_env", "auth_token", "browser_port", "controller_url",
    "cuda_visible_devices", "endpoint", "job_id", "manager_socket", "port",
})


def semantic_runtime_value(value):
    """Remove allocation coordinates while retaining images and task bindings."""
    if isinstance(value, dict):
        return {key: semantic_runtime_value(item) for key, item in value.items()
                if key not in _EPHEMERAL_RUNTIME_KEYS}
    if isinstance(value, list):
        return [semantic_runtime_value(item) for item in value]
    return value


def _referenced_digest(path, *, semantic_runtime=False):
    if not semantic_runtime:
        return sha256_file(path)
    import yaml
    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return sha256_file(path)
    return sha256_json(semantic_runtime_value(value))


class UsageLedger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, value):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def start(self, kind, details=None):
        request_id = uuid.uuid4().hex
        self.append({"request_id": request_id, "request_kind": kind, "state": "dispatch_started",
                     "time": time.time(), **(details or {})})
        return request_id

    def finish(self, request_id, kind, metrics=None):
        self.append({"request_id": request_id, "request_kind": kind, "state": "completed",
                     "time": time.time(), "metrics": metrics or {}})

    def error(self, request_id, kind, exc, metrics=None):
        self.append({"request_id": request_id, "request_kind": kind, "state": "error",
                     "time": time.time(), "error": f"{type(exc).__name__}: {exc}",
                     "billing_status": "unknown", "metrics": metrics or {}})

    def totals(self):
        return ledger_totals(self.path)


def ledger_totals(path):
    totals = Counter()
    if not Path(path).exists():
        return {}
    for line in Path(path).read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue  # a killed process can leave a partial final line
        kind = row["request_kind"]
        if row["state"] == "dispatch_started":
            totals[kind + "_requests"] += 1
            if kind in {
            }:
                totals["controller_calls"] += 1
        elif row["state"] == "completed":
            for key, value in row.get("metrics", {}).items():
                if type(value) in {int, float}:
                    totals[kind + "_" + key] += value
        elif row["state"] == "error":
            totals[kind + "_errors"] += 1
            for key, value in row.get("metrics", {}).items():
                if type(value) in {int, float}:
                    totals[kind + "_" + key] += value
    return dict(totals)


def input_fingerprint(config, task_ids, dataset_root):
    """Hash semantic inputs. Ephemeral pool endpoint/auth are runtime records."""
    import copy
    value = copy.deepcopy(config)
    controller = value.get("controller", {})
    for name in ("endpoint", "api_key_env", "cuda_visible_devices"):
        controller.pop(name, None)
    value.pop("output", None)
    value["execution"] = semantic_runtime_value(value.get("execution", {}))
    execution = value["execution"]
    execution.pop("resolved_openhands_base_image", None)
    package = Path(__file__).resolve().parents[1]
    sources = {str(p.relative_to(package)): sha256_file(p)
               for folder in ("attacks", "environments", "campaigns", "config")
               for p in sorted((package / folder).rglob("*"))
               if p.is_file() and p.suffix in {".py", ".md", ".json"}}
    multiple_roots = isinstance(dataset_root, dict)
    roots = ({task_id: Path(dataset_root[task_id]).resolve() for task_id in task_ids}
             if multiple_roots else
             {task_id: Path(dataset_root).resolve() for task_id in task_ids})
    assets = ({
        f"{tid}/{p.relative_to(roots[tid] / 'tasks' / tid)}": sha256_file(p)
        for tid in task_ids
        for p in sorted((roots[tid] / "tasks" / tid).rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    } if multiple_roots else {
        str(p.relative_to(dataset_root)): sha256_file(p)
        for tid in task_ids
        for p in sorted((Path(dataset_root) / "tasks" / tid).rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    })
    # Referenced environment/Judge/runtime files are part of the inputs too.
    referenced = {}
    def visit(item):
        if isinstance(item, dict):
            for key, val in item.items():
                if isinstance(val, str) and key in {"judge_config", "registry", "runtime_registry", "service_deployments", "tac_pool_config"}:
                    path = Path(val)
                    if path.is_file():
                        referenced[str(path)] = _referenced_digest(
                            path, semantic_runtime=key in {"registry", "runtime_registry", "tac_pool_config"})
                visit(val)
        elif isinstance(item, list):
            for val in item:
                visit(val)
    visit(value)
    payload = {"config": value, "task_ids": task_ids, "sources": sources,
               "assets": assets, "referenced": referenced}
    if multiple_roots:
        payload["dataset_roots"] = {
            task_id: str(roots[task_id]) for task_id in task_ids
        }
    return sha256_json(payload)
