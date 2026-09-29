"""Resolve explicit configuration references once, relative to their source file."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml


class UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def read_yaml(path):
    value = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=UniqueLoader)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


PATH_KEYS = frozenset({
    "dataset_root", "openhands_root", "worker_python", "python",
    "service_deployments", "service_deployments_path", "tac_pool_config",
    "registry", "profiles", "state_directory", "manager_socket", "model_path",
    "model_manifest", "manifest", "snapshot", "directory", "selection_path",
    "candidate_index_path", "oas_selection_path", "oas_candidate_index_path",
    "runtime_registry", "fixture_manifest",
    "judge_config", "deployment_ref", "catalog_path", "source_root", "tool_catalog_cache",
    "collection", "scheduling_config",
})

SCHEDULING_KEYS = frozenset({
    "max_workers", "service_task_limits", "tool_family_task_limits", "web_task_limit",
    "workers_per_controller_replica", "max_technical_attempts",
    "technical_retry_backoff_seconds", "controller_batch_size",
    "controller_batch_wait_ms",
})


def resolve_paths(value, base):
    if isinstance(value, dict):
        return {
            key: ([os.path.abspath(base / path) for path in item]
                  if key in {"seed_roots", "workspace_roots"} and isinstance(item, list) else
                  os.path.abspath(base / item)
                  if key in PATH_KEYS and isinstance(item, str) and item
                  and not os.path.isabs(item) else resolve_paths(item, base))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [resolve_paths(item, base) for item in value]
    return value


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _scheduling_suite(value):
    if not isinstance(value, dict) or set(value) - SCHEDULING_KEYS:
        raise ValueError("unknown scheduling configuration keys")
    for key in (
        "max_workers",
        "max_technical_attempts",
        "workers_per_controller_replica",
        "controller_batch_size",
    ):
        if key in value and (type(value[key]) is not int or value[key] < 1):
            raise ValueError(f"scheduling.{key} must be a positive integer")
    renames = {
        "max_workers": "max_concurrent_tasks",
        "web_task_limit": "max_concurrent_web_tasks",
        "workers_per_controller_replica": "workers_per_controller",
    }
    suite = {
        renames.get(key, key): item
        for key, item in value.items()
        if key != "max_technical_attempts"
    }
    if "max_technical_attempts" in value:
        suite["max_technical_retries"] = int(value["max_technical_attempts"]) - 1
    return suite


def load_scheduling_config(path):
    """Load the shared scheduling schema for legacy campaign configurations."""

    path = Path(path).absolute()
    value = read_yaml(path)
    if value.pop("schema_version", None) != "sead-scheduling-v1":
        raise ValueError(f"expected sead-scheduling-v1: {path}")
    return _scheduling_suite(resolve_paths(value, path.parent))


def load_config_data(path):
    """Accept legacy DART YAML and the shared experiment schema.

    Returns existing runner fields so adoption does not change replay/scoring
    contracts. Paths are absolute before any per-task snapshot is produced.
    """
    path = Path(path).absolute()
    raw = read_yaml(path)
    if raw.get("schema_version") != "sead-experiment-v1":
        value = resolve_paths(raw, path.parent)
        scheduling_path = value.pop("scheduling_config", None)
        if scheduling_path is not None:
            suite = value.get("suite", {})
            if not isinstance(suite, dict):
                raise ValueError("suite must be an object")
            value["suite"] = merge(load_scheduling_config(scheduling_path), suite)
        return freeze_environments(value)
    allowed = {"schema_version", "kind", "config_refs", "benchmark", "roles",
               "defense", "search", "evaluation", "output", "overrides", "attack"}
    if set(raw) - allowed:
        raise ValueError(f"unknown experiment keys: {sorted(set(raw) - allowed)}")
    if raw.get("kind") not in {"dart", "attack", "defender_online"}:
        raise ValueError("unknown experiment kind")
    refs = raw.get("config_refs", {})
    if not isinstance(refs, dict) or set(refs) - {"models", "inference", "execution", "scheduling"}:
        raise ValueError("invalid config_refs")
    sections = {}
    for section, relative in refs.items():
        source = (path.parent / relative).absolute()
        value = read_yaml(source)
        expected = {"models": "sead-model-catalog-v1",
                    "inference": "sead-inference-catalog-v1",
                    "execution": "sead-execution-v1",
                    "scheduling": "sead-scheduling-v1"}[section]
        if value.pop("schema_version", None) != expected or "config_refs" in value:
            raise ValueError(f"expected {expected}: {source}")
        sections[section] = resolve_paths(value, source.parent)
    raw = resolve_paths(raw, path.parent)
    sections = merge(sections, raw.get("overrides", {}))
    allowed_sections = {
        "models": {"models"}, "inference": {"profiles"},
        "execution": {"worker", "environments", "timeouts", "max_steps", "render_tree_html", "judge_config"},
        "scheduling": SCHEDULING_KEYS,
    }
    if set(sections) - set(allowed_sections):
        raise ValueError("unknown override section")
    for name, section in sections.items():
        if not isinstance(section, dict) or set(section) - allowed_sections[name]:
            raise ValueError(f"unknown {name} configuration keys")
    result = {"schema_version": "dart-tree-search-v4", "experiment_kind": raw["kind"],
              "benchmark": raw.get("benchmark", {}), "search": raw.get("search", {}),
              "evaluation": raw.get("evaluation", {}), "output": raw.get("output", {})}
    if "attack" in raw:
        result["attack"] = raw["attack"]
    execution = sections.get("execution", {})
    worker = execution.pop("worker", {})
    if set(worker) - {"python", "openhands_root"}:
        raise ValueError("unknown execution.worker keys")
    timeouts = execution.get("timeouts", {})
    if set(timeouts) - {"replay_seconds", "prefix_seconds", "tool_seconds", "cleanup_seconds"}:
        raise ValueError("unknown execution.timeouts keys")
    for key, value in timeouts.items():
        if isinstance(value, bool) or not 0 < float(value) < float("inf"):
            raise ValueError(f"invalid execution timeout: {key}")
    result["benchmark"].update({
        ("worker_python" if k == "python" else k): v for k, v in worker.items()
    })
    for name, field in {"replay_seconds": "sample_timeout_seconds",
                        "prefix_seconds": "prefix_replay_timeout_seconds",
                        "tool_seconds": "tool_timeout_seconds",
                        "cleanup_seconds": "cleanup_timeout_seconds"}.items():
        if name in execution.get("timeouts", {}):
            execution[field] = execution["timeouts"][name]
    execution.pop("timeouts", None)
    result["execution"] = execution
    scheduling = sections.get("scheduling", {})
    result["suite"] = _scheduling_suite(scheduling)
    roles = raw.get("roles", {})
    for role, reference in roles.items():
        if role not in {"controller", "target", "defender", "judge"}:
            raise ValueError(f"unknown model role: {role}")
        if not isinstance(reference, dict) or set(reference) != {"model_ref", "inference_ref"}:
            raise ValueError(f"invalid model role reference: {role}")
        try:
            model = sections["models"]["models"][reference["model_ref"]]
            inference = sections["inference"]["profiles"][reference["inference_ref"]]
        except KeyError as exc:
            raise ValueError(f"unresolved {role} model/inference reference") from exc
        if set(model) & set(inference):
            raise ValueError(f"model and inference profile overlap for {role}")
        settings = merge(model, inference)
        if role == "controller" and settings.get("backend") == "http_chat":
            settings["backend"] = "sglang_http"
        if role == "defender":
            settings["timeout_seconds"] = settings.pop("request_timeout_seconds", 300)
            settings["max_tokens_per_step"] = settings.pop("max_output_tokens", 1024)
        result[role] = settings
    defense = raw.get("defense", {"mode": "disabled"})
    mode = defense.get("mode", "disabled")
    if mode not in {"disabled", "sage"}:
        raise ValueError("unknown defender mode")
    result["defender_mode"] = mode
    tool = {**result.pop("defender", {}), **{k: v for k, v in defense.items() if k != "mode"}}
    tool.update(enabled=mode != "disabled", type="sage")
    if mode == "sage":
        tool["environment_investigation"] = {**tool.get("environment_investigation", {}), "enabled": True}
    result["defense"] = {"tool": tool}
    return freeze_environments(result)


def freeze_environments(value):
    from sead.environments.registry import environment_config, EnvironmentRegistry
    execution = value.get("execution", {})
    config = environment_config(execution)
    if config:
        registry = EnvironmentRegistry(config["registry"])
        if config.get("registry_digest", registry.digest) != registry.digest:
            raise ValueError("environment registry changed since configuration was frozen")
        execution["environments"] = {**config, "registry_digest": registry.digest}
    return value
