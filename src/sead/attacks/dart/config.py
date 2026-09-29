"""Strict loader for deterministic DART v3 configuration."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml
from sead.config import load_config_data

from .models import SCHEMA_VERSION, TreeSearchConfig

DEFAULT_CONTROLLER_MODEL = "huihui-ai/Huihui-Qwen3.8-27B-abliterated"


@dataclass(frozen=True)
class UnifiedDARTConfig:
    path: Path
    benchmark: Mapping[str, Any]
    controller: Mapping[str, Any]
    target: Mapping[str, Any]
    execution: Mapping[str, Any]
    search: TreeSearchConfig
    defense: Mapping[str, Any] = field(default_factory=dict)
    judge: Mapping[str, Any] = field(default_factory=dict)
    attack: Mapping[str, Any] = field(default_factory=dict)


def with_defense_config(
    config: UnifiedDARTConfig, path: Path | str
) -> UnifiedDARTConfig:
    """Replace only defense settings; benchmark paths keep their original base."""

    from ...defenses import validate_online_defense_config

    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, Mapping) or set(raw) != {"defense"}:
        raise ValueError("defender configuration must contain only a defense section")
    return replace(config, defense=validate_online_defense_config(raw["defense"]))


def _mapping(raw: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    value = raw.get(section)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing configuration section: {section}")  # noqa: TRY004
    return value


def validate_target_temperature(
    target: Mapping[str, Any], *, label: str = "target"
) -> None:
    """Validate benchmark temperatures, including supported model defaults."""

    model = str(target.get("model") or "").rsplit("/", 1)[-1].casefold()
    if target.get("temperature") is None:
        if target.get("provider") == "anthropic_foundry" and model == "claude-sonnet-5":
            return
        raise ValueError(f"{label}.temperature may be null only for claude-sonnet-5")
    temperature = float(target.get("temperature", 0))
    if model in {"gpt-5.6-luna", "gpt-5.6-terra"}:
        if temperature != 1:
            raise ValueError(f"{label}.temperature must be 1 for {model}")
    elif target.get("provider") == "gemini" and model in {
        "gemini-3.5-flash", "gemini-3.8-flash",
    }:
        # Keep zero-temperature runs valid while allowing the same temperature
        # as Luna/Terra for this comparison. Gemini does not require a fixed 1.
        if temperature not in {0, 1}:
            raise ValueError(f"{label}.temperature must be 0 or 1 for {model}")
    elif temperature != 0:
        raise ValueError(f"{label}.temperature must be 0")


def load_tree_search_config(
    path: Path | str,
    *,
    benchmark_override: Mapping[str, Any] | None = None,
) -> UnifiedDARTConfig:
    path = Path(path).resolve()
    raw = load_config_data(path)
    if not isinstance(raw, Mapping) or raw.get("schema_version") not in {
        "dart-tree-search-v3",
        SCHEMA_VERSION,
    }:
        raise ValueError(
            f"expected schema_version={SCHEMA_VERSION} "
            "(legacy dart-tree-search-v3 configs are also accepted)"
        )
    benchmark = dict(_mapping(raw, "benchmark"))
    if benchmark_override:
        benchmark.update(benchmark_override)
    attack = raw.get("attack", {})
    if not isinstance(attack, Mapping):
        raise ValueError("attack must be an object")
    method = attack.get("name", "dart")
    if method != "dart":
        raise ValueError("attack.name must be dart")
    if set(attack) - {"name"}:
        raise ValueError("unknown DART attack configuration fields")
    controller = _mapping(raw, "controller")
    target = _mapping(raw, "target")
    search = _mapping(raw, "search")
    execution_value = raw.get("execution", {})
    if not isinstance(execution_value, Mapping):
        raise ValueError("execution must be an object")  # noqa: TRY004
    execution = dict(execution_value)
    from ...environments.registry import environment_config
    environment_config(execution)
    from ...environments.leases import postgres_mode
    postgres_mode(execution)
    defense_value = raw.get("defense", {})
    if not isinstance(defense_value, Mapping):
        raise ValueError("defense must be an object")  # noqa: TRY004
    from ...defenses import validate_online_defense_config

    defense = validate_online_defense_config(defense_value)
    render_tree_html = execution.get("render_tree_html", False)
    if not isinstance(render_tree_html, bool):
        raise ValueError("execution.render_tree_html must be Boolean")
    execution["render_tree_html"] = render_tree_html
    kind = str(benchmark.get("kind") or "")
    if kind not in {"mtar", "oas"}:
        raise ValueError("benchmark.kind must be mtar or oas")
    if not str(benchmark.get("task_id") or "").strip():
        raise ValueError("benchmark.task_id is required")
    collection = benchmark.get("collection")
    if collection is None and benchmark.get("group") is not None:
        raise ValueError("benchmark.group requires benchmark.collection")
    if collection is not None:
        if kind != "mtar":
            raise ValueError("benchmark.collection is supported only for MTAR")
        group = benchmark.get("group")
        if not isinstance(group, str) or not group.strip():
            raise ValueError("benchmark.group is required with benchmark.collection")
        from ...benchmarks.mtar.collection import resolve_collection

        resolved = resolve_collection(Path(str(collection)), group)
        selected_root = resolved.task(str(benchmark["task_id"])).dataset_root
        configured_root = benchmark.get("dataset_root")
        if configured_root is not None and Path(str(configured_root)).resolve() != selected_root:
            raise ValueError(
                "benchmark.dataset_root disagrees with the selected collection task source"
            )
        benchmark["dataset_root"] = str(selected_root)
    controller = dict(controller)
    if not str(controller.get("model") or "").strip():
        controller["model"] = DEFAULT_CONTROLLER_MODEL
    controller_backend = str(controller.get("backend") or "sglang_subprocess")
    if controller_backend not in {
        "sglang_subprocess",
        "sglang_http",
        "openai_v1",
    }:
        raise ValueError(
            "controller.backend must be sglang_subprocess, sglang_http, or openai_v1"
        )
    if (
        controller_backend == "sglang_subprocess"
        and not str(controller.get("python") or "").strip()
    ):
        raise ValueError("controller.python is required for sglang_subprocess")
    if controller_backend == "sglang_http":
        endpoint = str(controller.get("endpoint") or "").strip()
        if not re.match(r"^https?://[^/\s]+", endpoint):
            raise ValueError(
                "controller.endpoint must be an HTTP(S) URL for sglang_http"
            )
        api_key_env = controller.get("api_key_env")
        if api_key_env is not None and not re.fullmatch(
            r"[A-Z_][A-Z0-9_]*", str(api_key_env)
        ):
            raise ValueError(
                "controller.api_key_env must be an uppercase environment variable name"
            )
    if controller_backend == "openai_v1":
        endpoint = str(controller.get("endpoint") or "").strip()
        if not endpoint.startswith("https://"):
            raise ValueError("controller.endpoint must be an HTTPS OpenAI v1 URL")
        api_key_env = str(controller.get("api_key_env") or "")
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", api_key_env):
            raise ValueError(
                "controller.api_key_env must be an uppercase environment variable name"
            )
    max_output_tokens = int(controller.get("max_output_tokens", 1024))
    if max_output_tokens < 1:
        raise ValueError("controller.max_output_tokens must be positive")
    if int(controller.get("context_length", 65536)) < 1:
        raise ValueError("controller.context_length must be positive")
    for name, default in (
        ("tensor_parallel_size", 1),
        ("data_parallel_size", 1),
        ("max_running_requests", 16),
        ("max_inflight_requests", 16),
        ("max_inflight_batches", 4),
    ):
        if int(controller.get(name, default)) < 1:
            raise ValueError(f"controller.{name} must be positive")
    mem_fraction_static = float(controller.get("mem_fraction_static", 0.82))
    if not 0 < mem_fraction_static < 1:
        raise ValueError("controller.mem_fraction_static must be between 0 and 1")
    if controller_backend == "sglang_subprocess":
        devices = controller.get("cuda_visible_devices")
        if devices is not None:
            visible_count = len(
                [item for item in str(devices).split(",") if item.strip()]
            )
            requested_count = int(controller.get("tensor_parallel_size", 1)) * int(
                controller.get("data_parallel_size", 1)
            )
            if visible_count != requested_count:
                raise ValueError(
                    "controller.cuda_visible_devices count must equal "
                    "tensor_parallel_size * data_parallel_size"
                )
    if int(controller.get("max_retries", 1)) not in {0, 1}:
        raise ValueError("controller.max_retries must be 0 or 1")
    if not str(target.get("model") or "").strip():
        raise ValueError("target.model is required")
    target_provider = str(target.get("provider") or "")
    if target_provider not in {
        "anthropic_foundry", "azure_openai_v1", "gemini",
    }:
        raise ValueError(
            "target.provider must be anthropic_foundry, azure_openai_v1, or gemini"
        )
    if target_provider == "anthropic_foundry":
        if kind != "mtar":
            raise ValueError("anthropic_foundry Target is currently supported only for MTAR")
        endpoint = str(target.get("endpoint") or "").strip()
        if not endpoint.startswith("https://") or not endpoint.rstrip("/").endswith(
            "/anthropic"
        ):
            raise ValueError(
                "target.endpoint must be a complete HTTPS Anthropic Foundry "
                "base URL ending in /anthropic"
            )
        api_key_env = str(target.get("api_key_env") or "")
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", api_key_env):
            raise ValueError(
                "target.api_key_env must be an uppercase environment variable name"
            )
        if "/" in str(target["model"]):
            raise ValueError("Anthropic Foundry target.model must be a deployment name")
    if target_provider == "azure_openai_v1":
        endpoint = str(target.get("endpoint") or "").strip()
        if not endpoint.startswith("https://") or not endpoint.rstrip("/").endswith(
            "/openai/v1"
        ):
            raise ValueError(
                "target.endpoint must be a complete HTTPS OpenAI v1 base URL "
                "ending in /openai/v1"
            )
        api_key_env = str(target.get("api_key_env") or "")
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", api_key_env):
            raise ValueError(
                "target.api_key_env must be an uppercase environment variable name"
            )
    if target_provider == "gemini":
        if not str(target.get("model") or "").startswith("gemini/"):
            raise ValueError("Gemini target.model must start with gemini/")
        if str(target.get("api_key_env") or "") != "GEMINI_API_KEY":
            raise ValueError("Gemini target.api_key_env must be GEMINI_API_KEY")
    validate_target_temperature(target)
    search_config = TreeSearchConfig(
        max_depth=int(search.get("max_depth", 3)),
        branching_factor=int(search.get("branching_factor", 2)),
        max_executed_nodes=int(search.get("max_executed_nodes", 8)),
        max_controller_calls=int(search.get("max_controller_calls", 4)),
        exploration_weight=float(search.get("exploration_weight", 1.414)),
        replay_retries=int(search.get("replay_retries", 1)),
        controller_max_retries=int(controller.get("max_retries", 1)),
        semantic_completion=search.get("semantic_completion", False),
        semantic_completion_min_confidence=float(
            search.get("semantic_completion_min_confidence", 0.9)
        ),
    )
    return UnifiedDARTConfig(
        path=path,
        benchmark=dict(benchmark),
        controller={
            **controller,
            "backend": controller_backend,
            "max_output_tokens": max_output_tokens,
        },
        target=dict(target),
        execution=execution,
        search=search_config,
        defense=defense,
        judge=dict(raw.get("judge", {})),
        attack=dict(attack),
    )
