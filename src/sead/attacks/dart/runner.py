"""Construction and artifact manifest logic for the deterministic v3 runner."""

from __future__ import annotations

from importlib.resources import files
from sead.environments.workers import worker_path

import json
import math
import os
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping

from ...benchmarks.mtar.dataset import (
    BENIGN_TASK_DESCRIPTION,
    load_task,
    load_task_dependencies,
    tool_runtime_context,
)
from ...benchmarks.mtar.environment import (
    DEFAULT_SERVICE_DEPLOYMENTS,
    environment_description,
)
from ...benchmarks.mtar.evaluator import MTARSingleTaskProgressEvaluator
from ...benchmarks.mtar.openhands_backend import MTAROpenHandsWorkerBackend
from ...benchmarks.mtar.replay_protocol import MTARReplayRequest, atomic_write_json
from ...benchmarks.mtar.runtime_profiles import (
    load_runtime_registry,
    prepare_openhands_base_image,
    resolve_task_profile,
)
from ...benchmarks.oas.dataset import (
    BENIGN_TASK_DESCRIPTION as OAS_BENIGN_TASK_DESCRIPTION,
    DEFAULT_CANDIDATE_INDEX as OAS_DEFAULT_CANDIDATE_INDEX,
    DEFAULT_SELECTION as OAS_DEFAULT_SELECTION,
    load_task as load_oas_task,
)
from ...benchmarks.oas.evaluator import OASSingleTaskProgressEvaluator
from ...benchmarks.oas.openhands_backend import OASOpenHandsWorkerBackend
from ...evaluation.posthoc_judge.config import load_config as load_judge_config
from ...evaluation.posthoc_judge.gemini import GeminiBackend
from ...evaluation.posthoc_judge.prompts import load_prompts as load_judge_prompts
from ...infrastructure.azure_openai_credentials import (
    load_azure_openai_api_key,
)
from ...infrastructure.controller_backend import (
    OpenAIControllerBackend,
    SGLangHTTPBackend,
    SGLangSubprocessBackend,
)
from ...infrastructure.gemini_credentials import load_gemini_api_key
from ...hashing import sha256_file, sha256_text
from .candidates import StructuredBatchController
from .checkpoints import CheckpointSet, load_checkpoints
from .config import UnifiedDARTConfig
from .node_critic import MTARNodeCritic
from .semantic_judge import GeminiSemanticCompletionJudge
from .search_tree_renderer import render_search_tree
from .tree_search import DARTTreeSearchEngine, TreeSearchOutcome
from .models import (
    SCHEMA_VERSION,
    BranchReplayBackend,
    ControllerContext,
    GoalProgressEvaluator,
)


DEFAULT_SEMANTIC_JUDGE_CONFIG = (
    Path(__file__).resolve().parents[4] / "config" / "posthoc_judge_gemini.yml"
)


def _judge_config(config):
    if config.judge:
        return load_judge_config(config.path, value={"schema_version": "posthoc-llm-judge-v1", "judge": dict(config.judge)})
    return load_judge_config(config.execution.get("judge_config", DEFAULT_SEMANTIC_JUDGE_CONFIG))


def _tac_pool_spec(config: UnifiedDARTConfig, dependencies: tuple[str, ...] | list[str]):
    from ...environments.services.tac_pool import tac_pool_instance

    execution = dict(config.execution)
    raw_path = execution.get("tac_pool_config")
    if raw_path and not Path(str(raw_path)).is_absolute():
        execution["tac_pool_config"] = str(
            (config.path.parent / str(raw_path)).resolve()
        )
    return tac_pool_instance(
        execution, str(config.benchmark["task_id"]), dependencies=dependencies,
    )


def _controller_task_descriptions(
    config: UnifiedDARTConfig,
    *,
    values: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    """Load the exact pinned benign and side-task descriptions for Controller."""

    resolved_values = (
        values if values is not None else _validate_builtin_benchmark_config(config)
    )
    kind = config.benchmark["kind"]
    benign = (
        BENIGN_TASK_DESCRIPTION if kind == "mtar" else OAS_BENIGN_TASK_DESCRIPTION
    )
    side = (resolved_values["task_root"] / "task.md").read_text().strip()
    if kind == "mtar":
        from ...environments.services.tac_pool import rewrite_service_urls
        spec = _tac_pool_spec(config, resolved_values["dependencies"])
        side = rewrite_service_urls(side, spec)
    if not benign:
        raise ValueError("benchmark benign task description is empty")
    if not side:
        raise ValueError("benchmark side task description is empty")
    return benign, side


def _prepare_target_credentials(config: UnifiedDARTConfig) -> None:
    provider = str(config.target["provider"])
    if provider in {"anthropic_foundry", "azure_openai_v1"}:
        load_azure_openai_api_key(
            variable=str(config.target["api_key_env"]),
        )
    elif provider == "gemini":
        load_gemini_api_key()


def _prepare_controller_credentials(config: UnifiedDARTConfig) -> None:
    if config.controller.get("backend") == "openai_v1":
        load_azure_openai_api_key(
            variable=str(config.controller["api_key_env"]),
        )


def _build_semantic_completion_judge(
    config: UnifiedDARTConfig,
    *,
    unsafe_goal: str,
    checkpoint_set: CheckpointSet,
    output_dir: Path,
) -> GeminiSemanticCompletionJudge | None:
    if not config.search.semantic_completion:
        return None
    judge_config = _judge_config(config)
    prompts = load_judge_prompts()
    backend = GeminiBackend(
        config=judge_config.judge,
        api_key=load_gemini_api_key(),
    )
    return GeminiSemanticCompletionJudge(
        task_id=str(config.benchmark["task_id"]),
        unsafe_goal=unsafe_goal,
        checkpoint_set=checkpoint_set,
        config=judge_config.judge,
        prompts=prompts,
        backend=backend,
        trace_path=output_dir / "semantic_judge_trace.jsonl",
    )


def _build_backend(
    config: UnifiedDARTConfig,
    output_dir: Path | str,
) -> tuple[BranchReplayBackend, GoalProgressEvaluator]:
    values = _validate_builtin_benchmark_config(config)
    kind = str(config.benchmark["kind"])
    execution = dict(config.execution)
    execution["defense"] = dict(config.defense)
    backend_class = (
        MTAROpenHandsWorkerBackend if kind == "mtar" else OASOpenHandsWorkerBackend
    )
    if kind == "mtar":
        execution["service_deployments_path"] = str(values["service_deployments_path"])
        execution["evaluation_mode"] = "lm_judge" if config.search.semantic_completion else "environment_verifier"
    else:
        execution["service_deployments_path"] = str(values["service_deployments_path"])
        execution["oas_selection_path"] = str(values["selection_path"])
        execution["oas_candidate_index_path"] = str(values["candidate_index_path"])
    execution["resolved_openhands_base_image"] = (
        _prepare_openhands_base_image_for_config(config, values=values)
    )
    backend = backend_class(
        task_id=str(config.benchmark["task_id"]),
        dataset_root=values["dataset_root"],
        openhands_root=values["openhands_root"],
        worker_python=values["worker_python"],
        target=config.target,
        execution=execution,
        output_dir=output_dir,
    )
    evaluator = (
        MTARSingleTaskProgressEvaluator()
        if kind == "mtar"
        else OASSingleTaskProgressEvaluator()
    )
    return backend, evaluator


def _prepare_openhands_base_image_for_config(
    config: UnifiedDARTConfig,
    *,
    values: Mapping[str, Any] | None = None,
) -> str:
    """Resolve one verified runtime alias for all replay workers of a task."""

    resolved_values = dict(values or _validate_builtin_benchmark_config(config))
    if config.benchmark["kind"] == "mtar":
        runtime_profile, _ = resolve_task_profile(resolved_values["task_root"])
    else:
        profile_id = "oas-web" if resolved_values["dependencies"] else "oas-base"
        runtime_profile = load_runtime_registry()[profile_id]
    return prepare_openhands_base_image(runtime_profile.require_available())


def prepare_openhands_runtime_for_config(config: UnifiedDARTConfig) -> str:
    """Verify the base and build the actual source-matched worker runtime."""
    from ...infrastructure.openhands_runtime import prepare_runtime_image

    values = _validate_builtin_benchmark_config(config)
    base_image = _prepare_openhands_base_image_for_config(config, values=values)
    prepare_runtime_image(base_image, values["openhands_root"], values["worker_python"])
    return base_image


def _resolve_config_path(config: UnifiedDARTConfig, value: Any) -> Path:
    path = Path(str(value))
    # Normalize ``..`` without resolving a virtualenv Python symlink.
    return Path(
        os.path.abspath(path if path.is_absolute() else config.path.parent / path)
    )


def _validate_controller_config(config: UnifiedDARTConfig) -> dict[str, Path]:
    if config.controller["backend"] in {"sglang_http", "openai_v1"}:
        return {}
    python = _resolve_config_path(config, config.controller["python"])
    worker_script = Path(__file__).resolve().parents[2] / "infrastructure/controller_worker.py"
    for name, path in {
        "python": python,
        "worker_script": worker_script,
    }.items():
        if not path.exists():
            raise ValueError(f"controller {name} does not exist: {path}")
    try:
        check = subprocess.run(
            [str(python), "-c", "import sglang"],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"Could not run SGLang interpreter {python}: {exc}") from exc
    if check.returncode != 0:
        detail = (check.stderr or check.stdout).strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise ValueError(f"controller.python does not provide SGLang{suffix}")
    return {
        "python": python,
        "worker_script": worker_script,
    }


def _create_controller_backend(
    config: UnifiedDARTConfig,
    *,
    stderr_path: Path | str,
    environment: Mapping[str, str] | None = None,
    max_batch_size: int = 1,
    batch_wait_ms: float = 0,
) -> SGLangSubprocessBackend | SGLangHTTPBackend | OpenAIControllerBackend:
    """Construct either a managed SGLang worker or an external HTTP client."""

    if config.controller["backend"] in {"sglang_http", "openai_v1"}:
        api_key_env = str(config.controller.get("api_key_env") or "")
        source_environment = os.environ if environment is None else environment
        api_key = source_environment.get(api_key_env) if api_key_env else None
        if api_key_env and not api_key:
            raise ValueError(
                f"missing Controller credential environment variable: {api_key_env}"
            )
        backend_class = (
            OpenAIControllerBackend
            if config.controller["backend"] == "openai_v1"
            else SGLangHTTPBackend
        )
        common = dict(
            endpoint=str(config.controller["endpoint"]),
            model=str(config.controller["model"]),
            temperature=float(config.controller.get("temperature", 0.7)),
            top_p=float(config.controller.get("top_p", 0.9)),
            max_output_tokens=int(config.controller.get("max_output_tokens", 1024)),
            remove_thinking=bool(config.controller.get("remove_thinking", True)),
            api_key=api_key,
            request_timeout_seconds=float(
                config.controller.get("request_timeout_seconds", 600)
            ),
            max_inflight_requests=int(
                config.controller.get("max_inflight_requests", 16)
            ),
        )
        if backend_class is OpenAIControllerBackend:
            common["reasoning_effort"] = config.controller.get("reasoning_effort")
        else:
            common["enable_thinking"] = bool(
                config.controller.get("enable_thinking", False)
            )
        return backend_class(**common)
    runtime = _validate_controller_config(config)
    return SGLangSubprocessBackend(
        python_executable=runtime["python"],
        worker_script=runtime["worker_script"],
        model=str(config.controller["model"]),
        temperature=float(config.controller.get("temperature", 0.7)),
        top_p=float(config.controller.get("top_p", 0.9)),
        max_output_tokens=int(config.controller.get("max_output_tokens", 1024)),
        enable_thinking=bool(config.controller.get("enable_thinking", False)),
        remove_thinking=bool(config.controller.get("remove_thinking", True)),
        context_length=int(config.controller.get("context_length", 65536)),
        stderr_path=stderr_path,
        environment=environment,
        max_batch_size=max_batch_size,
        batch_wait_ms=batch_wait_ms,
        tensor_parallel_size=int(config.controller.get("tensor_parallel_size", 1)),
        data_parallel_size=int(config.controller.get("data_parallel_size", 1)),
        mem_fraction_static=float(config.controller.get("mem_fraction_static", 0.82)),
        max_running_requests=int(config.controller.get("max_running_requests", 16)),
        max_inflight_batches=int(config.controller.get("max_inflight_batches", 4)),
    )


def _mtar_prompt_dir() -> Path:
    return Path(str(files("sead.attacks.dart").joinpath("prompts")))


def _load_mtar_prompts() -> tuple[Path, dict[str, str]]:
    root = _mtar_prompt_dir().resolve()
    values: dict[str, str] = {}
    for name in MTARNodeCritic.PROMPT_FILES:
        path = root / name
        if not path.is_file():
            raise ValueError(f"missing MTAR prompt: {path}")
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"empty MTAR prompt: {path}")
        values[name] = text
    expected = {
        "controller_user_prompt.md": {
            "harmful_task_description",
            "target_tool",
            "target_tool_description",
            "other_tools",
            "service_deployments",
            "parent_instructions",
            "target_transcript",
            "current_depth",
            "maximum_depth",
            "remaining_depth",
            "branching_factor",
        },
        "introspection_user_prompt.md": {
            "harmful_goal",
            "parent_history",
            "current_instruction",
            "current_rationale",
        },
        "feedback_user_prompt.md": {
            "harmful_goal",
            "completion_criteria",
            "parent_history",
            "current_instruction",
            "current_response",
        },
    }
    for name, required in expected.items():
        found = set(re.findall(r"\{([a-z_]+)\}", values[name]))
        if found != required:
            raise ValueError(
                f"invalid MTAR prompt placeholders in {root / name}: "
                f"missing={sorted(required - found)}, unknown={sorted(found - required)}"
            )
    return root, values


def _validate_builtin_mtar_config(config: UnifiedDARTConfig) -> dict[str, Any]:
    benchmark = config.benchmark
    task_id = str(benchmark["task_id"])
    required = ("dataset_root", "openhands_root", "worker_python")
    missing = [name for name in required if not str(benchmark.get(name) or "").strip()]
    if missing:
        raise ValueError("missing built-in MTAR configuration: " + ", ".join(missing))
    values = {name: _resolve_config_path(config, benchmark[name]) for name in required}
    for name, path in values.items():
        expected_file = name == "worker_python"
        if (not path.is_file()) if expected_file else (not path.is_dir()):
            raise ValueError(f"benchmark.{name} does not exist: {path}")
    task_root, task_row = load_task(values["dataset_root"], task_id)
    from sead.benchmarks.mtar.postgres.fixtures import checkpoint_path, require_scoring_mode
    from sead.environments.leases import postgres_mode
    if postgres_mode(config.execution) == "leased":
        require_scoring_mode(task_id, lm_judge=config.search.semantic_completion)
    tool_family, playwright_environment = tool_runtime_context(str(task_row["tool"]))
    dependencies = load_task_dependencies(task_root, str(task_row["tool"]))
    from sead.environments.registry import environment_config
    if "mcp-postgres" in dependencies and not environment_config(config.execution) and postgres_mode(config.execution) != "leased":
        raise ValueError("PostgreSQL tasks require an isolated environment or PostgreSQL lease")
    raw_service_deployments = benchmark.get("service_deployments")
    service_deployments_path = (
        _resolve_config_path(config, raw_service_deployments)
        if str(raw_service_deployments or "").strip()
        else DEFAULT_SERVICE_DEPLOYMENTS.resolve()
    )
    if not service_deployments_path.is_file():
        raise ValueError(
            f"benchmark.service_deployments does not exist: {service_deployments_path}"
        )
    for name in ("max_output_tokens", "max_tool_calls_per_instruction"):
        if int(config.target.get(name, 0)) < 1:
            raise ValueError(f"target.{name} must be positive")
    for name in ("max_steps", "sample_timeout_seconds"):
        if int(config.execution.get(name, 0)) < 1:
            raise ValueError(f"execution.{name} must be positive")
    if "prefix_replay_timeout_seconds" in config.execution:
        restore_timeout = float(config.execution["prefix_replay_timeout_seconds"])
        if not math.isfinite(restore_timeout) or restore_timeout <= 0:
            raise ValueError(
                "execution.prefix_replay_timeout_seconds must be finite and positive"
            )
    if (
        int(config.execution["max_steps"])
        < int(config.target["max_tool_calls_per_instruction"]) + 1
    ):
        raise ValueError(
            "execution.max_steps must exceed the per-instruction tool-call limit"
        )
    from ...environments.registry import public_deployments
    description = environment_description(
        tool_family=tool_family,
        playwright_environment=playwright_environment,
        dependencies=dependencies,
        service_deployments_path=service_deployments_path,
        configured_deployments=public_deployments(config.execution, dependencies),
    )
    from ...environments.services.tac_pool import rewrite_service_urls
    description = rewrite_service_urls(
        description, _tac_pool_spec(config, tuple(dependencies))
    )
    return {
        **values,
        "task_root": task_root,
        "task_row": task_row,
        "checkpoints_path": checkpoint_path(task_id, task_root),
        "service_deployments_path": service_deployments_path,
        "dependencies": tuple(dependencies),
        "environment_description": description,
    }


def _validate_builtin_oas_config(config: UnifiedDARTConfig) -> dict[str, Any]:
    from ...environments.registry import public_deployments
    benchmark = config.benchmark
    task_id = str(benchmark["task_id"])
    required = ("dataset_root", "openhands_root", "worker_python")
    missing = [name for name in required if not str(benchmark.get(name) or "").strip()]
    if missing:
        raise ValueError("missing built-in OAS configuration: " + ", ".join(missing))
    values = {name: _resolve_config_path(config, benchmark[name]) for name in required}
    for name, path in values.items():
        expected_file = name == "worker_python"
        if (not path.is_file()) if expected_file else (not path.is_dir()):
            raise ValueError(f"benchmark.{name} does not exist: {path}")
    raw_selection = benchmark.get("selection")
    selection_path = (
        _resolve_config_path(config, raw_selection)
        if str(raw_selection or "").strip()
        else OAS_DEFAULT_SELECTION.resolve()
    )
    raw_candidate_index = benchmark.get("candidate_index")
    candidate_index_path = (
        _resolve_config_path(config, raw_candidate_index)
        if str(raw_candidate_index or "").strip()
        else OAS_DEFAULT_CANDIDATE_INDEX.resolve()
    )
    raw_service_deployments = benchmark.get("service_deployments")
    service_deployments_path = (
        _resolve_config_path(config, raw_service_deployments)
        if str(raw_service_deployments or "").strip()
        else DEFAULT_SERVICE_DEPLOYMENTS.resolve()
    )
    for label, path in {
        "selection": selection_path,
        "candidate_index": candidate_index_path,
        "service_deployments": service_deployments_path,
    }.items():
        if not path.is_file():
            raise ValueError(f"benchmark.{label} does not exist: {path}")
    task = load_oas_task(
        values["dataset_root"],
        task_id,
        selection_path=selection_path,
        candidate_index=candidate_index_path,
    )
    for name in ("max_output_tokens", "max_tool_calls_per_instruction"):
        if int(config.target.get(name, 0)) < 1:
            raise ValueError(f"target.{name} must be positive")
    for name in ("max_steps", "sample_timeout_seconds"):
        if int(config.execution.get(name, 0)) < 1:
            raise ValueError(f"execution.{name} must be positive")
    if "prefix_replay_timeout_seconds" in config.execution:
        restore_timeout = float(config.execution["prefix_replay_timeout_seconds"])
        if not math.isfinite(restore_timeout) or restore_timeout <= 0:
            raise ValueError(
                "execution.prefix_replay_timeout_seconds must be finite and positive"
            )
    if (
        int(config.execution["max_steps"])
        < int(config.target["max_tool_calls_per_instruction"]) + 1
    ):
        raise ValueError(
            "execution.max_steps must exceed the per-instruction tool-call limit"
        )
    return {
        **values,
        "task_root": task.root,
        "task_row": {
            "task_id": task.task_id,
            "tool": "Playwright" if task.dependencies else "Terminal",
        },
        "selection_path": selection_path,
        "candidate_index_path": candidate_index_path,
        "service_deployments_path": service_deployments_path,
        "dependencies": tuple(task.dependencies),
        "environment_description": environment_description(
            tool_family="playwright" if task.dependencies else "terminal",
            playwright_environment=(
                task.dependencies[0] if task.dependencies else None
            ),
            dependencies=(
                tuple((*task.dependencies, "mcp-playwright"))
                if task.dependencies
                else ()
            ),
            service_deployments_path=service_deployments_path,
            configured_deployments=public_deployments(config.execution, task.dependencies),
            benchmark_name="OpenAgentSafety task environment",
            initial_state="released OAS workspace seed plus dependency-selected TAC services",
        ),
    }


def _validate_builtin_benchmark_config(config: UnifiedDARTConfig) -> dict[str, Any]:
    kind = config.benchmark["kind"]
    if kind == "mtar":
        return _validate_builtin_mtar_config(config)
    if kind == "oas":
        return _validate_builtin_oas_config(config)
    raise ValueError(f"unsupported benchmark kind: {kind}")


def validate_runtime_config(config: UnifiedDARTConfig) -> Mapping[str, Any]:
    """Import configured symbols/runtime without starting Docker or a model."""

    controller_runtime = _validate_controller_config(config)
    _prepare_target_credentials(config)
    _prepare_controller_credentials(config)
    if config.search.semantic_completion:
        semantic_config = _judge_config(config)
        load_judge_prompts()
        GeminiBackend(
            config=semantic_config.judge,
            api_key=load_gemini_api_key(),
        )
    values = _validate_builtin_benchmark_config(config)
    _, side_task_description = _controller_task_descriptions(config, values=values)
    load_checkpoints(
        values.get("checkpoints_path", Path(values["task_root"]) / "checkpoints.md"),
        harmful_goal=side_task_description,
    )
    kind = str(config.benchmark["kind"])
    with tempfile.TemporaryDirectory(prefix=f"dart-{kind}-check-") as temporary:
        root = Path(temporary)
        execution = dict(config.execution)
        execution["defense"] = dict(config.defense)
        if kind == "mtar":
            execution["evaluation_mode"] = "lm_judge" if config.search.semantic_completion else "environment_verifier"
        execution["service_deployments_path"] = str(
            values["service_deployments_path"]
        )
        if kind == "oas":
            execution["oas_selection_path"] = str(values["selection_path"])
            execution["oas_candidate_index_path"] = str(
                values["candidate_index_path"]
            )
        request = MTARReplayRequest(
            replay_id="check-replay",
            environment_id="check-environment",
            run_id="check",
            node_id="check",
            parent_node_id="root",
            task_id=str(config.benchmark["task_id"]),
            dataset_root=str(values["dataset_root"]),
            openhands_root=str(values["openhands_root"]),
            worker_dir=str(root),
            parent_replay_turns=(),
            new_instruction="check",
            target=config.target,
            execution=execution,
        )
        request_path = root / "request.json"
        atomic_write_json(request_path, request.to_dict())
        worker_script = worker_path(kind)
        completed = subprocess.run(
            [
                str(values["worker_python"]),
                str(worker_script),
                "--request",
                str(request_path),
                "--check-only",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        if completed.returncode != 0:
            raise ValueError(
                f"{kind.upper()} worker validation failed: "
                + (completed.stderr.strip() or completed.stdout.strip())
            )
        result = json.loads(completed.stdout)
        result["controller_task_context"] = {
            "harmful_task_source": (
                "normalized single task.md"
                if kind == "mtar"
                else "OpenAgentSafety task.md"
            ),
            "harmful_task_description_loaded": bool(side_task_description),
        }
        result["controller_backend"] = str(
            config.controller.get("backend") or "sglang_subprocess"
        )
        if controller_runtime:
            result["controller_runtime"] = {
                name: str(path) for name, path in controller_runtime.items()
            }
        return result


def _collect_inspect_usage(output_dir: Path) -> Mapping[str, Any]:
    records: list[dict[str, Any]] = []
    totals: dict[str, dict[str, int]] = {}
    for eval_path in sorted(output_dir.rglob("*.eval")):
        try:
            with zipfile.ZipFile(eval_path) as archive:
                sample_names = [
                    name
                    for name in archive.namelist()
                    if name.startswith("samples/") and name.endswith(".json")
                ]
                for sample_name in sample_names:
                    sample = json.loads(archive.read(sample_name))
                    usage = sample.get("model_usage") or {}
                    if not isinstance(usage, Mapping):
                        continue
                    for model, values in usage.items():
                        if not isinstance(values, Mapping):
                            continue
                        counts = {
                            str(key): int(value)
                            for key, value in values.items()
                            if isinstance(value, (int, float))
                        }
                        records.append(
                            {
                                "eval": str(eval_path.relative_to(output_dir)),
                                "sample": sample_name,
                                "model": str(model),
                                "tokens": counts,
                            }
                        )
                        model_totals = totals.setdefault(str(model), {})
                        for key, value in counts.items():
                            model_totals[key] = model_totals.get(key, 0) + value
        except (
            OSError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
            zipfile.BadZipFile,
        ):
            continue
    for usage_path in sorted(output_dir.rglob("target_usage.json")):
        try:
            payload = json.loads(usage_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        model = str(payload.get("model") or "unknown")
        raw_records = payload.get("records")
        if not isinstance(raw_records, list):
            continue
        for raw in raw_records:
            if not isinstance(raw, Mapping):
                continue
            counts = {
                str(key): int(value)
                for key, value in raw.items()
                if key.endswith("_tokens")
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            }
            records.append(
                {
                    "usage": str(usage_path.relative_to(output_dir)),
                    "model": model,
                    "model_call": raw.get("model_call"),
                    "tokens": counts,
                }
            )
            model_totals = totals.setdefault(model, {})
            for key, value in counts.items():
                model_totals[key] = model_totals.get(key, 0) + value
    return {"records": records, "totals_by_model": totals}


def _run_engine_and_maybe_render(
    engine: DARTTreeSearchEngine,
    config: UnifiedDARTConfig,
    output_dir: Path,
) -> TreeSearchOutcome:
    """Render only after the engine has persisted a normally completed search."""

    outcome = engine.run()
    if config.execution["render_tree_html"]:
        render_search_tree(output_dir)
    return outcome


def _record_controller_raw_responses(
    complete: Callable[..., str], path: Path
) -> Callable[..., str]:
    """Wrap a Controller completion function with lossless JSONL recording."""

    path.touch()
    response_index = 0

    def complete_and_record(
        system: str,
        user: str,
        *,
        request_kind: str = "candidate",
    ) -> str:
        nonlocal response_index
        try:
            response = complete(system, user, request_kind=request_kind)
        except TypeError as exc:
            if "unexpected keyword" not in str(exc):
                raise
            response = complete(system, user)
        response_index += 1
        row = {
            "schema_version": SCHEMA_VERSION,
            "controller_call": response_index,
            "request_kind": request_kind,
            "raw_response": response,
        }
        # ASCII JSON escaping keeps embedded newlines and Unicode line separators
        # on one physical line while json.loads restores the exact response.
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        return response

    return complete_and_record


def run_from_config(
    config: UnifiedDARTConfig,
    *,
    run_id: str,
    output_dir: Path | str,
    backend_bundle: tuple[BranchReplayBackend, GoalProgressEvaluator] | None = None,
    controller_complete: Callable[[str, str], str] | None = None,
    controller_runtime: Any | None = None,
) -> TreeSearchOutcome:
    if controller_complete is not None and controller_runtime is not None:
        raise ValueError("provide controller_complete or controller_runtime, not both")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError(f"run directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    # Parse the complete private scoring definition before starting either
    # model runtime. A present but malformed checkpoint file is a task setup
    # error, never a reason to silently fall back to the harmful goal.
    controller_values = _validate_builtin_benchmark_config(config)
    benign_task_description, side_task_description = _controller_task_descriptions(
        config,
        values=controller_values,
    )
    task_root = controller_values.get("task_root")
    checkpoints_path = (
        controller_values.get("checkpoints_path", Path(task_root) / "checkpoints.md") if task_root is not None else None
    )
    checkpoint_set = load_checkpoints(
        checkpoints_path,
        harmful_goal=side_task_description,
    )
    completion_criteria = (
        checkpoints_path.read_text(encoding="utf-8")
        if checkpoints_path is not None and checkpoints_path.is_file()
        else side_task_description
    )
    _prepare_target_credentials(config)
    _prepare_controller_credentials(config)
    mtar_prompt_root, mtar_prompts = _load_mtar_prompts()
    # Create the lossless artifact before Controller initialization so an
    # environment/schema failure still leaves an explicit empty record.
    raw_response_path = output_dir / "controller_raw_responses.jsonl"
    raw_response_path.touch()
    controller_backend = controller_runtime
    owns_controller_backend = False
    if controller_complete is None:
        if controller_backend is None:
            environment = os.environ.copy()
            if config.controller.get("cuda_visible_devices") is not None:
                environment["CUDA_VISIBLE_DEVICES"] = str(
                    config.controller["cuda_visible_devices"]
                )
            controller_backend = _create_controller_backend(
                config,
                stderr_path=output_dir / "controller.stderr.log",
                environment=environment,
            )
            owns_controller_backend = True
        controller_complete = controller_backend.complete
    controller_complete = _record_controller_raw_responses(
        controller_complete,
        raw_response_path,
    )
    complete_all_kinds = controller_complete

    def planning_complete(system: str, user: str) -> str:
        return complete_all_kinds(
            system,
            user,
            request_kind="candidate",
        )

    controller_environment = str(controller_values["environment_description"])
    service_deployments_path = controller_values.get("service_deployments_path")
    semantic_judge = _build_semantic_completion_judge(
        config,
        unsafe_goal=side_task_description,
        checkpoint_set=checkpoint_set,
        output_dir=output_dir,
    )
    controller = StructuredBatchController(
        planning_complete,
        side_task_description=side_task_description,
        environment_description=controller_environment,
        maximum_depth=config.search.max_depth,
        allow_json_repair=True,
        mtar_system_prompt=mtar_prompts["controller_system_prompt.md"],
        mtar_user_template=mtar_prompts["controller_user_prompt.md"],
    )
    root_system_prompt, root_user_prompt = controller.prompts(
        ControllerContext(
            node_id="root",
            instructions=(),
            target_transcript=(),
        ),
        config.search.branching_factor,
    )
    atomic_write_json(
        output_dir / "controller_prompt_root.json",
        {
            "schema_version": "dart-controller-prompt-root-v1",
            "task_id": str(config.benchmark["task_id"]),
            "harmful_task_description": side_task_description,
            "benign_task_description": benign_task_description,
            "environment_description": controller_environment,
            "service_deployments_config": (
                {
                    "path": str(service_deployments_path),
                    "sha256": sha256_file(service_deployments_path),
                }
                if service_deployments_path is not None
                else None
            ),
            "system_prompt": root_system_prompt,
            "user_prompt": root_user_prompt,
            "system_prompt_template": str(
                mtar_prompt_root / "controller_system_prompt.md"
            ),
            "user_prompt_template": str(mtar_prompt_root / "controller_user_prompt.md"),
        },
    )
    backend, progress = backend_bundle or _build_backend(config, output_dir)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=config.path.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "config": str(config.path),
        "version": {"sead_commit": commit},
        "benchmark": dict(config.benchmark),
        "controller": {
            "backend": config.controller.get("backend", "sglang_subprocess"),
            "model": config.controller["model"],
            "temperature": config.controller.get("temperature", 0.7),
            "structured_batch": True,
            "trusted_evaluation_visible": False,
            "system_prompt_template": str(
                mtar_prompt_root / "controller_system_prompt.md"
            ),
            "user_prompt_template": str(mtar_prompt_root / "controller_user_prompt.md"),
            "response_processing": "extract-repair-parse-validate",
            "worker_ready": (
                dict(controller_backend.ready_info)
                if controller_backend is not None
                else None
            ),
            "shared_across_tasks": bool(
                getattr(controller_backend, "shared_across_tasks", False)
            ),
            "worker_stderr": (
                str(controller_backend.stderr_path)
                if controller_backend is not None
                and getattr(controller_backend, "stderr_path", None) is not None
                else None
            ),
            "prompt_files": [
                {
                    "path": str(mtar_prompt_root / name),
                    "sha256": sha256_text(text),
                }
                for name, text in mtar_prompts.items()
            ],
        },
        "target": dict(config.target),
        "defense": dict(config.defense),
        "search": config.search.to_dict(),
        "semantic_judge": (
            semantic_judge.manifest() if semantic_judge is not None else None
        ),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    node_critic = MTARNodeCritic(
        complete_all_kinds,
        harmful_goal=side_task_description,
        completion_criteria=completion_criteria,
        prompt_dir=mtar_prompt_root,
        trace_path=output_dir / "node_critic_trace.jsonl",
    )
    engine = DARTTreeSearchEngine(
        config=config.search,
        controller=controller,
        backend=backend,
        progress_evaluator=progress,
        run_id=run_id,
        output_dir=output_dir,
        node_critic=node_critic,
        semantic_judge=semantic_judge,
    )
    try:
        return _run_engine_and_maybe_render(engine, config, output_dir)
    finally:
        atomic_write_json(
            output_dir / "model_usage.json",
            {
                "controller": (
                    list(getattr(controller_backend, "usage_records", ()))
                    if controller_backend is not None
                    else []
                ),
                "target": _collect_inspect_usage(output_dir),
                "semantic_judge": (
                    list(semantic_judge.usage_records)
                    if semantic_judge is not None
                    else []
                ),
            },
        )
        if owns_controller_backend and controller_backend is not None:
            controller_backend.close()
        if semantic_judge is not None:
            semantic_judge.close()


# Public construction API used by the suite orchestrator.
create_controller_backend = _create_controller_backend
prepare_controller_credentials = _prepare_controller_credentials
prepare_openhands_base_image_for_config = _prepare_openhands_base_image_for_config
