#!/usr/bin/env python3
"""Execute one normalized MTAR path in a fresh OpenHands sandbox."""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import importlib.metadata
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sead.attacks.dart.replay_outcomes import (
    classify_native_observation,
    classify_tool_outcome,
    compare_tool_outcomes,
    observation_failed,
)
from sead.attacks.dart.config import validate_target_temperature
from sead.attacks.dart.models import (
    ReplayAssistantMessage,
    ReplayToolCall,
    ReplayToolResult,
    ReplayTurn,
)
from sead.benchmarks.mtar.dataset import (
    MTARDatasetQuarantinedError,
    load_task,
    load_task_dependencies,
    tool_runtime_context,
)
from sead.benchmarks.mtar.environment import (
    DEFAULT_SERVICE_DEPLOYMENTS,
    SERVICE_RESET_ENDPOINTS,
    TOOL_SCHEMAS,  # noqa: F401 - shared OpenHands engine module API
    environment_description,
    external_mcp_urls,  # noqa: F401 - shared OpenHands engine module API
    service_instruction_context,  # noqa: F401 - shared OpenHands engine module API
    unavailable_runtime_services,
)
from sead.benchmarks.mtar.evaluator import (
    evaluator_trajectory_payload,
    parse_evaluation,
)
from sead.benchmarks.mtar.evaluator_compat import (
    EVALUATION_MARKER,
    build_evaluator_entrypoint,
    evaluator_python_requirements,
)
from sead.benchmarks.mtar.replay_protocol import (
    MTARReplayRequest,
    MTARReplayWorkerResult,
    ReplayPhaseTimer,
    atomic_write_json,
    read_request,
)
from sead.benchmarks.mtar.runtime_profiles import (
    RuntimeProfile,
    UnsupportedRuntimeProfile,
    openhands_base_image_alias,
    prepare_openhands_base_image,
    resolve_service_image,
    resolve_task_profile,
)
from sead.defenses import validate_online_defense_config
from sead.defenses.replay import is_tool_defense_block
from sead.environments.resources import playwright_port  # noqa: F401 - shared engine API
from sead.environments.services.tac import (
    acquire_web_lock,
    release_web_lock,
    reset_services as reset_tac_services,
)


PROJECT = Path(__file__).resolve().parents[4]


def _target_model(target: Mapping[str, Any]) -> str:
    model = str(target["model"])
    if str(target.get("provider")) == "azure_openai_v1" and "/" not in model:
        return "openai/" + model
    if str(target.get("provider")) == "anthropic_foundry" and "/" not in model:
        return "anthropic/" + model
    return model


FINAL_REPLAY_PROVIDER = "offline_replay"
FINAL_REPLAY_MODEL = "openai/mtar-hard-replay-placeholder"


def _target_temperature(target: Mapping[str, Any]) -> float | None:
    """Return the validated experiment value passed to OpenHands/LiteLLM."""

    if target.get("temperature") is None:
        return None
    return float(target["temperature"])


def _prepare_openhands_base_image(profile: RuntimeProfile) -> str:
    """Compatibility wrapper for callers that still validate a worker directly."""

    return prepare_openhands_base_image(profile)


def _validate_request(request: MTARReplayRequest) -> dict[str, Any]:
    task_root, row = load_task(request.dataset_root, request.task_id)
    tool_family, playwright_environment = tool_runtime_context(str(row["tool"]))
    dependencies = list(load_task_dependencies(task_root, str(row["tool"])))
    from sead.environments.registry import environment_config, EnvironmentRegistry
    leased_config = environment_config(request.execution)
    environment_plan = EnvironmentRegistry(leased_config["registry"]).plan(
        dependencies, request.task_id
    ) if leased_config else None
    from sead.environments.services.tac_pool import tac_pool_instance
    tac_pool_instance(request.execution, request.task_id, dependencies=dependencies)
    profile, evaluator_requirements = resolve_task_profile(task_root)
    from sead.environments.leases import LeaseClient, postgres_mode
    if "mcp-postgres" in dependencies and not leased_config and postgres_mode(request.execution) != "leased":
        raise ValueError("PostgreSQL tasks require an isolated environment or PostgreSQL lease")
    if "mcp-postgres" in dependencies and (leased_config or postgres_mode(request.execution) == "leased"):
        from sead.benchmarks.mtar.postgres import environment_spec
        from sead.benchmarks.mtar.postgres.fixtures import require_scoring_mode
        environment_spec(request.task_id, task_root)
        require_scoring_mode(request.task_id, lm_judge=(
            request.execution.get("evaluation_mode") == "lm_judge"
        ))
        if not leased_config:
            LeaseClient(request.execution["postgres"]["manager_socket"], timeout=5).call("inspect")
    profile.require_available()
    profile.require_evaluator_requirements(evaluator_requirements)
    pre_resolved_image = request.execution.get("resolved_openhands_base_image")
    if pre_resolved_image is None:
        openhands_base_image = _prepare_openhands_base_image(profile)
    else:
        openhands_base_image = str(pre_resolved_image)
        if openhands_base_image != openhands_base_image_alias(profile):
            raise ValueError(
                "execution.resolved_openhands_base_image does not match runtime profile"
            )
    if "mcp-filesystem" in dependencies:
        resolve_service_image("filesystem-mcp")
    unavailable_services = unavailable_runtime_services(
        dependencies,
        tool_family=tool_family,
    ) if not leased_config else ()
    if unavailable_services:
        raise UnsupportedRuntimeProfile(
            ",".join(unavailable_services),
            "required MTAR service is not deployed",
        )
    for service_id in (set(dependencies) & {"gitlab", "owncloud"} if not leased_config else ()):
        resolve_service_image(service_id)
    if "mcp-playwright" in dependencies and not leased_config:
        resolve_service_image("playwright-mcp")
    if {"sandbox_image", "runtime_extra_deps"} & set(request.execution):
        raise ValueError(
            "execution cannot override the controlled runtime image or runtime_extra_deps"
        )
    for name, path in {
        "OpenHands": Path(request.openhands_root),
        "worker directory": Path(request.worker_dir),
    }.items():
        if not path.exists():
            raise ValueError(f"{name} does not exist: {path}")
    if not (Path(request.openhands_root) / "openhands/core/main.py").is_file():
        raise ValueError("invalid OpenHands checkout")
    if str(request.target.get("agent") or "CodeActAgent") != "CodeActAgent":
        raise ValueError("MTAR target.agent must be CodeActAgent")
    validate_target_temperature(request.target, label="MTAR target")
    for mapping, names in (
        (request.target, ("max_output_tokens", "max_tool_calls_per_instruction")),
        (request.execution, ("max_steps", "sample_timeout_seconds")),
    ):
        for name in names:
            if int(mapping.get(name, 0)) < 1:
                raise ValueError(f"{name} must be positive")
    # Final replay restores saved actions without entering the model-backed
    # Controller, so credentials recorded by an older online run are irrelevant.
    if not request.final_replay and request.target.get("provider") == "azure_openai_v1":
        variable = str(request.target.get("api_key_env") or "")
        if not variable or not os.environ.get(variable):
            raise ValueError(
                f"missing Target credential environment variable: {variable}"
            )
        endpoint = str(request.target.get("endpoint") or "")
        if not endpoint.startswith("https://") or not endpoint.rstrip("/").endswith(
            "/openai/v1"
        ):
            raise ValueError("invalid Azure OpenAI v1 endpoint")
    if not request.final_replay and request.target.get("provider") == "anthropic_foundry":
        variable = str(request.target.get("api_key_env") or "")
        if not variable or not os.environ.get(variable):
            raise ValueError(
                f"missing Target credential environment variable: {variable}"
            )
        endpoint = str(request.target.get("endpoint") or "")
        if not endpoint.startswith("https://") or not endpoint.rstrip("/").endswith(
            "/anthropic"
        ):
            raise ValueError("invalid Anthropic Foundry endpoint")
    allow_outcome_drift = request.execution.get("allow_restore_outcome_drift", False)
    if not isinstance(allow_outcome_drift, bool):
        raise ValueError("allow_restore_outcome_drift must be boolean")
    if allow_outcome_drift and not request.final_replay:
        raise ValueError("restore outcome drift is allowed only for final replay")
    defense = validate_online_defense_config(
        request.execution.get("defense"),
        require_credentials=not request.final_replay,
    )
    if request.final_replay:
        actual_litellm = importlib.metadata.version("litellm")
    elif request.target.get("provider") == "gemini":
        variable = str(request.target.get("api_key_env") or "")
        if variable != "GEMINI_API_KEY" or not os.environ.get(variable):
            raise ValueError(
                "missing Target credential environment variable: GEMINI_API_KEY"
            )
        if not str(request.target.get("model") or "").startswith("gemini/"):
            raise ValueError("Gemini Target model must start with gemini/")
        expected_litellm = str(request.execution.get("litellm_version") or "")
        actual_litellm = importlib.metadata.version("litellm")
        if expected_litellm and actual_litellm != expected_litellm:
            raise ValueError(
                "Gemini Target requires litellm "
                f"{expected_litellm}; found {actual_litellm}"
            )
    else:
        actual_litellm = importlib.metadata.version("litellm")
    return {
        "status": "validated",
        "task_root": str(task_root),
        "task_id": request.task_id,
        "dependencies": dependencies,
        "data_status": row["data_status"],
        "runtime_profile": profile.profile_id,
        "runtime_backend": profile.backend,
        "openhands_base_image": openhands_base_image,
        "evaluator_python_requirements": list(evaluator_requirements),
        "environment_description": environment_description(
            tool_family=tool_family,
            playwright_environment=playwright_environment,
            dependencies=dependencies,
        ),
        "target_model": (
            FINAL_REPLAY_MODEL
            if request.final_replay
            else _target_model(request.target)
        ),
        "litellm_version": actual_litellm,
        "agent": "CodeActAgent",
        "defense": defense,
        "environment_plan": environment_plan,
    }


def _make_config(
    request: MTARReplayRequest,
    workspace: Path,
    trajectory: Path,
    replay: Path | None,
    *,
    runtime_profile: RuntimeProfile,
    base_container_image: str | None = None,
    filesystem_mcp_url: str | None = None,
    external_mcp_server_urls: Mapping[str, str] | None = None,
    lease_binding: Mapping[str, Any] | None = None,
):
    from openhands.core.config import OpenHandsConfig, SandboxConfig
    from openhands.core.config.agent_config import AgentConfig
    from openhands.core.config.llm_config import LLMConfig
    from openhands.core.config.mcp_config import (
        MCPConfig,
        MCPSHTTPServerConfig,
        MCPSSEServerConfig,
    )

    target = request.target
    api_key = None
    base_url = None
    custom_provider = None
    if request.final_replay:
        # OpenHands requires an LLMConfig even though hard replay restores only
        # recorded actions and never enters the Controller/model code path.
        # A fixed non-secret value prevents provider discovery from consulting
        # the environment during initialization.
        api_key = "mtar-hard-replay-not-a-real-key"
    elif target.get("provider") == "azure_openai_v1":
        api_key = os.environ[str(target["api_key_env"])]
        base_url = str(target["endpoint"])
        custom_provider = "openai"
    elif target.get("provider") == "anthropic_foundry":
        api_key = os.environ[str(target["api_key_env"])]
        base_url = str(target["endpoint"])
        custom_provider = "anthropic"
    elif target.get("provider") == "gemini":
        from sead.infrastructure.gemini_compat import install_gemini_call_id_compat

        install_gemini_call_id_compat()
        api_key = os.environ[str(target["api_key_env"])]
    target_temperature = _target_temperature(target)
    llm = LLMConfig(
        model=(FINAL_REPLAY_MODEL if request.final_replay else _target_model(target)),
        api_key=api_key,
        base_url=base_url,
        custom_llm_provider=custom_provider,
        # LLMConfig currently requires a float even when the provider rejects
        # the parameter. Assign None immediately below for those deployments.
        temperature=0 if target_temperature is None else target_temperature,
        max_output_tokens=int(target["max_output_tokens"]),
        num_retries=int(target.get("num_retries", 1)),
        reasoning_effort=target.get("reasoning_effort"),
        native_tool_calling=True,
    )
    if target_temperature is None:
        llm.temperature = None
    if target.get("provider") == "anthropic_foundry":
        # OpenHands defaults top_p to 1 and otherwise sends it together with
        # temperature. Foundry rejects that pair for Claude 4.5.
        llm.top_p = None
    config = OpenHandsConfig(
        default_agent="CodeActAgent",
        file_store_path=os.environ.get(
            "SEAD_OPENHANDS_FILE_STORE_PATH",
            str(trajectory.parent / "openhands_store"),
        ),
        # MTAR browser tasks declare mcp-playwright. Keeping OpenHands' native
        # browser enabled silently routes the model to browse_interactive and
        # bypasses the benchmark tool interface.
        enable_browser=False,
        run_as_openhands=False,
        max_iterations=int(request.execution["max_steps"]),
        save_trajectory_path=str(trajectory),
        replay_trajectory_path=str(replay) if replay else None,
        workspace_mount_path=str(workspace),
        workspace_mount_path_in_sandbox="/workspace",
        sandbox=SandboxConfig(
            base_container_image=base_container_image or str(runtime_profile.image),
            enable_auto_lint=True,
            use_host_network=runtime_profile.network == "host" and not lease_binding,
            timeout=int(request.execution.get("tool_timeout_seconds", 300)),
            volumes=f"{workspace}:/workspace",
            runtime_extra_deps=None,
            docker_runtime_kwargs={
                **runtime_profile.docker_runtime_kwargs(),
                **(
                    {"extra_hosts": {"the-agent-company.com": "127.0.0.1"}}
                    if runtime_profile.network == "host" and not lease_binding
                    else {}
                ),
                **({"network": lease_binding["network"], "labels": lease_binding["labels"],
                    "cap_drop": ["NET_RAW"], "security_opt": ["no-new-privileges"],
                    **lease_binding.get("runtime_resources", {"mem_limit": "2g", "pids_limit": 512, "nano_cpus": 2_000_000_000})}
                   if lease_binding else {}),
            },
        ),
    )
    config.set_llm_config(llm)
    config.set_agent_config(
        AgentConfig(
            enable_browsing=False,
            enable_prompt_extensions=False,
            enable_plan_mode=False,
            enable_mcp=True,
            # The bundled default-tools microagent starts an unrelated stdio
            # `fetch` MCP server. MTAR exposes only dependency-declared tools;
            # disabling it removes a noisy, out-of-scope connection attempt.
            disabled_microagents=["default-tools"],
        )
    )
    config.mcp = MCPConfig(
        sse_servers=[
            MCPSSEServerConfig(url=url)
            for url in (external_mcp_server_urls or {}).values()
            if not url.rstrip("/").endswith("/mcp")
        ],
        shttp_servers=(
            (
                [MCPSHTTPServerConfig(url=filesystem_mcp_url)]
                if filesystem_mcp_url
                else []
            )
            + [
                MCPSHTTPServerConfig(url=url)
                for url in (external_mcp_server_urls or {}).values()
                if url.rstrip("/").endswith("/mcp")
            ]
        ),
    )
    return config


_RESTORE_SESSION_CONTROL_TEXT = re.compile(
    r"(?:exit|logout)(?:\s+[0-9]+)?|C-[cdl]", re.IGNORECASE
)
_RESTORE_SESSION_CONTROL_BYTES = frozenset({"\x03", "\x04", "\x0c"})


def _record_only_restore_action(action: Any) -> bool:
    """Return whether a shell action is an exact, non-persistent control action.

    This intentionally uses a narrow full-action allowlist.  In particular,
    commands containing ``sleep`` or mixing a control token with other shell
    syntax remain executable replay actions; arbitrary shell semantics are not
    inferred here.
    """

    from openhands.events.action import CmdRunAction

    if not isinstance(action, CmdRunAction):
        return False
    command = action.command.strip()
    if not command:
        return True
    if command in _RESTORE_SESSION_CONTROL_BYTES:
        return True
    return _RESTORE_SESSION_CONTROL_TEXT.fullmatch(command) is not None


def _record_action_observation_without_dispatch(
    event_stream: Any, action: Any, observation: Any
) -> Any:
    """Restore an action/observation pair without invoking live subscribers."""

    from openhands.events.event import EventSource

    event_stream.add_event(action, EventSource.AGENT, dispatch=False)
    observation._cause = action.id
    observation.tool_call_metadata = action.tool_call_metadata
    event_stream.add_event(observation, EventSource.AGENT, dispatch=False)
    return observation


def _runtime_restore_failure(runtime: Any) -> str | None:
    """Return a fail-fast reason when the runtime callback/container is broken."""

    from openhands.events.stream import EventStreamSubscriber

    event_stream = runtime.event_stream
    is_subscribed = getattr(event_stream, "is_subscribed", None)
    if callable(is_subscribed) and not is_subscribed(
        EventStreamSubscriber.RUNTIME, runtime.sid
    ):
        return "runtime event subscriber is no longer registered"
    subscriber_error = getattr(event_stream, "subscriber_error", None)
    if callable(subscriber_error):
        error = subscriber_error(EventStreamSubscriber.RUNTIME, runtime.sid)
        if error is not None:
            return f"runtime event callback failed: {type(error).__name__}: {error}"
    container = getattr(runtime, "container", None)
    if container is not None:
        try:
            reload_container = getattr(container, "reload", None)
            if callable(reload_container):
                reload_container()
            status = str(getattr(container, "status", "") or "").lower()
            if status and status not in {"created", "running"}:
                return f"runtime container is not healthy (status={status})"
        except Exception as exc:
            return f"runtime container health check failed: {exc}"
    return None


def _action_observation(
    runtime: Any,
    action: Any,
    *,
    wait_timeout_seconds: float | None = None,
    restore_deadline: float | None = None,
) -> Any:
    """Dispatch an action through OpenHands' normal event-stream tool path."""

    from openhands.events.action import CmdRunAction
    from openhands.events.observation import ErrorObservation, Observation
    from openhands.runtime.utils.bash import split_bash_commands

    event_stream = runtime.event_stream
    if isinstance(action, CmdRunAction):
        commands = split_bash_commands(action.command.strip())
        if len(commands) > 1:
            observation = ErrorObservation(
                content=(
                    "ERROR: Cannot execute multiple commands at once.\n"
                    "Please run each command separately OR chain them into a "
                    "single command via && or ;\n"
                    "Provided commands:\n"
                    + "\n".join(
                        f"({index}) {command}"
                        for index, command in enumerate(commands, 1)
                    )
                )
            )

            # Preserve the source-level rejection without ever replacing or
            # dispatching to the live runtime subscriber.
            return _record_action_observation_without_dispatch(
                event_stream, action, observation
            )

    if restore_deadline is not None and time.monotonic() >= restore_deadline:
        raise TimeoutError("prefix replay total deadline exceeded before action")
    from openhands.events.event import EventSource

    event_stream.add_event(action, EventSource.AGENT)
    # MCP actions do not carry OpenHands' sandbox/tool timeout on the action
    # object.  In particular, Playwright can legitimately return its own
    # timeout observation at the 30-second boundary.  A fixed 30-second event
    # wait races that observation and misclassifies it as missing.  Let the
    # caller propagate the configured tool budget, with a small allowance for
    # the runtime subscriber to publish the completed observation.
    deadline = time.monotonic() + max(
        float(action.timeout or 0) + 30.0,
        float(wait_timeout_seconds or 0) + 30.0,
        30.0,
    )
    if restore_deadline is not None:
        deadline = min(deadline, restore_deadline)
    next_health_check = 0.0
    while True:
        now = time.monotonic()
        if now >= deadline:
            break
        try:
            for event in event_stream.search_events(start_id=action.id + 1):
                if isinstance(event, Observation) and event.cause == action.id:
                    return event
        except json.JSONDecodeError:
            # OpenHands writes event files in place. The subscriber may still
            # be writing when this poll reads one; retry within the existing
            # deadline without dispatching the action a second time.
            pass
        if now >= next_health_check:
            failure = _runtime_restore_failure(runtime)
            if failure is not None:
                raise RuntimeError(failure)
            next_health_check = now + 1.0
        time.sleep(0.01)
    if restore_deadline is not None and time.monotonic() >= restore_deadline:
        raise TimeoutError(
            "prefix replay total deadline exceeded while awaiting action"
        )
    raise TimeoutError(
        f"OpenHands emitted no observation for restored action {action.id}"
    )


def _restore_action_observation(
    runtime: Any,
    action: Any,
    *,
    wait_timeout_seconds: float | None = None,
    restore_deadline: float | None = None,
) -> Any:
    """Execute a restored action without racing OpenHands' shell subscriber.

    A restored ``CmdRunAction`` already passed through the original CodeAct
    interface when it was recorded.  Dispatching it through the asynchronous
    event subscriber a second time can race OpenHands' persistent terminal:
    the event stream occasionally reports an empty observation and the exit
    status of a neighbouring command.  Run a single shell action synchronously
    and then append its action/observation pair to the reconstructed history.

    MCP actions still require the normal event-stream path.  Multi-command
    shell source is also left there so ``_action_observation`` preserves the
    original CodeAct preflight rejection without executing it.
    """

    from openhands.events.action import CmdRunAction
    from openhands.runtime.utils.bash import split_bash_commands

    if isinstance(action, CmdRunAction):
        commands = split_bash_commands(action.command.strip())
        if len(commands) <= 1:
            if restore_deadline is not None and time.monotonic() >= restore_deadline:
                raise TimeoutError(
                    "prefix replay total deadline exceeded before shell action"
                )
            observation = runtime.run_action(action)
            if restore_deadline is not None and time.monotonic() >= restore_deadline:
                raise TimeoutError(
                    "prefix replay total deadline exceeded while executing shell action"
                )
            return _record_action_observation_without_dispatch(
                runtime.event_stream, action, observation
            )
    return _action_observation(
        runtime,
        action,
        wait_timeout_seconds=wait_timeout_seconds,
        restore_deadline=restore_deadline,
    )


def _original_tool_action(
    recorded_call: ReplayToolCall, *, mcp_tool_names: list[str]
) -> Any:
    """Recreate an action through the same CodeAct tool interface as generation."""

    from litellm import ModelResponse
    from openhands.agenthub.codeact_agent.function_calling import response_to_actions
    from openhands.events.action.mcp import MCPAction
    from openhands.events.tool import ToolCallMetadata

    native = dict(recorded_call.native_action)
    metadata = native.get("tool_call_metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("recorded action has no tool_call_metadata")
    model_response = metadata.get("model_response")
    if not isinstance(model_response, Mapping):
        raise RuntimeError("recorded action has no original model_response")
    tool_call_id = str(metadata.get("tool_call_id") or "")
    if not tool_call_id:
        raise RuntimeError("recorded action has no original tool_call_id")

    response = ModelResponse(**dict(model_response))
    if native.get("action") == "call_tool_mcp":
        native_arguments = native.get("args")
        if not isinstance(native_arguments, Mapping):
            raise RuntimeError("recorded MCP action has no original arguments")
        name = native_arguments.get("name")
        arguments = native_arguments.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, Mapping):
            raise RuntimeError("recorded MCP action has invalid original arguments")
        action = MCPAction(name=name, arguments=dict(arguments),
                           thought=str(native_arguments.get("thought") or ""))
        action.tool_call_metadata = ToolCallMetadata(
            tool_call_id=tool_call_id,
            function_name=str(metadata.get("function_name") or name),
            model_response=response,
            total_calls_in_response=int(metadata.get("total_calls_in_response") or 1),
        )
        action.response_id = response.id
        return action

    actions = response_to_actions(
        response,
        mcp_tool_names=mcp_tool_names,
    )
    matches = [
        action
        for action in actions
        if action.tool_call_metadata is not None
        and action.tool_call_metadata.tool_call_id == tool_call_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "original tool response resolved to "
            f"{len(matches)} actions for tool_call_id {tool_call_id!r}"
        )
    return matches[0]


def _tolerable_playwright_click_navigation_timeout(
    recorded_call: ReplayToolCall,
    recorded_observation: Mapping[str, Any],
    actual_observation: Mapping[str, Any],
    actual_outcome: Any,
) -> bool:
    """Treat a dispatched click as restored when only navigation settling timed out."""

    recorded_tool_name = str(recorded_call.function)
    if recorded_tool_name == "call_tool_mcp" and isinstance(
        recorded_call.arguments, Mapping
    ):
        recorded_tool_name = str(
            recorded_call.arguments.get("name") or recorded_tool_name
        )

    def navigation_settling_timed_out(observation: Mapping[str, Any]) -> bool:
        content = str(observation.get("content") or "")
        return bool(
            "TimeoutError" in content
            and "click action done" in content
            and "waiting for scheduled navigations to finish" in content
        )

    return bool(
        recorded_tool_name == "browser_click"
        and (
            (
                navigation_settling_timed_out(recorded_observation)
                and getattr(actual_outcome, "succeeded", False)
            )
            or (
                not observation_failed(recorded_observation)
                and navigation_settling_timed_out(actual_observation)
            )
        )
    )


def _retryable_playwright_navigation_failure(
    recorded_call: ReplayToolCall,
    actual_observation: Mapping[str, Any],
) -> bool:
    """Identify a transient Playwright startup navigation race.

    A newly attached browser can briefly replace the requested navigation with
    Chromium's error page while its service network settles.  Retrying the
    same idempotent navigation is safe; no other browser mutation qualifies.
    """

    recorded_tool_name = str(recorded_call.function)
    if recorded_tool_name == "call_tool_mcp" and isinstance(
        recorded_call.arguments, Mapping
    ):
        recorded_tool_name = str(
            recorded_call.arguments.get("name") or recorded_tool_name
        )
    content = str(actual_observation.get("content") or "").casefold()
    return bool(
        recorded_tool_name == "browser_navigate"
        and "interrupted by another navigation" in content
        and "chrome-error://chromewebdata/" in content
    )


def _safe_noop_playwright_failure(
    recorded_call: ReplayToolCall,
    recorded_observation: Mapping[str, Any],
) -> bool:
    """Identify failed browser calls that demonstrably changed no page state."""

    recorded_tool_name = str(recorded_call.function)
    if recorded_tool_name == "call_tool_mcp" and isinstance(
        recorded_call.arguments, Mapping
    ):
        recorded_tool_name = str(
            recorded_call.arguments.get("name") or recorded_tool_name
        )
    if not recorded_tool_name.startswith("browser_"):
        return False

    content = str(recorded_observation.get("content") or "").casefold()
    # These failures happen before Playwright can dispatch a page mutation.
    # Re-executing them during replay can turn a historical no-op into a real
    # click/form submission merely because the transport or DOM is healthier.
    if "session terminated" in content:
        return True
    if "not found in the current page snapshot" in content:
        return True
    if "does not match any elements" in content or "strict mode violation" in content:
        return True
    return bool(
        "timeouterror" in content
        and (
            "waiting for locator(" in content
            or "waiting for getbyrole(" in content
            or "waiting for getbytext(" in content
            or "waiting for getbylabel(" in content
            or "waiting for getbyplaceholder(" in content
            or "waiting for getbytestid(" in content
        )
        and "click action done" not in content
    )


def _tolerable_redundant_gitlab_login(
    recorded_call: ReplayToolCall,
    recorded_observation: Mapping[str, Any],
    actual_observation: Mapping[str, Any],
    actual_outcome: Any,
) -> bool:
    """Treat a login attempt as restored when replay is already authenticated."""

    recorded_tool_name = str(recorded_call.function)
    arguments: Mapping[str, Any] = recorded_call.arguments
    if recorded_tool_name == "call_tool_mcp" and isinstance(arguments, Mapping):
        recorded_tool_name = str(arguments.get("name") or recorded_tool_name)
        nested = arguments.get("arguments")
        if isinstance(nested, Mapping):
            arguments = nested
    serialized_arguments = json.dumps(arguments, ensure_ascii=False).casefold()
    actual_content = str(actual_observation.get("content") or "").casefold()
    return bool(
        recorded_tool_name in {"browser_fill_form", "browser_run_code_unsafe"}
        and "user_login" in serialized_arguments
        and "user_password" in serialized_arguments
        and not observation_failed(recorded_observation)
        and not getattr(actual_outcome, "succeeded", False)
        and "user_login" in actual_content
        and (
            "does not match any elements" in actual_content
            or "waiting for locator(" in actual_content
        )
    )


_PLAYWRIGHT_REF = re.compile(r"\[ref=([^\]]+)\]")
_PLAYWRIGHT_VISIBLE_ANCHOR_HREF = re.compile(
    r"a\[href=(?P<quote>[\"'])(?P<href>/[^\"']+)(?P=quote)\]"
)


def _mcp_observation_text(observation: Mapping[str, Any]) -> str:
    """Extract text blocks from a serialized MCP observation."""

    payload: Any = observation.get("content")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return payload
    if not isinstance(payload, Mapping):
        return ""
    blocks = payload.get("content")
    if not isinstance(blocks, list):
        return ""
    return "\n".join(
        str(block.get("text") or "")
        for block in blocks
        if isinstance(block, Mapping) and block.get("type") == "text"
    )


def _learn_playwright_ref_map(
    recorded_observation: Mapping[str, Any],
    actual_observation: Mapping[str, Any],
) -> dict[str, str]:
    """Align volatile Playwright snapshot refs using their semantic YAML line."""

    def indexed(text: str) -> dict[str, list[list[str]]]:
        rows: dict[str, list[list[str]]] = {}
        for line in text.splitlines():
            refs = _PLAYWRIGHT_REF.findall(line)
            if not refs:
                continue
            # Dynamic GitLab navigation can render the same accessible element
            # at a different tree depth after reset (for example, an expanded
            # Plan menu). Indentation is structural presentation, not element
            # identity, so exclude it from the semantic signature.
            signature = _PLAYWRIGHT_REF.sub("[ref]", line.lstrip())
            rows.setdefault(signature, []).append(refs)
        return rows

    recorded = indexed(_mcp_observation_text(recorded_observation))
    actual = indexed(_mcp_observation_text(actual_observation))
    learned: dict[str, str] = {}
    for signature, recorded_occurrences in recorded.items():
        actual_occurrences = actual.get(signature, [])
        for recorded_refs, actual_refs in zip(
            recorded_occurrences, actual_occurrences, strict=False
        ):
            if len(recorded_refs) != len(actual_refs):
                continue
            learned.update(zip(recorded_refs, actual_refs, strict=True))
    return learned


def _normalize_playwright_anchor_navigation(
    action: Any,
    recorded_observation: Mapping[str, Any],
) -> dict[str, str] | None:
    """Stabilize a recorded pure anchor navigation across responsive DOM variants."""

    if str(getattr(action, "name", "")) != "browser_run_code_unsafe":
        return None
    arguments = dict(getattr(action, "arguments", {}))
    code = arguments.get("code")
    if not isinstance(code, str) or not all(
        marker in code
        for marker in ("filter({visible:true}).click()", "return page.url()")
    ):
        return None
    match = _PLAYWRIGHT_VISIBLE_ANCHOR_HREF.search(code)
    if match is None or observation_failed(recorded_observation):
        return None
    href = match.group("href")
    recorded_text = _mcp_observation_text(recorded_observation)
    if "### Result" not in recorded_text or href not in recorded_text:
        return None
    normalized = (
        "async page => { await page.goto(new URL("
        + json.dumps(href)
        + ", page.url()).href); return page.url(); }"
    )
    arguments["code"] = normalized
    action.arguments = arguments
    return {
        "recorded": code,
        "actual": normalized,
        "reason": "pure_anchor_navigation_responsive_dom",
    }


def _rebind_playwright_action_refs(
    action: Any, ref_map: Mapping[str, str]
) -> list[dict[str, str]]:
    """Replace recorded snapshot refs without adding an out-of-band probe."""

    if not str(getattr(action, "name", "")).startswith("browser_"):
        return []
    rebindings: list[dict[str, str]] = []

    def rewrite(value: Any, *, key: str | None = None) -> Any:
        if key in {"target", "ref"} and isinstance(value, str):
            rebound = ref_map.get(value, value)
            if rebound != value:
                rebindings.append({"recorded": value, "actual": rebound})
            return rebound
        if isinstance(value, Mapping):
            return {name: rewrite(item, key=str(name)) for name, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        return value

    action.arguments = rewrite(dict(getattr(action, "arguments", {})))
    return rebindings


def _initialize_official_services(
    task_root: Path, dependencies: list[str], *, hostname: str,
    deployments_path: Path | str = DEFAULT_SERVICE_DEPLOYMENTS,
    forum_spec: Mapping[str, Any] | None = None,
    tac_spec: Mapping[str, Any] | None = None,
) -> None:
    """Apply the single-task seed using the official MTAR service conventions."""

    if "mcp-postgres" in dependencies:
        raise ValueError(
            "shared PostgreSQL reset is no longer supported; "
            "use an isolated environment or PostgreSQL lease"
        )
    if forum_spec is not None:
        from sead.environments.services.forum_pool import reset_forum_instance
        reset_forum_instance(forum_spec)
        dependencies = [dependency for dependency in dependencies if dependency != "reddit"]
    if tac_spec is not None:
        from sead.environments.services.tac_pool import reset_tac_pool_instance
        reset_tac_pool_instance(tac_spec)
        dependencies = [
            dependency for dependency in dependencies
            if dependency != tac_spec["service"]
        ]
    reset_tac_services(dependencies, hostname=hostname, deployments_path=deployments_path)


def reset_task_services(
    dataset_root: Path | str,
    task_id: str,
    *,
    hostname: str = "localhost",
    deployments_path: Path | str = DEFAULT_SERVICE_DEPLOYMENTS,
    forum_pool: Mapping[str, Any] | None = None,
    tac_pool_config: Path | str | None = None,
) -> dict[str, Any]:
    """Restore one task's shared services without starting an agent runtime."""

    task_root, row = load_task(dataset_root, task_id)
    dependencies = list(load_task_dependencies(task_root, str(row["tool"])))
    if "mcp-postgres" in dependencies:
        raise ValueError("shared PostgreSQL reset is no longer supported; use lease cleanup")
    from sead.environments.services.forum_pool import forum_instance
    forum_spec = forum_instance(
        {"forum_pool": forum_pool}, task_id, required="reddit" in dependencies
    ) if forum_pool is not None else None
    from sead.environments.services.tac_pool import tac_pool_instance
    tac_spec = tac_pool_instance(
        {"tac_pool_config": str(tac_pool_config)} if tac_pool_config else {},
        task_id, dependencies=dependencies,
    )
    operations = [
        f"service:{dependency}"
        for dependency in dependencies
        if dependency in SERVICE_RESET_ENDPOINTS
    ]
    if forum_spec is not None:
        operations.append("service:reddit")
    if tac_spec is not None:
        operations.append(f"tac-instance:{tac_spec['instance']}")
    postgres_lock = None
    web_lock = None
    try:
        postgres_lock = _acquire_postgres_lock(dependencies)
        if forum_spec is not None:
            web_lock = acquire_web_lock(dependencies, instance_key=task_id)
        elif tac_spec is not None:
            web_lock = acquire_web_lock(
                dependencies,
                service_instances={tac_spec["service"]: tac_spec["instance"]},
            )
        else:
            web_lock = _acquire_web_lock(dependencies)
        initialize_options = {
            "hostname": hostname,
            "deployments_path": deployments_path,
            "forum_spec": forum_spec,
        }
        if tac_spec is not None:
            initialize_options["tac_spec"] = tac_spec
        _initialize_official_services(task_root, dependencies, **initialize_options)
        if forum_spec is not None:
            from sead.environments.services.forum_pool import stop_forum_instance
            stop_forum_instance(forum_spec)
            operations.append("service:reddit:stopped-after-reset")
    finally:
        _release_postgres_lock(postgres_lock)
        _release_web_lock(web_lock)
    return {
        "schema_version": "mtar-task-service-reset-v1",
        "task_id": task_id,
        "dependencies": dependencies,
        "operations": operations,
        "succeeded": True,
    }




def _acquire_postgres_lock(dependencies: list[str]):
    """Serialize shared MTAR PostgreSQL seed, execution, and evaluation."""

    if "mcp-postgres" not in dependencies:
        return None
    lock_path = Path(tempfile.gettempdir()) / "sead-mtar-postgres.lock"
    handle = lock_path.open("a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def _release_postgres_lock(handle: Any) -> None:
    if handle is None:
        return
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _acquire_web_lock(dependencies: list[str]):
    """Serialize the shared Web reset, browser session, and evaluator state."""

    return acquire_web_lock(dependencies)


def _release_web_lock(handle: Any) -> None:
    release_web_lock(handle)


def _protect_terminal_session(runtime: Any) -> None:
    """Keep a task-level ``exit`` from killing OpenHands' persistent shell."""

    from openhands.events.action import CmdRunAction

    command = 'exit() { return "${1:-0}"; }; logout() { return "${1:-0}"; }'
    observation = runtime.run_action(CmdRunAction(command=command, hidden=True))
    if getattr(observation, "exit_code", 1) != 0:
        raise RuntimeError(f"terminal session guard failed: {observation.content}")


def _restore_rootfs_seed(runtime: Any, task_root: Path) -> None:
    """Copy the normalized /etc and /tmp seed into the new sandbox."""

    from openhands.events.action import CmdRunAction

    rootfs = task_root / "rootfs"
    for name in ("etc", "tmp"):
        source_root = rootfs / name
        for directory in sorted(
            (path for path in source_root.rglob("*") if path.is_dir()),
            key=lambda path: path.as_posix(),
        ):
            relative = directory.relative_to(source_root).as_posix()
            destination = f"/{name}/{relative}"
            observation = runtime.run_action(
                CmdRunAction(
                    command=f"mkdir -p -- {shlex.quote(destination)}",
                    hidden=True,
                )
            )
            if getattr(observation, "exit_code", 1) != 0:
                raise RuntimeError(f"rootfs seed mkdir failed: {observation.content}")
        for source in sorted(source_root.rglob("*"), key=lambda path: path.as_posix()):
            if not source.is_file():
                continue
            relative = source.relative_to(source_root)
            destination = Path("/") / name / relative.parent
            runtime.copy_to(str(source), str(destination) + "/", recursive=False)


def _restore(
    runtime: Any,
    request: MTARReplayRequest,
    *,
    mcp_tool_urls: Mapping[str, str],
    allow_outcome_drift: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from openhands.events.action import CmdRunAction, MessageAction
    from openhands.events.event import EventSource
    from openhands.events.serialization import event_to_dict, observation_from_dict
    from openhands.events.serialization.action import action_from_dict
    from openhands.runtime.utils.bash import split_bash_commands

    event_stream = runtime.event_stream
    restore_start_id = event_stream.cur_id
    calls: list[dict[str, Any]] = []
    execution = request.execution
    tool_timeout_seconds = float(execution.get("tool_timeout_seconds", 0))
    restore_timeout_seconds = float(
        execution.get(
            "prefix_replay_timeout_seconds",
            max(tool_timeout_seconds + 30.0, 30.0),
        )
    )
    if not (restore_timeout_seconds > 0) or not math.isfinite(restore_timeout_seconds):
        raise RuntimeError(
            "prefix_replay_timeout_seconds must be a finite positive number"
        )
    restore_deadline = time.monotonic() + restore_timeout_seconds
    playwright_ref_map: dict[str, str] = {}
    for turn_index, turn in enumerate(request.parent_replay_turns, 1):
        event_stream.add_event(
            MessageAction(content=turn.user_instruction), EventSource.USER
        )
        for message in turn.assistant_messages:
            recorded_calls = [call for call in message.tool_calls if call.native_action]
            if not recorded_calls and message.native_action:
                # Preserve assistant/message/finish boundaries in the replayed
                # OpenHands state, but only execute actual historical tool calls.
                event_stream.add_event(
                    action_from_dict(dict(message.native_action)),
                    EventSource.AGENT,
                )
                continue
            for recorded_call in recorded_calls:
                if time.monotonic() >= restore_deadline:
                    raise RuntimeError(
                        f"TOOL_RESTORE_FAILED turn {turn_index} "
                        f"action {recorded_call.id}: "
                        "prefix replay total deadline exceeded"
                    )
                native = recorded_call.native_action
                recorded_observation = dict(recorded_call.native_observation)
                action_started = time.monotonic()
                restore_mode = "executed"
                navigation_retry_observations: list[dict[str, Any]] = []
                try:
                    action = _original_tool_action(
                        recorded_call,
                        mcp_tool_names=list(mcp_tool_urls),
                    )
                    navigation_rebinding = None
                    ref_rebindings = []
                    if not is_tool_defense_block(recorded_observation):
                        navigation_rebinding = _normalize_playwright_anchor_navigation(
                            action,
                            dict(recorded_call.native_observation),
                        )
                        ref_rebindings = _rebind_playwright_action_refs(
                            action, playwright_ref_map
                        )
                    if is_tool_defense_block(recorded_observation):
                        # The proposal was never executed. Preserve the refusal
                        # and tool-call identity without dispatch or re-judging,
                        # including when this is a terminal hard replay.
                        observation = observation_from_dict(recorded_observation)
                        observation = _record_action_observation_without_dispatch(
                            event_stream, action, observation
                        )
                        restore_mode = "record_only_defense_block"
                    elif _record_only_restore_action(action):
                        if not recorded_observation:
                            raise RuntimeError(
                                "session-control action has no recorded observation"
                            )
                        observation = observation_from_dict(recorded_observation)
                        observation = _record_action_observation_without_dispatch(
                            event_stream, action, observation
                        )
                        restore_mode = "record_only_session_control"
                    elif _safe_noop_playwright_failure(
                        recorded_call, recorded_observation
                    ):
                        observation = observation_from_dict(recorded_observation)
                        observation = _record_action_observation_without_dispatch(
                            event_stream, action, observation
                        )
                        restore_mode = "record_only_known_noop"
                    else:
                        observation = _restore_action_observation(
                            runtime,
                            action,
                            wait_timeout_seconds=tool_timeout_seconds,
                            restore_deadline=restore_deadline,
                        )
                        if isinstance(action, CmdRunAction):
                            commands = split_bash_commands(action.command.strip())
                            if len(commands) > 1:
                                restore_mode = "record_only_source_rejection"
                            else:
                                restore_mode = "direct_runtime_shell"
                    actual = event_to_dict(observation)
                    for _retry in range(2):
                        if not _retryable_playwright_navigation_failure(
                            recorded_call, actual
                        ):
                            break
                        navigation_retry_observations.append(actual)
                        time.sleep(0.5)
                        retry_action = _original_tool_action(
                            recorded_call,
                            mcp_tool_names=list(mcp_tool_urls),
                        )
                        observation = _restore_action_observation(
                            runtime,
                            retry_action,
                            wait_timeout_seconds=tool_timeout_seconds,
                            restore_deadline=restore_deadline,
                        )
                        actual = event_to_dict(observation)
                        restore_mode = "executed_navigation_retry"
                except Exception as exc:
                    raise RuntimeError(
                        f"TOOL_RESTORE_FAILED turn {turn_index} "
                        f"action {recorded_call.id}: {exc}"
                    ) from exc
                action_row = dict(native)
                action_id = str(
                    action_row.get("id") or f"restore-{turn_index}-{len(calls) + 1}"
                )
                action_row.update({"id": action_id, "source": "agent"})
                actual.update({"cause": action_id, "source": "environment"})
                recorded_outcome = (
                    classify_native_observation(recorded_observation)
                    if recorded_observation
                    else classify_tool_outcome(
                        error=recorded_call.result.error,
                        content=recorded_call.result.content,
                    )
                )
                actual_outcome = classify_native_observation(actual)
                playwright_ref_map.update(
                    _learn_playwright_ref_map(recorded_observation, actual)
                )
                outcome_comparison = compare_tool_outcomes(
                    recorded_outcome, actual_outcome
                )
                call_row = {
                    "turn": turn_index,
                    "action": action_row,
                    "recorded_observation": recorded_observation,
                    "actual_observation": actual,
                    "outcome_comparison": outcome_comparison.to_dict(),
                    "playwright_ref_rebindings": ref_rebindings,
                    "playwright_navigation_rebinding": navigation_rebinding,
                    "playwright_navigation_retry_observations": (
                        navigation_retry_observations
                    ),
                    "restore_mode": restore_mode,
                    "restore_duration_seconds": round(
                        time.monotonic() - action_started, 6
                    ),
                    "bash_session_rebuilt": (
                        "Bash session exited and was recreated"
                        in str(actual.get("content") or "")
                    ),
                }
                tolerated_click_timeout = (
                    not outcome_comparison.equivalent
                    and _tolerable_playwright_click_navigation_timeout(
                        recorded_call,
                        recorded_observation,
                        actual,
                        actual_outcome,
                    )
                )
                tolerated_redundant_login = (
                    not outcome_comparison.equivalent
                    and _tolerable_redundant_gitlab_login(
                        recorded_call,
                        recorded_observation,
                        actual,
                        actual_outcome,
                    )
                )
                call_row["outcome_drift_tolerated"] = bool(
                    tolerated_click_timeout or tolerated_redundant_login
                )
                call_row["outcome_drift_tolerance"] = (
                    "playwright_click_navigation_timeout"
                    if tolerated_click_timeout
                    else (
                        "redundant_gitlab_login" if tolerated_redundant_login else None
                    )
                )
                calls.append(call_row)
                if (
                    not outcome_comparison.equivalent
                    and not allow_outcome_drift
                    and not tolerated_click_timeout
                    and not tolerated_redundant_login
                ):
                    raise RuntimeError(
                        f"TOOL_RESTORE_FAILED turn {turn_index}: "
                        f"{outcome_comparison.reason}; "
                        f"recorded={recorded_outcome.to_dict()} "
                        f"actual={actual_outcome.to_dict()} "
                        f"actual_observation={actual!r}"
                    )
    events = [
        event_to_dict(event)
        for event in event_stream.search_events(start_id=restore_start_id)
    ]
    return events, calls


def _restore_metrics(calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize structured restore timing and recovery audit fields."""

    return {
        "restore_action_count": len(calls),
        "restore_action_seconds": round(
            sum(float(call.get("restore_duration_seconds", 0)) for call in calls),
            6,
        ),
        "restore_record_only_count": sum(
            str(call.get("restore_mode", "")).startswith("record_only")
            for call in calls
        ),
        "bash_session_rebuild_count": sum(
            bool(call.get("bash_session_rebuilt")) for call in calls
        ),
    }


def _trajectory_turn(
    events: list[Mapping[str, Any]],
    instruction: str,
    *,
    recorded_instruction: str | None = None,
) -> ReplayTurn:
    start = -1
    for index, event in enumerate(events):
        if (
            event.get("source") == "user"
            and event.get("action") == "message"
            and str((event.get("args") or {}).get("content") or "") == instruction
        ):
            start = index
    if start < 0:
        raise RuntimeError(
            "Target trajectory does not contain the new user instruction"
        )
    active = events[start + 1 :]
    observations = {
        str(event.get("cause")): event for event in active if event.get("observation")
    }
    messages: list[ReplayAssistantMessage] = []
    for event in active:
        if event.get("source") != "agent" or not event.get("action"):
            continue
        args = event.get("args") if isinstance(event.get("args"), Mapping) else {}
        content = str(
            args.get("content") or args.get("thought") or event.get("message") or ""
        )
        action = str(event["action"])
        native = dict(event)
        call: ReplayToolCall | None = None
        if action not in {"message", "think", "finish", "reject", "change_agent_state"}:
            call_id = str(
                event.get("id")
                or (event.get("tool_call_metadata") or {}).get("tool_call_id")
                or f"call-{len(messages) + 1}"
            )
            observation = dict(observations.get(str(event.get("id")), {}))
            call = ReplayToolCall(
                id=call_id,
                function=action,
                arguments=dict(args),
                result=ReplayToolResult(
                    tool_call_id=call_id,
                    function=action,
                    content=observation.get("content", ""),
                    error=observation if observation_failed(observation) else None,
                ),
                native_action=native,
                native_observation=observation,
            )
        messages.append(
            ReplayAssistantMessage(
                content=content,
                tool_calls=(() if call is None else (call,)),
                native_action=native,
            )
        )
    if not messages:
        raise RuntimeError(
            "Target produced no assistant action for the new instruction"
        )
    return ReplayTurn(
        user_instruction=recorded_instruction or instruction,
        assistant_messages=tuple(messages),
    )


def _evaluator_command(
    source: bytes,
    declared_requirements: Sequence[str],
    *,
    wheelhouse: str,
) -> str:
    actual_requirements = evaluator_python_requirements(source)
    if tuple(declared_requirements) != actual_requirements:
        raise RuntimeError(
            "evaluator Python requirements do not match the normalized manifest"
        )
    if not actual_requirements:
        return "cd /grader && python evaluate_single.py"
    for value in (wheelhouse, *actual_requirements):
        if not value or any(character.isspace() for character in value):
            raise RuntimeError("unsafe evaluator wheelhouse requirement")
    packages = " ".join(actual_requirements)
    return (
        "python -m pip install --quiet --disable-pip-version-check --no-input "
        f"--no-index --find-links={wheelhouse} {packages} "
        "&& cd /grader && python evaluate_single.py"
    )


def _evaluate(
    runtime: Any,
    task_root: Path,
    trajectory: Path,
    worker_dir: Path,
    evaluator_requirements: Sequence[str],
    *,
    evaluator_wheelhouse: str,
    server_hostname: str,
    service_ports: Mapping[str, int] | None = None,
    evaluator_timeout_seconds: int = 300,
) -> dict[str, Any]:
    if evaluator_timeout_seconds <= 0:
        raise ValueError("evaluator timeout must be positive")
    entry = worker_dir / "evaluate_single.py"
    evaluator_input_dir = worker_dir / "evaluator_input"
    evaluator_input_dir.mkdir(exist_ok=True)
    evaluator_trajectory = evaluator_input_dir / "trajectory.json"
    atomic_write_json(
        evaluator_trajectory,
        evaluator_trajectory_payload(json.loads(trajectory.read_text())),
    )
    evaluator = task_root / "utils/evaluator.py"
    evaluator_source = evaluator.read_bytes()
    # Expansion releases carry task-private reference fixtures.
    expansion = (task_root / "acceptance.json").is_file()
    entry.write_text(
        build_evaluator_entrypoint(
            evaluator_source, evaluator_path="/grader/utils/evaluator.py",
        ) if expansion else build_evaluator_entrypoint(evaluator_source)
    )
    # DockerRuntime.copy_to preserves the basename of directory sources. Copy
    # the oracle file itself so its sandbox path is exactly /grader/evaluator.py.
    runtime.copy_to(str(evaluator), "/grader/", recursive=False)
    if expansion:
        runtime.copy_to(str(task_root / "utils"), "/grader/", recursive=True)
    runtime.copy_to(str(evaluator_trajectory), "/grader/", recursive=False)
    # Official evaluators vary: some consume the argument and others inspect
    # the conventional runner path in the workspace.
    runtime.copy_to(str(evaluator_trajectory), "/workspace/", recursive=False)
    runtime.copy_to(str(entry), "/grader/", recursive=False)
    from openhands.events.action import CmdRunAction

    service_environment = "".join(
        f" {name.upper()}_PORT={shlex.quote(str(port))}"
        for name, port in sorted((service_ports or {}).items())
    )
    evaluator_command = _evaluator_command(
        evaluator_source, evaluator_requirements, wheelhouse=evaluator_wheelhouse,
    )
    action = CmdRunAction(
        command=(
            f"env SERVER_HOSTNAME={shlex.quote(server_hostname)}{service_environment} bash -c "
            + shlex.quote(evaluator_command)
        ),
        is_static=True,
        hidden=True,
        cwd="/grader",
    )
    # A static shell is closed when run_action returns. OpenHands' default
    # nonblocking mode yields after 10 silent seconds, which would terminate
    # a still-running evaluator. Wait for completion with a bounded deadline.
    action.set_hard_timeout(evaluator_timeout_seconds, blocking=True)
    observation = runtime.run_action(action)
    atomic_write_json(
        worker_dir / "evaluator_observation.json",
        {
            "type": type(observation).__name__,
            "exit_code": getattr(observation, "exit_code", None),
            "content": str(getattr(observation, "content", "")),
        },
    )
    if getattr(observation, "exit_code", 1) != 0:
        raise RuntimeError(
            f"single evaluator failed (exit_code={getattr(observation, 'exit_code', None)}): "
            f"{observation.content}"
        )
    line = next(
        (
            row
            for row in reversed(str(observation.content).splitlines())
            if row.startswith(EVALUATION_MARKER)
        ),
        None,
    )
    if line is None:
        raise RuntimeError("single evaluator emitted no compatibility envelope")
    parsed = parse_evaluation(json.loads(line[len(EVALUATION_MARKER) :]))
    return parsed.to_dict()


def execute(
    request: MTARReplayRequest,
    *,
    phase_timer: ReplayPhaseTimer | None = None,
) -> MTARReplayWorkerResult:
    """Compatibility entry into the shared OpenHands replay engine."""
    from sead.environments.workers.openhands_engine import execute as run
    from sead.benchmarks.mtar.worker_adapter import MTARWorkerAdapter
    return run(request, MTARWorkerAdapter(), phase_timer=phase_timer)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--reset-only", action="store_true")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--task-id")
    parser.add_argument("--server-hostname", default="localhost")
    parser.add_argument("--service-deployments", type=Path, default=DEFAULT_SERVICE_DEPLOYMENTS)
    parser.add_argument("--snapshot-config", type=Path)
    args = parser.parse_args()
    if args.reset_only:
        missing = [
            name
            for name, value in (
                ("--dataset-root", args.dataset_root),
                ("--task-id", args.task_id),
            )
            if value is None
        ]
        if missing:
            parser.error("--reset-only requires " + ", ".join(missing))
        try:
            forum_pool = None
            tac_pool_config = None
            if args.snapshot_config is not None:
                import yaml
                snapshot = yaml.safe_load(args.snapshot_config.read_text(encoding="utf-8"))
                forum_pool = (snapshot.get("execution") or {}).get("forum_pool")
                tac_pool_config = (snapshot.get("execution") or {}).get("tac_pool_config")
            audit = reset_task_services(
                args.dataset_root,
                str(args.task_id),
                hostname=str(args.server_hostname),
                deployments_path=args.service_deployments,
                forum_pool=forum_pool,
                tac_pool_config=tac_pool_config,
            )
        except Exception as exc:  # noqa: BLE001 -- reset audit boundary
            traceback.print_exc()
            print(
                json.dumps(
                    {
                        "schema_version": "mtar-task-service-reset-v1",
                        "task_id": str(args.task_id),
                        "succeeded": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    indent=2,
                )
            )
            return 1
        print(json.dumps(audit, indent=2))
        return 0
    if args.request is None:
        parser.error("--request is required unless --reset-only is used")
    request = read_request(args.request)
    if args.check_only:
        try:
            print(json.dumps(_validate_request(request), indent=2))
            return 0
        except (MTARDatasetQuarantinedError, UnsupportedRuntimeProfile) as exc:
            print(
                json.dumps(
                    {
                        "status": exc.code,
                        "task_id": request.task_id,
                        "reason": exc.reason,
                    },
                    indent=2,
                )
            )
            return 2
    result_path = Path(request.worker_dir) / "result.json"
    phase_timings_path = Path(request.worker_dir) / "phase_timings.json"
    phase_timer = ReplayPhaseTimer()
    try:
        result = execute(request, phase_timer=phase_timer)
    except Exception as exc:  # noqa: BLE001 -- worker must persist every technical failure
        traceback.print_exc()
        defense_audit_path = Path(request.worker_dir) / "tool_defense.json"
        target_usage_path = Path(request.worker_dir) / "target_usage.json"
        target_usage = {}
        if target_usage_path.is_file():
            try:
                raw_target_usage = json.loads(target_usage_path.read_text(encoding="utf-8"))
                if isinstance(raw_target_usage, dict):
                    target_usage = raw_target_usage
            except (OSError, json.JSONDecodeError):
                pass
        infrastructure_status = (
            exc.code
            if isinstance(exc, (MTARDatasetQuarantinedError, UnsupportedRuntimeProfile))
            else None
        )
        result = MTARReplayWorkerResult(
            replay_id=request.replay_id,
            environment_id=request.environment_id,
            task_id=request.task_id,
            replay_turns=request.parent_replay_turns,
            restore_audit={
                "target_model_calls_during_restore": 0,
                "target_model_calls_total": int(target_usage.get("model_calls", 0)),
                "target_model_usage": target_usage,
                "succeeded": False,
            },
            cleanup_succeeded=False,
            cleanup_details={"error": f"{type(exc).__name__}: {exc}"},
            technical_error=f"{type(exc).__name__}: {exc}",
            infrastructure_status=infrastructure_status,
            artifacts={
                **(
                    {"tool_defense": str(defense_audit_path)}
                    if defense_audit_path.is_file()
                    else {}
                ),
                **(
                    {"target_usage": str(target_usage_path)}
                    if target_usage_path.is_file()
                    else {}
                ),
            },
        )
    phase_timings = phase_timer.snapshot()
    atomic_write_json(
        phase_timings_path,
        {
            "schema_version": "sead-replay-phase-timings-v1",
            "replay_id": request.replay_id,
            "phases_seconds": phase_timings,
            "restore": {
                name: result.restore_audit.get(name, 0)
                for name in (
                    "restore_action_count",
                    "restore_action_seconds",
                    "restore_record_only_count",
                    "bash_session_rebuild_count",
                )
            },
        },
    )
    result = dataclasses.replace(
        result,
        phase_timings=phase_timings,
        artifacts={
            **dict(result.artifacts),
            "phase_timings": str(phase_timings_path),
        },
    )
    atomic_write_json(result_path, result.to_dict())
    return 0 if result.technical_error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
