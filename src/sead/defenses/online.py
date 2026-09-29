"""Live Target gates; shared defender construction lives in defender.py."""

from __future__ import annotations
import asyncio
import json
import os
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any
from .sage import SAGEDefenseTool
from .base import Action, BaseToolDefender
from .inputs import (
    CONTEXT_POLICY_ID,
    NORMALIZER_VERSION,
    build_tool_defense_input,
)
from .replay import TOOL_DEFENSE_BLOCK_ERROR_ID
from .defender import (
    OnlineToolDefenseError as OnlineToolDefenseError,
    OpenAICompatibleBatchLLMClient as OpenAICompatibleBatchLLMClient,
    ONLINE_TOOL_DEFENDER_TYPES as ONLINE_TOOL_DEFENDER_TYPES,
    ONLINE_DEFENDER_TYPES as ONLINE_DEFENDER_TYPES,
    DEFAULT_BLOCKED_OBSERVATION as DEFAULT_BLOCKED_OBSERVATION,
    validate_online_tool_defense_config as validate_online_tool_defense_config,
    validate_online_defense_config as validate_online_defense_config,
    build_tool_defender as build_online_tool_defender,
)


class OnlineToolDefenseGate:
    """Apply decisions and persist an audit linked to trajectory action IDs."""

    def __init__(
        self,
        defender: BaseToolDefender,
        *,
        audit_path: Path | str | None = None,
        history_event_limit: int | None = None,
        history_max_chars: int | None = None,
        preserve_latest_user_message: bool = False,
        blocked_observation: str = DEFAULT_BLOCKED_OBSERVATION,
    ) -> None:
        if history_event_limit is not None and history_event_limit < 0:
            raise ValueError("history_event_limit must be non-negative or None")
        if history_max_chars is not None and history_max_chars < 1_000:
            raise ValueError("history_max_chars must be at least 1000 or None")
        self.defender = defender
        self.audit_path = Path(audit_path) if audit_path is not None else None
        self.history_event_limit = history_event_limit
        self.history_max_chars = history_max_chars
        self.preserve_latest_user_message = preserve_latest_user_message
        self.blocked_observation = blocked_observation
        self.records: list[dict[str, Any]] = []
        self.failure: str | None = None


    def _write(self) -> None:
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        counts = {"pass": 0, "block": 0, "error": 0}
        for record in self.records:
            counts[str(record["decision"])] += 1
        payload = {
            "schema_version": "online-tool-defense-audit-v1",
            "normalizer_version": NORMALIZER_VERSION,
            "context_policy_id": CONTEXT_POLICY_ID,
            "gate_stage": "tool_pre_execution",
            "defender": type(self.defender).__name__,
            "history_policy": {
                "history_event_limit": self.history_event_limit,
                "history_max_chars": self.history_max_chars,
                "preserve_latest_user_message": self.preserve_latest_user_message,
            },
            "summary": {
                "total_checked": len(self.records),
                "passed": counts["pass"],
                "blocked": counts["block"],
                "errors": counts["error"],
            },
            "checks": self.records,
        }
        temporary = self.audit_path.with_name(
            f".{self.audit_path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.audit_path)

    def decide(
        self,
        prior_events: Sequence[Mapping[str, Any]],
        action_event: Mapping[str, Any],
    ) -> Action | None:
        """Return ``None`` for non-tool events, otherwise a PASS/BLOCK decision."""

        sample = build_tool_defense_input(
            prior_events,
            action_event,
            history_event_limit=self.history_event_limit,
            history_max_chars=self.history_max_chars,
            preserve_latest_user_message=self.preserve_latest_user_message,
        )
        if sample is None:
            return None
        history, tool_action = sample["history"], sample["tool_action"]
        started = time.perf_counter()
        details: dict[str, Any] = {}
        try:
            predictions = list(self.defender.predict_batch_with_details([sample]))
            if len(predictions) != 1:
                raise ValueError("defender returned the wrong number of predictions")
            prediction, raw_details = predictions[0]
            if not isinstance(raw_details, Mapping):
                raise TypeError("defender audit details must be an object")
            details = dict(raw_details)
            # Malformed model responses are technical errors, not safety blocks.
            if details.get("parse_error"):
                raise ValueError(str(details["parse_error"]))
            decision = (
                prediction if isinstance(prediction, Action) else Action(prediction)
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            self.failure = f"{type(exc).__name__}: {exc}"
            self.records.append(
                {
                    "gate_stage": "tool_pre_execution",
                    "action_id": tool_action["action_id"],
                    "tool_action": tool_action,
                    "decision": "error",
                    "history_event_count": len(history),
                    "latency_seconds": round(elapsed, 6),
                    "error": self.failure,
                    "allowed": False,
                    "defender_details": details,
                    # Preserve the audit field used by defended DART runs.
                    "details": details,
                }
            )
            self._write()
            raise OnlineToolDefenseError(self.failure) from exc
        elapsed = time.perf_counter() - started
        self.records.append(
            {
                "gate_stage": "tool_pre_execution",
                "action_id": tool_action["action_id"],
                "tool_action": tool_action,
                "decision": decision.value,
                "history_event_count": len(history),
                "latency_seconds": round(elapsed, 6),
                "error": None,
                "allowed": decision is Action.PASS,
                "defender_details": details,
                "details": details,
            }
        )
        self._write()
        return decision




def _restore_instance_method(
    runtime: Any,
    name: str,
    *,
    had_instance_value: bool,
    instance_value: Any,
) -> None:
    if had_instance_value:
        setattr(runtime, name, instance_value)
    else:
        runtime.__dict__.pop(name, None)


@contextmanager
def guard_openhands_runtime(
    runtime: Any,
    defender: BaseToolDefender,
    *,
    audit_path: Path | str | None = None,
    history_event_limit: int | None = None,
    history_max_chars: int | None = None,
    preserve_latest_user_message: bool = False,
    blocked_observation: str = DEFAULT_BLOCKED_OBSERVATION,
):
    """Gate agent tool actions before either OpenHands execution path runs."""

    from openhands.events.observation import ErrorObservation
    from openhands.events.serialization import event_to_dict

    gate = OnlineToolDefenseGate(
        defender,
        audit_path=audit_path,
        history_event_limit=history_event_limit,
        history_max_chars=history_max_chars,
        preserve_latest_user_message=preserve_latest_user_message,
        blocked_observation=blocked_observation,
    )
    original_run_action = runtime.run_action
    original_call_tool_mcp = runtime.call_tool_mcp
    run_action_was_instance = "run_action" in runtime.__dict__
    call_tool_was_instance = "call_tool_mcp" in runtime.__dict__
    run_action_instance_value = runtime.__dict__.get("run_action")
    call_tool_instance_value = runtime.__dict__.get("call_tool_mcp")
    original_handle_action = getattr(runtime, "_handle_action", None)
    handle_action_was_instance = "_handle_action" in runtime.__dict__
    handle_action_instance_value = runtime.__dict__.get("_handle_action")

    def decide(action: Any) -> Action | None:
        action_row = event_to_dict(action)
        action_id = getattr(action, "id", None)
        prior_rows = [
            event_to_dict(event)
            for event in runtime.event_stream.search_events(start_id=0)
            if action_id is None or getattr(event, "id", action_id) < action_id
        ]
        return gate.decide(prior_rows, action_row)

    def run_action(self: Any, action: Any) -> Any:
        if decide(action) is Action.BLOCK:
            return ErrorObservation(
                content=gate.blocked_observation,
                error_id=TOOL_DEFENSE_BLOCK_ERROR_ID,
            )
        return original_run_action(action)

    async def call_tool_mcp(self: Any, action: Any) -> Any:
        # Keep the MCP owner loop alive while synchronous model/investigation
        # calls run. Browser investigation schedules reads onto this same loop.
        self._sead_mcp_loop = asyncio.get_running_loop()
        if await asyncio.to_thread(decide, action) is Action.BLOCK:
            return ErrorObservation(
                content=gate.blocked_observation,
                error_id=TOOL_DEFENSE_BLOCK_ERROR_ID,
            )
        return await original_call_tool_mcp(action)

    async def handle_action(self: Any, action: Any) -> Any:
        # Ordinary tools run in an executor too. Capture their owning loop even
        # if the first Target action is bash and no MCP call has happened yet.
        self._sead_mcp_loop = asyncio.get_running_loop()
        return await original_handle_action(action)

    runtime.run_action = MethodType(run_action, runtime)
    runtime.call_tool_mcp = MethodType(call_tool_mcp, runtime)
    if callable(original_handle_action):
        runtime._handle_action = MethodType(handle_action, runtime)
    completed_normally = False
    try:
        yield gate
        completed_normally = True
    finally:
        _restore_instance_method(
            runtime,
            "run_action",
            had_instance_value=run_action_was_instance,
            instance_value=run_action_instance_value,
        )
        _restore_instance_method(
            runtime,
            "call_tool_mcp",
            had_instance_value=call_tool_was_instance,
            instance_value=call_tool_instance_value,
        )
        if callable(original_handle_action):
            _restore_instance_method(
                runtime,
                "_handle_action",
                had_instance_value=handle_action_was_instance,
                instance_value=handle_action_instance_value,
            )
        gate._write()
    if completed_normally and gate.failure is not None:
        raise OnlineToolDefenseError(gate.failure)


def configured_openhands_tool_defense(
    runtime: Any,
    value: Mapping[str, Any] | None,
    *,
    audit_path: Path | str,
    investigation_domain: str | None = None,
    filesystem_container_name: str | None = None,
):
    """Return the configured live guard, or a no-op context when disabled."""

    config = validate_online_tool_defense_config(
        value,
        require_credentials=True,
    )
    if not config["enabled"]:
        return nullcontext()
    extra_tools: Sequence[SAGEDefenseTool] = ()
    investigation = config.get("environment_investigation", {})
    if investigation is not None and not isinstance(investigation, Mapping):
        raise TypeError("defense.tool.environment_investigation must be an object")
    investigation = dict(investigation or {})
    enabled = investigation.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("environment_investigation.enabled must be boolean")
    if enabled:
        if config["type"] != "sage":
            raise ValueError("environment investigation requires SAGE")
        from .environment_tools import (
            DEFAULT_ENVIRONMENT_ROOTS,
            OpenHandsEnvironmentInspector,
        )

        roots = investigation.get("allowed_roots", DEFAULT_ENVIRONMENT_ROOTS)
        if not isinstance(roots, list | tuple) or any(
            not isinstance(root, str) for root in roots
        ):
            raise TypeError("environment_investigation.allowed_roots must be strings")
        if investigation_domain is not None:
            from .domain_tools import contract_defender, domain_tools

            inspection_runtime = runtime
            if investigation_domain == "Filesystem":
                if not filesystem_container_name:
                    raise ValueError("filesystem investigation requires the replay's sidecar")
                inspection_runtime = SimpleNamespace(
                    container=runtime.docker_client.containers.get(filesystem_container_name)
                )
            if investigation_domain == "PostgreSQL" and getattr(
                runtime, "sead_postgres_lease", None
            ) is None:
                raise ValueError("database investigation requires the replay's PostgreSQL lease")
            defender = contract_defender(
                config,
                domain_tools(inspection_runtime, investigation_domain, investigation),
                investigation_domain,
            )
        else:
            extra_tools = OpenHandsEnvironmentInspector(
                runtime,
                allowed_roots=roots,
                max_read_lines=int(investigation.get("max_read_lines", 400)),
                max_search_matches=int(investigation.get("max_search_matches", 200)),
            ).tools()
            defender = build_online_tool_defender(config, extra_tools=extra_tools)
    else:
        defender = build_online_tool_defender(config)
    return guard_openhands_runtime(
        runtime,
        defender,
        audit_path=audit_path,
        history_event_limit=config.get("history_event_limit"),
        history_max_chars=config.get("history_max_chars"),
        preserve_latest_user_message=config.get("preserve_latest_user_message", False),
        blocked_observation=str(
            config.get("blocked_observation") or DEFAULT_BLOCKED_OBSERVATION
        ),
    )




__all__ = [
    "DEFAULT_BLOCKED_OBSERVATION",
    "ONLINE_DEFENDER_TYPES",
    "ONLINE_TOOL_DEFENDER_TYPES",
    "OnlineToolDefenseError",
    "OnlineToolDefenseGate",
    "OpenAICompatibleBatchLLMClient",
    "build_online_tool_defender",
    "configured_openhands_tool_defense",
    "guard_openhands_runtime",
    "validate_online_defense_config",
    "validate_online_tool_defense_config",
]
