"""One OpenHands replay lifecycle for MTAR and OAS.

Each invocation runs in its own subprocess. Adapters own task semantics;
this engine owns services, runtime, replay, model dispatch and cleanup.
"""

from __future__ import annotations

import asyncio
import json
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

from . import mtar as common
from .protocol import (
    ReplayPhaseTimer,
    MTARReplayWorkerResult,
    atomic_write_json,
)
from ..registry import environment_config
from ..session import EnvironmentSession


_CUMULATIVE_TOKEN_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
)


def _legacy_target_usage(metrics):
    """Read usage from the controller state used by older OpenHands versions."""
    if metrics is None:
        return {"model_calls": 0, "accumulated_cost": 0, "records": []}
    raw = metrics.get()
    accumulated = dict(raw.get("accumulated_token_usage") or {})
    records = [dict(row) for row in raw.get("token_usages", [])]
    return {
        "model_calls": len(records),
        "accumulated_cost": raw.get("accumulated_cost", 0),
        **{key: value for key, value in accumulated.items()
           if key != "response_id" and type(value) in {int, float}},
        "records": records,
    }


def target_usage_from_events(rows, *, first_new_event_id, model, fallback_metrics=None):
    """Recover per-call Target usage from action metric snapshots.

    Current OpenHands keeps the live model metrics in ``Action.llm_metrics``
    while ``ControllerState.metrics`` can remain empty.  Each action snapshot is
    cumulative for the current controller invocation; consecutive distinct
    snapshots therefore identify model calls.  Restored prefix events are
    excluded by the event-stream boundary captured immediately before rollout.
    """
    previous = {key: 0 for key in _CUMULATIVE_TOKEN_FIELDS}
    previous_cost = 0.0
    final_accumulated = {}
    records = []
    for row in rows:
        try:
            event_id = int(row.get("id"))
        except (TypeError, ValueError):
            continue
        if event_id < first_new_event_id or row.get("source") != "agent":
            continue
        snapshot = row.get("llm_metrics")
        if not isinstance(snapshot, dict):
            continue
        accumulated = snapshot.get("accumulated_token_usage")
        if not isinstance(accumulated, dict):
            continue
        current = {
            key: accumulated.get(key, 0)
            for key in _CUMULATIVE_TOKEN_FIELDS
        }
        if any(type(value) not in {int, float} for value in current.values()):
            continue
        cost = snapshot.get("accumulated_cost", previous_cost)
        if type(cost) not in {int, float}:
            cost = previous_cost
        if any(current[key] < previous[key] for key in _CUMULATIVE_TOKEN_FIELDS) or cost < previous_cost:
            raise ValueError("Target action metrics are not cumulative and monotonic")
        if current == previous and cost == previous_cost:
            continue

        metadata = row.get("tool_call_metadata")
        response = metadata.get("model_response") if isinstance(metadata, dict) else None
        response = response if isinstance(response, dict) else {}
        record = {
            "model": str(response.get("model") or model),
            **{key: current[key] - previous[key] for key in _CUMULATIVE_TOKEN_FIELDS},
            "context_window": accumulated.get("context_window", 0),
            "per_turn_token": accumulated.get(
                "per_turn_token",
                current["prompt_tokens"] - previous["prompt_tokens"]
                + current["completion_tokens"] - previous["completion_tokens"],
            ),
            "response_id": str(response.get("id") or ""),
            "event_id": event_id,
            "cost": cost - previous_cost,
        }
        records.append(record)
        previous = current
        previous_cost = cost
        final_accumulated = {
            key: value for key, value in accumulated.items()
            if key != "response_id" and type(value) in {int, float}
        }

    if records:
        return {
            "model_calls": len(records),
            "accumulated_cost": previous_cost,
            **final_accumulated,
            "records": records,
        }
    return _legacy_target_usage(fallback_metrics)


def has_agent_action_since(rows, first_event_id):
    """Return whether the current rollout, rather than its prefix, acted."""
    for row in rows:
        try:
            event_id = int(row.get("id"))
        except (AttributeError, TypeError, ValueError):
            continue
        if (
            event_id >= first_event_id
            and row.get("source") == "agent"
            and row.get("action") not in {"system", None}
        ):
            return True
    return False


def _services(c):
    request = c.request
    deps = c.dependencies
    if "mcp-filesystem" in deps:
        from sead.infrastructure.filesystem_mcp import FilesystemMCPService, prepare_filesystem_workspace
        prepare_filesystem_workspace(c.workspace)
        c.filesystem = FilesystemMCPService(c.workspace, environment_id=request.environment_id,
            output_dir=c.worker_dir, node_image=common.resolve_service_image("filesystem-mcp"))
        c.filesystem_url = c.filesystem.start()
    if environment_config(request.execution):
        c.session = EnvironmentSession(request, deps, c.task_root, c.workspace)
        c.session.__enter__()
        c.session.install_runtime_network()
        if c.session.service:
            dependency = "mcp-postgres" if c.session.service == "postgres" else "mcp-playwright"
            c.mcp_urls[dependency] = c.session.binding["host_mcp_url"]
        if c.session.service == "postgres":
            c.postgres_lease = c.session.lease
        return
    # Explicit legacy compatibility: no leased request ever enters this branch.
    from sead.environments.leases import LeaseClient, postgres_mode
    from sead.environments.services.forum_pool import forum_instance
    from sead.environments.services.tac_pool import tac_pool_instance
    c.mcp_urls = common.external_mcp_urls(deps, hostname=c.server_hostname)
    pg_leased = "mcp-postgres" in deps and postgres_mode(request.execution) == "leased"
    shared_deps = [d for d in deps if not (pg_leased and d == "mcp-postgres")]
    if pg_leased:
        from sead.benchmarks.mtar.postgres import environment_spec
        from sead.environments.services.docker_relay import install_openhands_lease_network
        if set(deps) != {"mcp-postgres"} or c.profile.cap_add:
            raise ValueError("leased PostgreSQL requires a standalone, unprivileged task")
        pg = request.execution["postgres"]
        c.postgres_lease = LeaseClient(pg["manager_socket"]).acquire(
            environment_spec(request.task_id, c.task_root),
            {key: getattr(request, key) for key in ("run_id", "task_id", "node_id", "replay_id")},
            request_id=request.replay_id, wait_seconds=float(pg.get("acquire_timeout_seconds", 300)))
        atomic_write_json(c.worker_dir / "postgres_lease.json", c.postgres_lease.audit())
        c.mcp_urls["mcp-postgres"] = c.postgres_lease.binding["host_mcp_url"]
        c.relays = install_openhands_lease_network(c.postgres_lease.binding["network"], c.postgres_lease.binding["labels"])
    c.postgres_lock = common._acquire_postgres_lock(shared_deps)
    c.forum_spec = forum_instance(request.execution, request.task_id,
        required="reddit" in deps and "forum_pool" in request.execution)
    c.tac_spec = tac_pool_instance(request.execution, request.task_id, dependencies=deps)
    if c.forum_spec:
        c.web_lock = common.acquire_web_lock(deps, instance_key=request.task_id)
    elif c.tac_spec:
        c.web_lock = common.acquire_web_lock(deps, service_instances={c.tac_spec["service"]: c.tac_spec["instance"]})
    else:
        c.web_lock = common._acquire_web_lock(deps)
    common._initialize_official_services(c.task_root, shared_deps, hostname=c.server_hostname,
        deployments_path=request.execution.get("service_deployments_path", common.DEFAULT_SERVICE_DEPLOYMENTS),
        forum_spec=c.forum_spec, tac_spec=c.tac_spec)
    if "mcp-playwright" in deps:
        from sead.infrastructure.playwright_mcp import PlaywrightMCPService
        c.browser = PlaywrightMCPService(c.workspace, environment_id=request.environment_id,
            output_dir=c.worker_dir, image=common.resolve_service_image("playwright-mcp"),
            port=(c.forum_spec["browser_port"] if c.forum_spec else
                  c.tac_spec["browser_port"] if c.tac_spec else common.playwright_port(deps)))
        c.mcp_urls["mcp-playwright"] = c.browser.start()


def _rollout(c):
    from openhands.core.main import run_controller
    from openhands.events.action import MessageAction
    from openhands.events.serialization import event_to_dict
    from sead.defenses import configured_openhands_tool_defense
    from sead.environments.services.tac_pool import rewrite_service_urls
    request = c.request
    context = (c.session.service_context() if c.session else common.service_instruction_context(c.dependencies,
        deployments_path=request.execution.get("service_deployments_path", common.DEFAULT_SERVICE_DEPLOYMENTS)))
    instruction = request.new_instruction + context
    instruction = c.session.transform(instruction) if c.session else rewrite_service_urls(instruction, c.tac_spec)
    first_new_event_id = c.runtime.event_stream.cur_id
    with c.timer.phase("target_execution"), configured_openhands_tool_defense(
        c.runtime, c.defense["tool"], audit_path=c.defense_audit, investigation_domain=c.domain,
        filesystem_container_name=c.filesystem.name if c.filesystem else None,
    ):
        state = asyncio.run(run_controller(config=c.config, sid=request.environment_id,
            initial_user_action=MessageAction(content=instruction), runtime=c.runtime, exit_on_message=True))
    state_rows = ([event_to_dict(event) for event in state.history]
                  if state is not None else [])
    if state is not None:
        c.target_usage = target_usage_from_events(
            state_rows,
            first_new_event_id=first_new_event_id,
            model=str(request.target["model"]),
            fallback_metrics=getattr(state, "metrics", None),
        )
        atomic_write_json(c.worker_dir / "target_usage.json", {
            "schema_version": "sead-target-usage-v1",
            "model": str(request.target["model"]),
            **c.target_usage,
        })
    if not c.trajectory.is_file():
        if state is None:
            raise RuntimeError("OpenHands returned no state or trajectory")
        c.trajectory.write_text(json.dumps(state_rows, indent=2) + "\n")
    rows = json.loads(c.trajectory.read_text())
    if state is not None and str(getattr(state, "last_error", "")).strip():
        from sead.infrastructure.gemini_compat import raise_on_signature_error
        raise_on_signature_error(state.last_error)
        if not has_agent_action_since(rows, first_new_event_id):
            message = "Target model returned no action: " + str(state.last_error).strip()
            rows.append({"id": max((int(row.get("id", -1)) for row in rows if str(row.get("id", "")).lstrip("-").isdigit()), default=-1)+1,
                         "source": "agent", "action": "message", "message": message,
                         "args": {"content": message, "wait_for_response": False}})
    c.trajectory.write_text(json.dumps(rows, indent=2) + "\n")
    return common._trajectory_turn(rows, instruction, recorded_instruction=request.new_instruction)




def _restore_prefix(c, *, mcp_tools):
    """Restore the recorded replay prefix."""

    with c.timer.phase("prefix_restore"):
        return common._restore(
            c.runtime,
            c.request,
            mcp_tool_urls=mcp_tools,
            allow_outcome_drift=(
                c.request.execution.get("allow_restore_outcome_drift") is True
            ),
        )


def execute(request, adapter, *, phase_timer=None):
    if request.benchmark_kind is not None and request.benchmark_kind != adapter.kind:
        raise ValueError("request benchmark does not match worker adapter")
    timer = phase_timer or ReplayPhaseTimer()
    worker_dir = Path(request.worker_dir)
    c = SimpleNamespace(request=request, timer=timer, worker_dir=worker_dir,
        workspace=worker_dir / "workspace", trajectory=worker_dir / "trajectory.json",
        runtime=None, session=None, postgres_lease=None, postgres_oracle=None,
        filesystem=None, filesystem_url=None, browser=None, mcp_urls={},
        target_usage={},
        postgres_lock=None, web_lock=None, relays=[], forum_spec=None, tac_spec=None,
        server_hostname=str(request.execution.get("server_hostname") or "localhost"),
        defense_audit=worker_dir / "tool_defense.json")
    with timer.phase("validation"):
        adapter.prepare(c)
        c.defense = c.validation["defense"]
    restore_calls = []
    replay_turns = list(request.parent_replay_turns)
    previous_alarm = None
    try:
        with timer.phase("worker_imports"):
            sys.path.insert(0, request.openhands_root)
            import openhands.memory.memory as memory_module
            from openhands.core.main import create_runtime
            from openhands.utils.async_utils import call_async_from_sync
        microagents = worker_dir / "openhands_microagents"
        microagents.mkdir()
        memory_module.GLOBAL_MICROAGENTS_DIR = str(microagents)
        memory_module.USER_MICROAGENTS_DIR = str(microagents)
        with timer.phase("environment_acquire"):
            _services(c)
        if c.session:
            def replay_expired(_signum, _frame):
                raise TimeoutError("REPLAY_TIMEOUT: replay execution budget exceeded")
            previous_alarm = signal.signal(signal.SIGALRM, replay_expired)
            signal.setitimer(signal.ITIMER_REAL, float(request.execution["sample_timeout_seconds"]))
        binding = c.session.binding if c.session and c.session.lease else c.postgres_lease.binding if c.postgres_lease else None
        c.config = common._make_config(request, c.workspace, c.trajectory, None,
            runtime_profile=c.profile, base_container_image=str(c.validation["openhands_base_image"]),
            filesystem_mcp_url=c.filesystem_url, external_mcp_server_urls=c.mcp_urls, lease_binding=binding)
        from sead.infrastructure.docker_cleanup import install_openhands_scoped_cleanup
        from sead.infrastructure.openhands_mcp_sessions import install_persistent_mcp_sessions
        with timer.phase("runtime_boot"):
            install_openhands_scoped_cleanup(request.environment_id)
            c.runtime = create_runtime(c.config, sid=request.environment_id, headless_mode=True)
            if c.postgres_lease:
                c.runtime.sead_postgres_lease = c.postgres_lease
            install_persistent_mcp_sessions(c.runtime, timeout_seconds=float(request.execution.get("tool_timeout_seconds", 300)))
            call_async_from_sync(c.runtime.connect)
            if c.session:
                c.session.attach_runtime(c.runtime)
        with timer.phase("environment_prepare"):
            adapter.initialize(c)
        mcp_tools = {tool: url for dependency, url in c.mcp_urls.items() for tool in common.TOOL_SCHEMAS[dependency]}
        if c.filesystem_url:
            mcp_tools.update({tool: c.filesystem_url for tool in common.TOOL_SCHEMAS["mcp-filesystem"]})
        restored_events, restore_calls = _restore_prefix(
            c, mcp_tools=mcp_tools
        )
        c.restored_events = restored_events
        if request.new_instruction is not None:
            replay_turns.append(_rollout(c))
        else:
            c.trajectory.write_text(json.dumps(restored_events, indent=2) + "\n")
        if c.session:
            c.session.check()
        with timer.phase("evaluator"):
            evaluation = adapter.evaluate(c)
        if c.session:
            c.session.check()
        elif c.postgres_lease:
            c.postgres_lease.check()
        result_fields = dict(environment_id=request.environment_id,
            task_id=request.task_id, replay_turns=tuple(replay_turns),
            restore_audit={"target_model_calls_during_restore": 0,
                "target_model_calls_total": int(c.target_usage.get("model_calls", 0)),
                "target_model_usage": dict(c.target_usage),
                "restored_tool_calls": restore_calls, **common._restore_metrics(restore_calls),
                "outcome_drift_allowed": request.execution.get("allow_restore_outcome_drift") is True,
                "outcome_drift_count": sum(not bool(row["outcome_comparison"]["equivalent"]) for row in restore_calls),
                "succeeded": True}, evaluation=evaluation, cleanup_succeeded=True,
            cleanup_details={"runtime_close_attempted": True},
            artifacts={"trajectory": str(c.trajectory), "workspace": str(c.workspace),
                **{key: str(worker_dir / filename) for key, filename in {
                    "tool_defense": "tool_defense.json",
                    "initial_metadata": "initial_metadata.json", "environment_lease": "environment_lease.json",
                    "target_usage": "target_usage.json",
                }.items() if (worker_dir / filename).is_file()}})
        return MTARReplayWorkerResult(replay_id=request.replay_id, **result_fields)
    finally:
        if previous_alarm is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_alarm)
        errors = []
        with timer.phase("cleanup"):
            callbacks = []
            if c.runtime:
                callbacks.append(c.runtime.close)
            callbacks.extend(service.stop for service in (c.filesystem, c.browser) if service)
            callbacks.extend(relay.close for relay in c.relays)
            if c.session:
                callbacks.append(c.session.close)
            elif c.postgres_lease:
                callbacks.append(c.postgres_lease.close)
            for callback in callbacks:
                try:
                    callback()
                except Exception as exc:
                    errors.append(exc)
            if c.postgres_lease and not c.session and c.postgres_lease.released:
                atomic_write_json(worker_dir / "postgres_cleanup.json", {**c.postgres_lease.audit(), "succeeded": True})
            common._release_postgres_lock(c.postgres_lock)
            common._release_web_lock(c.web_lock)
        if errors:
            raise RuntimeError("environment cleanup failed: " + str(errors[0])) from errors[0]
