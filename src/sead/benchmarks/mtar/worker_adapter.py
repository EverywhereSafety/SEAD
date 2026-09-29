"""MTAR task/seed/oracle hooks for the shared OpenHands execution engine."""

import shutil


class MTARWorkerAdapter:
    kind = "mtar"

    def prepare(self, context):
        from sead.environments.workers import mtar as worker
        self.worker = worker
        request = context.request
        context.validation = worker._validate_request(request)
        context.task_root, context.row = worker.load_task(request.dataset_root, request.task_id)
        context.profile, context.evaluator_requirements = worker.resolve_task_profile(context.task_root)
        context.dependencies = list(worker.load_task_dependencies(context.task_root, str(context.row["tool"])))
        source = context.task_root / "workspace"
        if source.is_dir():
            shutil.copytree(source, context.workspace)
        else:
            context.workspace.mkdir()
        from sead.defenses.domain_tools import domain_for_tool
        context.domain = domain_for_tool(context.row["tool"])

    def initialize(self, context):
        worker = self.worker
        runtime = context.runtime
        if context.postgres_lease:
            from sead.benchmarks.mtar.postgres.task_oracle import has_hard_oracle, create_postgres_oracle
            if has_hard_oracle(context.task_root):
                context.postgres_oracle = create_postgres_oracle(
                    context.request.task_id, context.task_root, context.postgres_lease,
                    workspace=context.workspace)
        worker._restore_rootfs_seed(runtime, context.task_root)
        worker._protect_terminal_session(runtime)
        runtime.copy_to(str(context.task_root / "utils/dependencies.yml"), "/utils/", recursive=False)
        pre_init = context.task_root / "pre_init.py"
        if pre_init.is_file():
            from openhands.events.action import CmdRunAction
            runtime.copy_to(str(pre_init), "/utils/", recursive=False)
            observation = runtime.run_action(CmdRunAction(command="python /utils/pre_init.py"))
            if getattr(observation, "exit_code", 1) != 0:
                raise RuntimeError(f"pre_init.py failed: {observation.content}")

    def evaluate(self, context):
        worker = self.worker
        if context.postgres_oracle:
            oracle = context.postgres_oracle
            result = oracle.evaluate()
            worker.atomic_write_json(context.worker_dir / "postgres_evidence.json", {
                "baseline": oracle.baseline, "final": oracle.last_snapshot, "evaluation": result})
            return result
        if context.postgres_lease:
            from sead.benchmarks.mtar.postgres.fixtures import unscored_evaluation
            if context.postgres_lease.sql("SELECT 1;") != "1":
                raise RuntimeError("PostgreSQL lease health check failed")
            result = unscored_evaluation()
            worker.atomic_write_json(context.worker_dir / "postgres_evidence.json", {
                "evaluation": result, "execution": context.postgres_lease.evidence()})
            return result
        settings = context.session.evaluator_settings() if context.session else {}
        return worker._evaluate(context.runtime, context.task_root, context.trajectory,
            context.worker_dir, context.evaluator_requirements,
            evaluator_wheelhouse=context.profile.evaluator_wheelhouse,
            server_hostname=settings.get("server_hostname", context.server_hostname),
            service_ports=settings.get("service_ports", {context.tac_spec["service"]: int(context.tac_spec["port"])}
                if context.tac_spec else None),
            evaluator_timeout_seconds=int(context.request.execution.get("tool_timeout_seconds", 300)))
