"""OAS task and independent oracle hooks; environment ownership is shared."""


class OASWorkerAdapter:
    kind = "oas"

    def prepare(self, context):
        from sead.environments.workers import oas as worker
        self.worker = worker
        request = context.request
        context.validation = worker._validate_request(request)
        context.task = worker.load_task(request.dataset_root, request.task_id,
            selection_path=worker._selection_path(request), candidate_index=worker._candidate_index_path(request))
        context.task_root = context.task.root
        context.dependencies = list(worker._runtime_dependencies(context.task.dependencies))
        context.profile = worker._oas_profile(context.task.dependencies)
        worker.materialize_workspace(context.task, context.workspace)
        context.domain = None  # Preserve OAS's existing investigation contract.

    def initialize(self, context):
        self.worker.shared_worker._protect_terminal_session(context.runtime)

    def evaluate(self, context):
        options = {}
        if context.session and context.session.lease:
            options["environment_binding"] = context.session.evaluator_settings()
        return self.worker._evaluate(context.task, context.trajectory, context.worker_dir,
            context.workspace, evaluator_image=str(context.validation["evaluator_image"]), **options)
