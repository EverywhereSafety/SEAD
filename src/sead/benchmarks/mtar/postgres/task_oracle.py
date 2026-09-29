"""Select task-owned expansion oracles without changing frozen legacy datasets."""
import hashlib
import importlib.util
from pathlib import Path

from .fixtures import HARD_ORACLE_TASKS, _lease
from .oracle import PostgresOracle


def has_hard_oracle(task_root: Path) -> bool:
    task_root = Path(task_root)
    spec = _lease(task_root)
    return bool(spec and spec['hard_oracle'] and (
        task_root.name in HARD_ORACLE_TASKS or
        (task_root / 'utils/postgres_state_oracle.py').is_file()))


def create_postgres_oracle(task_id, task_root, lease, *, workspace=None):
    """Called only after the normal dataset/release validation gate.

    The module is a trusted, digest-bound benchmark asset, never Target code.
    Existing tasks keep the exact original implementation and result schema.
    """
    task_root = Path(task_root)
    if task_id != task_root.name or not has_hard_oracle(task_root):
        raise ValueError('no verified hard oracle for ' + task_id)
    source = task_root / 'utils/postgres_state_oracle.py'
    if not source.is_file():
        return PostgresOracle(task_id, lease, workspace=workspace)
    if source.is_symlink() or not source.resolve().is_relative_to(task_root.resolve()):
        raise ValueError('task oracle escapes its dataset')
    name = 'mtar_task_oracle_' + hashlib.sha256(source.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ValueError('cannot load task-owned oracle')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PostgresOracle(task_id, lease, workspace=workspace)
