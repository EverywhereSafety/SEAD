"""Load the unchanged PostgreSQL oracle implementation from MTAR-CKA."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[5] / "data/mtar/oracles/postgres/oracle.py"
_SPEC = importlib.util.spec_from_file_location("mtar_dart_postgres_oracle", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"MTAR-CKA oracle is unavailable: {_SOURCE}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

PostgresOracle = _MODULE.PostgresOracle
ORACLE_VERSION = _MODULE.ORACLE_VERSION
STATE_EFFECT_TASKS = _MODULE.STATE_EFFECT_TASKS
MUTATION_COHORT = _MODULE.MUTATION_COHORT
