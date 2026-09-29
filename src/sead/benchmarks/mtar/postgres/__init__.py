"""Versioned PostgreSQL fixtures and state-based grading, outside Target."""

from .fixtures import (
    HARD_ORACLE_TASKS,
    JUDGE_ONLY_TASKS,
    SUPPORTED_TASKS,
    environment_spec,
)
from .oracle import PostgresOracle

__all__ = [
    "HARD_ORACLE_TASKS",
    "JUDGE_ONLY_TASKS",
    "SUPPORTED_TASKS",
    "PostgresOracle",
    "environment_spec",
]
