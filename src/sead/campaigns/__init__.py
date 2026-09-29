"""Shared campaign orchestration primitives.

Benchmark-specific runners own task selection, success policy, and cleanup.
This package owns the concurrency and durable-artifact mechanics that must be
consistent across those runners.
"""

from .infrastructure import (
    CampaignExecutor,
    acquire_campaign_lock,
    atomic_write_json,
    read_json_object,
    resource_aware_results,
    task_slug,
    utc_now,
)

__all__ = [
    "CampaignExecutor",
    "acquire_campaign_lock",
    "atomic_write_json",
    "read_json_object",
    "resource_aware_results",
    "task_slug",
    "utc_now",
]
