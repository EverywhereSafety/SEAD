"""Persistent execution semantics for actions rejected by the tool gate."""

from collections.abc import Mapping, Sequence
from typing import Any


# OpenHands serializes ErrorObservation.error_id in extras and preserves it
# through observation_from_dict. Tool output text is never a replay control flag.
TOOL_DEFENSE_BLOCK_ERROR_ID = "sead_tool_defense_block_v1"


def is_tool_defense_block(observation: Mapping[str, Any]) -> bool:
    extras = observation.get("extras")
    return (
        observation.get("observation") == "error"
        and isinstance(extras, Mapping)
        and extras.get("error_id") == TOOL_DEFENSE_BLOCK_ERROR_ID
    )


def blocked_action_ids(events: Sequence[Any]) -> set[str]:
    """Identify proposals that must not count as executed evaluator actions."""

    return {
        str(event["cause"])
        for event in events
        if isinstance(event, Mapping)
        and event.get("cause") is not None
        and is_tool_defense_block(event)
    }
