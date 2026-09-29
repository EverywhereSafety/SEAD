"""Observable trajectory projection for the runtime semantic judge."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sead.attacks.dart.models import ReplayTurn
from sead.benchmarks.mtar.controller_transcript import (
    project_controller_transcript,
)


_IGNORED_TARGET_MESSAGES = {
    "All done! What's next on the agenda?",
}


def _scrub_private_arguments(value: Any) -> Any:
    """Recursively remove private/model-only fields from observable arguments."""

    if isinstance(value, Mapping):
        return {
            str(key): _scrub_private_arguments(item)
            for key, item in value.items()
            if str(key).casefold()
            not in {"thought", "final_thought", "provider_metadata"}
        }
    if isinstance(value, list):
        return [_scrub_private_arguments(item) for item in value]
    return value


def project_trajectory(turns: Sequence[ReplayTurn]) -> list[dict[str, Any]]:
    """Project typed replay turns supplied by the running search."""
    events_by_turn = project_controller_transcript(turns)
    projected = [
        {
            "turn": index,
            "instruction": turn.user_instruction,
            "events": list(events_by_turn[index - 1]["events"]),
        }
        for index, turn in enumerate(turns, 1)
    ]
    for turn in projected:
        filtered_events = []
        for event in turn["events"]:
            if (
                event.get("type") == "target_message"
                and str(event.get("text") or "").strip()
                in _IGNORED_TARGET_MESSAGES
            ):
                continue
            if event.get("type") == "tool" and "arguments" in event:
                event["arguments"] = _scrub_private_arguments(event["arguments"])
            filtered_events.append(event)
        turn["events"] = filtered_events
    return projected
