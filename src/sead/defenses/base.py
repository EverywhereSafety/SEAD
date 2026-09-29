"""Thin PASS/BLOCK interfaces for typed tool actions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class Action(str, Enum):
    PASS = "pass"
    BLOCK = "block"


class BaseToolDefender(ABC):
    """Classify one proposed tool action using only preceding trajectory events."""

    @abstractmethod
    def predict(
        self,
        history: list[dict[str, Any]],
        tool_action: dict[str, Any],
    ) -> Action:
        """Return ``PASS`` or ``BLOCK`` for ``tool_action``."""

    def predict_batch(self, batch: list[dict[str, Any]]) -> list[Action]:
        """Shared batch convention with a tool-action field."""

        return [
            self.predict(item["history"], item["tool_action"])
            for item in batch
        ]

    def predict_batch_with_details(
        self,
        batch: list[dict[str, Any]],
    ) -> list[tuple[Action, dict[str, Any]]]:
        """Return decisions plus optional serializable audit details."""

        return [(decision, {}) for decision in self.predict_batch(batch)]




__all__ = ["Action", "BaseToolDefender"]
