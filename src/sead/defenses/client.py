"""Shared model-client protocol for the SAGE defender."""

from collections.abc import Sequence
from typing import Protocol


class BatchLLMClient(Protocol):
    """Model-client interface used by SAGE."""

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int | None = None,
        temperature: float = 0.0,
        top_p: float | None = None,
    ) -> Sequence[str]: ...

    def generate_batch_with_messages(
        self,
        messages_list: list[list[dict[str, str]]],
        max_tokens: int | None = None,
        temperature: float = 0.0,
        top_p: float | None = None,
    ) -> Sequence[str]: ...
