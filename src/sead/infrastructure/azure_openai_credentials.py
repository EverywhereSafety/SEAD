"""Repository-wide Azure OpenAI credential loading."""

from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path

from .gemini_credentials import _ENV_UTILS

_load_azure_openai_api_key = _ENV_UTILS.load_azure_openai_api_key


def load_azure_openai_api_key(
    *,
    environ: MutableMapping[str, str] | None = None,
    project_env: Path | None = None,
    variable: str = "AZURE_OPENAI_API_KEY",
) -> str:
    """Load the Azure OpenAI key from the launch directory and propagate it."""

    return _load_azure_openai_api_key(
        environ=environ,
        project_env=project_env or Path.cwd() / ".env",
        variable=variable,
        prefer_project_env=True,
    )
