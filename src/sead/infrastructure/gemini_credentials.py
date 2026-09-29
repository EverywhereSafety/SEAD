"""Gemini credential loading for SEAD."""

from __future__ import annotations

from pathlib import Path
from typing import MutableMapping

from . import credentials as _ENV_UTILS

CredentialLoadError = _ENV_UTILS.CredentialLoadError
_load_gemini_api_key = _ENV_UTILS.load_gemini_api_key


def load_gemini_api_key(
    *,
    environ: MutableMapping[str, str] | None = None,
    home_env: Path | None = None,
    project_env: Path | None = None,
) -> str:
    """Load the Gemini key and update the supplied environment.

    Precedence is ``GEMINI_ENV_FILE``, the supplied project file or the launch
    directory's ``.env``, ``~/.gemini.env``, then an exported key.
    """

    return _load_gemini_api_key(
        environ=environ,
        home_env=home_env,
        project_env=project_env or Path.cwd() / ".env",
        prefer_project_env=True,
    )
