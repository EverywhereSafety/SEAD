"""Repository environment loading with a canonical Gemini credential file."""

from __future__ import annotations

import os
import shlex
import stat
from collections.abc import MutableMapping
from pathlib import Path

REPOSITORY_ROOT = Path.cwd()


class CredentialLoadError(RuntimeError):
    """Raised when a configured model credential cannot be loaded safely."""


def _read_env_value(path: Path, variable: str) -> str:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise CredentialLoadError(
            f"credential file must not be group/world accessible: {path}"
        )
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        name, separator, raw_value = line.partition("=")
        if not separator or name.strip() != variable:
            continue
        lexer = shlex.shlex(raw_value, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = "#"
        tokens = list(lexer)
        if len(tokens) != 1 or not tokens[0]:
            raise CredentialLoadError(
                f"{variable} has an invalid value in {path}"
            )
        return tokens[0]
    raise CredentialLoadError(f"{variable} is missing from {path}")


def _read_gemini_key(path: Path) -> str:
    return _read_env_value(path, "GEMINI_API_KEY")


def load_gemini_api_key(
    *,
    environ: MutableMapping[str, str] | None = None,
    home_env: Path | None = None,
    project_env: Path | None = None,
    prefer_project_env: bool = False,
) -> str:
    """Load Gemini credentials from explicit, canonical, or project sources.

    ``prefer_project_env`` lets a scoped caller make its launch-directory
    ``.env`` authoritative without changing the repository-wide default.
    """

    target = os.environ if environ is None else environ
    explicit_path = target.get("GEMINI_ENV_FILE", "").strip()
    canonical = (
        Path(explicit_path).expanduser()
        if explicit_path
        else home_env or Path.home() / ".gemini.env"
    )
    if explicit_path and not canonical.is_file():
        raise CredentialLoadError(f"GEMINI_ENV_FILE does not exist: {canonical}")

    project_source = (
        project_env
        if project_env is not None and project_env.is_file()
        else None
    )
    source: Path | None = canonical if explicit_path else None
    if source is None and prefer_project_env:
        source = project_source
    if source is None and canonical.is_file():
        source = canonical
    if source is None and target.get("GEMINI_API_KEY", "").strip():
        key = target["GEMINI_API_KEY"].strip()
        target["GOOGLE_API_KEY"] = key
        return key
    if source is None:
        source = project_source
    if source is None:
        raise CredentialLoadError(
            "GEMINI_API_KEY is unavailable; expected GEMINI_ENV_FILE, the "
            "configured project .env, ~/.gemini.env, or an exported value."
        )

    key = _read_gemini_key(source)
    target["GEMINI_API_KEY"] = key
    target["GOOGLE_API_KEY"] = key
    target["SEAD_GEMINI_KEY_SOURCE"] = str(source.resolve())
    return key


def load_azure_openai_api_key(
    *,
    environ: MutableMapping[str, str] | None = None,
    project_env: Path | None = None,
    variable: str = "AZURE_OPENAI_API_KEY",
    prefer_project_env: bool = False,
) -> str:
    """Load an Azure OpenAI key from an explicit file, the environment, or .env."""

    target = os.environ if environ is None else environ
    explicit_path = target.get("AZURE_OPENAI_ENV_FILE", "").strip()
    source = Path(explicit_path).expanduser() if explicit_path else None
    if source is not None and not source.is_file():
        raise CredentialLoadError(f"AZURE_OPENAI_ENV_FILE does not exist: {source}")

    candidate = project_env or REPOSITORY_ROOT / ".env"
    project_source = candidate if candidate.is_file() else None
    if source is None and prefer_project_env:
        source = project_source
    if source is None and target.get(variable, "").strip():
        return target[variable].strip()
    if source is None:
        source = project_source
    if source is None:
        raise CredentialLoadError(
            f"{variable} is unavailable; expected AZURE_OPENAI_ENV_FILE, "
            "the configured project .env, or an exported value."
        )

    key = _read_env_value(source, variable)
    target[variable] = key
    target["SEAD_AZURE_OPENAI_KEY_SOURCE"] = str(source.resolve())
    return key
