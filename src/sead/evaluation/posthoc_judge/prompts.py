"""Load, validate, render, and fingerprint the external Judge prompts."""

from __future__ import annotations

import json
import string
from dataclasses import dataclass
from pathlib import Path

from sead.hashing import sha256_text

EXPECTED_USER_FIELDS = {
    "task_id",
    "unsafe_goal",
    "completion_criteria",
    "trajectory",
}
DEFAULT_PROMPT_DIR = Path(__file__).resolve().parent / "prompt_templates"


class PromptError(ValueError):
    pass


@dataclass(frozen=True)
class JudgePrompts:
    system: str
    user_template: str
    system_sha256: str
    user_sha256: str
    combined_sha256: str

    def render(
        self,
        *,
        task_id: str,
        unsafe_goal: str,
        completion_criteria: str,
        trajectory: list[dict[str, object]],
    ) -> str:
        return self.user_template.format(
            task_id=task_id,
            unsafe_goal=unsafe_goal,
            completion_criteria=completion_criteria,
            trajectory=json.dumps(trajectory, ensure_ascii=False, indent=2),
        )

    def hashes(self) -> dict[str, str]:
        return {
            "system_prompt_sha256": self.system_sha256,
            "user_prompt_sha256": self.user_sha256,
            "combined_prompt_sha256": self.combined_sha256,
        }


def load_prompts(prompt_dir: Path | str = DEFAULT_PROMPT_DIR) -> JudgePrompts:
    prompt_dir = Path(prompt_dir)
    try:
        system = (prompt_dir / "system_prompt.md").read_text(encoding="utf-8")
        user = (prompt_dir / "user_prompt.md").read_text(encoding="utf-8")
    except OSError as exc:
        raise PromptError(f"cannot read Judge prompts from {prompt_dir}: {exc}") from exc
    if not system.strip() or not user.strip():
        raise PromptError("Judge prompts must be non-empty")
    fields: list[str] = []
    try:
        parsed = list(string.Formatter().parse(user))
    except ValueError as exc:
        raise PromptError(f"invalid user prompt template: {exc}") from exc
    for _, field, format_spec, conversion in parsed:
        if field is None:
            continue
        if format_spec or conversion:
            raise PromptError("user prompt placeholders cannot use formats/conversions")
        fields.append(field)
    if set(fields) != EXPECTED_USER_FIELDS or len(fields) != len(EXPECTED_USER_FIELDS):
        raise PromptError(
            "user prompt must contain each of task_id, unsafe_goal, "
            "completion_criteria, trajectory exactly once"
        )
    system_hash = sha256_text(system)
    user_hash = sha256_text(user)
    combined = sha256_text(system_hash + ":" + user_hash)
    return JudgePrompts(system, user, system_hash, user_hash, combined)
