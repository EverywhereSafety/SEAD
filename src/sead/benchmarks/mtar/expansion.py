"""Read-only acceptance validation for separately versioned MTAR releases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .dataset import EXPECTED_SOURCE_COMMIT, MTARDatasetError

ACCEPTANCE_CHECKS = (
    "initial_zero", "known_positive", "negative_controls", "reset_zero",
    "prefix_replay", "blocked_replay", "isolation",
)


def task_asset_digest(task_root: Path) -> str:
    """Same digest as the original intake; acceptance/release are separate."""
    digest = hashlib.sha256()
    for path in sorted(task_root.rglob("*")):
        if not path.is_file() or path.name in {"release.yml", "acceptance.json"}:
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        digest.update(path.relative_to(task_root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _verify_evidence(root: Path, relative: str, checksum: str) -> None:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not relative:
        raise MTARDatasetError("release evidence must use a contained relative path")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise MTARDatasetError(f"missing or escaping release evidence: {relative}")
    if hashlib.sha256(resolved.read_bytes()).hexdigest() != checksum:
        raise MTARDatasetError(f"release evidence digest mismatch: {relative}")


def validate_acceptance(task_root: Path) -> None:
    try:
        acceptance = json.loads((task_root / "acceptance.json").read_text())
        provenance = json.loads((task_root / "provenance.json").read_text())
    except (OSError, ValueError) as exc:
        raise MTARDatasetError(f"missing expansion acceptance/provenance: {task_root.name}") from exc
    portable = acceptance.get("schema_version") == "mtar-task-acceptance-v2"
    if (
        acceptance.get("schema_version") not in {"mtar-task-acceptance-v1", "mtar-task-acceptance-v2"}
        or acceptance.get("task_asset_digest") != task_asset_digest(task_root)
        or provenance.get("source_commit") != EXPECTED_SOURCE_COMMIT
        or provenance.get("source_complete") is not True
        or provenance.get("metadata_verified") is not True
        or provenance.get("normalization_blockers")
    ):
        raise MTARDatasetError(f"unverified or stale expansion acceptance: {task_root.name}")
    checks = acceptance.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(ACCEPTANCE_CHECKS):
        raise MTARDatasetError(f"incomplete expansion acceptance checks: {task_root.name}")
    for key, check in checks.items():
        if not isinstance(check, dict) or check.get("passed") is not True:
            raise MTARDatasetError(f"expansion acceptance pending: {task_root.name}/{key}")
        evidence = check.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            raise MTARDatasetError(f"expansion acceptance lacks evidence: {task_root.name}/{key}")
        if portable:
            _verify_evidence(task_root.parent.parent, evidence, check.get("evidence_sha256", ""))
    if portable:
        _verify_evidence(
            task_root.parent.parent, acceptance.get("metadata_receipt", ""),
            acceptance.get("metadata_receipt_sha256", ""),
        )


def validate_release_snapshot(root: Path, manifest: dict) -> None:
    """Keep published selection and task revisions bound to their release."""
    path = root / "release.json"
    if not path.is_file():
        return  # Historical candidate intake, not a portable publication.
    try:
        record = json.loads(path.read_text())
        digests = record["task_asset_digests"]
        ids = {row["task_id"] for row in manifest["tasks"]}
        if (
            record["schema_version"] != "mtar-release-snapshot-v1"
            or record["counts"] != manifest["counts"]
            or record["source"] != manifest["mtar_source"]
            or set(digests) != ids
        ):
            raise ValueError("snapshot selection or source mismatch")
        for tid in ids:
            task = root / "tasks" / tid
            if task_asset_digest(task) != digests[tid]:
                raise ValueError(f"snapshot task revision changed: {tid}")
            acceptance = json.loads((task / "acceptance.json").read_text())
            if acceptance.get("schema_version") != "mtar-task-acceptance-v2":
                raise ValueError(f"published acceptance is not portable: {tid}")
        for relative, checksum in record["evidence_digests"].items():
            _verify_evidence(root, relative, checksum)
        for relative, checksum in record.get("baseline_digests", {}).items():
            _verify_evidence(root, relative, checksum)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise MTARDatasetError(f"invalid frozen expansion release: {exc}") from exc
