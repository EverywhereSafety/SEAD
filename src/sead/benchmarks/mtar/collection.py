"""Resolve versioned MTAR collections without mutating dataset selection files."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import yaml

from sead.hashing import sha256_file, sha256_json

from .dataset import validate_dataset


class MTARCollectionError(ValueError):
    """The collection metadata or requested group is invalid."""


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MTARCollectionError(f"cannot read MTAR collection metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MTARCollectionError(f"MTAR collection metadata must be an object: {path}")
    return value


def _yaml_object(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise MTARCollectionError(f"cannot read MTAR collection metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MTARCollectionError(f"MTAR collection metadata must be a mapping: {path}")
    return value


@dataclass(frozen=True)
class MTARCollectionTask:
    task_id: str
    cohort: str
    release: str
    dataset_root: Path
    task_asset_digest: str


@dataclass(frozen=True)
class MTARCollectionResolution:
    root: Path
    name: str
    group: str
    tasks: tuple[MTARCollectionTask, ...]
    source_manifest_digests: tuple[tuple[str, str], ...]
    metadata_digest: str

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.tasks)

    @property
    def dataset_roots(self) -> tuple[Path, ...]:
        return tuple(dict.fromkeys(task.dataset_root for task in self.tasks))

    def task(self, task_id: str) -> MTARCollectionTask:
        matches = [task for task in self.tasks if task.task_id == task_id]
        if len(matches) != 1:
            raise MTARCollectionError(
                f"task {task_id!r} is not a member of MTAR collection group {self.group!r}"
            )
        return matches[0]

    def provenance(self) -> dict[str, Any]:
        sources: dict[str, dict[str, Any]] = {}
        digests = dict(self.source_manifest_digests)
        for task in self.tasks:
            source = sources.setdefault(
                task.release,
                {
                    "release": task.release,
                    "dataset_root": str(task.dataset_root),
                    "manifest_sha256": digests[task.release],
                    "task_ids": [],
                },
            )
            source["task_ids"].append(task.task_id)
        return {
            "schema_version": "sead-mtar-collection-resolution-v1",
            "collection": self.name,
            "collection_root": str(self.root),
            "group": self.group,
            "task_count": len(self.tasks),
            "task_ids": list(self.task_ids),
            "sources": list(sources.values()),
            "metadata_digest": self.metadata_digest,
        }


def _source_roots(root: Path, contract: dict[str, Any]) -> dict[str, Path]:
    official = contract.get("official")
    expansion = contract.get("expansion")
    if not isinstance(official, dict) or not isinstance(official.get("root"), str):
        raise MTARCollectionError("collection.yml must declare official.root")
    if not isinstance(expansion, dict) or not isinstance(expansion.get("root"), str):
        raise MTARCollectionError("collection.yml must declare expansion.root")
    expansion_root = (root / expansion["root"]).resolve()
    return {
        "Official-100": (root / official["root"]).resolve(),
        expansion_root.name: expansion_root,
    }


def resolve_collection(path: Path | str, group: str) -> MTARCollectionResolution:
    """Resolve one explicit group into validated, source-bound task records."""

    root = Path(path).resolve()
    manifest_path = root / "manifest.json"
    groups_path = root / "groups.json"
    contract_path = root / "collection.yml"
    manifest = _json_object(manifest_path)
    groups = _json_object(groups_path)
    contract = _yaml_object(contract_path)
    if manifest.get("schema_version") != "mtar-ready187-manifest-v1":
        raise MTARCollectionError("unsupported MTAR collection manifest schema")
    if groups.get("schema_version") != "mtar-collection-groups-v1":
        raise MTARCollectionError("unsupported MTAR collection groups schema")
    if contract.get("schema_version") != "mtar-cross-release-collection-v3":
        raise MTARCollectionError("unsupported MTAR collection contract schema")
    definitions = groups.get("groups")
    if not isinstance(definitions, dict) or group not in definitions:
        choices = sorted(definitions) if isinstance(definitions, dict) else []
        raise MTARCollectionError(
            f"unknown MTAR collection group {group!r}; choose one of {choices}"
        )
    definition = definitions[group]
    if not isinstance(definition, dict) or not isinstance(definition.get("cohorts"), list):
        raise MTARCollectionError(f"invalid MTAR collection group definition: {group}")
    rows = manifest.get("tasks")
    if not isinstance(rows, list):
        raise MTARCollectionError("MTAR collection manifest tasks must be a list")
    cohorts = set(definition["cohorts"])
    selected = [row for row in rows if isinstance(row, dict) and row.get("cohort") in cohorts]
    expected = definition.get("expected_count")
    if type(expected) is not int or len(selected) != expected:
        raise MTARCollectionError(
            f"MTAR collection group {group!r} resolved to {len(selected)} tasks, expected {expected}"
        )
    task_ids = [str(row.get("task_id")) for row in selected]
    if any(task_id in {"", "None"} for task_id in task_ids) or len(set(task_ids)) != len(task_ids):
        raise MTARCollectionError(f"MTAR collection group {group!r} has invalid task identities")

    roots = _source_roots(root, contract)
    releases = {str(row.get("release")) for row in selected}
    unknown_releases = releases - set(roots)
    if unknown_releases:
        raise MTARCollectionError(
            f"MTAR collection group {group!r} references unknown releases: {sorted(unknown_releases)}"
        )
    source_rows: dict[str, dict[str, Any]] = {}
    source_digests: list[tuple[str, str]] = []
    for release in sorted(releases):
        dataset_root = roots[release]
        source_manifest = validate_dataset(dataset_root)
        ready = {
            str(row["task_id"]): row
            for row in source_manifest["tasks"]
            if row["data_status"] == "ready"
        }
        selected_for_release = {
            str(row["task_id"]) for row in selected if row.get("release") == release
        }
        if selected_for_release != set(ready):
            raise MTARCollectionError(
                f"MTAR collection group {group!r} does not equal the ready index for {release}"
            )
        source_rows.update(ready)
        source_digests.append((release, sha256_file(dataset_root / "manifest.json")))

    tasks: list[MTARCollectionTask] = []
    for row in selected:
        task_id = str(row["task_id"])
        source = source_rows.get(task_id)
        if source is None or source.get("tool") != row.get("tool"):
            raise MTARCollectionError(f"MTAR collection/source binding mismatch: {task_id}")
        digest = row.get("task_asset_digest")
        if not isinstance(digest, str) or len(digest) != 64:
            raise MTARCollectionError(f"invalid collection task digest: {task_id}")
        tasks.append(
            MTARCollectionTask(
                task_id=task_id,
                cohort=str(row["cohort"]),
                release=str(row["release"]),
                dataset_root=roots[str(row["release"])],
                task_asset_digest=digest,
            )
        )
    tasks.sort(key=lambda task: int(task.task_id.split(".", 1)[1]))
    metadata_digest = sha256_json(
        {
            "manifest": sha256_file(manifest_path),
            "groups": sha256_file(groups_path),
            "contract": sha256_file(contract_path),
            "group": group,
            "sources": source_digests,
        }
    )
    return MTARCollectionResolution(
        root=root,
        name=str(manifest.get("name") or root.name),
        group=group,
        tasks=tuple(tasks),
        source_manifest_digests=tuple(source_digests),
        metadata_digest=metadata_digest,
    )
