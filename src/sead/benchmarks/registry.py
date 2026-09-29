"""Read-only benchmark catalog used by campaign planning.

Selection, quarantine and environment requirements belong to the data adapters;
task scheduling and attempt budgets belong to scripts/run_dart_suite.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
import sqlite3

from sead.benchmarks.mtar.dataset import (
    load_task as load_mtar_task,
    load_task_dependencies,
    validate_dataset,
)
from sead.benchmarks.mtar.collection import (
    MTARCollectionResolution,
    resolve_collection as resolve_mtar_collection,
)
from sead.benchmarks.oas.dataset import (
    DEFAULT_CANDIDATE_INDEX as DEFAULT_OAS_CANDIDATE_INDEX,
    DEFAULT_SELECTION as DEFAULT_OAS_SELECTION,
    load_selection as load_oas_selection,
    load_task as load_oas_task,
)
from sead.environments.resources import web_resources


def partition_task_ids(
    task_ids: list[str],
    dataset_root: Path | str,
    benchmark_kind: str = "mtar",
    selection_path: Path | str = DEFAULT_OAS_SELECTION,
    *,
    candidate_index_path: Path | str = DEFAULT_OAS_CANDIDATE_INDEX,
) -> tuple[list[str], dict[str, str]]:
    dataset_root = Path(dataset_root)
    if benchmark_kind == "oas":
        allowed = set(load_oas_selection(selection_path))
        unknown = sorted(set(task_ids) - allowed)
        if unknown:
            raise ValueError(f"tasks are outside the selected OAS subset: {unknown}")
        active = list(dict.fromkeys(task_ids))
        for task_id in active:
            load_oas_task(
                dataset_root,
                task_id,
                selection_path=selection_path,
                candidate_index=candidate_index_path,
            )
        return active, {}
    if benchmark_kind != "mtar":
        raise ValueError(f"unsupported benchmark kind: {benchmark_kind}")
    rows = {str(row["task_id"]): row for row in validate_dataset(dataset_root)["tasks"]}
    catalog_file = dataset_root / "catalog.sqlite"
    if catalog_file.is_file():
        with sqlite3.connect(f"file:{catalog_file}?mode=ro", uri=True) as catalog:
            published = dict(catalog.execute("SELECT task_id, status FROM tasks"))
        expected = {task_id: "ready" if row["data_status"] == "ready" else "blocked" for task_id, row in rows.items()}
        if published != expected:
            raise ValueError("MTAR-CKA catalog and generated manifest disagree")
    active: list[str] = []
    excluded: dict[str, str] = {}
    for task_id in dict.fromkeys(task_ids):
        row = rows.get(task_id)
        if row is not None and row["data_status"] == "quarantined":
            excluded[task_id] = "DATASET_QUARANTINED: " + str(row["quarantine_reason"])
            continue
        if row is None:
            raise ValueError(f"unknown normalized MTAR task: {task_id}")
        active.append(task_id)
    return active, excluded


def task_resources(
    task_id: str,
    dataset_root: Path | str,
    *,
    benchmark_kind: str,
    selection_path: Path | str = DEFAULT_OAS_SELECTION,
    candidate_index_path: Path | str = DEFAULT_OAS_CANDIDATE_INDEX,
    postgres_mode: str = "legacy_shared",
) -> tuple[str, ...]:
    """Return host-wide mutable resources required for a complete task run."""

    if benchmark_kind == "oas":
        dependencies = load_oas_task(
            dataset_root,
            task_id,
            selection_path=selection_path,
            candidate_index=candidate_index_path,
        ).dependencies
    elif benchmark_kind == "mtar":
        task_root, row = load_mtar_task(dataset_root, task_id)
        dependencies = load_task_dependencies(task_root, str(row["tool"]))
    else:
        raise ValueError(f"unsupported benchmark kind: {benchmark_kind}")
    resources = set()
    if postgres_mode not in {"legacy_shared", "leased"}:
        raise ValueError("unknown PostgreSQL resource mode")
    if "mcp-postgres" in dependencies and postgres_mode != "leased":
        resources.add("postgres")
    resources.update(web_resources(dependencies))
    return tuple(sorted(resources))


@dataclass(frozen=True)
class BenchmarkCatalog:
    kind: str
    root: Path | None
    selection: Path
    candidate_index: Path = DEFAULT_OAS_CANDIDATE_INDEX
    collection: Path | None = None
    group: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "mtar" and self.collection is not None:
            if self.group is None:
                raise ValueError("benchmark.group is required with benchmark.collection")
            if self.root is not None:
                raise ValueError(
                    "benchmark.collection conflicts with benchmark.dataset_root"
                )
        elif self.root is None:
            raise ValueError("benchmark.dataset_root is required")

    @cached_property
    def mtar_collection(self) -> MTARCollectionResolution | None:
        if self.collection is None:
            return None
        if self.kind != "mtar":
            raise ValueError("benchmark.collection is supported only for MTAR")
        return resolve_mtar_collection(self.collection, str(self.group))

    def task_dataset_root(self, task_id: str) -> Path:
        if self.mtar_collection is not None:
            return self.mtar_collection.task(task_id).dataset_root
        assert self.root is not None
        return Path(self.root)

    def task_release(self, task_id: str) -> str | None:
        if self.mtar_collection is None:
            return None
        return self.mtar_collection.task(task_id).release

    @cached_property
    def _mtar_rows_by_root(self) -> dict[Path, dict[str, dict]]:
        if self.kind != "mtar":
            return {}
        roots = (
            self.mtar_collection.dataset_roots
            if self.mtar_collection is not None
            else (Path(self.root).resolve(),)
        )
        return {
            root: {
                str(row["task_id"]): row
                for row in validate_dataset(root)["tasks"]
            }
            for root in roots
        }

    def mtar_task(self, task_id: str) -> tuple[Path, dict]:
        root = self.task_dataset_root(task_id).resolve()
        row = self._mtar_rows_by_root[root].get(task_id)
        if row is None:
            raise ValueError(f"unknown normalized MTAR task: {task_id}")
        return root / str(row["path"]), row

    def provenance(self) -> dict:
        if self.mtar_collection is not None:
            return self.mtar_collection.provenance()
        assert self.root is not None
        return {
            "schema_version": "sead-dataset-root-selection-v1",
            "dataset_root": str(Path(self.root).resolve()),
        }

    def dependencies(self, task_id: str) -> tuple[str, ...]:
        if self.kind == "mtar":
            root, row = self.mtar_task(task_id)
            return tuple(load_task_dependencies(root, str(row["tool"])))
        if self.kind == "oas":
            return tuple(load_oas_task(self.root, task_id, selection_path=self.selection,
                                      candidate_index=self.candidate_index).dependencies)
        raise ValueError(f"unsupported benchmark kind: {self.kind}")

    def selected_task_ids(self) -> tuple[str, ...]:
        if self.kind == "mtar":
            if self.mtar_collection is not None:
                return self.mtar_collection.task_ids
            assert self.root is not None
            rows = self._mtar_rows_by_root[Path(self.root).resolve()]
            return tuple(rows)
        if self.kind == "oas":
            return load_oas_selection(self.selection)
        raise ValueError(f"unsupported benchmark kind: {self.kind}")

    def partition(self, task_ids: list[str]) -> tuple[list[str], dict[str, str]]:
        if self.kind == "mtar" and self.mtar_collection is not None:
            allowed = set(self.mtar_collection.task_ids)
            requested = list(dict.fromkeys(task_ids))
            unknown = sorted(set(requested) - allowed)
            if unknown:
                raise ValueError(
                    f"tasks are outside MTAR collection group {self.group!r}: {unknown}"
                )
            for task_id in requested:
                _, row = self.mtar_task(task_id)
                if row["data_status"] != "ready":
                    raise ValueError(f"MTAR collection task is not ready: {task_id}")
            return requested, {}
        assert self.root is not None
        return partition_task_ids(
            task_ids,
            self.root,
            self.kind,
            self.selection,
            candidate_index_path=self.candidate_index,
        )

    def resources(self, task_id: str, *, postgres_mode: str = "legacy_shared") -> tuple[str, ...]:
        root = self.task_dataset_root(task_id) if self.kind == "mtar" else self.root
        assert root is not None
        return task_resources(
            task_id,
            root,
            benchmark_kind=self.kind,
            selection_path=self.selection,
            candidate_index_path=self.candidate_index,
            postgres_mode=postgres_mode,
        )
