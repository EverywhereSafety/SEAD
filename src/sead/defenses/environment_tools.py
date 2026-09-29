"""Read-only investigation tools backed by a live OpenHands Docker runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
import threading
import json
from typing import Any

from .sage import SAGEDefenseTool

DEFAULT_ENVIRONMENT_ROOTS = ("/workspace", "/etc", "/tmp", "/var", "/opt", "/home")


def _output(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


class OpenHandsEnvironmentInspector:
    """Execute a narrow read-only command vocabulary outside the event stream."""

    def __init__(
        self,
        runtime: Any,
        *,
        allowed_roots: Sequence[str] = DEFAULT_ENVIRONMENT_ROOTS,
        max_read_lines: int = 400,
        max_search_matches: int = 200,
    ) -> None:
        container = getattr(runtime, "container", None)
        if container is None or not callable(getattr(container, "exec_run", None)):
            raise RuntimeError("OpenHands Docker container is unavailable")
        roots = tuple(str(PurePosixPath(value)) for value in allowed_roots)
        if not roots or any(not value.startswith("/") for value in roots):
            raise ValueError("environment investigation roots must be absolute")
        if max_read_lines < 1 or max_search_matches < 1:
            raise ValueError("environment investigation limits must be positive")
        self.container = container
        self.postgres_lease = getattr(runtime, "sead_postgres_lease", None)
        self.allowed_roots = roots
        self.max_read_lines = max_read_lines
        self.max_search_matches = max_search_matches
        # Docker's exec API is not guaranteed to be thread-safe on a shared
        # container object, so serialize read-only tool calls.
        self._exec_lock = threading.Lock()

    def _exec(
        self, argv: list[str], *, accepted_codes: set[int] | frozenset[int] = frozenset({0})
    ) -> str:
        with self._exec_lock:
            result = self.container.exec_run(argv, demux=False)
        exit_code = int(getattr(result, "exit_code", result[0]))
        output = getattr(result, "output", result[1])
        text = _output(output)
        if exit_code not in accepted_codes:
            raise RuntimeError(f"read-only environment command failed ({exit_code}): {text}")
        return text

    def _resolve(self, raw_path: Any) -> str:
        path = str(raw_path or "").strip()
        if not path.startswith("/"):
            raise ValueError("path must be absolute")
        resolved = self._exec(["readlink", "-f", "--", path]).strip()
        if not resolved.startswith("/"):
            raise ValueError("path does not resolve inside the container")
        if not any(
            resolved == root or resolved.startswith(root.rstrip("/") + "/")
            for root in self.allowed_roots
        ):
            raise ValueError("path is outside the configured investigation roots")
        return resolved

    def read_file(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        path = self._resolve(arguments.get("path"))
        start_line = int(arguments.get("start_line", 1))
        line_count = min(
            max(int(arguments.get("line_count", 200)), 1), self.max_read_lines
        )
        if start_line < 1:
            raise ValueError("start_line must be positive")
        end_line = start_line + line_count - 1
        content = self._exec(
            ["sed", "-n", f"{start_line},{end_line}p", "--", path]
        )
        return {
            "path": path,
            "start_line": start_line,
            "line_count": len(content.splitlines()),
            "content": content,
        }

    def list_directory(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        path = self._resolve(arguments.get("path"))
        content = self._exec(
            [
                "find",
                path,
                "-mindepth",
                "1",
                "-maxdepth",
                "1",
                "-printf",
                "%y %p\n",
            ]
        )
        return {"path": path, "entries": content.splitlines()}

    def search_files(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        path = self._resolve(arguments.get("path"))
        query = str(arguments.get("query") or "")
        if not query:
            raise ValueError("query must be non-empty")
        max_matches = min(
            max(int(arguments.get("max_matches", 50)), 1), self.max_search_matches
        )
        content = self._exec(
            ["grep", "-R", "-n", "-F", "-m", str(max_matches), "--", query, path],
            accepted_codes={0, 1},
        )
        return {"path": path, "query": query, "matches": content.splitlines()}

    def stat_path(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        path = self._resolve(arguments.get("path"))
        content = self._exec(
            [
                "stat",
                "--printf",
                "type=%F mode=%a size=%s owner=%U group=%G mtime=%y\n",
                "--",
                path,
            ]
        )
        return {"path": path, "stat": content.strip()}

    def processes(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if arguments:
            raise ValueError("env_processes accepts no arguments")
        content = self._exec(["ps", "-eo", "pid,user,stat,comm,args"])
        return {"processes": content.splitlines()}

    def tools(self) -> tuple[SAGEDefenseTool, ...]:
        roots = list(self.allowed_roots)
        return (
            SAGEDefenseTool(
                "env_read_file",
                f"Read lines from a file in the live pre-action environment under {roots}.",
                {
                    "path": "absolute path",
                    "start_line": "optional positive integer",
                    "line_count": f"optional positive integer, max {self.max_read_lines}",
                },
                self.read_file,
            ),
            SAGEDefenseTool(
                "env_list_directory",
                f"List one directory in the live pre-action environment under {roots}.",
                {"path": "absolute directory path"},
                self.list_directory,
            ),
            SAGEDefenseTool(
                "env_search_files",
                f"Literal recursive search in the live pre-action environment under {roots}.",
                {
                    "path": "absolute path",
                    "query": "non-empty literal string",
                    "max_matches": f"optional positive integer, max {self.max_search_matches}",
                },
                self.search_files,
            ),
            SAGEDefenseTool(
                "env_stat_path",
                "Inspect type, permissions, size, ownership, and mtime of one path.",
                {"path": "absolute path"},
                self.stat_path,
            ),
            SAGEDefenseTool(
                "env_processes",
                "Inspect the live pre-action process table without changing it.",
                {},
                self.processes,
            ),
        ) + ((SAGEDefenseTool(
            "env_postgres_catalog",
            "Inspect public database object names, ownership and role attributes in the live pre-action environment.",
            {}, self.postgres_catalog,
        ),) if self.postgres_lease else ())

    def postgres_catalog(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if arguments or self.postgres_lease is None:
            raise ValueError("catalog inspection accepts no SQL or other arguments")
        raw = self.postgres_lease.sql("""
            BEGIN READ ONLY;
            SET LOCAL statement_timeout = '5s';
            SET LOCAL search_path = pg_catalog;
            SELECT json_build_object(
                'objects', (SELECT coalesce(json_agg(x),'[]'::json) FROM
                    (SELECT n.nspname AS schema, c.relname AS name, c.relkind AS kind,
                            pg_get_userbyid(c.relowner) AS owner
                     FROM pg_class c JOIN pg_namespace n ON c.relnamespace=n.oid
                     WHERE n.nspname='public' ORDER BY c.relname LIMIT 200) x),
                'roles', (SELECT coalesce(json_agg(x),'[]'::json) FROM
                    (SELECT rolname, rolsuper, rolcanlogin FROM pg_roles ORDER BY rolname LIMIT 200) x)
            );
            ROLLBACK;
        """)
        return json.loads(raw)


__all__ = ["DEFAULT_ENVIRONMENT_ROOTS", "OpenHandsEnvironmentInspector"]
