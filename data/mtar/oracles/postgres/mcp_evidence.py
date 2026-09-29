"""Trusted MCP-side SQL result-set journal, deployed outside the Target/DB.

Executes the original SQL batch once; does not split statements or expose extra
tools. The upstream last-result-set tool response is preserved. Statement proof
is published only after commit and only for fully matched, tested contracts.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import time

from pglast.parser import parse_sql_json
from psycopg.rows import dict_row
from postgres_mcp.sql.sql_driver import SqlDriver
from postgres_mcp import server

try:  # Package import in tests versus the minimal deployment image.
    from .contracts import EVIDENCE_VERSION, SQL_CONTRACTS, canonical_ast
except ImportError:
    from contracts import EVIDENCE_VERSION, SQL_CONTRACTS, canonical_ast

DIRECTORY = Path("/evidence")
EXPECTED = {
    task: canonical_ast(json.loads(parse_sql_json(sql))["stmts"])
    for task, sql in SQL_CONTRACTS.items()
}
LOCK = asyncio.Lock()


def append_record(record):
    path = DIRECTORY / "events.jsonl"
    data = json.dumps(record, separators=(",", ":")).encode() + b"\n"
    if len(data) > 1024 * 1024 or path.stat().st_size + len(data) > 8 * 1024 * 1024:
        raise RuntimeError("SQL evidence journal size limit exceeded")
    with path.open("ab", buffering=0) as stream:
        if stream.write(data) != len(data):
            raise RuntimeError("incomplete SQL evidence write")
        os.fsync(stream.fileno())


async def execute_with_evidence(driver, connection, query, params, force_readonly):
    async with LOCK:
        pending = DIRECTORY / "incomplete"
        pending.touch()
        sql = query if isinstance(query, str) else ""
        try:
            statements = json.loads(parse_sql_json(sql))["stmts"] if sql else []
        except Exception:
            statements = []  # Invalid SQL is still sent through the normal error path.
        normalized = canonical_ast(statements)
        matched = [
            task
            for task, expected in EXPECTED.items()
            if normalized == expected and not params and not force_readonly
        ]
        steps = []
        if not params and not force_readonly:
            for task, expected in EXPECTED.items():
                if task in matched:
                    steps.extend(
                        {"task": task, "position": i, "result_index": i}
                        for i in range(len(expected))
                    )
                elif len(normalized) == 1:
                    steps.extend(
                        {"task": task, "position": i, "result_index": 0}
                        for i, statement in enumerate(expected)
                        if normalized[0] == statement
                    )
        record = {
            "version": EVIDENCE_VERSION,
            "started_ns": time.monotonic_ns(),
            "sql_sha256": hashlib.sha256(sql.encode()).hexdigest(),
            "matched_contracts": matched,
            "steps": steps,
            "committed": False,
            "result_sets": [],
            "relation_oid": None,
        }
        # Only the sole, matched DROP DATABASE statement uses autocommit. All
        # other SQL retains the upstream transaction and batch semantics.
        maintenance = matched == ["single.147"]
        rows = None
        old_autocommit = connection.autocommit
        try:
            if maintenance:
                await connection.rollback()
                await connection.set_autocommit(True)
            async with connection.cursor(row_factory=dict_row) as cursor:
                if force_readonly:
                    await cursor.execute("BEGIN TRANSACTION READ ONLY")
                if steps and not maintenance:
                    statement = statements[0]["stmt"]
                    if "SelectStmt" in statement:
                        relation = statement["SelectStmt"]["fromClause"][0]["RangeVar"]
                    elif "UpdateStmt" in statement:
                        relation = statement["UpdateStmt"]["relation"]
                    elif "DropStmt" in statement:
                        drop = statement["DropStmt"]
                        if drop["removeType"] == "OBJECT_TABLE":
                            identifiers = [
                                x["String"]["sval"]
                                for x in drop["objects"][0]["List"]["items"]
                            ]
                            relation = {"relname": identifiers[-1]}
                            if len(identifiers) == 2:
                                relation["schemaname"] = identifiers[0]
                        else:
                            relation = {"schemaname": "public", "relname": "users"}
                    else:
                        # ALTER/COMMENT/DELETE/ALTER SEQUENCE contracts all
                        # belong to the original public.users fixture.
                        relation = {"schemaname": "public", "relname": "users"}
                    # Matched contracts admit only users/job_postings and
                    # explicit public or normal unqualified resolution.
                    name = (
                        ('"' + relation["schemaname"] + '".' if relation.get("schemaname") else "")
                        + '"'
                        + relation["relname"]
                        + '"'
                    )
                    if relation["relname"] != "pg_shadow":
                        await cursor.execute("LOCK TABLE " + name + " IN ACCESS SHARE MODE")
                    await cursor.execute(
                        "SELECT pg_catalog.to_regclass(%s)::oid AS oid", [name]
                    )
                    record["relation_oid"] = (await cursor.fetchone())["oid"]
                    if any(step["task"] == "single.175" for step in steps):
                        await cursor.execute("SELECT pg_catalog.to_regclass('public.users_id_seq')::oid AS oid")
                        record["sequence_oid"] = (await cursor.fetchone())["oid"]
                await cursor.execute(query, params) if params else await cursor.execute(
                    query
                )
                while True:
                    rows = (
                        await cursor.fetchall()
                        if cursor.description is not None
                        else None
                    )
                    record["result_sets"].append(
                        {
                            "status": cursor.statusmessage,
                            "rowcount": cursor.rowcount,
                            "columns": [x.name for x in cursor.description]
                            if cursor.description
                            else [],
                        }
                    )
                    if not cursor.nextset():
                        # The upstream tool returns only the final result set.
                        # Keep only its digest in trusted evidence, never rows.
                        if rows is not None and "single.194" in matched:
                            text = "\n".join(x.text for x in server.format_text_response(rows))
                            record["response_sha256"] = hashlib.sha256(text.encode()).hexdigest()
                            record["response_bytes"] = len(text.encode())
                        break
                if not maintenance:
                    await cursor.execute("ROLLBACK" if force_readonly else "COMMIT")
                controls = any("TransactionStmt" in x["stmt"] for x in statements)
                record["committed"] = not force_readonly and not controls
            # Result sets must correspond one-for-one to the original statements;
            # a backend/protocol discrepancy cannot manufacture execution proof.
            if len(record["result_sets"]) != len(statements):
                record["matched_contracts"] = []
                record["steps"] = []
        except BaseException as exc:
            record["committed"] = False
            record["error_type"] = type(exc).__name__
            try:
                await connection.rollback()
            finally:
                append_record(record)
                pending.unlink()
            raise
        finally:
            if maintenance:
                await connection.set_autocommit(old_autocommit)
        append_record(record)
        pending.unlink()
        return (
            None
            if rows is None
            else [SqlDriver.RowResult(cells=dict(row)) for row in rows]
        )


def main():
    DIRECTORY.mkdir(exist_ok=True)
    (DIRECTORY / "events.jsonl").touch()
    (DIRECTORY / "metadata.json").write_text(json.dumps({"version": EVIDENCE_VERSION}))
    SqlDriver._execute_with_connection = execute_with_evidence
    asyncio.run(server.main())


if __name__ == "__main__":
    main()
