"""Versioned, deliberately narrow SQL execution-evidence contracts.

Compared as PostgreSQL parse trees, not text/keywords. Equivalent queries outside
these shapes remain unverified until an explicit contract extension is tested.
"""

from pathlib import Path

EVIDENCE_VERSION = "postgres-mcp-evidence-v2"
_DATASET_ROOT = Path(__file__).resolve().parents[5] / "data/mtar"
SQL_CONTRACTS = {
    path.parent.parent.name: path.read_text(encoding="utf-8").rstrip("\n")
    for path in (_DATASET_ROOT / "tasks").glob("single.*/utils/sql_contract.sql")
}


def canonical_ast(value):
    if isinstance(value, list):
        return [canonical_ast(x) for x in value]
    if not isinstance(value, dict):
        return value
    result = {
        k: canonical_ast(v)
        for k, v in value.items()
        if k not in {"location", "stmt_location", "stmt_len"}
    }
    if "RangeVar" in result:
        result["RangeVar"].setdefault(
            "schemaname", "pg_catalog" if result["RangeVar"].get("relname") == "pg_shadow" else "public"
        )
    # DROP TABLE users and DROP TABLE public.users name the same object under
    # the collector's verified resolution/lock. Other schemas remain distinct.
    drop = result.get("DropStmt")
    if drop and drop.get("removeType") == "OBJECT_TABLE":
        for obj in drop.get("objects", []):
            items = obj.get("List", {}).get("items", [])
            if len(items) == 1:
                items.insert(0, {"String": {"sval": "public"}})
    return result
