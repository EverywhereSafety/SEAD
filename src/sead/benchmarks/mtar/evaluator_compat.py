"""Versioned compatibility boundary for released MT-AgentRisk evaluators.

Evaluator files are trusted benchmark inputs, but the released dataset contains
several generations of the ``scoring`` API.  This module classifies the source
without changing it and builds the one entrypoint used by every MTAR runner.
"""

from __future__ import annotations

import ast
import io
import json
import tokenize
from dataclasses import dataclass


ADAPTER_VERSION = "mtar-evaluator-compat-v2"
EVALUATION_MARKER = "MTAR_EVALUATION="
_DIALECTS = {"canonical", "legacy", "mixed"}
_REQUIREMENT_BY_IMPORT = {
    "psycopg2": "psycopg2-binary==2.9.10",
}
_CONTROLLED_EVALUATOR_IMPORTS = {
    "__future__",
    "ast",
    "common",
    "contextlib",
    "csv",
    "datetime",
    # Standard-library modules present in the pinned evaluator image. Evaluator
    # execution (including task-generated Python) remains sandbox-only.
    "importlib",
    "html",
    "json",
    "logging",
    "math",
    "os",
    "pandas",
    "pathlib",
    "psycopg2",
    "re",
    "requests",
    "scoring",
    "shlex",
    "sqlite3",
    "stat",
    "subprocess",
    "sys",
    "typing",
    "xml",
}


class EvaluatorCompatibilityError(ValueError):
    """The evaluator cannot safely be adapted."""


@dataclass(frozen=True)
class EvaluatorSource:
    dialect: str


def inspect_evaluator_source(source: str | bytes) -> EvaluatorSource:
    """Parse evaluator source and identify the scoring API generation.

    Syntax errors are deliberately surfaced here, before a sandbox is started.
    The classification is based on imports and constructor call shapes rather
    than fragile substring replacement.
    """

    if isinstance(source, bytes):
        try:
            text = source.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EvaluatorCompatibilityError("evaluator is not UTF-8") from exc
    else:
        text = source
    try:
        tree = ast.parse(text, filename="evaluator.py")
    except SyntaxError as exc:
        raise EvaluatorCompatibilityError(f"invalid evaluator syntax: {exc}") from exc

    legacy_import = False
    legacy_shapes = False
    canonical_shapes = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "common.grader":
            legacy_import = True
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = function.id if isinstance(function, ast.Name) else None
        keywords = {item.arg for item in node.keywords if item.arg is not None}
        if name == "Result":
            if "score" in keywords or "message" in keywords or "explanation" in keywords:
                legacy_shapes = True
            if "checkpoints" in keywords or (node.args and not keywords & {"score"}):
                canonical_shapes = True
        elif name == "Checkpoint":
            if keywords & {"name", "weight", "score", "explanation", "metadata"}:
                legacy_shapes = True
            if "total" in keywords or len(node.args) >= 2:
                canonical_shapes = True

    if canonical_shapes and (legacy_import or legacy_shapes):
        dialect = "mixed"
    elif legacy_import or legacy_shapes:
        dialect = "legacy"
    else:
        dialect = "canonical"
    return EvaluatorSource(dialect=dialect)


def evaluator_python_requirements(source: str | bytes) -> tuple[str, ...]:
    """Return pinned sandbox packages required by a released evaluator."""

    if isinstance(source, bytes):
        try:
            text = source.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EvaluatorCompatibilityError("evaluator is not UTF-8") from exc
    else:
        text = source
    try:
        tree = ast.parse(text, filename="evaluator.py")
    except SyntaxError:
        # Dependency metadata must still be buildable for released evaluators
        # that are classified separately as syntax-incompatible. Tokenization
        # ignores comments and string contents while tolerating partial input.
        imported = set()
        mode: str | None = None
        expect_name = False
        try:
            tokens = tokenize.generate_tokens(io.StringIO(text).readline)
            for token in tokens:
                if token.type == tokenize.NAME and token.string in {"import", "from"}:
                    mode = token.string
                    expect_name = True
                elif mode and expect_name and token.type == tokenize.NAME:
                    imported.add(token.string)
                    expect_name = False
                    if mode == "from":
                        mode = None
                elif mode == "import" and token.string == ",":
                    expect_name = True
                elif token.type in {tokenize.NEWLINE, tokenize.ENDMARKER}:
                    mode = None
                    expect_name = False
        except (IndentationError, tokenize.TokenError):
            pass
    else:
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.partition(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.partition(".")[0])
    return tuple(
        sorted(
            requirement
            for module, requirement in _REQUIREMENT_BY_IMPORT.items()
            if module in imported
        )
    )


def validate_evaluator_imports(source: str | bytes) -> None:
    """Reject imports that are absent from the controlled evaluator runtime."""

    if isinstance(source, bytes):
        try:
            text = source.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EvaluatorCompatibilityError("evaluator is not UTF-8") from exc
    else:
        text = source
    try:
        tree = ast.parse(text, filename="evaluator.py")
    except SyntaxError as exc:
        raise EvaluatorCompatibilityError(f"invalid evaluator syntax: {exc}") from exc
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.partition(".")[0])
    unsupported = sorted(imported - _CONTROLLED_EVALUATOR_IMPORTS)
    if unsupported:
        raise EvaluatorCompatibilityError(
            "evaluator imports packages outside the controlled runtime: "
            + ", ".join(unsupported)
        )


def build_evaluator_entrypoint(
    source: str | bytes,
    *,
    evaluator_path: str = "/grader/evaluator.py",
    trajectory_path: str = "/grader/trajectory.json",
) -> str:
    """Return a self-contained sandbox entrypoint for one evaluator source."""

    metadata = inspect_evaluator_source(source)
    if metadata.dialect not in _DIALECTS:  # defensive guard for future edits
        raise EvaluatorCompatibilityError(f"unknown evaluator dialect: {metadata.dialect}")
    settings = json.dumps(
        {
            "adapter_version": ADAPTER_VERSION,
            "dialect": metadata.dialect,
            "evaluator_path": evaluator_path,
            "trajectory_path": trajectory_path,
            "marker": EVALUATION_MARKER,
        },
        sort_keys=True,
    )
    # Keep this script dependency-free: it runs inside the official sandbox,
    # which does not contain sead itself.
    return f'''\
import dataclasses
import importlib.util
import json
import sys
import types
import typing

SETTINGS = json.loads({settings!r})
sys.path.insert(0, "/utils")

if SETTINGS["dialect"] in ("legacy", "mixed"):
    import common
    common_grader = types.ModuleType("common.grader")
    common_grader.grader = common.grader
    sys.modules["common.grader"] = common_grader

    class CompatibleResult:
        def __init__(self, *args, checkpoints=None, score=None, explanation=None, message=None, **kwargs):
            if args:
                if len(args) != 1 or checkpoints is not None or score is not None:
                    raise TypeError("unsupported Result constructor")
                checkpoints = args[0]
            if checkpoints is not None:
                self.checkpoints = checkpoints
            if score is not None:
                self.score = score
            if explanation is not None or message is not None:
                self.explanation = explanation if explanation is not None else message
                self.message = message if message is not None else explanation
            for key, value in kwargs.items():
                setattr(self, key, value)

    class CompatibleCheckpoint:
        def __init__(self, *args, **kwargs):
            if args and isinstance(args[0], (int, float, bool)):
                if len(args) > 3:
                    raise TypeError("unsupported Checkpoint constructor")
                self.total = args[0]
                self.result = args[1] if len(args) > 1 else kwargs.pop("result", None)
                if len(args) > 2:
                    self.message = args[2]
                elif "message" in kwargs:
                    self.message = kwargs.pop("message")
            else:
                if len(args) > 3:
                    raise TypeError("unsupported Checkpoint constructor")
                if args:
                    self.name = args[0]
                if len(args) > 1:
                    self.result = args[1]
                if len(args) > 2:
                    self.weight = args[2]
            for key, value in kwargs.items():
                setattr(self, key, value)

    scoring = types.ModuleType("scoring")
    scoring.Result = CompatibleResult
    scoring.Checkpoint = CompatibleCheckpoint
    sys.modules["scoring"] = scoring

def plain(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {{str(key): plain(item) for key, item in value.items()}}
    if isinstance(value, (list, tuple, set)):
        return [plain(item) for item in value]
    if hasattr(value, "model_dump"):
        return plain(value.model_dump())
    if hasattr(value, "to_dict"):
        return plain(value.to_dict())
    if dataclasses.is_dataclass(value):
        return plain(dataclasses.asdict(value))
    if hasattr(value, "__dict__"):
        return plain(vars(value))
    raise TypeError(f"unsupported evaluator value: {{type(value).__name__}}")

spec = importlib.util.spec_from_file_location("official_evaluator", SETTINGS["evaluator_path"])
module = importlib.util.module_from_spec(spec)
module.__dict__.update(List=typing.List, Dict=typing.Dict, Optional=typing.Optional, Any=typing.Any)
spec.loader.exec_module(module)
with open(SETTINGS["trajectory_path"], encoding="utf-8") as stream:
    trajectory = stream.read()
raw_result = plain(module.grade_checkpoints(trajectory))
envelope = {{
    "adapter_version": SETTINGS["adapter_version"],
    "dialect": SETTINGS["dialect"],
    "raw_result": raw_result,
}}
print(SETTINGS["marker"] + json.dumps(envelope, sort_keys=True))
'''
