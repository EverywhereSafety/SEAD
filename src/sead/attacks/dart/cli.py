"""Installed console entry point for deterministic DART tree search v3."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

from .config import load_tree_search_config, with_defense_config
from .runner import run_from_config, validate_runtime_config
from .models import SCHEMA_VERSION


def _main(argv: list[str] | None = None, *, require_tool_defense: bool = False) -> int:
    parser = argparse.ArgumentParser(
        description="Run DART tree search with optional tool defense"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--defense-config",
        type=Path,
        help="YAML containing a defense section that replaces the attack config's gates",
    )
    parser.add_argument(
        "--run-id",
        default=time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        + "-"
        + uuid.uuid4().hex[:8],
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "reports/dart-defended"
            if require_tool_defense
            else "reports/dart-tree-search-v4"
        ),
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.run_id or args.run_id in {".", ".."} or "/" in args.run_id:
        parser.error("--run-id must be one path component")
    config = load_tree_search_config(args.config)
    if args.defense_config is not None:
        config = with_defense_config(config, args.defense_config)
    if require_tool_defense and not config.defense.get("tool", {}).get("enabled"):
        parser.error(
            "defended DART requires defense.tool.enabled: true; "
            "provide it in --config or --defense-config"
        )
    if args.check_only:
        runtime = validate_runtime_config(config)
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "benchmark": dict(config.benchmark),
                    "controller": dict(config.controller),
                    "target": dict(config.target),
                    "defense": dict(config.defense),
                    "search": config.search.to_dict(),
                    "execution": dict(config.execution),
                    "runtime": dict(runtime),
                    "check_only": True,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    outcome = run_from_config(
        config,
        run_id=args.run_id,
        output_dir=args.output_root / args.run_id,
    )
    print(json.dumps(outcome.summary, indent=2, ensure_ascii=False))
    return 0 if outcome.summary["final_replay_success"] else 2


def main(argv: list[str] | None = None) -> int:
    return _main(argv)


def defended_main(argv: list[str] | None = None) -> int:
    return _main(argv, require_tool_defense=True)


if __name__ == "__main__":
    raise SystemExit(main())
