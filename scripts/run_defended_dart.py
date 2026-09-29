#!/usr/bin/env python3
"""Run the canonical DART search with a required pre-execution tool defender."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sead.attacks.dart.cli import defended_main


if __name__ == "__main__":
    raise SystemExit(defended_main())
