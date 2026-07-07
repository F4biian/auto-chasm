#!/usr/bin/env python3
"""Check that no tracked Python file exceeds the maximum allowed line count.

Usage:
    python scripts/check_file_length.py [DIR ...]

Exits with code 1 if any .py file in the given directories exceeds the limit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MAX_LINES = 750
DEFAULT_DIRS = ("src", "tests")
SKIP_PATTERNS = ("__pycache__", ".venv", ".git")


def _should_skip(path: Path) -> bool:
    """Return True if the path should be ignored."""
    parts = path.parts
    return any(skip in parts for skip in SKIP_PATTERNS) or path.name.startswith(".")


def _count_lines(path: Path) -> int:
    """Return the number of lines in the file."""
    return len(path.read_text(encoding="utf-8").splitlines())


def check(dirs: tuple[str, ...], max_lines: int) -> list[tuple[Path, int]]:
    """Scan directories and return files that exceed *max_lines*."""
    offenders: list[tuple[Path, int]] = []
    for dirname in dirs:
        root = Path(dirname)
        if not root.exists():
            continue
        for py_file in root.rglob("*.py"):
            if _should_skip(py_file):
                continue
            count = _count_lines(py_file)
            if count > max_lines:
                offenders.append((py_file, count))
    return offenders


def main(argv: list[str] | None = None) -> int:
    """Run the check as a CLI command."""
    parser = argparse.ArgumentParser(description="Enforce a maximum line count per file.")
    parser.add_argument(
        "--max-lines",
        type=int,
        default=MAX_LINES,
        help=f"Maximum allowed lines (default: {MAX_LINES})",
    )
    parser.add_argument(
        "dirs",
        nargs="*",
        default=list(DEFAULT_DIRS),
        help=f"Directories to scan (default: {' '.join(DEFAULT_DIRS)})",
    )
    args = parser.parse_args(argv)

    offenders = check(tuple(args.dirs), args.max_lines)

    if offenders:
        print(f"ERROR: {len(offenders)} file(s) exceed {args.max_lines} lines:\n")
        for path, count in sorted(offenders):
            over = count - args.max_lines
            print(f"  {path}: {count} lines (+{over})")
        print("\nSplit large modules into smaller sub-modules to keep them maintainable.")
        return 1

    print(f"All Python files in {' '.join(args.dirs)} are within {args.max_lines} lines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
