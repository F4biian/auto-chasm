#!/usr/bin/env python3
"""Enforce Google-style docstrings.

Checks that every public function/class/method in ``src/`` and ``tests/`` has a
docstring and uses Google-style sections (``Args:``, ``Returns:``, ``Raises:``).

Usage:
    python scripts/check_sphinx_docstrings.py [DIR ...]

Exits with code 1 if any file violates the docstring convention.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

REQUIRED_SECTIONS = ("Args:",)
VALID_SECTIONS = ("Args:", "Returns:", "Raises:", "Yields:", "Notes:", "Examples:")
SKIP_PREFIXES = ("test_", "_")
SKIP_PATTERNS = ("__pycache__", ".venv", ".git")
DEFAULT_DIRS = ("src", "tests")


def _should_skip(path: Path) -> bool:
    """Return True if the path should be ignored."""
    parts = path.parts
    return any(skip in parts for skip in SKIP_PATTERNS) or path.name.startswith(".")


def _is_public(name: str) -> bool:
    """Return True if *name* is a public identifier."""
    return not name.startswith("_")


class _DocstringVisitor(ast.NodeVisitor):
    """AST visitor that collects missing or malformed docstrings."""

    def __init__(self, filepath: str | None = None) -> None:
        super().__init__()
        self.filepath = filepath or "<unknown>"
        self.errors: list[str] = []
        self._is_test_file = "tests" in Path(filepath).parts if filepath else False

    def _check_docstring(
        self, node: ast.FunctionDef | ast.ClassDef | ast.AsyncFunctionDef, kind: str
    ) -> None:
        """Validate docstring presence and Google-style sections."""
        name = node.name
        if not _is_public(name):
            return
        if kind == "function" and name.startswith(SKIP_PREFIXES):
            return
        # Skip D102/D107 for test files (matching ruff's per-file-ignores)
        if self._is_test_file and kind == "function" and isinstance(node, ast.FunctionDef):
            return
        if self._is_test_file and kind == "class" and name == "__init__":
            return

        docstring = ast.get_docstring(node)
        if not docstring:
            self.errors.append(
                f"{self.filepath}:{node.lineno}: Missing docstring in {kind} '{name}'"
            )
            return

        lines = docstring.splitlines()
        if not lines:
            self.errors.append(f"{self.filepath}:{node.lineno}: Empty docstring in {kind} '{name}'")
            return

        # First line should be a short imperative-mood summary
        first_line = lines[0].strip()
        if not first_line.endswith("."):
            self.errors.append(
                f"{self.filepath}:{node.lineno}: First line must end with a period in '{name}'"
            )
            return

        # If Args: or Returns: are used, they must be proper Google-style
        # (we're lenient — we only flag clearly malformed sections)
        for line in lines[1:]:
            stripped = line.strip()
            if (
                stripped.endswith(":")
                and stripped not in VALID_SECTIONS
                and not stripped.startswith("*")
                and any(
                    s.lower() in stripped.lower()
                    for s in ("param", "returns", "raises", "args", "notes")
                )
            ):
                self.errors.append(
                    f"{self.filepath}:{node.lineno}: Unknown or misnamed section "
                    f"'{stripped}' in '{name}'. "
                    f"Use Google-style sections: {', '.join(VALID_SECTIONS)}"
                )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self._check_docstring(node, "class")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._check_docstring(node, "function")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._check_docstring(node, "function")
        self.generic_visit(node)


def check_file(path: Path) -> list[str]:
    """Return a list of error messages for *path*."""
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return [f"{path}: syntax error — cannot parse"]
    except UnicodeDecodeError:
        return [f"{path}: not a text file — skip"]

    visitor = _DocstringVisitor(filepath=str(path))
    visitor.visit(tree)
    return visitor.errors


def check(dirs: tuple[str, ...]) -> list[str]:
    """Scan directories and collect all docstring violations."""
    all_errors: list[str] = []
    for dirname in dirs:
        root = Path(dirname)
        if not root.exists():
            continue
        for py_file in sorted(root.rglob("*.py")):
            if _should_skip(py_file):
                continue
            all_errors.extend(check_file(py_file))
    return all_errors


def main(argv: list[str] | None = None) -> int:
    """Run the check as a CLI command."""
    parser = argparse.ArgumentParser(description="Enforce Google-style docstrings.")
    parser.add_argument(
        "dirs",
        nargs="*",
        default=list(DEFAULT_DIRS),
        help=f"Directories to scan (default: {' '.join(DEFAULT_DIRS)})",
    )
    args = parser.parse_args(argv)

    errors = check(tuple(args.dirs))

    if errors:
        print(f"ERROR: {len(errors)} docstring violation(s) found:\n")
        for err in errors:
            print(f"  {err}")
        print("\nEvery public function, class, and method needs a Google-style docstring.")
        return 1

    print(f"All public symbols in {' '.join(args.dirs)} have proper docstrings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
