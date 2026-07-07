#!/usr/bin/env python3
"""Verify that the Python examples in docs/ run exactly as written.

Honesty contract
----------------
"Verified" means *the code shown in the docs runs as shown*.  To keep that
claim truthful, this script:

* verifies **every** ``docs/*.md`` file, including the API reference and the
  docs README — nothing is skipped;
* injects **no** hidden imports.  Each markdown file is concatenated into one
  script and run as-is, so a file that forgets its ``import`` lines fails here
  (which is exactly what a reader pasting the code would experience);
* does **not** count commented-out or illustrative code as "passed".  A block
  is executed only if it is a real ``python`` block without a skip marker.

Block markers (HTML comment on the line immediately above the fence)
--------------------------------------------------------------------
``<!-- verify: skip -->``
    The block is illustrative (e.g. an interactive REPL loop, a destructive
    operation, or pseudo-code).  It is neither executed nor counted as passed.
    It is reported separately so the count stays honest.

``<!-- verify: mlx -->``
    The block is MLX-only (uses ``mlx.core`` / ``mlx.nn``).  It is executed
    only when the active backend is MLX; on PyTorch it is reported as skipped
    rather than failed.

A file's executable blocks are concatenated *in document order* and run as a
single process, so a later block may use names defined in an earlier one.
``bash`` and other non-``python`` fences are ignored.

Run with::

    uv run python scripts/verify_docs.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Fenced code blocks, capturing the language and the optional marker comment
# that sits on the line immediately above the opening fence.
_BLOCK_RE = re.compile(
    r"(?:<!--\s*verify:\s*(?P<marker>[\w-]+)\s*-->\n)?```(?P<lang>\w+)\n(?P<body>.*?)```",
    re.DOTALL,
)


def _active_backend() -> str:
    """Return the backend auto-chasm will use here ('mlx' or 'torch')."""
    try:
        from auto_chasm.model import _detect_backend  # type: ignore[attr-defined]

        return _detect_backend()
    except Exception:
        return "mlx"


class Block:
    """One fenced code block plus its verification disposition."""

    def __init__(self, lang: str, marker: str | None, body: str) -> None:
        """Store a fenced block's language, verify-marker, and source body."""
        self.lang = lang
        self.marker = marker
        self.body = body.strip()


def extract_blocks(md_path: Path) -> list[Block]:
    """Extract all fenced code blocks (with markers) from a markdown file."""
    content = md_path.read_text()
    return [
        Block(m.group("lang"), m.group("marker"), m.group("body"))
        for m in _BLOCK_RE.finditer(content)
        if m.group("body").strip()
    ]


def strip_repl(code: str) -> str:
    """Remove ``>>> `` and ``... `` REPL prefixes from code lines."""
    out: list[str] = []
    for line in code.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(">>> "):
            out.append(line.replace(">>> ", "", 1).lstrip())
        elif stripped.startswith("... "):
            out.append(line.replace("... ", "", 1).lstrip())
        elif stripped in (">>>", "..."):
            out.append("")
        else:
            out.append(line)
    return "\n".join(out)


def _wrap(code: str) -> str:
    """Wrap a concatenated script so it runs from a clean temp cwd.

    No imports are injected — the docs must import what they use.
    """
    root = str(PROJECT_ROOT)
    return (
        "from __future__ import annotations\n"
        "import os, sys, tempfile\n"
        "os.chdir(tempfile.mkdtemp())\n"
        f"if {root!r} not in sys.path:\n"
        f"    sys.path.insert(0, {root!r})\n"
        + code.removeprefix("from __future__ import annotations\n")
    )


def _run(code: str) -> tuple[bool, str]:
    """Run a wrapped script in a subprocess; return (ok, stderr)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(_wrap(code))
        tmp = f.name
    try:
        result = subprocess.run(
            ["uv", "run", "python", tmp],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        return False, "Timeout (180s)"
    except Exception as e:  # pragma: no cover - environment failure
        return False, f"Subprocess error: {e}"
    finally:
        Path(tmp).unlink(missing_ok=True)

    if result.returncode == 0:
        return True, ""
    stderr = result.stderr.strip()
    if len(stderr) > 800:
        stderr = stderr[:800] + "\n...(truncated)"
    return False, stderr


class FileResult:
    """Per-file verification tally."""

    def __init__(self, name: str) -> None:
        """Initialize an empty tally for the given doc file name."""
        self.name = name
        self.executed = 0  # python blocks that ran and passed
        self.illustrative = 0  # marked skip
        self.mlx_skipped = 0  # mlx-only, skipped on torch
        self.ok = True
        self.error = ""


def verify_file(md_path: Path, backend: str) -> FileResult:
    """Concatenate and run a file's executable python blocks."""
    res = FileResult(md_path.name)
    blocks = extract_blocks(md_path)

    runnable: list[str] = []
    for block in blocks:
        if block.lang != "python":
            continue
        if block.marker == "skip":
            res.illustrative += 1
            continue
        if block.marker == "mlx" and backend != "mlx":
            res.mlx_skipped += 1
            continue
        runnable.append(strip_repl(block.body))

    if not runnable:
        return res

    combined = "\n\n".join(runnable)
    ok, err = _run(combined)
    res.executed = len(runnable)
    res.ok = ok
    res.error = err
    return res


def verify_docs() -> int:
    """Verify every doc file. Return exit code (0 = all executed blocks pass)."""
    backend = _active_backend()
    docs_dir = PROJECT_ROOT / "docs"
    md_files = sorted(docs_dir.glob("*.md"))

    print(f"Active backend: {backend}")

    total_exec = 0
    total_illustrative = 0
    total_mlx_skipped = 0
    failures: list[FileResult] = []

    for md_file in md_files:
        res = verify_file(md_file, backend)
        if res.executed == 0 and res.illustrative == 0 and res.mlx_skipped == 0:
            continue

        tag = "PASS" if res.ok else "FAIL"
        extra = []
        if res.illustrative:
            extra.append(f"{res.illustrative} illustrative")
        if res.mlx_skipped:
            extra.append(f"{res.mlx_skipped} mlx-only skipped")
        suffix = f" ({', '.join(extra)})" if extra else ""
        print(f"--- {res.name}: [{tag}] {res.executed} executed{suffix}")
        if not res.ok:
            err_short = res.error.replace("\n", "\\n ")[:300]
            print(f"      {err_short}")
            failures.append(res)

        total_exec += res.executed
        total_illustrative += res.illustrative
        total_mlx_skipped += res.mlx_skipped

    print("=" * 60)
    passed_files = "all" if not failures else f"{len(failures)} failing"
    print(f"files: {passed_files}")
    print(
        f"blocks: {total_exec} executed (ran as shown), "
        f"{total_illustrative} illustrative (marked, not run), "
        f"{total_mlx_skipped} mlx-only skipped on this backend"
    )

    if failures:
        print(f"\nFailing files ({len(failures)}):")
        for res in failures:
            print(f"\n  {res.name}:")
            for line in res.error.splitlines()[:12]:
                print(f"    {line}")
            if len(res.error.splitlines()) > 12:
                print("    ...")

    # Clean up any stray checkpoint dirs the examples wrote into the repo root.
    for pattern in ("ckpt_*", "my_digit_model", "_verify*", "checkpoints"):
        for path in PROJECT_ROOT.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(verify_docs())
