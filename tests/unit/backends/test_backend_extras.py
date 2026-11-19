"""Proves MLX and PyTorch are EQUAL, OPTIONAL peers (Phase 5 packaging).

Core `auto-chasm` depends only on numpy; a backend is chosen by installing an
extra (`auto-chasm[mlx]` or `auto-chasm[torch]`). The contract that makes that
honest: `import auto_chasm` — and importing the whole public API — must succeed
with EITHER backend installed alone, never requiring both.

We can't uninstall a backend on the dev box (both are installed), so each test
spawns a clean subprocess with a meta-path finder that makes one backend's import
raise `ImportError` (simulating "not installed"), then imports the package and its
full public surface. A leak check confirms the blocked backend was never imported.
"""

from __future__ import annotations

import subprocess
import sys

# Imports the package + every public symbol, with each name in ``BLOCKED`` made
# unimportable (simulating that backend not being installed).
_CHILD = """
import sys, importlib.abc
BLOCKED = {blocked!r}


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        top = name.split(".")[0]
        if top in BLOCKED:
            raise ImportError(f"simulated: {{name}} not installed")
        return None


sys.meta_path.insert(0, _Blocker())

import auto_chasm  # noqa: F401
from auto_chasm import (  # noqa: F401
    Dataset,
    JointLoss,
    LayerSweep,
    Model,
    ModuleSpec,
    ProbeConfig,
    Task,
    Trainer,
    ops,
)

leaked = [m for m in sys.modules if m.split(".")[0] in BLOCKED]
assert not leaked, f"blocked backend(s) imported anyway: {{leaked[:5]}}"
print("OK")
"""


def _import_with_backends_blocked(*blocked: str) -> subprocess.CompletedProcess[str]:
    """Import auto_chasm in a clean subprocess with each ``blocked`` name unimportable."""
    return subprocess.run(
        [sys.executable, "-c", _CHILD.format(blocked=list(blocked))],
        capture_output=True,
        text=True,
        check=False,
    )


def test_imports_with_only_torch_installed() -> None:
    """With MLX unavailable, the package + full public API still import (torch peer)."""
    result = _import_with_backends_blocked("mlx")
    assert result.returncode == 0, f"import failed with mlx blocked:\n{result.stderr}"
    assert result.stdout.strip() == "OK"


def test_imports_with_only_mlx_installed() -> None:
    """With PyTorch unavailable, the package + full public API still import (mlx peer)."""
    result = _import_with_backends_blocked("torch")
    assert result.returncode == 0, f"import failed with torch blocked:\n{result.stderr}"
    assert result.stdout.strip() == "OK"


def test_imports_with_no_backend_installed() -> None:
    """`import auto_chasm` (+ public API) succeeds with NEITHER backend installed.

    A core-only install (`pip install auto-chasm`, no extra) must import cleanly;
    a backend is required only to load a model, where ``Model.from_pretrained``
    raises a clear error naming the extras. Import must never die with a raw
    ``ModuleNotFoundError`` for mlx/torch.
    """
    result = _import_with_backends_blocked("mlx", "torch")
    assert result.returncode == 0, f"import failed with no backend:\n{result.stderr}"
    assert result.stdout.strip() == "OK"
