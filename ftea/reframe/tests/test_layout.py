"""Guards for the reframe package as a whole.

The stage was carried over from an earlier in-house layout. These tests
keep the copy honest: no path from that layout survives, importing a
stage module never loads torch, OpenCV, numpy or the encoder package,
every test runs isolated from the developer's own settings, and the
modules the launchers run really exist.
"""

from __future__ import annotations

import importlib
import inspect
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from ftea import paths

REFRAME_DIR = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()
STALE_LAYOUT = re.compile(
    r"pipeline[./](?:reframe|gpu)\b"
    r"|pipeline/(?:config\.env|input|work|install\.sh)"
    r"|\bREPO_ROOT\b"
)
NOT_AT_IMPORT = ("torch", "torchvision", "cv2", "numpy")


def _package_files() -> list[Path]:
    return sorted(
        p for p in REFRAME_DIR.rglob("*")
        if p.suffix in (".py", ".txt") and p != THIS_FILE
        and "__pycache__" not in p.parts
    )


def _rel(path: Path) -> str:
    return path.relative_to(REFRAME_DIR).as_posix()


def test_should_isolate_every_test_from_developer_settings():
    home = Path(os.environ[paths.ENV_HOME])
    assert home.is_dir()
    assert list(home.iterdir()) == []
    assert [k for k in os.environ if k.startswith("REFRAME")] == []


def test_should_import_every_stage_module_without_heavy_dependencies():
    """torch, cv2 and numpy load lazily, and never the encoder package."""
    code = (
        "import importlib, pkgutil, sys\n"
        "import ftea.reframe as package\n"
        "prefix = 'ftea.reframe.'\n"
        "for info in pkgutil.iter_modules(package.__path__, prefix):\n"
        "    if not info.ispkg:\n"
        "        importlib.import_module(info.name)\n"
        f"heavy = {NOT_AT_IMPORT!r}\n"
        "print(','.join(sorted(\n"
        "    m for m in sys.modules\n"
        "    if m.split('.')[0] in heavy or m.startswith('ftea.encode'))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=paths.source_root(),
        env=paths.reframe_subprocess_env(os.environ),
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


@pytest.mark.parametrize("path", _package_files(), ids=_rel)
def test_should_not_reference_the_previous_layout(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    hits = [
        f"{number}: {line.strip()}"
        for number, line in enumerate(lines, 1)
        if STALE_LAYOUT.search(line)
    ]
    assert hits == [], f"{_rel(path)} names the previous layout: {hits}"


@pytest.mark.parametrize(
    "module", [paths.REFRAME_MODULE, paths.VERIFY_ENV_MODULE],
)
def test_should_provide_the_entry_modules_the_launchers_run(module):
    imported = importlib.import_module(module)
    assert callable(getattr(imported, "main", None))
    assert 'if __name__ == "__main__":' in inspect.getsource(imported)
