"""The encoder's boundaries, checked from its source and imports.

No network access, no delivery step, no reframe import, exactly one
place that starts processes, and a scene module the reframe stage can
import on its own.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

ENCODE = Path(__file__).resolve().parents[1]
REPO = ENCODE.parents[1]
SOURCES = sorted(ENCODE.glob("*.py"))
NETWORK_MODULES = {
    "socket", "ssl", "http", "urllib", "ftplib", "smtplib", "xmlrpc",
    "asyncio", "requests", "httpx", "aiohttp",
}
HEAVY_MODULES = ("torch", "torchvision", "cv2", "numpy")


def _name(path: Path) -> str:
    return path.name


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            found.add((node.module or "").split(".")[0])
    return found


def _modules_loaded_by(statement: str) -> set[str]:
    code = f"import sys\n{statement}\nprint('\\n'.join(sorted(sys.modules)))"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True,
        encoding="utf-8", errors="replace",
    )
    assert result.returncode == 0, result.stderr
    return set(result.stdout.split())


def test_should_find_every_encoder_module():
    assert {p.name for p in SOURCES} >= {
        "config.py", "encode.py", "geometry.py", "main.py", "package.py",
        "preflight.py", "probe.py", "report.py", "scene.py", "tools.py",
    }


@pytest.mark.parametrize("path", SOURCES, ids=_name)
def test_should_import_no_network_module(path):
    assert not (_imported_modules(path) & NETWORK_MODULES)


@pytest.mark.parametrize("path", SOURCES, ids=_name)
def test_should_start_processes_only_through_run_tool(path):
    if path.name == "tools.py":
        pytest.skip("tools.py is the one place that starts processes")
    text = path.read_text(encoding="utf-8")
    assert "subprocess" not in _imported_modules(path)
    assert not re.search(
        r"\bos\.(?:system|popen|spawn\w*|exec\w*|startfile)\b", text,
    )


@pytest.mark.parametrize("path", SOURCES, ids=_name)
def test_should_have_no_delivery_step(path):
    """There is no upload or registration phase: output stays on disk."""
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"(?i)\b(?:upload|regist|rclone|bucket)\w*", text)


def test_should_never_load_the_reframe_stage_in_the_encoder():
    loaded = _modules_loaded_by("import ftea.encode.main")
    assert not {
        name for name in loaded
        if name == "ftea.reframe" or name.startswith("ftea.reframe.")
    }
    assert not {n for n in loaded if n.split(".")[0] in HEAVY_MODULES}


def test_should_let_the_reframe_stage_import_scene_on_its_own():
    loaded = _modules_loaded_by("import ftea.encode.scene")
    assert "ftea.encode.main" not in loaded
    assert not {n for n in loaded if n.split(".")[0] in HEAVY_MODULES}
