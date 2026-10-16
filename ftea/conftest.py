"""Run every test inside a throwaway data root.

FTEA_HOME points at a fresh tmp_path folder for each test, so a missed
patch can never reach the real checkout's input/, output/, work/ or
models/. A test that needs FTEA_HOME unset uses monkeypatch.delenv;
autouse fixtures here run before a test module's own, so it wins.

The session guard fails the run if the real checkout's data folders
changed anyway. It can also trip if ftea runs in this same checkout
while the tests run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ftea import paths

_GUARDED = ("input", "output", "work", "models")


def _snapshot(root: Path) -> dict[str, int | None]:
    found: dict[str, int | None] = {}
    for name in _GUARDED:
        top = root / name
        if not top.exists():
            continue
        for item in top.rglob("*"):
            key = item.relative_to(root).as_posix()
            found[key] = item.stat().st_mtime_ns if item.is_file() else None
    return found


@pytest.fixture(scope="session", autouse=True)
def _real_checkout_data_untouched():
    root = paths.source_root()
    before = _snapshot(root)
    yield
    after = _snapshot(root)
    changed = sorted(
        k for k in before.keys() | after.keys()
        if before.get(k, -1) != after.get(k, -1)
    )
    assert not changed, f"tests changed the real checkout: {changed[:10]}"


@pytest.fixture(autouse=True)
def _sandbox_data_root(tmp_path, monkeypatch):
    home = tmp_path / "ftea-home"
    home.mkdir()
    monkeypatch.setenv(paths.ENV_HOME, str(home))
    return home
