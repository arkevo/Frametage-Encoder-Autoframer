"""Shared setup for the reframe tests: no developer setting leaks in."""

from __future__ import annotations

import os

import pytest

from ftea import paths

# Captured at import, before any test changes it, for the few tests that
# need the developer's real data root (the latent GPU test).
_REAL_FTEA_HOME = os.environ.get(paths.ENV_HOME)


@pytest.fixture(autouse=True)
def _hermetic_environment(monkeypatch, tmp_path_factory):
    """Point FTEA_HOME at an empty folder and drop every REFRAME_* variable.

    Without this, a config.env or models/ folder in the checkout, or a
    REFRAME_* override in the developer's shell, would change what the
    stage reads under test.
    """
    for key in [k for k in os.environ if k.startswith("REFRAME")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv(paths.ENV_HOME, str(tmp_path_factory.mktemp("home")))


@pytest.fixture
def real_ftea_home(monkeypatch):
    """Restore the developer's own FTEA_HOME (or its absence)."""
    if _REAL_FTEA_HOME is None:
        monkeypatch.delenv(paths.ENV_HOME, raising=False)
    else:
        monkeypatch.setenv(paths.ENV_HOME, _REAL_FTEA_HOME)
