"""Fixtures shared by every encoder test."""

from __future__ import annotations

import os

import pytest

from ftea import paths


@pytest.fixture(autouse=True)
def ftea_home(tmp_path_factory, monkeypatch):
    """Give every test its own data root, and no REFRAME* overrides.

    Without this a test that reaches work/ or output/ would write into
    the checkout itself, and a REFRAME variable exported in the
    developer's shell would change what load_config returns. The root is
    a separate temp folder, not inside tmp_path, so tests that list
    tmp_path see only what they created.
    """
    home = tmp_path_factory.mktemp("ftea-home")
    monkeypatch.setenv(paths.ENV_HOME, str(home))
    for key in [name for name in os.environ if name.startswith("REFRAME")]:
        monkeypatch.delenv(key)
    return home


@pytest.fixture(autouse=True)
def _sandbox_data_root(ftea_home):
    """Replace the suite-wide sandbox from ftea/conftest.py here.

    That one makes tmp_path/ftea-home, so every tmp_path would start with
    a folder the test did not create. ftea_home already gives each test
    its own data root, outside tmp_path.
    """
    return ftea_home


@pytest.fixture
def gpu_run(request):
    """Skip unless this pytest run selected GPU tests with -m gpu."""
    if "gpu" not in (request.config.getoption("markexpr") or ""):
        pytest.skip("end-to-end GPU test: run with -m gpu on an NVENC machine")
