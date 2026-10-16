"""Pins the data-root sandbox in ftea/conftest.py.

This module has no fixtures of its own, so it sees exactly what every
other test sees. Tests rely on that sandbox instead of patching each
data path, so losing it would let them write into the real checkout.
"""

from __future__ import annotations

from ftea import paths


def test_should_run_every_test_in_a_throwaway_data_root(tmp_path):
    root = paths.data_root()
    assert root == (tmp_path / "ftea-home").resolve()
    assert root != paths.source_root()
