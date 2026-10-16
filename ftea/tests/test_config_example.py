"""Guards for config.env.example, the settings template users copy."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ftea import paths
from ftea.encode.config import (
    ENCODER_KEYS,
    REFRAME_STAGE_KEYS,
    PipelineConfig,
    load_config,
    unknown_keys,
)

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "config.env.example"
SET_LINE = re.compile(r"^([A-Z][A-Z0-9_]*)=", re.MULTILINE)
DOC_LINE = re.compile(r"^#?\s*([A-Z][A-Z0-9_]*)=", re.MULTILINE)
CODE_KEY = re.compile(r'"(REFRAME_[A-Z0-9_]+)"')
# Read by the reframe stage but deliberately left out of the template:
# the legacy single detection floor, replaced by the per-class floors.
UNDOCUMENTED_REFRAME_KEYS = {"REFRAME_MIN_CONFIDENCE"}


def _text() -> str:
    return EXAMPLE.read_text(encoding="utf-8")


def _reframe_keys_read_by_the_stage() -> set[str]:
    found: set[str] = set()
    for path in (REPO / "ftea" / "reframe").glob("*.py"):
        found |= set(CODE_KEY.findall(path.read_text(encoding="utf-8")))
    return found


def test_should_sit_where_the_path_contract_expects_it():
    assert paths.config_example_path() == EXAMPLE
    assert EXAMPLE.is_file()


def test_should_ship_exactly_the_encoder_defaults():
    assert load_config(EXAMPLE) == PipelineConfig()


def test_should_turn_reframing_on():
    assert load_config(EXAMPLE).reframe_enabled is True


def test_should_set_every_encoder_key():
    assert set(ENCODER_KEYS) <= set(SET_LINE.findall(_text()))


def test_should_hold_no_key_the_encoder_does_not_know():
    assert unknown_keys(EXAMPLE) == []


def test_should_leave_the_models_folder_to_the_data_root():
    assert re.search(r"^REFRAME_MODELS_DIR=$", _text(), re.MULTILINE)


def test_should_only_list_reframe_keys_that_something_reads():
    stage = _reframe_keys_read_by_the_stage()
    if not stage:
        pytest.skip("the reframe stage is not in this checkout yet")
    listed = {k for k in DOC_LINE.findall(_text()) if k.startswith("REFRAME_")}
    assert listed - stage - set(ENCODER_KEYS) == set()


def test_should_document_every_reframe_key_the_stage_reads():
    stage = _reframe_keys_read_by_the_stage()
    if not stage:
        pytest.skip("the reframe stage is not in this checkout yet")
    listed = set(DOC_LINE.findall(_text()))
    assert stage - listed - UNDOCUMENTED_REFRAME_KEYS == set()


def test_should_let_the_encoder_know_every_key_the_stage_reads():
    """unknown_keys is the one place that reports a misspelt REFRAME_
    setting, and the encoder never imports the stage, so it keeps its
    own copy of the stage's keys. This keeps that copy exact. The removed
    single confidence floor is reported as unknown on purpose."""
    stage = _reframe_keys_read_by_the_stage()
    if not stage:
        pytest.skip("the reframe stage is not in this checkout yet")
    assert set(REFRAME_STAGE_KEYS) == stage - UNDOCUMENTED_REFRAME_KEYS
