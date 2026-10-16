"""Tests for ftea.install_checks, the FFmpeg check install.sh runs."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from ftea import install_checks, paths


@pytest.fixture
def seen(monkeypatch):
    """Record the config check_toolchain receives; answer with `problems`."""
    calls = {"configs": [], "problems": []}

    def fake(config):
        calls["configs"].append(config)
        return list(calls["problems"])

    monkeypatch.setattr(install_checks, "check_toolchain", fake)
    return calls


def test_should_report_a_ready_toolchain(seen, capsys):
    assert install_checks.main(["toolchain"]) == 0

    out = capsys.readouterr().out
    assert "✓ FFmpeg: h264_nvenc, av1_nvenc and hevc_nvenc" in out
    assert len(seen["configs"]) == 1


def test_should_print_every_problem_and_exit_1(seen, capsys):
    seen["problems"] = [
        "this FFmpeg build has no libvmaf filter, which ftea uses",
        "the GPU could not start an AV1 encode (unsupported device)",
    ]

    assert install_checks.main(["toolchain"]) == 1

    out = capsys.readouterr().out
    assert "  ✗ this FFmpeg build has no libvmaf filter" in out
    assert "  ✗ the GPU could not start an AV1 encode" in out


def test_should_test_with_the_settings_in_config_env(seen, capsys):
    paths.config_path().write_text("NVENC_TUNE=hq\n", encoding="utf-8")

    assert install_checks.main(["toolchain"]) == 0

    assert seen["configs"][0].nvenc_tune == "hq"
    assert f"testing the settings in {paths.config_path()}" in (
        capsys.readouterr().out)


def test_should_test_the_defaults_without_config_env(seen, capsys):
    assert install_checks.main(["toolchain"]) == 0

    assert seen["configs"][0] == install_checks.PipelineConfig()
    assert "testing the default settings" in capsys.readouterr().out


def test_should_stop_on_an_invalid_config_env(seen, capsys):
    paths.config_path().write_text("BATCH_ON_ERROR=halt\n", encoding="utf-8")

    assert install_checks.main(["toolchain"]) == 1

    assert "✗ config.env:" in capsys.readouterr().out
    assert seen["configs"] == []


def test_should_explain_a_config_env_that_is_not_utf8(seen, capsys):
    """Notepad's "Unicode" is UTF-16; it must not end install.sh in a
    traceback."""
    paths.config_path().write_text("VMAF_FLOOR=91", encoding="utf-16")

    assert install_checks.main(["toolchain"]) == 1

    out = capsys.readouterr().out
    assert "✗ config.env:" in out and "not UTF-8" in out
    assert seen["configs"] == []


def test_should_use_the_defaults_when_the_data_folder_is_not_made_yet(
    seen, monkeypatch, tmp_path, capsys,
):
    monkeypatch.setenv(paths.ENV_HOME, str(tmp_path / "not yet"))

    assert install_checks.main(["toolchain"]) == 0

    assert seen["configs"][0] == install_checks.PipelineConfig()
    assert "does not exist yet" in capsys.readouterr().out


def test_should_refuse_a_data_folder_that_is_a_file(
    seen, monkeypatch, tmp_path, capsys,
):
    notes = tmp_path / "notes.txt"
    notes.write_text("x", encoding="utf-8")
    monkeypatch.setenv(paths.ENV_HOME, str(notes))

    assert install_checks.main(["toolchain"]) == 1

    out = capsys.readouterr().out
    assert "✗ data folder: " in out and "a file, not a folder" in out
    assert "does not exist yet" not in out
    assert seen["configs"] == []


def test_should_build_the_ready_line_from_the_checks_it_runs():
    for name in (*install_checks.REQUIRED_ENCODERS,
                 install_checks.REQUIRED_FILTER):
        assert name in install_checks.READY, name


@pytest.mark.parametrize("argv", [[], ["bogus"], ["toolchain", "x"]],
                         ids=repr)
def test_should_reject_anything_but_toolchain(argv, capsys):
    assert install_checks.main(argv) == 2
    assert "usage" in capsys.readouterr().err


def test_should_run_as_a_module_from_the_checkout():
    result = subprocess.run(
        [sys.executable, "-m", "ftea.install_checks"],
        cwd=paths.source_root(),
        env=paths.reframe_subprocess_env(os.environ),
        capture_output=True, text=True, encoding="utf-8",
    )

    assert result.returncode == 2
    assert "usage" in result.stderr
