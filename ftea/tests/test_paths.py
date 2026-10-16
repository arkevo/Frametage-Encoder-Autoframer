"""Tests for ftea.paths, the repository path contract."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ftea import paths

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _no_ftea_home(monkeypatch):
    """Every test starts with FTEA_HOME unset unless it sets it."""
    monkeypatch.delenv(paths.ENV_HOME, raising=False)


def _clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k != paths.ENV_HOME}


class TestRoots:
    def test_should_use_the_checkout_when_ftea_home_is_unset(self):
        assert paths.source_root() == REPO
        assert paths.checkout_root() == REPO
        assert paths.data_root() == REPO

    def test_should_ignore_the_working_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert paths.data_root() == REPO
        assert paths.input_dir() == REPO / "input"
        assert paths.config_path() == REPO / "config.env"

    def test_should_move_data_when_ftea_home_is_set(
        self, tmp_path, monkeypatch,
    ):
        home = tmp_path / "data root"  # contains a space on purpose
        home.mkdir()
        monkeypatch.setenv(paths.ENV_HOME, str(home))
        assert paths.data_root() == home.resolve()
        assert paths.output_dir() == home.resolve() / "output"

    def test_should_treat_blank_ftea_home_as_unset(self, monkeypatch):
        monkeypatch.setenv(paths.ENV_HOME, "   ")
        assert paths.data_root() == REPO

    def test_should_resolve_relative_ftea_home_once_against_cwd(
        self, tmp_path, monkeypatch,
    ):
        (tmp_path / "rel").mkdir()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(paths.ENV_HOME, "rel")
        assert paths.data_root() == (tmp_path / "rel").resolve()

    def test_should_reject_ftea_home_that_does_not_exist(
        self, tmp_path, monkeypatch,
    ):
        missing = tmp_path / "ftea-dta"  # a typo, never created
        monkeypatch.setenv(paths.ENV_HOME, str(missing))
        with pytest.raises(paths.DataRootError, match=paths.ENV_HOME) as exc:
            paths.data_root()
        assert str(missing.resolve()) in str(exc.value)
        assert not missing.exists()

    def test_should_reject_ftea_home_that_names_a_file(
        self, tmp_path, monkeypatch,
    ):
        settings = tmp_path / "ftea.env"  # a file, not a folder
        settings.write_text("", encoding="utf-8")
        monkeypatch.setenv(paths.ENV_HOME, str(settings))
        with pytest.raises(paths.DataRootError, match="not a folder"):
            paths.data_root()

    def test_should_expand_the_home_directory_in_ftea_home(
        self, tmp_path, monkeypatch,
    ):
        (tmp_path / "data").mkdir()
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        monkeypatch.setenv(paths.ENV_HOME, "~/data")
        assert paths.data_root() == (tmp_path / "data").resolve()


@pytest.fixture
def installed_copy(tmp_path, monkeypatch):
    """ftea as a plain `pip install .` would leave it."""
    package = tmp_path / "site-packages" / "ftea"
    package.mkdir(parents=True)
    monkeypatch.setattr(paths, "_PACKAGE_DIR", package)
    return package


class TestNotACheckout:
    def test_should_raise_actionable_error_without_a_checkout(
        self, installed_copy,
    ):
        with pytest.raises(paths.RepoRootError, match="pip install -e"):
            paths.data_root()

    def test_should_name_ftea_home_in_the_error(self, installed_copy):
        with pytest.raises(paths.RepoRootError, match=paths.ENV_HOME):
            paths.checkout_root()

    def test_should_accept_ftea_home_without_a_checkout(
        self, installed_copy, tmp_path, monkeypatch,
    ):
        (tmp_path / "data").mkdir()
        monkeypatch.setenv(paths.ENV_HOME, str(tmp_path / "data"))
        assert paths.data_root() == (tmp_path / "data").resolve()

    def test_should_still_need_a_checkout_for_virtualenvs(
        self, installed_copy, tmp_path, monkeypatch,
    ):
        (tmp_path / "data").mkdir()
        monkeypatch.setenv(paths.ENV_HOME, str(tmp_path / "data"))
        with pytest.raises(paths.RepoRootError):
            paths.reframe_venv_dir()


class TestLayout:
    def test_should_place_runtime_data_under_the_data_root(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv(paths.ENV_HOME, str(tmp_path))
        root = tmp_path.resolve()
        assert paths.config_path() == root / "config.env"
        assert paths.input_dir() == root / "input"
        assert paths.output_dir() == root / "output"
        assert paths.work_dir() == root / "work"
        assert paths.lock_file() == root / "work" / ".pipeline.lock"
        assert paths.progress_file() == root / "work" / "batch_progress.json"
        assert paths.models_dir() == root / "models"

    def test_should_keep_installed_files_with_the_checkout(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv(paths.ENV_HOME, str(tmp_path))
        assert paths.config_example_path() == REPO / "config.env.example"
        assert paths.encoder_venv_dir() == REPO / ".venv"
        assert paths.reframe_venv_dir() == REPO / ".venv-reframe"

    def test_should_try_the_windows_interpreter_first(self, tmp_path):
        assert paths.venv_python_candidates(tmp_path) == (
            tmp_path / "Scripts" / "python.exe",
            tmp_path / "bin" / "python",
        )

    def test_should_name_the_reframe_entry_modules(self):
        assert paths.REFRAME_MODULE == "ftea.reframe.main"
        assert paths.VERIFY_ENV_MODULE == "ftea.reframe.verify_env"


class TestResolveSettingPath:
    @pytest.mark.parametrize("value", ["", "   ", "\t"])
    def test_should_return_none_when_value_is_blank(self, value):
        assert paths.resolve_setting_path(value) is None

    def test_should_keep_an_absolute_path(self, tmp_path):
        target = tmp_path / "weights"
        assert paths.resolve_setting_path(str(target)) == target

    def test_should_anchor_a_relative_path_at_the_data_root(
        self, tmp_path, monkeypatch,
    ):
        (tmp_path / "root").mkdir()
        monkeypatch.setenv(paths.ENV_HOME, str(tmp_path / "root"))
        monkeypatch.chdir(tmp_path)
        expected = (tmp_path / "root").resolve() / "models" / "custom"
        assert paths.resolve_setting_path(" models/custom ") == expected

    def test_should_expand_the_home_directory(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        assert paths.resolve_setting_path("~/weights") == tmp_path / "weights"

    @pytest.mark.skipif(os.name != "nt", reason="Windows path forms only")
    @pytest.mark.parametrize(
        "value", ["/d/ftea/models", "\\models", "C:models"],
    )
    def test_should_reject_an_ambiguous_windows_path(self, value):
        """Git Bash style, driveless-root and rootless-drive forms.

        Left alone they become <current drive>:\\d\\ftea\\models,
        <current drive>:\\models, or a path quietly relative to the data
        root.
        """
        with pytest.raises(ValueError, match="D:/ftea/models"):
            paths.resolve_setting_path(value)


class TestReframeSubprocessEnv:
    def test_should_replace_an_inherited_pythonpath(self):
        env = paths.reframe_subprocess_env(
            {"PYTHONPATH": "/elsewhere", "KEEP": "1"},
        )
        assert env["PYTHONPATH"] == str(REPO)
        assert env["KEEP"] == "1"
        assert paths.ENV_HOME not in env

    def test_should_not_modify_the_given_mapping(self):
        base = {"PYTHONPATH": "/elsewhere"}
        paths.reframe_subprocess_env(base)
        assert base == {"PYTHONPATH": "/elsewhere"}

    def test_should_take_ftea_home_from_the_given_mapping(
        self, tmp_path,
    ):
        (tmp_path / "data").mkdir()
        env = paths.reframe_subprocess_env(
            {paths.ENV_HOME: str(tmp_path / "data")},
        )
        assert env[paths.ENV_HOME] == str((tmp_path / "data").resolve())

    def test_should_not_add_ftea_home_the_mapping_lacks(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv(paths.ENV_HOME, str(tmp_path))
        assert paths.ENV_HOME not in paths.reframe_subprocess_env({})

    def test_should_drop_a_blank_ftea_home(self):
        env = paths.reframe_subprocess_env({paths.ENV_HOME: "   "})
        assert paths.ENV_HOME not in env

    def test_should_pass_a_relative_ftea_home_on_as_absolute(
        self, tmp_path, monkeypatch,
    ):
        (tmp_path / "rel").mkdir()
        monkeypatch.chdir(tmp_path)
        env = paths.reframe_subprocess_env({paths.ENV_HOME: "rel"})
        assert env[paths.ENV_HOME] == str((tmp_path / "rel").resolve())

    def test_should_point_at_the_code_when_data_moves(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv(paths.ENV_HOME, str(tmp_path))
        env = paths.reframe_subprocess_env({})
        assert env["PYTHONPATH"] == str(REPO)

    def test_should_hand_the_child_the_parents_data_root(
        self, tmp_path, monkeypatch,
    ):
        """A relative FTEA_HOME must not be re-resolved in the child.

        The child starts at cwd=source_root(), so passing "rel" on as
        written would give it <checkout>/rel instead of <cwd>/rel.
        """
        (tmp_path / "rel").mkdir()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(paths.ENV_HOME, "rel")
        code = "import ftea.paths; print(ftea.paths.data_root())"
        result = subprocess.run(
            [sys.executable, "-S", "-c", code],
            cwd=paths.source_root(),
            env=paths.reframe_subprocess_env(dict(os.environ)),
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert Path(result.stdout.strip()) == (tmp_path / "rel").resolve()

    def test_should_refuse_an_install_that_is_not_a_checkout(
        self, installed_copy, tmp_path, monkeypatch,
    ):
        """REFRAME_PYTHON plus FTEA_HOME on a `pip install .` copy.

        No other RepoRootError fires first there, so the helper must
        not hand the child site-packages as PYTHONPATH.
        """
        (tmp_path / "data").mkdir()
        monkeypatch.setenv(paths.ENV_HOME, str(tmp_path / "data"))
        with pytest.raises(paths.RepoRootError):
            paths.reframe_subprocess_env({})

    def test_should_import_ftea_without_an_install(self, tmp_path):
        """The reframe venv's situation: ftea is not installed there.

        -S hides site-packages, and the editable install with it, and
        cwd is elsewhere, so only PYTHONPATH can find the package.
        """
        code = (
            "import ftea.paths, ftea.reframe; "
            "print(ftea.paths.source_root())"
        )
        result = subprocess.run(
            [sys.executable, "-S", "-c", code],
            cwd=tmp_path, env=paths.reframe_subprocess_env(_clean_env()),
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert Path(result.stdout.strip()) == REPO

    def test_should_import_ftea_with_the_checkout_as_cwd(self):
        """cwd alone also works: the second half of the launch rule."""
        env = _clean_env()
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONSAFEPATH", None)
        result = subprocess.run(
            [sys.executable, "-S", "-c", "import ftea.reframe"],
            cwd=REPO, env=env, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr


class TestImportWeight:
    def test_should_not_load_heavy_modules_from_the_light_api(self):
        code = (
            "import sys, ftea, ftea.paths, ftea.cli, ftea.encode, "
            "ftea.reframe\n"
            "heavy = ('torch', 'torchvision', 'cv2', 'numpy')\n"
            "print(','.join(sorted(m for m in sys.modules "
            "if m.split('.')[0] in heavy)))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == ""
