"""Where ftea keeps its files: the repository path contract.

Every default location resolves against a fixed root, never the current
working directory, so `ftea` behaves the same from any folder.

    <data root>/config.env          local settings (gitignored)
    <data root>/input/              what `ftea run` processes by default
    <data root>/output/             finished packages and every report
    <data root>/work/               per-clip scratch, run lock, progress
    <data root>/models/             detector weights; models/torch is
                                    the torch hub cache
    <checkout>/config.env.example   the settings template (tracked)
    <checkout>/.venv/               encoder virtualenv, ftea editable
    <checkout>/.venv-reframe/       reframe virtualenv, ftea NOT installed

The checkout is the folder that holds pyproject.toml and this package.
The data root is the checkout, unless FTEA_HOME names another folder.

Standard library only: the reframe interpreter imports this module
from the checkout without ftea being installed.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

ENV_HOME = "FTEA_HOME"
REFRAME_MODULE = "ftea.reframe.main"
VERIFY_ENV_MODULE = "ftea.reframe.verify_env"

_PACKAGE_DIR = Path(__file__).resolve().parent
_CHECKOUT_MARKER = "pyproject.toml"


class RepoRootError(RuntimeError):
    """ftea is not running from a cloned repository."""


class DataRootError(RuntimeError):
    """FTEA_HOME is set but does not name an existing folder."""


class MissingDataRootError(DataRootError):
    """FTEA_HOME names nothing yet (as opposed to a file)."""


def source_root() -> Path:
    """The folder that contains the ftea package (unchecked)."""
    return _PACKAGE_DIR.parent


def checkout_root() -> Path:
    """The cloned repository; RepoRootError for any other install."""
    root = source_root()
    if not (root / _CHECKOUT_MARKER).is_file():
        raise RepoRootError(
            f"ftea is not running from a cloned repository ({root} has "
            f"no {_CHECKOUT_MARKER}). Clone the repository and run "
            "`pip install -e .` inside it. To keep config.env, input/ "
            f"and output/ elsewhere, set {ENV_HOME} to that folder."
        )
    return root


def data_root() -> Path:
    """FTEA_HOME when set and not blank, otherwise the checkout.

    A set FTEA_HOME must already exist: a typo must fail here, not
    become a new data root that models are downloaded into.
    """
    root = _resolve_home(os.environ.get(ENV_HOME, ""))
    return checkout_root() if root is None else root


def _resolve_home(value: str) -> Path | None:
    """A FTEA_HOME value as an existing folder; None when blank."""
    override = value.strip()
    if not override:
        return None
    root = Path(override).expanduser().resolve()
    if not root.exists():
        raise MissingDataRootError(
            f"{ENV_HOME} is set to {override!r}, which resolves to "
            f"{root}, but that folder does not exist. Create it, or "
            f"fix {ENV_HOME}."
        )
    if not root.is_dir():
        raise DataRootError(
            f"{ENV_HOME} is set to {override!r}, which resolves to "
            f"{root}, but that is a file, not a folder. Point "
            f"{ENV_HOME} at a folder."
        )
    return root


def config_path() -> Path:
    return data_root() / "config.env"


def input_dir() -> Path:
    return data_root() / "input"


def output_dir() -> Path:
    return data_root() / "output"


def work_dir() -> Path:
    return data_root() / "work"


def lock_file() -> Path:
    return work_dir() / ".pipeline.lock"


def progress_file() -> Path:
    return work_dir() / "batch_progress.json"


def models_dir() -> Path:
    return data_root() / "models"


def config_example_path() -> Path:
    return checkout_root() / "config.env.example"


def encoder_venv_dir() -> Path:
    return checkout_root() / ".venv"


def reframe_venv_dir() -> Path:
    return checkout_root() / ".venv-reframe"


def venv_python_candidates(venv: Path) -> tuple[Path, Path]:
    """Interpreter locations in a virtualenv: Windows first, then POSIX."""
    return (venv / "Scripts" / "python.exe", venv / "bin" / "python")


def resolve_setting_path(value: str) -> Path | None:
    """Turn a path-valued config.env setting into a Path.

    Blank means "use the default" and returns None. `~` expands. A
    relative value is anchored at the data root, never the working
    directory.

    On Windows a value with a root but no drive (/c/..., \\x) or a
    drive but no root (C:x) raises ValueError: Git Bash never converts
    file contents, so these would silently land somewhere else. Callers
    add the setting's name to the message.
    """
    text = value.strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if os.name == "nt" and bool(path.drive) != bool(path.root):
        raise ValueError(
            f"{text!r} is not a full Windows path: write a full Windows "
            "path such as D:/ftea/models"
        )
    return path if path.is_absolute() else data_root() / path


def reframe_subprocess_env(base: Mapping[str, str]) -> dict[str, str]:
    """Environment for launching the reframe interpreter.

    That interpreter never has ftea installed, so PYTHONPATH is set to
    the checkout and nothing else; an inherited PYTHONPATH could shadow
    the package. checkout_root() raises RepoRootError for an install
    that is not a checkout; launchers let it surface. When `base` sets
    FTEA_HOME it is passed on as an absolute folder, so the child, which
    starts at cwd=source_root(), resolves the same folder as the
    parent. Pass every path argument as an absolute path.
    """
    env = dict(base)
    env["PYTHONPATH"] = str(checkout_root())
    home = _resolve_home(env.pop(ENV_HOME, ""))
    if home is not None:
        env[ENV_HOME] = str(home)
    return env
