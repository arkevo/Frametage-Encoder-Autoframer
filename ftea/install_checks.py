"""The FFmpeg check install.sh runs before it creates anything.

    python -m ftea.install_checks toolchain

install.sh runs this with the base Python, from the checkout and with
PYTHONPATH set to it, before either virtual environment exists; ftea
needs only the standard library for it. It applies the encoder's own
startup check, so installing and every later `ftea run` judge FFmpeg
the same way: each NVENC encoder must really encode a frame, libvmaf
must be present, and a short sample must encode with the settings in
config.env.

Exit code 0 when FFmpeg is ready, 1 when it is not, 2 for a usage error.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence, Tuple

from . import paths
from .encode.config import (
    ConfigValidationError,
    PipelineConfig,
    load_config,
)
from .encode.preflight import (
    REQUIRED_ENCODERS,
    REQUIRED_FILTER,
    check_toolchain,
)

# Windows gives Python a cp1252 stdout, which cannot print the marks.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

USAGE = "usage: python -m ftea.install_checks toolchain"
READY = (f"FFmpeg: {', '.join(REQUIRED_ENCODERS[:-1])} and "
         f"{REQUIRED_ENCODERS[-1]} each encoded a test frame, "
         f"{REQUIRED_FILTER} is present, and AV1 and H.264 test encodes "
         "with your settings succeeded")


def install_config() -> Tuple[PipelineConfig, str]:
    """The settings to test with, and where they came from.

    Raises ConfigValidationError for a config.env with a bad value, and
    DataRootError for a FTEA_HOME that names a file. A FTEA_HOME that
    does not exist yet is fine here: `bash install.sh --check` creates
    nothing, and a full install creates it before this runs.
    """
    try:
        path = paths.config_path()
    except paths.MissingDataRootError:
        return (PipelineConfig(),
                "the data folder does not exist yet; testing the default "
                "settings")
    if not path.is_file():
        return PipelineConfig(), f"no {path}; testing the default settings"
    return (load_config(path, env=os.environ),
            f"testing the settings in {path}")


def toolchain() -> int:
    print("  testing FFmpeg with real encodes on the GPU ...")
    try:
        config, note = install_config()
    except ConfigValidationError as exc:
        print(f"  ✗ config.env: {exc}")
        return 1
    except paths.DataRootError as exc:
        print(f"  ✗ data folder: {exc}")
        return 1
    print(f"    ({note})")
    problems: List[str] = check_toolchain(config)
    if not problems:
        print(f"  ✓ {READY}")
        return 0
    for problem in problems:
        print(f"  ✗ {problem}")
    return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args != ["toolchain"]:
        print(USAGE, file=sys.stderr)
        return 2
    return toolchain()


if __name__ == "__main__":
    raise SystemExit(main())
