"""Install-time steps for the reframe stage, run by install.sh.

    python -m ftea.reframe.provision models
    python -m ftea.reframe.provision torch-build
    python -m ftea.reframe.provision torchvision-url

models           downloads the three model files into the models folder
                 and checks each against its pinned size and SHA-256. It
                 needs only the standard library, so install.sh runs it
                 with the base Python before .venv-reframe exists.
torch-build      checks that the installed torch and torchvision are
                 exactly the CUDA builds requirements-torch.txt pins,
                 which catches a CPU-only torch pulled in by a later pip
                 install.
torchvision-url  checks that torchvision looks for the person-detector
                 weights at the pinned URL, so it loads the verified file
                 and never downloads one of its own.

Launch it the way every reframe-stage module is launched: from the
checkout, with PYTHONPATH set to the checkout. Exit code 0 when all is
well, 1 when something is not, 2 for a usage error.
"""

from __future__ import annotations

import hashlib
import http.client
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Sequence

from .. import paths
from .detect import torch_home
from .errors import ReframeError
from .settings import load_settings

# Windows gives Python a cp1252 stdout, which cannot print the marks.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

TORCH_REQUIREMENTS = Path(__file__).with_name("requirements-torch.txt")
ATTEMPTS = 3
RETRY_DELAY_SEC = 2.0
TIMEOUT_SEC = 60
CHUNK = 1 << 20
LFS_POINTER = b"version https://git-lfs"
FetchStatus = Literal["present", "downloaded", "replaced"]
STATUS_TEXT: Dict[FetchStatus, str] = {
    "present": "already present",
    "downloaded": "downloaded",
    "replaced": "replaced a damaged copy",
}


@dataclass(frozen=True)
class ModelFile:
    """One pinned download.

    An in_torch_hub file goes into torchvision's download cache under the
    models folder, where torchvision finds it instead of downloading.
    """

    name: str
    url: str
    sha256: str
    size: int
    license: str
    in_torch_hub: bool = False


# Pinned 2026-10-15. Each URL names an exact upstream commit (OpenCV Zoo)
# or a versioned file (PyTorch), and each size and hash was measured on
# the bytes it served. OpenCV Zoo keeps its weights in git-lfs: the
# github.com/<owner>/<repo>/raw/<commit>/ form serves the real file,
# while raw.githubusercontent.com serves a 132-byte pointer instead.
MODELS = (
    ModelFile(
        name="face_detection_yunet_2023mar.onnx",
        url="https://github.com/opencv/opencv_zoo/raw/"
            "f12e12798e8314f7c074a6656816c048dcc95b7a/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        sha256="8f2383e4dd3cfbb4553ea8718107fc04"
               "23210dc964f9f4280604804ed2552fa4",
        size=232_589,
        license="MIT",
    ),
    ModelFile(
        name="text_detection_en_ppocrv3_2023may.onnx",
        url="https://github.com/opencv/opencv_zoo/raw/"
            "b32d27c8f138ffd625efc52e2d82f8b7d54dabc7/models/"
            "text_detection_ppocr/text_detection_en_ppocrv3_2023may.onnx",
        sha256="03f550c6b406fda8bf54bd8327815f6c"
               "7e2edd98cea02348c93d879254366587",
        size=2_423_490,
        license="Apache-2.0",
    ),
    ModelFile(
        name="fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth",
        url="https://download.pytorch.org/models/"
            "fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth",
        sha256="fb6a3cc702b1df54c18a44b26708cd08"
               "3614211062d0c36d2ca7bf9270df3533",
        size=77_844_807,
        license="BSD-3-Clause",
        in_torch_hub=True,
    ),
)


class FetchError(RuntimeError):
    """A pinned file could not be downloaded and verified."""


class ConnectionDropped(ConnectionError):
    """The transfer stopped before the whole file arrived (retried)."""

    def __init__(self, received: int, expected: int):
        super().__init__(f"the connection dropped after {received:,} of "
                         f"{expected:,} bytes")


def models_dir() -> Path:
    """The models folder the reframe stage reads, config.env included."""
    return load_settings(paths.config_path(), env=os.environ).models_dir


def destination(model: ModelFile, models: Path) -> Path:
    """Where the reframe stage looks for this file by default.

    A REFRAME_FACE_MODEL, REFRAME_TITLE_MODEL or REFRAME_PERSON_MODEL
    path set in config.env takes precedence in the stage, and install.sh
    neither fetches nor checks that file.
    """
    if model.in_torch_hub:
        return torch_home(models) / "hub" / "checkpoints" / model.name
    return models / model.name


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _open(url: str, timeout: float):
    return urllib.request.urlopen(url, timeout=timeout)


def _declared_length(response) -> Optional[int]:
    """The Content-Length the server sent, or None."""
    headers = getattr(response, "headers", None)
    value = headers.get("Content-Length") if headers is not None else None
    try:
        return None if value is None else int(value)
    except ValueError:
        return None


def _download(model: ModelFile, part: Path, opener: Callable) -> None:
    """Stream the file into `part`, refusing more bytes than pinned.

    A transfer that ends early raises ConnectionDropped. http.client
    does not report that itself: a response with a Content-Length that
    is cut short just reads as b"" (a chunked one raises IncompleteRead).
    The body counts as complete only when it is the pinned size, or the
    exact size the server declared, so a short file the server meant to
    send (a git-lfs pointer, a proxy's page) still reaches _mismatch.
    """
    received = 0
    with opener(model.url, timeout=TIMEOUT_SEC) as response:
        with part.open("wb") as out:
            while True:
                try:
                    block = response.read(CHUNK)
                except http.client.IncompleteRead as exc:
                    raise ConnectionDropped(
                        received + len(exc.partial), model.size) from exc
                if not block:
                    break
                received += len(block)
                if received > model.size:
                    raise FetchError(
                        f"{model.name}: the server sent more than the "
                        f"{model.size} bytes expected; stopped")
                out.write(block)
        declared = _declared_length(response)
    if received < model.size and received != declared:
        raise ConnectionDropped(received, model.size)


def _reason(exc: BaseException) -> str:
    if isinstance(exc, ConnectionDropped):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


def _download_with_retries(
    model: ModelFile, part: Path, opener: Callable, sleep: Callable,
) -> None:
    last = ""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            _download(model, part, opener)
            return
        # OSError: URLError, HTTPError, timeouts, resets, ConnectionDropped.
        # HTTPException: http.client's own, such as IncompleteRead or
        # BadStatusLine, which are not OSError.
        except (OSError, http.client.HTTPException) as exc:
            last = _reason(exc)
        if attempt < ATTEMPTS:
            sleep(RETRY_DELAY_SEC)
    raise FetchError(
        f"{model.name}: could not download {model.url} after {ATTEMPTS} "
        f"attempts ({last})")


def _mismatch(model: ModelFile, part: Path) -> Optional[str]:
    """Why the downloaded bytes are not the pinned file, or None."""
    digest = file_sha256(part)
    if digest == model.sha256:
        return None
    with part.open("rb") as handle:
        head = handle.read(len(LFS_POINTER))
    if head == LFS_POINTER:
        return ("the server sent a git-lfs pointer file instead of the "
                "weights (GitHub did not serve the stored file). Try again "
                "later; if it keeps happening, the pinned URL in "
                "ftea/reframe/provision.py needs updating")
    return (
        f"downloaded {part.stat().st_size} bytes with SHA-256 {digest}, "
        f"but {model.size} bytes with SHA-256 {model.sha256} were "
        "expected. The file changed upstream, or something between you "
        "and the server (a proxy, a captive portal) altered it. It was "
        "not used.")


def fetch(
    model: ModelFile,
    target: Path,
    *,
    opener: Optional[Callable] = None,
    sleep: Optional[Callable[[float], None]] = None,
) -> FetchStatus:
    """Make `target` hold exactly the pinned bytes.

    Returns "present" when it already does, otherwise "downloaded" or
    "replaced". Nothing unverified ever lands at `target`: the bytes go
    to <name>.part beside it, are checked, and only then moved into
    place. Raises FetchError.
    """
    if target.is_file() and file_sha256(target) == model.sha256:
        return "present"
    existed = target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    try:
        _download_with_retries(
            model, part, opener or _open, sleep or time.sleep)
        problem = _mismatch(model, part)
        if problem:
            raise FetchError(f"{model.name}: {problem}")
        os.replace(part, target)
    finally:
        part.unlink(missing_ok=True)
    return "replaced" if existed else "downloaded"


def models_command() -> int:
    try:
        folder = models_dir()
    except (ReframeError, paths.RepoRootError, paths.DataRootError) as exc:
        print(f"  ✗ settings: {exc}")
        return 1
    print(f"  models folder: {folder}")
    failed = 0
    for model in MODELS:
        try:
            status = fetch(model, destination(model, folder))
        except FetchError as exc:
            print(f"  ✗ {exc}")
            failed += 1
            continue
        except Exception as exc:  # still one line per file, then the next
            print(f"  ✗ {model.name}: {type(exc).__name__}: {exc}")
            failed += 1
            continue
        print(f"  ✓ {model.name}: {STATUS_TEXT[status]}, SHA-256 verified"
              f" ({model.size:,} bytes)")
    return 1 if failed else 0


def read_pins(path: Path) -> Dict[str, str]:
    """name -> version for every `name==version` line of a requirements
    file. Options and comments are skipped."""
    pins: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and not line.startswith("-") and "==" in line:
            name, _, version = line.partition("==")
            pins[name.strip().lower()] = version.strip()
    return pins


def torch_build_problems(
    torch, torchvision, pins: Dict[str, str], expected_cuda: str,
) -> List[str]:
    """Why the installed torch stack is not the pinned CUDA build."""
    problems: List[str] = []
    if torch.version.cuda is None:
        problems.append(f"torch is a CPU-only build ({torch.__version__})")
    elif torch.version.cuda != expected_cuda:
        problems.append(f"torch is built for CUDA {torch.version.cuda}, "
                        f"not {expected_cuda}")
    for module, name in ((torch, "torch"), (torchvision, "torchvision")):
        wanted = pins.get(name)
        if wanted is None:
            problems.append(f"{TORCH_REQUIREMENTS.name} has no == pin for "
                            f"{name}, so its build cannot be checked")
        elif module.__version__ != wanted:
            problems.append(f"{name} is {module.__version__}, but "
                            f"{TORCH_REQUIREMENTS.name} pins {wanted}")
    return problems


def _import_stack():
    """Import torch and torchvision (the tests replace this)."""
    import torch
    import torchvision

    return torch, torchvision


REFRAME_VENV_NAME = ".venv-reframe"


def delete_and_rerun() -> str:
    """The fix for a broken reframe environment.

    install.sh runs torch-build with the reframe environment's own
    interpreter, so sys.prefix is that folder and the advice names it.
    Run any other way, sys.prefix can be a Python installation (or
    another environment) that must never be deleted, so the advice
    names no folder.
    """
    folder = Path(sys.prefix)
    in_virtualenv = sys.prefix != sys.base_prefix
    if in_virtualenv and folder.name == REFRAME_VENV_NAME:
        return (f"Delete {REFRAME_VENV_NAME} and run install.sh again: "
                f'rm -rf "{folder.as_posix()}"')
    return (f"Delete the {REFRAME_VENV_NAME} folder in the ftea checkout "
            "and run install.sh again.")


def import_failure_hint(
    exc: BaseException, platform: Optional[str] = None,
) -> str:
    """A remedy for a torch that will not import, fitted to the cause."""
    platform = sys.platform if platform is None else platform
    if isinstance(exc, OSError) and platform == "win32":
        return ("A DLL would not load. On Windows the usual cause is a "
                "missing Microsoft Visual C++ Redistributable: install "
                "the x64 one from "
                "https://aka.ms/vs/17/release/vc_redist.x64.exe, then "
                "run install.sh again.")
    return "The installation is incomplete. " + delete_and_rerun()


def torch_build_command() -> int:
    from .verify_env import EXPECTED_CUDA

    try:
        torch, torchvision = _import_stack()
    except Exception as exc:
        print(f"  ✗ torch import: {type(exc).__name__}: {exc}")
        print(f"    {import_failure_hint(exc)}")
        return 1
    problems = torch_build_problems(
        torch, torchvision, read_pins(TORCH_REQUIREMENTS), EXPECTED_CUDA)
    if not problems:
        print(f"  ✓ torch {torch.__version__} and torchvision "
              f"{torchvision.__version__} (CUDA {torch.version.cuda})")
        return 0
    for problem in problems:
        print(f"  ✗ {problem}")
    print("    A later pip install most likely replaced the CUDA build. "
          + delete_and_rerun())
    return 1


def person_detector() -> ModelFile:
    return next(model for model in MODELS if model.in_torch_hub)


def torchvision_url_command() -> int:
    model = person_detector()
    try:
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_FPN_Weights as weights,
        )
    except Exception as exc:
        print(f"  ✗ torchvision import: {type(exc).__name__}: {exc}")
        return 1
    expected = weights.COCO_V1.url
    if expected == model.url:
        print("  ✓ torchvision will load the verified person-detector "
              "weights")
        return 0
    print(f"  ✗ torchvision expects the person-detector weights at "
          f"{expected}, but install.sh verified {model.url}. The pinned "
          "table in ftea/reframe/provision.py must be updated to match.")
    return 1


COMMANDS: Dict[str, Callable[[], int]] = {
    "models": models_command,
    "torch-build": torch_build_command,
    "torchvision-url": torchvision_url_command,
}
USAGE = ("usage: python -m ftea.reframe.provision "
         "{models|torch-build|torchvision-url}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1 or args[0] not in COMMANDS:
        print(USAGE, file=sys.stderr)
        return 2
    return COMMANDS[args[0]]()


if __name__ == "__main__":
    raise SystemExit(main())
