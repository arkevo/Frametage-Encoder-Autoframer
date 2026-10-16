"""Environment check for the reframe stage.

Run it with the reframe virtualenv's interpreter from the checkout, the
way install.sh does as its last step:

    cd <checkout>
    PYTHONPATH=<checkout> \\
        .venv-reframe/Scripts/python.exe -m ftea.reframe.verify_env

(.venv-reframe/bin/python on Linux). It reads the same config.env as the
stage itself, so REFRAME_MODELS_DIR, the model-file keys, REFRAME_DEVICE,
REFRAME_CODEC and REFRAME_FFMPEG are honoured, and it prints "  ✓ label"
or "  ✗ label" for each check.

Checks:
  1. torch imports, is the CUDA build requirements-torch.txt pins, and
     sees the GPU REFRAME_DEVICE names (cuda is the first, cuda:N the
     one numbered N)
  2. torch carries native kernels for that GPU (no run-time PTX compile)
  3. a real CUDA kernel runs on it and agrees with a CPU reference
     (1 to 3 check only that torch imports when REFRAME_DEVICE=cpu)
  4. torchvision and cv2 import, and the person detector (the weights in
     REFRAME_PERSON_MODEL when it is set), the face detector and the text
     detector load from the models directory
  5. ffprobe and ffmpeg are on PATH (shot detection and probing use them),
     and every ffmpeg the stage runs has the REFRAME_CODEC encoder and
     h264_nvenc (the debug video's first choice), a crop filter whose
     position can change at run time, and sendcmd
  6. every such ffmpeg really encodes a frame with REFRAME_CODEC, because
     a listed encoder is not a working encoder

Exit code 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .. import paths
from .detect import COCO_NUM_CLASSES, face_model_path, torch_home
from .errors import ReframeError
from .settings import ReframeSettings, load_settings
from .text import text_model_path

# Windows gives Python a cp1252 stdout, which cannot encode the ✓/✗ marks;
# printing one would raise UnicodeEncodeError in the middle of the check.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

# The CUDA build requirements-torch.txt installs (its cuNNN index URL). A
# test keeps the two in step.
EXPECTED_CUDA = "12.8"

MISSING = 0
_NATIVE_ARCH = re.compile(r"^sm_(\d+)$")
# The render's encoder when config.env names none (settings.py's default).
DEFAULT_CODEC = "hevc_nvenc"
# The debug video tries this first and falls back to libx264.
DEBUG_CODEC = "h264_nvenc"


def ok(label: str, detail: str = "") -> None:
    print(f"  ✓ {label}" + (f": {detail}" if detail else ""))


def fail(label: str, detail: str = "") -> None:
    global MISSING
    MISSING += 1
    print(f"  ✗ {label}" + (f": {detail}" if detail else ""))


def has_native_kernels(
    arch_list: Sequence[str], capability: Tuple[int, int],
) -> bool:
    """True when torch ships machine code this GPU runs directly.

    CUDA machine code built for sm_XY runs on any GPU with the same major
    version X and a minor version of at least Y. So an RTX 40-series card
    (8.9) runs the sm_86 kernels in a cu128 wheel natively, while an RTX
    50-series card (12.0) needs sm_120. compute_XY entries are PTX, which
    the driver compiles on first use, so they do not count.
    """
    major, minor = capability
    for entry in arch_list:
        match = _NATIVE_ARCH.match(entry)
        if match:
            built_major, built_minor = divmod(int(match.group(1)), 10)
            if built_major == major and built_minor <= minor:
                return True
    return False


def check_cuda_build(torch) -> None:
    ok("torch", f"{torch.__version__} (CUDA {torch.version.cuda})")
    if torch.version.cuda != EXPECTED_CUDA:
        fail(
            f"CUDA {EXPECTED_CUDA} build",
            f"got {torch.version.cuda}; a later install probably replaced "
            "the CUDA wheel with a CPU-only build from PyPI. Run install.sh "
            "again.",
        )


def check_arch(torch, index: int = 0) -> None:
    major, minor = torch.cuda.get_device_capability(index)
    arch = list(torch.cuda.get_arch_list())
    label = f"native kernels for compute capability {major}.{minor}"
    if has_native_kernels(arch, (major, minor)):
        ok(label, ", ".join(arch))
    else:
        fail(label, f"torch was built for {arch}; kernels would be "
                    "compiled at run time or fail")


def check_kernel(torch, device: str = "cuda") -> None:
    """A real kernel must run on `device` AND agree with a CPU reference.

    This is the check that catches "no kernel image is available for
    execution on the device".
    """
    try:
        a, b = torch.randn(512, 512), torch.randn(512, 512)
        on_gpu = a.to(device) @ b.to(device)
        err = (on_gpu.cpu() - (a @ b)).abs().max().item()
    except Exception as exc:
        fail("CUDA kernel execution", f"{type(exc).__name__}: {exc}")
        return
    if err < 1e-3:
        ok("CUDA kernel vs CPU reference", f"max abs err {err:.2e}")
    else:
        fail("CUDA kernel vs CPU reference", f"max abs err {err:.2e}")


def check_torch(device: str = "cuda") -> None:
    print("=== torch / CUDA ===")
    try:
        import torch
    except Exception as exc:
        fail("torch import", f"{type(exc).__name__}: {exc}")
        return
    check_torch_module(torch, device)


def check_torch_module(torch, device: str) -> None:
    """The torch checks for the device the stage will detect on.

    REFRAME_DEVICE=cpu needs neither the CUDA build nor a visible GPU,
    so only the import is checked then. The render's NVENC encoder is
    checked separately, under ffmpeg.
    """
    if not device.startswith("cuda"):
        ok("torch", f"{torch.__version__}; GPU checks skipped "
                    f"(REFRAME_DEVICE={device})")
        return
    check_cuda_build(torch)
    if not torch.cuda.is_available():
        fail("torch.cuda.is_available()", "no CUDA device visible")
        return
    index = gpu_index(device)
    count = torch.cuda.device_count()
    if index >= count:
        fail(f"REFRAME_DEVICE={device}",
             f"this machine shows {count} GPU(s), numbered from 0")
        return
    ok("CUDA device", torch.cuda.get_device_name(index))
    check_arch(torch, index)
    check_kernel(torch, device)


def gpu_index(device: str) -> int:
    """The GPU number in cuda:N; plain cuda is the first GPU, 0."""
    _, _, number = device.partition(":")
    return int(number) if number else 0


def model_paths(settings: ReframeSettings) -> Tuple[Path, Path]:
    """(face weights, text weights), resolved exactly as the stage does."""
    return (
        face_model_path(settings.models_dir, settings.face_model),
        text_model_path(settings.models_dir, settings.title_model),
    )


def check_model_files(face: Path, text_model: Path) -> Tuple[bool, bool]:
    """Whether each weights file exists. One line per file."""
    found = []
    for label, path in (("YuNet face weights", face),
                        ("PP-OCRv3 text weights", text_model)):
        if path.is_file():
            ok(label, f"{path} ({path.stat().st_size} bytes)")
            found.append(True)
        else:
            fail(label, f"missing at {path}; run install.sh")
            found.append(False)
    return found[0], found[1]


def check_person_detector(person_model: str) -> None:
    """Load the person detector the way the stage will.

    With REFRAME_PERSON_MODEL set, the stage loads that file and refuses
    a missing one, so the check does the same rather than testing the
    default COCO weights the stage will never use.
    """
    label = "person detector"
    path = Path(person_model) if person_model else None
    if path is not None and not path.is_file():
        fail(label, f"REFRAME_PERSON_MODEL not found: {path}")
        return
    try:
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_fpn,
        )

        if path is None:
            fasterrcnn_mobilenet_v3_large_fpn(
                weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
            )
            ok(label, "torchvision faster-rcnn mobilenet_v3 (BSD-3)")
            return
        import torch

        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=None, weights_backbone=None,
            num_classes=COCO_NUM_CLASSES,
        )
        model.load_state_dict(torch.load(str(path), map_location="cpu"))
        ok(label, f"{path} (REFRAME_PERSON_MODEL)")
    except Exception as exc:
        fail(label, f"{type(exc).__name__}: {exc}")


def check_vision(
    face: Path, text_model: Path, person_model: str = "",
) -> None:
    print("")
    print("=== vision stack ===")
    try:
        import torchvision

        ok("torchvision", torchvision.__version__)
    except Exception as exc:
        fail("torchvision import", f"{type(exc).__name__}: {exc}")

    try:
        import cv2

        ok("cv2", cv2.__version__)
    except Exception as exc:
        fail("cv2 import", f"{type(exc).__name__}: {exc}")
        return

    # Person detector: torchvision COCO Faster R-CNN (BSD-3-Clause). The
    # default weights load from TORCH_HOME, which main() points at the
    # models directory.
    check_person_detector(person_model)

    has_face, has_text = check_model_files(face, text_model)
    if has_face:
        try:
            cv2.FaceDetectorYN_create(str(face), "", (320, 320))
            ok("face detector", "YuNet (MIT)")
        except Exception as exc:
            fail("face detector", f"{type(exc).__name__}: {exc}")
    if has_text:
        try:
            cv2.dnn.TextDetectionModel_DB(cv2.dnn.readNet(str(text_model)))
            ok("text detector", "PP-OCRv3 DB (Apache-2.0)")
        except Exception as exc:
            fail("text detector", f"{type(exc).__name__}: {exc}")


def ffmpeg_binaries(configured: str) -> List[Tuple[str, Optional[str]]]:
    """(label, resolved path or None) for every ffmpeg the stage runs.

    Shot detection and probing always run `ffmpeg` and `ffprobe` from
    PATH. Rendering runs REFRAME_FFMPEG when it is set.
    """
    found = [("ffmpeg on PATH", shutil.which("ffmpeg"))]
    if configured:
        found.append((f"REFRAME_FFMPEG ({configured})",
                      shutil.which(configured)))
    return found


def _check_session(exe: str, codec: str) -> None:
    """Encode one frame with the render's codec to prove it really works.

    A static ffmpeg build newer than the installed driver still LISTS
    an NVENC encoder and then fails the first encode, so the listing
    alone would pass an environment that cannot reframe a single clip.
    256x256 is above every NVENC encoder's smallest frame.
    """
    label = (
        f"{codec} session opens" if "nvenc" in codec
        else f"{codec} encodes a frame"
    )
    try:
        done = subprocess.run(
            [exe, "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=s=256x256:d=0.04",
             "-frames:v", "1", "-c:v", codec, "-f", "null", "-"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as exc:
        fail(label, f"{type(exc).__name__}: {exc}")
        return
    if done.returncode == 0:
        ok(label)
        return
    lines = [line for line in done.stderr.splitlines() if line.strip()]
    fail(label, lines[-1] if lines else f"exit code {done.returncode}")


def _listed(name: str, listing: str) -> bool:
    """Whether `ffmpeg -encoders` lists exactly this encoder."""
    pattern = rf"(?m)^\s*\S+\s+{re.escape(name)}\s"
    return re.search(pattern, listing) is not None


def _check_ffmpeg_features(exe: str, codec: str = DEFAULT_CODEC) -> None:
    try:
        enc = subprocess.run(
            [exe, "-hide_banner", "-encoders"], capture_output=True,
            text=True,
        ).stdout
    except Exception as exc:
        fail("ffmpeg -encoders", f"{type(exc).__name__}: {exc}")
        return

    for name in dict.fromkeys((codec, DEBUG_CODEC)):
        if _listed(name, enc):
            ok(f"encoder {name}")
        else:
            fail(f"encoder {name}", "not built into this ffmpeg")
    if _listed(codec, enc):
        _check_session(exe, codec)

    # crop's x/y must be settable at run time for the sendcmd-driven crop.
    helptext = subprocess.run(
        [exe, "-hide_banner", "-h", "filter=crop"], capture_output=True,
        text=True,
    ).stdout
    runtime_x = any(
        line.strip().startswith("x ") and "T." in line
        for line in helptext.splitlines()
    )
    if runtime_x:
        ok("crop x/y runtime-commandable", "dynamic crop supported")
    else:
        fail("crop x/y runtime-commandable", "dynamic crop would not work")

    filters = subprocess.run(
        [exe, "-hide_banner", "-filters"], capture_output=True, text=True,
    ).stdout
    if " sendcmd " in filters:
        ok("filter sendcmd")
    else:
        fail("filter sendcmd", "not available")


def check_ffmpeg(configured: str = "", codec: str = DEFAULT_CODEC) -> None:
    print("")
    print("=== ffmpeg / NVENC / filters ===")
    probe = shutil.which("ffprobe")
    if probe:
        ok("ffprobe on PATH", probe)
    else:
        fail("ffprobe on PATH", "NOT FOUND")
    for label, exe in ffmpeg_binaries(configured):
        if not exe:
            fail(label, "NOT FOUND")
            continue
        ok(label, exe)
        _check_ffmpeg_features(exe, codec)


def main(argv: Optional[List[str]] = None) -> int:
    """Run every check. 0 when all pass, 1 otherwise."""
    global MISSING
    MISSING = 0
    print("=== ftea reframe stage: environment check ===")
    print(f"    interpreter: {sys.executable}")
    try:
        settings = load_settings(paths.config_path(), env=os.environ)
    except (ReframeError, paths.RepoRootError, paths.DataRootError) as exc:
        fail("settings", str(exc))
        return 1
    face, text_model = model_paths(settings)
    os.environ["TORCH_HOME"] = str(torch_home(settings.models_dir))
    print(f"    models: {settings.models_dir}")
    print("")
    check_torch(settings.device)
    check_vision(face, text_model, settings.person_model)
    check_ffmpeg(settings.ffmpeg, codec=settings.codec.strip().lower())
    print("")
    if MISSING:
        print(f"=== {MISSING} check(s) FAILED ===")
        return 1
    print("=== all checks passed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
