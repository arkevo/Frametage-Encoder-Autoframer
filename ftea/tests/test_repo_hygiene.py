"""Guards for repository hygiene.

What git ignores, line endings, the license text, the encoder's empty
dependency list, and the Python version floor. These protect the
publishing boundary: a user's config.env, source videos and encoded
output must never be staged, and shell scripts must keep LF endings so
they run on Linux.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
THIS_FILE = Path(__file__).resolve()
PYTHON_FLOOR = (3, 10)

# Checked against the ignore rules only, so none of these need exist.
MUST_IGNORE = [
    "config.env",
    "ftea/reframe/config.env",
    ".env",
    "input/clip.mp4",
    "input/nested/clip.mov",
    "output/clip/master.m3u8",
    "output/batch_report.json",
    "work/clip/source.mkv",
    "work/.pipeline.lock",
    "models/face_detection_yunet_2023mar.onnx",
    "models/torch/hub/checkpoints/weights.pth",
    ".venv/Scripts/python.exe",
    ".venv-reframe/bin/python",
    "ftea/__pycache__/paths.cpython-312.pyc",
    "ftea.egg-info/PKG-INFO",
    ".pytest_cache/v/cache/lastfailed",
    "docs/images/clip.mp4",
    "ftea/reframe/tests/data/clip.webm",
    "stills/clip_f00012.webp",
    "ftea/reframe/tests/frame.png",
    "output/clip/audio_extracted.m4a",
    "weights/yunet.onnx",
    "config.env.bak",
    "local.env",
    # Upper and mixed case: phones and cameras write CLIP.MOV, IMG.PNG.
    "CLIP.MOV",
    "docs/IMG_0001.MOV",
    "clip.Mp4",
    "stills/frame.PNG",
    "stills/frame.JPG",
    "stills/audio.WAV",
    "weights/yunet.ONNX",
    "docs/images/clip.MP4",
]

MUST_TRACK = [
    "config.env.example",
    "input/.gitkeep",
    "output/.gitkeep",
    "install.sh",
    "pyproject.toml",
    "ftea/paths.py",
    "ftea/player/player.html",
    "ftea/reframe/requirements.txt",
    "ftea/reframe/models/__init__.py",  # proves /models/ is root-anchored
    "ftea/encode/work/__init__.py",  # proves /work/ is root-anchored
    "docs/images/before-after.webp",
    "docs/images/Screenshot.PNG",
    ".github/workflows/tests.yml",
]

LF_PATHS = [
    "install.sh",
    "ftea/paths.py",
    "README.md",
    "config.env.example",
    ".github/workflows/tests.yml",
]

BANNED_NEWER_APIS = {
    r"^\s*(?:import|from)\s+tomllib\b": "tomllib needs Python 3.11",
    r"\bExceptionGroup\b": "exception groups need Python 3.11",
    r"from\s+typing\s+import\s+[^\n]*\b(?:Self|override)\b":
        "typing.Self / typing.override need Python 3.11 / 3.12",
    r"\bStrEnum\b": "enum.StrEnum needs Python 3.11",
    r"\bdatetime\.UTC\b|from\s+datetime\s+import\s+[^\n]*\bUTC\b":
        "datetime.UTC needs Python 3.11",
    r"\bTaskGroup\b": "asyncio.TaskGroup needs Python 3.11",
    r"\bitertools\.batched\b|from\s+itertools\s+import\s+[^\n]*\bbatched\b":
        "itertools.batched needs Python 3.12",
}


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True,
    )


def _check_ignore(path: str) -> subprocess.CompletedProcess[str]:
    # Linux clones match case-sensitively; force that here so a Windows
    # or macOS run (core.ignorecase=true) also catches CLIP.MOV leaking.
    return _git(
        "-c", "core.ignorecase=false",
        "check-ignore", "-q", "--no-index", "--", path,
    )


def _rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def _python_sources() -> list[Path]:
    return sorted(
        p for p in (REPO / "ftea").rglob("*.py")
        if "__pycache__" not in p.parts
    )


@pytest.fixture(scope="module")
def git_checkout() -> None:
    try:
        probe = _git("rev-parse", "--is-inside-work-tree")
    except FileNotFoundError:
        pytest.skip("git is not installed")
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        pytest.skip(f"not a git checkout: {probe.stderr.strip()}")


@pytest.mark.parametrize("path", MUST_IGNORE)
def test_should_ignore_private_and_generated_paths(git_checkout, path):
    result = _check_ignore(path)
    assert result.returncode == 0, f"{path} is not ignored"


@pytest.mark.parametrize("path", MUST_TRACK)
def test_should_keep_source_paths_committable(git_checkout, path):
    result = _check_ignore(path)
    assert result.returncode == 1, (
        f"{path} is ignored (exit {result.returncode}) {result.stderr}"
    )


@pytest.mark.parametrize("path", LF_PATHS)
def test_should_check_out_text_files_with_lf(git_checkout, path):
    result = _git("check-attr", "eol", "--", path)
    assert result.stdout.strip() == f"{path}: eol: lf"


def test_should_treat_images_as_binary(git_checkout):
    path = "docs/images/before-after.webp"
    result = _git("check-attr", "text", "--", path)
    assert result.stdout.strip() == f"{path}: text: unset"


def test_should_hold_no_crlf_bytes_in_the_index_or_working_tree(
    git_checkout,
):
    # check-attr proves what a fresh clone gets; WSL reads this working
    # tree directly, so the bytes on disk must be LF too.
    result = _git("ls-files", "--eol")
    assert result.returncode == 0, result.stderr
    bad = [
        line for line in result.stdout.splitlines()
        if re.search(r"(?:^|\s)[iw]/(?:crlf|mixed)\b", line)
    ]
    assert not bad, "CRLF line endings on disk:\n" + "\n".join(bad)


def test_should_ship_the_full_apache_2_license_text():
    text = (REPO / "LICENSE").read_text(encoding="utf-8")
    for phrase in (
        "Apache License",
        "Version 2.0, January 2004",
        "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION",
        "END OF TERMS AND CONDITIONS",
    ):
        assert phrase in text, f"LICENSE lost {phrase!r}"


def test_should_keep_the_encoder_free_of_runtime_dependencies():
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r"^dependencies = \[\]$", text, re.MULTILINE), (
        "The encoder is standard-library only by design; the reframe "
        "stack belongs in ftea/reframe/requirements.txt"
    )
    assert re.search(r'^license = "Apache-2.0"$', text, re.MULTILINE)


@pytest.mark.parametrize("path", _python_sources(), ids=_rel)
def test_should_parse_with_the_python_floor_grammar(path):
    source = path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(path), feature_version=PYTHON_FLOOR)


@pytest.mark.parametrize(
    "path", [p for p in _python_sources() if p != THIS_FILE], ids=_rel,
)
def test_should_avoid_stdlib_apis_newer_than_the_floor(path):
    source = path.read_text(encoding="utf-8")
    hits = [
        why for pattern, why in BANNED_NEWER_APIS.items()
        if re.search(pattern, source, re.MULTILINE)
    ]
    assert not hits, f"{_rel(path)}: {hits}"


def test_should_describe_the_package_for_its_index_page():
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    for minor in range(10, 15):
        classifier = f"Programming Language :: Python :: 3.{minor}"
        assert f'"{classifier}"' in text, classifier
    # hls.min.js ships in the wheel, so its license must too.
    assert re.search(
        r'^license-files = \["LICENSE", '
        r'"ftea/player/hls\.js-LICENSE\.txt"\]$',
        text, re.MULTILINE,
    )
    # The test packages stay out of the wheel.
    assert re.search(
        r'^exclude = \["\*\.tests", "\*\.tests\.\*"\]$', text, re.MULTILINE,
    )
