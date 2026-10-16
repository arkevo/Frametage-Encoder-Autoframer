"""The reframe stage's two requirement files and the order they encode.

install.sh installs requirements-torch.txt (torch and torchvision from
the CUDA index) first and requirements.txt (everything else) after,
checking the torch build after each step. These tests keep that split
honest and the pinned versions unchanged.
"""

from __future__ import annotations

from pathlib import Path

from ftea.reframe import verify_env

HERE = Path(verify_env.__file__).parent
REPO = HERE.parents[1]
TORCH_FILE = HERE / "requirements-torch.txt"
REST_FILE = HERE / "requirements.txt"
PINNED = {
    "torch": "2.11.0+cu128",
    "torchvision": "0.26.0+cu128",
    "opencv-python": "5.0.0.93",
}


def _options_and_pins(path: Path) -> tuple[list[str], dict[str, str]]:
    options: list[str] = []
    pins: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-"):
            options.append(line)
            continue
        name, _, version = line.partition("==")
        pins[name.strip().lower()] = version.strip()
    return options, pins


def _build() -> str:
    return "cu" + verify_env.EXPECTED_CUDA.replace(".", "")


def test_should_install_torch_from_the_cuda_index_first():
    options, pins = _options_and_pins(TORCH_FILE)

    assert options == [
        f"--index-url https://download.pytorch.org/whl/{_build()}",
        "--extra-index-url https://pypi.org/simple",
    ]
    assert set(pins) == {"torch", "torchvision"}
    assert all(v.endswith("+" + _build()) for v in pins.values())


def test_should_install_everything_else_from_the_default_index_after():
    options, pins = _options_and_pins(REST_FILE)

    assert options == []
    assert pins
    assert not {"torch", "torchvision"} & set(pins)


def test_should_keep_every_pin_unchanged():
    _, first = _options_and_pins(TORCH_FILE)
    _, second = _options_and_pins(REST_FILE)

    assert {**first, **second} == PINNED


def test_should_never_list_an_agpl_detector_package():
    for path in sorted(HERE.glob("requirements*.txt")):
        _, pins = _options_and_pins(path)
        assert "ultralytics" not in pins, path.name


def test_should_ship_both_files_with_the_package():
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")

    assert '"ftea.reframe" = ["requirements*.txt"]' in text
