"""Guard the CI contract: every push is tested on Windows and Linux."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "tests.yml"


def test_should_test_every_push_on_windows_and_linux():
    text = WORKFLOW.read_text(encoding="utf-8")
    for needle in (
        "push:",
        "pull_request:",
        "windows-latest",
        "ubuntu-latest",
        '"3.10"',
        'python -m pip install -e ".[test]"',
        "python -m pytest",
        "contents: read",
        "timeout-minutes",
    ):
        assert needle in text, f"tests.yml lost {needle!r}"
