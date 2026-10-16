"""A stand-in for the reframe stage, for the tests of `ftea reframe`.

Run as `python -m ftea.tests.fake_stage` with the stage's arguments, or
call main() directly. It follows the stage's contract: progress on
stderr, one JSON object on stdout, exit 0 (reframed), 2 (skipped) or 1
(failed). FTEA_FAKE_STAGE picks the outcome: reframed (default),
no-crops (reframed without a crop track), skipped, failed or
debug-error. Its report records where and how it was started, so a test
can check the launch rule.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

EXIT = {"reframed": 0, "skipped": 2, "failed": 1}


def main(argv: list[str]) -> int:
    args = dict(zip(argv[::2], argv[1::2]))
    mode = os.environ.get("FTEA_FAKE_STAGE", "reframed")
    print("  reframe: fake stage running", file=sys.stderr)
    payload = {
        "schema": 1, "status": "reframed", "output": None, "crops": None,
        "report": {
            "cwd": os.getcwd(),
            "pythonpath": os.environ.get("PYTHONPATH"),
            "argv": argv,
        },
        "error": None, "debug": None, "debug_error": None,
    }
    if mode == "skipped":
        payload["status"] = "skipped"
        payload["report"]["reason"] = "already_vertical"
    elif mode == "failed":
        payload["status"] = "failed"
        payload["error"] = "no frames could be read"
    else:
        work = Path(args["--work-dir"])
        vertical = work / "vertical.mp4"
        vertical.write_bytes(b"vertical")
        payload["output"] = str(vertical)
        if mode != "no-crops":
            crops = work / "crops.json"
            crops.write_text(
                json.dumps({"source": {"name": Path(args["--source"]).name}}),
                encoding="utf-8",
            )
            payload["crops"] = str(crops)
        if mode == "debug-error":
            payload["debug_error"] = "RuntimeError: no NVENC session"
        elif "--debug-video" in args:
            debug = Path(args["--debug-video"])
            debug.write_bytes(b"debug")
            payload["debug"] = str(debug)
    print(json.dumps(payload))
    return EXIT[payload["status"]]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
