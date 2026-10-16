"""Entry point for the auto-reframe stage.

The encoder (ftea/encode/main.py, _run_reframe) and `ftea reframe` run it
as a SUBPROCESS with the reframe virtualenv's interpreter:

    <REFRAME_PYTHON> -m ftea.reframe.main \\
        --source <clip> --work-dir <dir> [--config <config.env>] \\
        [--debug-video <out.mp4>]

with cwd set to the checkout, PYTHONPATH set to the checkout
(ftea.paths.reframe_subprocess_env) and every path absolute. A
subprocess rather than an import because the encoder must not inherit
this stage's multi-gigabyte torch environment. Without --config the
stage reads <data root>/config.env (ftea.paths.config_path()).

With --debug-video <absolute .mp4>, a successful reframe is followed by
the side-by-side debug video; its path or its error is reported in the
payload's `debug` / `debug_error`, and a debug failure never fails the
clip. The debug.sendcmd file written beside it is removed after a
successful render.

Prints exactly one JSON object on stdout (the caller parses it) and
human-readable progress on stderr. Exit codes:

    0  reframed  -- intermediate and crops.json written
    2  skipped   -- input already vertical or square, pass through untouched
    1  failed    -- no output or crop track reported; a failed render
                    may leave crops.failed.json and crop.sendcmd in the
                    work directory for diagnosis

Stdout is reserved for that one JSON object. `run_stages` -- and
therefore every stage module it calls (shots/detect/subject/camera/
render) -- runs with stdout redirected to stderr for the duration of the
call, so those modules may print() progress freely without risking the
payload.

NOTE: ftea/reframe/probe.py's DIAGNOSTIC cli is a different program
with its own argument surface. Never wire reframe.probe where
reframe.main is expected.
"""

import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from .. import paths
from .debug_video import render_debug_video, spec_from_crops
from .errors import ReframeError
from .pipeline import run_stages
from .probe import (
    OUTCOME_FAILED,
    ProbeError,
    decide,
    reframe_report_entry,
)
from .render import failed_crops_path
from .settings import load_settings

SCHEMA_VERSION = 1

EXIT_REFRAMED = 0
EXIT_FAILED = 1
EXIT_SKIPPED = 2

CROPS_NAME = "crops.json"
OUTPUT_STEM = "vertical"

_USAGE = (
    "Usage: python -m ftea.reframe.main "
    "--source <video> --work-dir <dir> [--config <config.env>] "
    "[--debug-video <out.mp4>]"
)


def _partial(path: Path) -> Path:
    """The sibling temp name for an artifact.

    `vertical.mp4` becomes `vertical.partial.mp4`. Keeps the real
    extension (rather than appending `.partial` after it) so ffmpeg can
    infer the muxer from the filename — `ffmpeg ... /tmp/x.partial` fails
    with "Unable to choose an output format".
    """
    return path.with_name(f"{path.stem}.partial{path.suffix}")


def _parse_args(argv: list[str]) -> dict[str, str]:
    """Parse the entry point's four long options.

    Hand-rolled rather than argparse so a bad argument produces the same
    one-line stderr message as every other failure here, with no
    SystemExit escaping into the caller's process.
    """
    parsed: dict[str, str] = {}
    known = {"--source", "--work-dir", "--config", "--debug-video"}
    index = 0
    while index < len(argv):
        token = argv[index]
        if token not in known:
            raise ReframeError(f"unknown argument '{token}'")
        if index + 1 >= len(argv):
            raise ReframeError(f"{token} requires a value")
        parsed[token] = argv[index + 1]
        index += 2
    for required in ("--source", "--work-dir"):
        if required not in parsed:
            raise ReframeError(f"{required} is required")
    debug = parsed.get("--debug-video")
    if debug is not None and not (
        Path(debug).is_absolute() and Path(debug).suffix.lower() == ".mp4"
    ):
        raise ReframeError(
            "--debug-video must be an absolute path to an .mp4 file")
    return parsed


def _result(
    status: str,
    *,
    output: Optional[Path] = None,
    crops: Optional[Path] = None,
    report: Optional[dict] = None,
    error: Optional[str] = None,
    debug: Optional[Path] = None,
    debug_error: Optional[str] = None,
) -> dict:
    """Build the stdout payload. Shape is a published contract."""
    return {
        "schema": SCHEMA_VERSION,
        "status": status,
        "output": str(output) if output is not None else None,
        "crops": str(crops) if crops is not None else None,
        "report": report,
        "error": error,
        "debug": str(debug) if debug is not None else None,
        "debug_error": debug_error,
    }


def _emit(result: dict, stream) -> None:
    """Print the payload as the process's only stdout output.

    `stream` is the real stdout, captured by main() before run_stages
    redirects sys.stdout to stderr — so a stage module's print() calls can
    never land in the payload.
    """
    json.dump(result, stream)
    stream.write("\n")
    stream.flush()


def main(argv: Optional[list[str]] = None) -> int:
    """Reframe one clip. Returns the process exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    # Captured before run_stages redirects sys.stdout to stderr.
    out_stream = sys.stdout

    try:
        parsed = _parse_args(args)
    except ReframeError as exc:
        print(f"  reframe: {exc}", file=sys.stderr)
        print(_USAGE, file=sys.stderr)
        _emit(_result("failed", error=str(exc)), out_stream)
        return EXIT_FAILED

    source = Path(parsed["--source"])
    work_dir = Path(parsed["--work-dir"])

    decision = None
    report: Optional[dict] = None
    started = time.time()

    try:
        # Resolved here, inside the try, so even a missing checkout
        # still produces the one JSON payload the caller parses.
        config_path = (Path(parsed["--config"]) if "--config" in parsed
                       else paths.config_path())
        if not config_path.is_file():
            # Not an error: the encoder passes --config whether or not
            # config.env exists, and every key has a default. Said out
            # loud so a mistyped path cannot pass unnoticed.
            print(f"  reframe: no config file at {config_path}; using "
                  "the built-in defaults", file=sys.stderr)
        settings = load_settings(config_path, env=os.environ)
        decision = decide(source)
        print(decision.log_line, file=sys.stderr)

        if not decision.should_reframe:
            report = reframe_report_entry(decision, applied=False)
            _emit(_result("skipped", report=report), out_stream)
            return EXIT_SKIPPED

        report = reframe_report_entry(decision, applied=False)

        work_dir.mkdir(parents=True, exist_ok=True)
        out_path = work_dir / f"{OUTPUT_STEM}.{settings.container}"
        crops_path = work_dir / CROPS_NAME
        # Everything an earlier run may have left. A stale crops.json
        # would satisfy the check below; a stale crops.failed.json
        # would sit beside a good result as if this run had failed.
        for stale in (
            out_path, crops_path,
            _partial(out_path), _partial(crops_path),
            failed_crops_path(crops_path),
        ):
            if stale.exists():
                stale.unlink()

        # run_stages (and every stage module it calls: shots, detect,
        # subject, camera, render) may print() progress freely; stdout is
        # redirected to stderr for the duration of the call so those
        # prints can never pollute the JSON payload.
        with contextlib.redirect_stdout(sys.stderr):
            run_stages(
                source=source,
                meta=decision.meta,
                out_path=out_path,
                crops_path=crops_path,
                settings=settings,
            )

        if not out_path.exists() or out_path.stat().st_size == 0:
            raise ReframeError(
                f"reframe produced no output at {out_path}"
            )
        if not crops_path.exists():
            raise ReframeError(
                f"reframe produced no crop track at {crops_path}"
            )

        elapsed = time.time() - started
        report = reframe_report_entry(
            decision, applied=True, elapsed_sec=elapsed,
        )
        print(f"  reframe: wrote {out_path.name} in {elapsed:.1f}s",
              file=sys.stderr)
        debug, debug_error = None, None
        if "--debug-video" in parsed:
            debug, debug_error = _render_debug(
                source, out_path, crops_path,
                Path(parsed["--debug-video"]), settings,
            )
        _emit(_result(
            "reframed", output=out_path, crops=crops_path, report=report,
            debug=debug, debug_error=debug_error,
        ), out_stream)
        return EXIT_REFRAMED

    except (ReframeError, ProbeError, NotImplementedError) as exc:
        message = f"{type(exc).__name__}: {exc}"
        print(f"  reframe: FAILED {message}", file=sys.stderr)
        _emit(
            _result("failed", report=_failed_entry(decision), error=message),
            out_stream,
        )
        return EXIT_FAILED
    except Exception as exc:  # unexpected — keep the traceback for triage
        message = f"{type(exc).__name__}: {exc}"
        traceback.print_exc(file=sys.stderr)
        _emit(
            _result("failed", report=_failed_entry(decision), error=message),
            out_stream,
        )
        return EXIT_FAILED


def _render_debug(
    source: Path, vertical: Path, crops_path: Path, target: Path, settings,
) -> tuple[Optional[Path], Optional[str]]:
    """Render the side-by-side debug video. Never fails the reframe.

    The vertical intermediate is already written and checked when this
    runs, so a debug-render problem of any kind, a bug included, is
    reported in the payload's `debug_error` and on stderr (with its
    traceback) instead of discarding a good result. The sendcmd sidecar
    is removed after a good render and kept after a failed one.
    Returns (debug video path, None) or (None, error message).
    """
    try:
        crops = json.loads(crops_path.read_text(encoding="ascii"))
        spec = spec_from_crops(crops, source, vertical, target)
        with contextlib.redirect_stdout(sys.stderr):
            result = render_debug_video(
                spec, ffmpeg=settings.ffmpeg or None, require_inputs=True,
            )
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        message = f"{type(exc).__name__}: {exc}"
        print(f"  reframe: debug video FAILED {message}", file=sys.stderr)
        return None, message
    try:
        result.script.unlink(missing_ok=True)
    except OSError as exc:
        print(f"  reframe: could not remove {result.script.name}: {exc}",
              file=sys.stderr)
    print(f"  reframe: wrote {result.out_path.name} "
          f"({result.width}x{result.height}, {result.codec})",
          file=sys.stderr)
    return result.out_path, None


def _failed_entry(decision) -> Optional[dict]:
    """The diagnostic entry for a clip whose stage failed.

    None when the gate never ran (a probe failure has no geometry to
    report). Otherwise the gate's own entry, explicitly marked
    outcome="failed" — probe.reframe_report_entry requires the caller to
    distinguish a stage failure from a purely diagnostic gate run.
    """
    if decision is None:
        return None
    return reframe_report_entry(
        decision, applied=False, outcome=OUTCOME_FAILED,
    )


if __name__ == "__main__":
    sys.exit(main())
