"""`ftea reframe`: run only the reframe stage on one video.

The stage runs in its own virtualenv as a subprocess, launched the way
the encoder launches it: from the checkout, with the checkout on
PYTHONPATH and every path absolute. Its progress goes straight to the
terminal; its one JSON result comes back on stdout. The results go into
their own folder, output/<name>-vertical/ unless another is given, and
never into a clip's package folder, which `ftea run` replaces.

A results folder records the full path of the video it belongs to in
MARKER_NAME, written before the stage starts. A folder that holds
anything but this video's own earlier results is refused: files ftea
reframe did not write, results that name another video (two videos of
the same name in different folders share a default folder), and results
that name no video at all. A file another program holds open is named
too. All of that is checked before the stage starts.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, get_args

from . import paths
from .encode.tools import run_process_tree

OUTPUT_SUFFIX = "-vertical"
# The stage's scratch folder inside the results folder: emptied before
# each run, removed after a good one, kept after a failure to look at.
PARTIAL_DIR = ".partial"
VERTICAL_STEM = "vertical"
CROPS_NAME = "crops.json"
DEBUG_NAME = "debug.mp4"
# The stage writes this beside the debug video and removes it after a
# good render, so one found before a run belongs to an earlier run.
DEBUG_SIDECAR = "debug.sendcmd"
# Which video a results folder belongs to: {"schema": 1, "source": <the
# video's resolved path>}.
MARKER_NAME = "ftea-reframe.json"
MARKER_SCHEMA = 1
# The schema of the stage's JSON result (ftea.reframe.main.SCHEMA_VERSION,
# which a test keeps equal; the stage is not imported here because it
# runs in its own environment).
STAGE_SCHEMA = 1
Status = Literal["reframed", "failed", "skipped"]
STATUS_EXIT: dict[str, int] = {"reframed": 0, "failed": 1, "skipped": 2}
# Everything `ftea reframe` itself writes into a results folder, besides
# vertical.<ext>. Anything else there belongs to the user.
RESULT_NAMES = frozenset({
    CROPS_NAME, DEBUG_NAME, DEBUG_SIDECAR, PARTIAL_DIR, MARKER_NAME,
})
# Payload fields that are a string or null, as the stage writes them.
_TEXT_FIELDS = ("output", "crops", "debug", "error", "debug_error")

Runner = Callable[..., subprocess.CompletedProcess]


class VerticalError(RuntimeError):
    """The stage could not be run, or its result broke the contract."""


@dataclass(frozen=True)
class VerticalResult:
    """What one `ftea reframe` produced."""

    status: Status
    out_dir: Path
    vertical: Optional[Path] = None
    crops: Optional[Path] = None
    debug: Optional[Path] = None
    error: Optional[str] = None
    debug_error: Optional[str] = None
    reason: Optional[str] = None
    report: Optional[dict] = None

    def __post_init__(self) -> None:
        if self.status not in get_args(Status):
            raise ValueError(f"unknown reframe status {self.status!r}")
        if self.status == "reframed" and self.vertical is None:
            raise ValueError("a reframed result needs its vertical video")

    @property
    def exit_code(self) -> int:
        """1 when the reframe failed, or the debug video asked for did.

        A video that is not landscape is left alone (skipped) and exits
        0, like a reframed one; `ftea reframe --help` says so.
        """
        return 1 if self.status == "failed" or self.debug_error else 0


@dataclass(frozen=True)
class _Claim:
    """What this run did to the results folder before the stage ran, so
    a run that ends up writing nothing can undo exactly that."""

    target: Path
    work: Path
    created_folder: bool
    wrote_marker: bool


def default_out_dir(video: Path) -> Path:
    return paths.output_dir() / f"{video.stem}{OUTPUT_SUFFIX}"


def stage_command(
    python: Path, video: Path, work: Path, debug_target: Optional[Path],
) -> list[str]:
    """The stage's command line, every path absolute.

    There is no --config: without one the stage reads paths.config_path(),
    the same file this process reads.
    """
    cmd = [
        str(python), "-m", paths.REFRAME_MODULE,
        "--source", str(video.resolve()),
        "--work-dir", str(work.resolve()),
    ]
    if debug_target is not None:
        cmd += ["--debug-video", str(debug_target.resolve())]
    return cmd


def make_vertical(
    video: Path, *, out_dir: Optional[Path] = None, debug_video: bool = False,
    config=None, runner: Runner = run_process_tree,
) -> VerticalResult:
    """Reframe one video into its results folder.

    The stage runs ffmpeg itself, so the default runner stops those
    programs too when the time runs out (see run_process_tree).
    """
    from .encode.main import _reframe_env, _resolve_reframe_python
    from .encode.main import load_run_config

    video = Path(video)
    if not video.is_file():
        raise VerticalError(f"{video} is not a video file")
    config = load_run_config(reframe_only=True) if config is None else config
    try:
        python = _resolve_reframe_python(config)
    except RuntimeError as exc:
        raise VerticalError(str(exc)) from None
    target = Path(out_dir if out_dir is not None else default_out_dir(video))
    target = target.resolve()
    _check_results_folder(target, video)
    claim = _claim(target, video)
    work = claim.work
    debug_target = target / DEBUG_NAME if debug_video else None
    cmd = stage_command(python, video, work, debug_target)
    # The debug render is at most a second pass over the clip.
    limit = config.reframe_timeout * (2 if debug_video else 1)
    print(f"Reframing {video.name} (limit {limit}s)...")
    try:
        done = runner(
            cmd, cwd=str(paths.source_root()), env=_reframe_env(),
            stdout=subprocess.PIPE, encoding="utf-8", errors="replace",
            timeout=limit,
        )
    except subprocess.TimeoutExpired:
        doubled = ", doubled for --debug-video" if debug_video else ""
        raise VerticalError(
            f"the reframe stage did not finish within {limit}s "
            f"(REFRAME_TIMEOUT in config.env{doubled}) and was stopped; "
            f"its partial files are in {work}"
        ) from None
    except OSError as exc:
        raise VerticalError(
            f"the reframe interpreter {python} could not be started: {exc}"
        ) from None
    return _settle(_payload(done.stdout, done.returncode), claim)


def _unusable(target: Path, exc: OSError) -> str:
    return (
        f"{target} cannot be used as the results folder ({exc}). Choose "
        "another with --out <folder>."
    )


def _check_results_folder(target: Path, video: Path) -> None:
    """Refuse a folder that holds anything but this video's own earlier
    results, so a user's files and another video's results are never
    deleted or replaced."""
    if not target.exists():
        return
    if not target.is_dir():
        raise VerticalError(
            f"{target} is a file. Choose a folder with --out <new folder>."
        )
    try:
        names = sorted(path.name for path in target.iterdir())
    except OSError as exc:
        raise VerticalError(_unusable(target, exc)) from None
    foreign = [
        name for name in names
        if name not in RESULT_NAMES and Path(name).stem != VERTICAL_STEM
    ]
    if foreign:
        raise VerticalError(
            f"{target} holds files ftea reframe did not write "
            f"({', '.join(foreign[:3])}). Choose a new or empty folder "
            "with --out <new folder>."
        )
    if not names:
        return
    owner = _owner(target / MARKER_NAME)
    if owner is None:
        raise VerticalError(
            f"{target} holds results that do not say which video they "
            f"belong to (no readable {MARKER_NAME}). Choose a new or empty "
            "folder with --out <new folder>."
        )
    if owner != video.resolve():
        raise VerticalError(
            f"{target} holds the results for {owner}. Choose a new folder "
            "with --out <new folder>."
        )


def _owner(marker: Path) -> Optional[Path]:
    """The video a results folder belongs to, from its marker."""
    try:
        source = json.loads(marker.read_text(encoding="utf-8"))["source"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return Path(source) if isinstance(source, str) and source else None


def _claim(target: Path, video: Path) -> _Claim:
    """Create the results folder and its scratch folder, and record which
    video they belong to, before the stage starts."""
    created = not target.exists()
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise VerticalError(_unusable(target, exc)) from None
    work = _fresh_work_dir(target)
    marker = target / MARKER_NAME
    wrote = not marker.is_file()
    if wrote:
        text = json.dumps(
            {"schema": MARKER_SCHEMA, "source": str(video.resolve())},
        )
        try:
            marker.write_text(text + "\n", encoding="utf-8")
        except OSError as exc:
            raise VerticalError(_unusable(target, exc)) from None
    return _Claim(target, work, created, wrote)


def _held(path: Path) -> str:
    return (
        f"{path} is in use: close it in the program that has it open, "
        "then run again"
    )


def _fresh_work_dir(target: Path) -> Path:
    """Empty the scratch folder, drop debug files of an earlier run, and
    check that earlier results can be replaced, all before the stage
    starts, so a file another program holds open is reported before
    minutes of GPU work rather than after them."""
    work = target / PARTIAL_DIR
    try:
        shutil.rmtree(work)
    except FileNotFoundError:
        pass
    except OSError:
        raise VerticalError(_held(work)) from None
    for name in (DEBUG_NAME, DEBUG_SIDECAR):
        try:
            (target / name).unlink(missing_ok=True)
        except OSError:
            raise VerticalError(_held(target / name)) from None
    for path in _earlier_results(target):
        _check_replaceable(path)
    try:
        work.mkdir()
    except OSError as exc:
        raise VerticalError(_unusable(target, exc)) from None
    return work


def _earlier_results(target: Path) -> list[Path]:
    try:
        found = list(target.iterdir())
    except OSError as exc:
        raise VerticalError(_unusable(target, exc)) from None
    return sorted(
        path for path in found
        if path.name == CROPS_NAME or path.stem == VERTICAL_STEM
    )


def _check_replaceable(path: Path) -> None:
    """Rename it aside and straight back: Windows refuses the rename
    while another program has the file open."""
    aside = path.with_name(path.name + ".ftea-check")
    try:
        path.rename(aside)
    except OSError:
        raise VerticalError(_held(path)) from None
    try:
        aside.rename(path)
    except OSError:
        raise VerticalError(
            f"{path} was renamed to {aside} for a check and could not be "
            "renamed back; rename it back, then run again"
        ) from None


def _payload(stdout: Optional[str], returncode: int) -> dict:
    """The stage's result, checked against its published shape."""
    try:
        payload = json.loads(stdout or "")
    except json.JSONDecodeError:
        payload = None
    if not isinstance(payload, dict):
        raise VerticalError(
            f"the reframe stage printed no result (exit {returncode}); "
            "its messages are above"
        )
    check_stage_result(payload, returncode)
    return payload


def check_stage_result(payload: dict, returncode: int) -> None:
    """Check the stage's JSON result against its published shape.

    Both launchers use it, `ftea reframe` here and `ftea run` in
    ftea.encode.main, so a result one refuses the other refuses too.
    Raises VerticalError naming what is wrong.
    """
    if payload.get("schema") != STAGE_SCHEMA:
        raise VerticalError(
            f"the reframe stage reported schema {payload.get('schema')!r}; "
            f"this ftea reads schema {STAGE_SCHEMA}. Run `bash install.sh` "
            "again so both come from the same checkout."
        )
    status = payload.get("status")
    if not isinstance(status, str) or status not in STATUS_EXIT:
        raise VerticalError(
            f"the reframe stage reported an unknown status {status!r}"
        )
    if STATUS_EXIT[status] != returncode:
        raise VerticalError(
            f"the reframe stage exited {returncode} but reported {status!r}"
        )
    _check_fields(payload)


def _check_fields(payload: dict) -> None:
    for field in _TEXT_FIELDS:
        value = payload.get(field)
        if value is not None and not isinstance(value, str):
            raise VerticalError(
                f"the reframe stage reported a {field} that is not text: "
                f"{value!r}"
            )
    if not isinstance(payload.get("report"), (dict, type(None))):
        raise VerticalError(
            "the reframe stage reported a report that is not an object: "
            f"{payload['report']!r}"
        )
    if payload["status"] == "reframed" and not payload.get("output"):
        raise VerticalError(
            "the reframe stage reported success but named no output file"
        )


def _settle(payload: dict, claim: _Claim) -> VerticalResult:
    status, target = payload["status"], claim.target
    report = payload.get("report")
    if status == "failed":
        return VerticalResult(
            "failed", target, report=report,
            error=payload.get("error") or "reframe failed",
        )
    if status == "skipped":
        _release(claim)
        reason = (report or {}).get("reason")
        return VerticalResult("skipped", target, reason=reason, report=report)
    produced = Path(payload["output"])
    if not produced.is_file():
        raise VerticalError(
            f"the reframe stage reported {produced}, which does not exist"
        )
    result_path = target / f"{VERTICAL_STEM}{produced.suffix}"
    try:
        os.replace(produced, result_path)
    except OSError:
        raise VerticalError(
            f"the new result could not be moved into {target} because a "
            f"file there is in use. It is in {claim.work}; close the file "
            "in the program that has it open, then run again"
        ) from None
    crops = _settle_crops(payload.get("crops"), target, result_path)
    shutil.rmtree(claim.work, ignore_errors=True)
    debug = payload.get("debug")
    return VerticalResult(
        "reframed", target, vertical=result_path, crops=crops,
        debug=Path(debug) if debug else None,
        debug_error=payload.get("debug_error"), report=report,
    )


def _release(claim: _Claim) -> None:
    """Undo what this run did to the results folder, when the stage
    wrote nothing: earlier results and a folder the user made stay."""
    shutil.rmtree(claim.work, ignore_errors=True)
    if claim.wrote_marker:
        with contextlib.suppress(OSError):
            (claim.target / MARKER_NAME).unlink()
    if claim.created_folder:
        with contextlib.suppress(OSError):
            claim.target.rmdir()  # only when nothing else is in it


def _settle_crops(
    new: Optional[str], target: Path, result_path: Path,
) -> Optional[Path]:
    """Move the new crop track in, and drop earlier results the new run
    did not replace, so nothing stale sits beside the new video."""
    crops = target / CROPS_NAME
    has_new = bool(new) and Path(new).is_file()
    stale = [
        path for path in _earlier_results(target)
        if path != result_path and (path != crops or not has_new)
    ]
    if has_new:
        try:
            os.replace(new, crops)
        except OSError:
            raise VerticalError(
                f"the new vertical video is in {result_path}, but its crop "
                f"track could not be moved into {target}: {_held(crops)}. "
                f"The new crop track is {new}."
            ) from None
    for path in stale:
        try:
            path.unlink()
        except OSError:
            raise VerticalError(
                f"the new vertical video is in {result_path}, but the "
                f"earlier {path.name} beside it, which does not belong to "
                f"it, could not be removed: {_held(path)}"
            ) from None
    return crops if has_new else None
