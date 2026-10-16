"""Batch orchestrator: pre-scan, reframe and encode, retry, report.

run() is the entry point the `ftea run` command wraps; main() runs it
from a shell as `python -m ftea.encode.main`. Nothing here talks to a
network. Everything a run produces stays under the data root (see
ftea.paths): finished clips in output/<clip>/, scratch in work/<clip>/.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import shutil
import sys
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .. import paths
from ..console import shell_arg, tolerate_unprintable_text
from ..vertical import MARKER_NAME as REFRAME_RESULTS_MARKER
from ..vertical import STATUS_EXIT, VerticalError, check_stage_result
from . import preflight
from .config import (
    ConfigValidationError,
    PipelineConfig,
    load_config,
    load_reframe_launch_config,
    nvenc_supports_film_grain,
    unknown_keys,
)
from .encode import encode_and_concat
from .geometry import (
    conform,
    conform_entry,
    needs_conform,
    probe_geometry,
    target_frame,
)
from .package import package_hls
from .probe import run_probing
from .report import (
    ClipReport,
    generate_batch_report,
    generate_clip_report,
    generate_markdown_report,
    save_batch_report,
    save_clip_report,
)
from .scene import SourceAnalysis, analyze_and_split, detect_has_audio
from .tools import ToolError, ToolTimeout, clip_deadline, run_tool

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".avi"})
MASTER_PLAYLIST = "master.m3u8"
REPORT_JSON = "report.json"
REPORT_MD = "report.md"
BATCH_REPORT = "batch_report.json"
# Written in a failed attempt's work folder, next to what it left there.
TRACEBACK_NAME = "traceback.txt"
FFPROBE_TIMEOUT_SEC = 60
# The source fingerprint hashes the file's size plus its first and last
# MiB: milliseconds per file, and enough to tell two videos apart.
FINGERPRINT_CHUNK = 1024 * 1024

# A run creates its lock and writes its process id just after. A lock
# with no readable id that is newer than this counts as held.
LOCK_GRACE_SEC = 10

# Windows process queries, used by _windows_pid_alive.
_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
_ERROR_ACCESS_DENIED = 5


class PipelineError(RuntimeError):
    """A problem that stops a run before any clip is encoded."""


class UsageError(ValueError):
    """The command line could not be understood."""


# --- Settings ---------------------------------------------------------------

def load_run_config(
    reframe: Optional[bool] = None, *, reframe_only: bool = False,
) -> PipelineConfig:
    """The settings for one run: config.env, or the defaults without one.

    Prints a line when config.env is missing and one for every setting it
    does not recognise, so a typo never silently does nothing. `reframe`
    overrides REFRAME for this run when it is not None. reframe_only reads
    just what `ftea reframe` launches the stage with, so an encoder
    setting it never uses cannot stop it.
    """
    loader = load_reframe_launch_config if reframe_only else load_config
    config_path = paths.config_path()
    if not config_path.is_file():
        print(
            f"  No config.env at {config_path}; using the defaults "
            "(see config.env.example)."
        )
    try:
        for key in unknown_keys(config_path):
            print(
                f"  WARNING: config.env setting {key} is not a known "
                "setting and is ignored."
            )
        config = loader(config_path, env=os.environ, missing_ok=True)
    except ConfigValidationError as exc:
        raise PipelineError(f"config.env: {exc}") from None
    if reframe is not None:
        config = dataclasses.replace(config, reframe_enabled=reframe)
    return config


# --- The run lock -----------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    """Whether a process with this id is still running."""
    if pid <= 0:
        return False
    if os.name == "nt":
        return _windows_pid_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _windows_pid_alive(pid: int) -> bool:
    """Ask Windows directly whether a process is still running.

    Windows has no signal 0: os.kill(pid, 0) sends CTRL_C_EVENT, which
    can report success for a process that has already exited (whenever
    anything still holds a handle to it). A lock left by a killed run
    could then never be cleared.
    """
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (
        wintypes.DWORD, wintypes.BOOL, wintypes.DWORD,
    )
    kernel32.GetExitCodeProcess.argtypes = (
        wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD),
    )
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)

    handle = kernel32.OpenProcess(_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        # Access denied means it exists but belongs to someone else.
        return ctypes.get_last_error() == _ERROR_ACCESS_DENIED
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True
        return code.value == _STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _read_lock_pid(lock: Path) -> Optional[int]:
    try:
        return int(lock.read_text(encoding="utf-8").strip())
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def _lock_is_new(lock: Path) -> bool:
    """Whether the lock was written within the last LOCK_GRACE_SEC."""
    try:
        return time.time() - lock.stat().st_mtime < LOCK_GRACE_SEC
    except OSError:
        return False


def _clear_stale_lock(lock: Path, holder: Optional[int]) -> None:
    """Remove a lock no running process holds, or raise PipelineError.

    A lock without a readable process id may belong to a run that has
    created it and not yet written its id, so a new one is left alone.
    """
    if holder is not None and _pid_alive(holder):
        raise PipelineError(
            f"another ftea run (process {holder}) is using "
            f"{paths.data_root()}. If no run is active, delete {lock}."
        )
    if holder is None and _lock_is_new(lock):
        raise PipelineError(
            f"another ftea run is starting in {paths.data_root()}. If no "
            f"run is active, try again in {LOCK_GRACE_SEC}s or delete {lock}."
        )
    reason = (
        "it holds no process id" if holder is None
        else f"process {holder} is not running"
    )
    print(f"  Removing stale lock ({reason}).")
    try:
        lock.unlink(missing_ok=True)
    except OSError as exc:
        raise PipelineError(
            f"could not remove the stale lock {lock} ({exc}); delete it and "
            "try again."
        ) from None


def _acquire_lock() -> None:
    """Claim the data folder for this run, or raise PipelineError.

    The lock file holds the owning process id and is created atomically.
    A lock whose process is gone (a run that was killed) is removed and
    taken over.
    """
    lock = paths.lock_file()
    lock.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            _clear_stale_lock(lock, _read_lock_pid(lock))
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        return
    raise PipelineError(f"could not create {lock}; delete it and try again.")


def _release_lock() -> None:
    """Remove the lock, but only when this process owns it."""
    lock = paths.lock_file()
    if _read_lock_pid(lock) == os.getpid():
        lock.unlink(missing_ok=True)


# --- Locating the clips -----------------------------------------------------

def _is_safe_clip_id(video_id: str) -> bool:
    """A clip id becomes a folder name directly under output/."""
    return (
        bool(video_id.strip())
        and video_id not in (".", "..")
        and "/" not in video_id
        and "\\" not in video_id
    )


def _drop_unusable(files: list[Path]) -> list[Path]:
    """Skip files whose names cannot each become their own output folder.

    Names are compared without case, because output/Clip/ and
    output/clip/ are the same folder on Windows.
    """
    kept: list[Path] = []
    seen: dict[str, Path] = {}
    for path in files:
        video_id = path.stem
        if not _is_safe_clip_id(video_id):
            print(f"  SKIP (unusable file name): {path.name}")
            continue
        key = video_id.casefold()
        if key in seen:
            print(
                f"  SKIP (same name as {seen[key].name}; both would write "
                f"output/{video_id}/): {path.name}"
            )
            continue
        seen[key] = path
        kept.append(path)
    return kept


def _video_files(input_path: Path) -> list[Path]:
    """The named file itself, or a folder's video files in name order."""
    if input_path.is_file():
        return [input_path]
    return sorted(
        (
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        ),
        key=lambda p: p.name.casefold(),
    )


def _candidate_files(input_path: Path) -> list[Path]:
    """The video files whose names can each become an output folder."""
    return _drop_unusable(_video_files(input_path))


def _video_duration(path: Path) -> Optional[float]:
    """The file's length in seconds, or None without a video stream.

    One ffprobe call answers both. A length ffprobe cannot tell (N/A)
    reads as 0.0, which no length limit refuses. ffprobe failing to read
    the file at all raises ToolError with its own reason.
    """
    result = run_tool(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_type:format=duration",
            "-of", "json", str(path),
        ],
        timeout=FFPROBE_TIMEOUT_SEC,
    )
    try:
        data = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    streams = data.get("streams") or []
    if not any(s.get("codec_type") == "video" for s in streams):
        return None
    try:
        return float((data.get("format") or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _clock(seconds: float) -> str:
    """A length as minutes:seconds, such as 12:03."""
    whole = int(round(seconds))
    return f"{whole // 60}:{whole % 60:02d}"


def _pre_scan(
    candidates: list[Path], max_clip_sec: int = 0,
) -> tuple[dict[Path, float], list[Path]]:
    """The readable clips with their lengths, and the ones too long.

    A file without a readable video stream is skipped. A video longer
    than max_clip_sec (0 means no limit) is set aside instead of encoded:
    ftea is made for short clips, and a long one would only fail slowly,
    three times over. The lengths also size each clip's encode limit.
    """
    valid: dict[Path, float] = {}
    too_long: list[Path] = []
    for path in candidates:
        try:
            duration = _video_duration(path)
        except ToolError as exc:
            print(f"  SKIP ({exc}): {path.name}")
            continue
        if duration is None:
            print(f"  SKIP (no video stream): {path.name}")
        elif max_clip_sec and duration > max_clip_sec:
            print(
                f"  SKIP ({_clock(duration)} long; ftea is made for short "
                "clips — split it first, or raise MAX_CLIP_SEC): "
                f"{path.name}"
            )
            too_long.append(path)
        else:
            valid[path] = duration
    return valid, too_long


# --- Resume -----------------------------------------------------------------

def _load_existing_report(path: Path) -> Optional[ClipReport]:
    """Rebuild a ClipReport from a saved report.json, or return None.

    Unknown keys are dropped, so a report from a newer or older ftea still
    loads. A missing, unreadable or incomplete report gives None, and the
    clip is encoded again: slower, but always right.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    names = {f.name for f in dataclasses.fields(ClipReport)}
    try:
        return ClipReport(**{k: v for k, v in data.items() if k in names})
    except TypeError:
        return None


def _is_complete(clip_dir: Path) -> bool:
    """Whether a clip's output folder holds a finished, passing package.

    report.json is written last, after master.m3u8 and every segment, so
    a readable report whose status is not "failed", next to a master
    playlist, means the clip needs nothing more. A clip killed at any
    earlier point, or one that failed, is encoded again.
    """
    report = _load_existing_report(clip_dir / REPORT_JSON)
    return (
        report is not None
        and report.status != "failed"
        and (clip_dir / MASTER_PLAYLIST).is_file()
    )


def _source_fingerprint(path: Path) -> tuple[int, str]:
    """(size, SHA-256 of the size plus the first and last MiB) of a file."""
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode("ascii"))
    with path.open("rb") as handle:
        digest.update(handle.read(FINGERPRINT_CHUNK))
        if size > FINGERPRINT_CHUNK:
            handle.seek(max(size - FINGERPRINT_CHUNK, FINGERPRINT_CHUNK))
            digest.update(handle.read(FINGERPRINT_CHUNK))
    return size, digest.hexdigest()


def _same_source(report: ClipReport, clip: Path) -> bool:
    """Whether a finished report was made from this very file.

    A report without a fingerprint (made before fingerprints existed, or
    edited by hand) counts as a match, as matching by name always did.
    """
    if report.source_size is None or report.source_digest is None:
        return True
    recorded = (report.source_size, report.source_digest)
    return recorded == _source_fingerprint(clip)


def _holds_reframe_results(clip_dir: Path) -> bool:
    """Whether a clip's output folder is an `ftea reframe` results folder.

    `ftea reframe talk.mp4` writes output/talk-vertical/ by default,
    which is exactly where a clip named talk-vertical.mp4 is encoded.
    """
    return (clip_dir / REFRAME_RESULTS_MARKER).is_file()


def _resume_state(clip_dir: Path, clip: Path, force: bool = False) -> str:
    """Whether a clip is "done", a "clash", "reframe_results" or "todo".

    A clash is a finished folder made from a different file.

    A clash is never overwritten: the folder may already be published,
    and the file in the input may be different footage that only shares
    its name (IMG_0001.MOV recurs on every camera card). force encodes a
    finished or clashing clip again, but never replaces the results of
    `ftea reframe`, which ftea run did not make.
    """
    if _holds_reframe_results(clip_dir):
        return "reframe_results"
    if force or not _is_complete(clip_dir):
        return "todo"
    report = _load_existing_report(clip_dir / REPORT_JSON)
    return "done" if _same_source(report, clip) else "clash"


def _framing_note(clip_dir: Path, config: PipelineConfig) -> Optional[str]:
    """How a finished clip was framed, when this run would frame it
    otherwise; None when it matches.

    Resume keeps a finished package whatever the settings now say, as it
    does for every other setting. Reframing is the one that changes what
    the package looks like, so a run asking for the other framing says
    so instead of passing it over in silence.
    """
    report = _load_existing_report(clip_dir / REPORT_JSON)
    if report is None:
        return None
    reframed = bool(report.reframe and report.reframe.get("applied"))
    if (
        config.reframe_enabled and report.reframe is None
        and report.orientation == "landscape"
    ):
        return "was encoded without reframing, and reframing is on now"
    if not config.reframe_enabled and reframed:
        return "was reframed, and reframing is off now"
    return None


# --- Progress, failure folders, output folders -----------------------------

def _update_progress(
    total: int, completed: int, failed: int,
    current: str, started: float, failures: list[dict],
) -> None:
    """Write work/batch_progress.json so a long batch can be monitored."""
    elapsed = time.time() - started
    avg = elapsed / max(completed, 1)
    remaining = total - completed - failed
    progress = {
        "total": total,
        "completed": completed,
        "failed": failed,
        "remaining": remaining,
        "elapsed_sec": round(elapsed),
        "avg_per_clip_sec": round(avg, 1),
        "eta_sec": round(avg * remaining),
        "current_clip": current,
        "failures": failures[-5:],
    }
    progress_file = paths.progress_file()
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    progress_file.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def _manage_failure_dirs(work_base: Path, max_kept: int) -> None:
    """Keep only the newest `max_kept` failed-clip work folders.

    Only clip folders count: names starting with "." (such as the VMAF
    log folder) are left alone, and a missing work folder is fine.
    """
    if not work_base.is_dir():
        return
    failure_dirs = sorted(
        (
            d for d in work_base.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ),
        key=lambda d: d.stat().st_mtime,
    )
    while len(failure_dirs) > max_kept:
        shutil.rmtree(failure_dirs.pop(0), ignore_errors=True)


class OutputInUseError(RuntimeError):
    """A finished package could not be set aside: something holds it open."""


def _aside_dir(output_base: Path, video_id: str) -> Path:
    """Where a finished package waits while its clip is encoded again.

    A sibling inside output/, so the rename never crosses a disk.
    """
    return output_base / f".{video_id}.previous"


def _prepare_output_dir(
    output_base: Path, video_id: str,
) -> tuple[Path, Optional[Path]]:
    """Make an empty output/<clip>/ for an encode attempt.

    A finished package is renamed aside, not deleted, so a failed attempt
    loses nothing: _settle_previous deletes it after a success or puts it
    back after a failure. The second value says where it went, or is
    None. Unfinished leftovers are simply deleted. A package left aside
    by an interrupted run is the last good one, so it stays aside.
    """
    if not _is_safe_clip_id(video_id):
        raise ValueError(f"unusable clip id {video_id!r}")
    clip_dir = output_base / video_id
    if _holds_reframe_results(clip_dir):
        raise RuntimeError(
            f"output/{video_id}/ holds the results of `ftea reframe`, "
            "which are never replaced; rename the video to encode it"
        )
    aside = _aside_dir(output_base, video_id)
    if clip_dir.exists() and _is_complete(clip_dir):
        shutil.rmtree(aside, ignore_errors=True)
        try:
            clip_dir.rename(aside)
        except OSError as exc:
            # On Windows an open file (ftea preview, a video player)
            # blocks the rename. Failing here costs no GPU time.
            raise OutputInUseError(
                f"cannot set {clip_dir} aside ({exc}). Close what is using "
                f"output/{video_id}/ and run again."
            ) from exc
    elif clip_dir.exists():
        shutil.rmtree(clip_dir)
    clip_dir.mkdir(parents=True)
    return clip_dir, (aside if aside.is_dir() else None)


def _settle_previous(
    clip_dir: Path, previous: Optional[Path], succeeded: bool,
) -> None:
    """Delete the set-aside package after a success, or put it back.

    Never raises: a package that cannot be moved back stays aside, and
    the next attempt at the clip finds it there.
    """
    if previous is None:
        return
    try:
        if succeeded:
            shutil.rmtree(previous)
            return
        shutil.rmtree(clip_dir, ignore_errors=True)
        previous.rename(clip_dir)
    except OSError as exc:
        print(f"  WARNING: could not tidy {previous}: {exc}")


# --- The reframe stage: a subprocess in its own virtualenv ------------------

REFRAME_STATUS_EXIT = STATUS_EXIT
# The report entry of a clip whose reframe failed without the stage
# describing it (time limit, no result, the interpreter would not start,
# a result that broke the contract), so the batch report still counts
# it as a reframe failure.
REFRAME_FAILED_ENTRY = {"applied": False, "outcome": "failed"}
# What may follow a folder in a message for it to count as a whole name:
# the end, a path separator, white space, or punctuation around a path.
_FOLDER_END = r"(?=$|[\\/\s'\",;:|)])"
CROPS_NAME = "crops.json"
VERIFY_ENV_TIMEOUT_SEC = 600


class ReframeStageError(RuntimeError):
    """The reframe stage itself reported status "failed".

    Carries the stage's own diagnostic entry (reason, source geometry,
    rotation, fps), so the failed clip's report keeps it instead of only
    an error string.
    """

    def __init__(self, message: str, *, entry: Optional[dict]) -> None:
        super().__init__(message)
        self.entry = entry


@dataclass(frozen=True)
class ReframeOutcome:
    """What _run_reframe hands back to the per-clip flow."""

    # The original clip when skipped; the vertical intermediate otherwise.
    source: Path
    # The stage's report entry, or None.
    entry: Optional[dict]
    # The crop track as the stage reported it; None when it wrote none.
    crops: Optional[Path]


def _reframe_env() -> dict[str, str]:
    """Environment for the reframe interpreter (see ftea.paths).

    PYTHONIOENCODING makes the child write UTF-8 even though its output
    is a pipe, which is how run_tool decodes it.
    """
    env = paths.reframe_subprocess_env(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _setting_path(key: str, value: str) -> Optional[Path]:
    """paths.resolve_setting_path, with the setting's name in its error.

    On Windows a value such as Git Bash's /c/... is refused rather than
    read as a different folder; the RuntimeError says which key it was.
    """
    try:
        return paths.resolve_setting_path(value)
    except ValueError as exc:
        raise RuntimeError(f"{key}: {exc}") from None


def _resolve_reframe_python(config: PipelineConfig) -> Path:
    """Find the reframe virtualenv's interpreter.

    REFRAME_PYTHON wins when set; a relative value is read from the data
    folder. Otherwise .venv-reframe in the repository. There is no
    fallback to `python` or `py`: the encoder's own interpreter has no
    torch, and falling back would turn a clear setup problem into an
    ImportError deep inside a detection module.
    """
    configured = _setting_path("REFRAME_PYTHON", config.reframe_python)
    if configured is not None:
        if configured.is_file():
            return configured
        raise RuntimeError(
            f"REFRAME_PYTHON points at {configured}, which does not exist"
        )
    venv = paths.reframe_venv_dir()
    for candidate in paths.venv_python_candidates(venv):
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        f"no reframe interpreter found in {venv}. Run `bash install.sh` in "
        f"the repository folder, or set REFRAME_PYTHON in "
        f"{paths.config_path()}."
    )


def _verify_reframe_env(python: Path) -> None:
    """Run the reframe stage's own environment check, once per run.

    Missing models, a CPU-only torch or no visible GPU then stop the run
    with one message, instead of failing every clip in turn.
    """
    print("  Checking the reframe environment...")
    try:
        result = run_tool(
            [str(python), "-m", paths.VERIFY_ENV_MODULE],
            cwd=paths.source_root(),
            env=_reframe_env(),
            timeout=VERIFY_ENV_TIMEOUT_SEC,
            check=False,
            kill_tree=True,
        )
    except ToolError as exc:
        raise PipelineError(
            f"the reframe environment check could not run: {exc}"
        ) from None
    if result.returncode == 0:
        return
    lines = [
        line for line in (result.stdout + "\n" + result.stderr).splitlines()
        if line.strip()
    ]
    raise PipelineError(
        "the reframe environment check failed:\n"
        + "\n".join(lines[-15:])
        + "\nRun `bash install.sh` in the repository folder, or run with "
        "--no-reframe."
    )


def _run_reframe(
    clip: Path, config: PipelineConfig, work_dir: Path,
) -> ReframeOutcome:
    """Run the auto-reframe stage for one clip.

    A skip hands back the ORIGINAL clip; pass-through is not a copy.
    Raises ReframeStageError (with the stage's diagnostic entry) when the
    stage reports "failed", and RuntimeError for every other problem:
    time limit, interpreter, malformed result, or a broken contract. The
    caller turns either into a failed clip report; the batch carries on.
    A stage that runs out of time is stopped together with the ffmpeg it
    started, so nothing keeps writing into the work folder the retry
    empties.
    """
    python = _resolve_reframe_python(config)
    cmd = [
        str(python), "-m", paths.REFRAME_MODULE,
        "--source", str(clip.resolve()),
        "--work-dir", str((work_dir / "reframe").resolve()),
        "--config", str(paths.config_path()),
    ]
    print(f"  Reframing (limit {config.reframe_timeout}s)...")
    try:
        result = run_tool(
            cmd,
            cwd=paths.source_root(),
            env=_reframe_env(),
            timeout=config.reframe_timeout,
            check=False,
            kill_tree=True,
        )
    except ToolTimeout:
        raise RuntimeError(
            f"reframe timed out after {config.reframe_timeout}s"
        ) from None
    except ToolError as exc:
        raise RuntimeError(
            f"reframe interpreter {python} could not be started: {exc}"
        ) from None

    if result.stderr:
        print(result.stderr.rstrip())

    try:
        payload = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        tail = (result.stderr or result.stdout or "").strip()[-500:]
        raise RuntimeError(
            f"reframe emitted no JSON result (exit {result.returncode}): "
            f"{tail}"
        ) from None
    if not isinstance(payload, dict):
        raise RuntimeError(f"reframe emitted an unexpected result: {payload!r}")
    try:
        # A report entry that is not an object would crash the batch
        # report once every clip is done; the check refuses it here.
        check_stage_result(payload, result.returncode)
    except VerticalError as exc:
        raise RuntimeError(str(exc)) from None

    status = payload.get("status")
    entry = payload.get("report")
    if status == "failed":
        raise ReframeStageError(
            payload.get("error") or "reframe failed", entry=entry,
        )
    if status == "skipped":
        return ReframeOutcome(source=clip, entry=entry, crops=None)

    output = Path(payload.get("output") or "")
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"reframe output {output} does not exist")
    # The encode takes its audio from the encode source, so a silent
    # intermediate would quietly drop the clip's sound. Refuse it.
    if detect_has_audio(clip) and not detect_has_audio(output):
        raise RuntimeError(
            "reframe produced a silent intermediate (source has audio)"
        )
    crops_raw = payload.get("crops")
    return ReframeOutcome(
        source=output, entry=entry,
        crops=Path(crops_raw) if crops_raw else None,
    )


def _retain_crops(
    video_id: str, crops: Optional[Path], config: PipelineConfig,
    clip_dir: Path,
) -> None:
    """Keep the reframe crop track once the stage succeeds.

    By default it goes in the clip's own output folder as crops.json;
    REFRAME_CROPS_DIR sends it to <folder>/<clip>.crops.json instead.
    Never under work/, which failure pruning empties. `crops` is the path
    the stage reported, never a guess; None means the stage wrote none.
    """
    if crops is None or not config.reframe_keep_crops:
        return
    if not crops.exists():
        raise RuntimeError(
            f"reframe reported a crop track at {crops}, but it is missing"
        )
    folder = _setting_path("REFRAME_CROPS_DIR", config.reframe_crops_dir)
    if folder is None:
        destination = clip_dir / CROPS_NAME
    else:
        folder.mkdir(parents=True, exist_ok=True)
        destination = folder / f"{video_id}.crops.json"
    shutil.copy2(crops, destination)


# --- One clip -----------------------------------------------------------------

@dataclass
class _ClipState:
    """What one clip attempt has learned so far, kept for its report."""

    source: Path
    reframe: Optional[dict] = None
    conform: Optional[dict] = None
    analysis: Optional[SourceAnalysis] = None


def _prepare_source(
    clip: Path, config: PipelineConfig, work_dir: Path, clip_dir: Path,
    state: _ClipState,
) -> None:
    """Reframe, before the clip's deadline starts (it has its own limit).

    A reframe failure is an ordinary clip failure: there is never a
    fallback to a centre crop.
    """
    if config.reframe_enabled:
        try:
            outcome = _run_reframe(clip, config, work_dir)
        except ReframeStageError as exc:
            state.reframe = exc.entry or dict(REFRAME_FAILED_ENTRY)
            raise
        except Exception:
            state.reframe = dict(REFRAME_FAILED_ENTRY)
            raise
        state.source = outcome.source
        state.reframe = outcome.entry
        # Kept as soon as the stage succeeds: the clip someone most wants
        # to inspect is the one that then fails during the encode.
        _retain_crops(clip.stem, outcome.crops, config, clip_dir)


def _conform_if_needed(state: _ClipState, work_dir: Path) -> None:
    """Pad a clip whose shape is neither 16:9 nor 9:16 onto its frame.

    Runs after reframing and inside the clip's deadline, so it judges
    what will actually be encoded; a reframed intermediate is already
    exactly 9:16 and is left alone.
    """
    shape = probe_geometry(state.source)
    if not needs_conform(shape):
        return
    width, height = target_frame(shape)
    print(
        f"  Conforming {shape.label} to {width}x{height} "
        "(black bars, no stretching)..."
    )
    output = conform(state.source, work_dir, shape)
    if detect_has_audio(state.source) and not detect_has_audio(output):
        raise RuntimeError(
            "conform produced a silent intermediate (source has audio)"
        )
    state.source = output
    state.conform = conform_entry(shape)


def _encode_stages(
    clip: Path, config: PipelineConfig, work_dir: Path, clip_dir: Path,
    state: _ClipState, started: float,
) -> ClipReport:
    """Analyse, probe, encode and package; return the clip's report.

    The report carries the original file's fingerprint, which is what
    resume compares on the next run.
    """
    print("  Analyzing source...")
    state.analysis = analysis = analyze_and_split(state.source, config, work_dir)
    print(
        f"  {analysis.orientation}, {analysis.fps}fps, "
        f"{'B&W' if analysis.is_bw else 'color'}, "
        f"{len(analysis.scenes)} scenes, "
        f"{'audio' if analysis.has_audio else 'no audio'}"
    )

    variants = config.build_variants(analysis.orientation)
    print(
        f"  Probing CQ ({len(analysis.scenes)} scenes x "
        f"{len(variants)} variants)..."
    )
    probe_results = run_probing(
        scenes=analysis.scenes, config=config, variants=variants,
        film_grain=analysis.film_grain,
        keyframe_interval=analysis.keyframe_interval, work_dir=work_dir,
    )

    print("  Final encode + concat...")
    variant_outputs = encode_and_concat(
        scenes=analysis.scenes, probe_results=probe_results,
        source_analysis=analysis, config=config, variants=variants,
        work_dir=work_dir,
    )

    print("  HLS packaging...")
    # The encode source supplies both the audio and the thumbnail, so a
    # reframed clip gets a vertical poster; the silent-intermediate
    # checks are what make that safe.
    package_hls(
        variant_outputs=variant_outputs, source=state.source,
        source_analysis=analysis, config=config, output_dir=clip_dir,
    )
    report = generate_clip_report(
        video_id=clip.stem, source=clip, source_analysis=analysis,
        probe_results=probe_results, variant_outputs=variant_outputs,
        encode_time_sec=time.time() - started, vmaf_floor=config.vmaf_floor,
        reframe=state.reframe, conform=state.conform,
    )
    size, digest = _source_fingerprint(clip)
    return dataclasses.replace(report, source_size=size, source_digest=digest)


def _local_folders() -> list[tuple[str, str]]:
    """(folder as written, placeholder) for the data folder, the checkout
    and the home folder, longest first, so a folder inside another is
    replaced first. A folder that is a drive or filesystem root is left
    out: replacing it would mangle every path."""
    found: list[tuple[str, str]] = []
    for where, placeholder in (
        (paths.data_root, "<data>"), (paths.source_root, "<checkout>"),
        (Path.home, "~"),
    ):
        try:
            folder = where()
        except (RuntimeError, OSError, KeyError):
            continue
        if folder == Path(folder.anchor):
            continue
        for spelling in dict.fromkeys((str(folder), folder.as_posix())):
            found.append((spelling, placeholder))
    return sorted(found, key=lambda item: len(item[0]), reverse=True)


def _without_local_folders(text: str) -> str:
    """`text` with the data folder, the checkout and the home folder
    written as <data>, <checkout> and ~.

    A failed clip's report keeps the error message, and messages from
    ftea and from ffmpeg name full paths. The output folder, report
    included, is meant to be published, and a local path often holds
    the user's account name (see generate_clip_report). A folder counts
    only as a whole name: /home/al never matches inside /home/alice.
    """
    flags = re.IGNORECASE if os.name == "nt" else 0
    for spelling, placeholder in _local_folders():
        text = re.sub(
            re.escape(spelling) + _FOLDER_END,
            lambda _match, value=placeholder: value, text, flags=flags,
        )
    return text


def _failure_report(
    clip: Path, config: PipelineConfig, state: _ClipState,
    exc: BaseException, elapsed: float,
) -> ClipReport:
    """The report for a clip attempt that raised, keeping what was learned.

    Its error names no local folder (see _without_local_folders).
    """
    error = _without_local_folders(f"{type(exc).__name__}: {exc}")
    if state.analysis is not None:
        return generate_clip_report(
            video_id=clip.stem, source=clip, source_analysis=state.analysis,
            probe_results=[], variant_outputs=[], encode_time_sec=elapsed,
            vmaf_floor=config.vmaf_floor, error=error,
            reframe=state.reframe, conform=state.conform,
        )
    return ClipReport(
        video_id=clip.stem, source=clip.name, orientation="unknown", fps=0,
        is_bw=False, film_grain=0, has_audio=False, scene_count=0,
        duration_s=0.0, encode_time_sec=round(elapsed, 1), scenes=[],
        variants=[], status="failed", error=error, reframe=state.reframe,
        conform=state.conform,
    )


def _write_clip_reports(report: ClipReport, clip_dir: Path) -> None:
    """report.md, then report.json, the "finished" record: written last."""
    generate_markdown_report(report, clip_dir / REPORT_MD)
    save_clip_report(report, clip_dir / REPORT_JSON)


def _write_failure_reports(report: ClipReport, folder: Path) -> None:
    """Write a failed attempt's reports; a write error only warns.

    The batch report still records the failure, so a full disk or a
    locked file costs this clip its report files, never the whole run.
    """
    try:
        _write_clip_reports(report, folder)
    except OSError as exc:
        print(
            f"  WARNING: could not write the report for "
            f"{report.video_id} in {folder}: {exc}"
        )


def _save_traceback(work_dir: Path) -> None:
    """Keep the exception being handled, in full, in work/<clip>/.

    The report keeps only its type and message, which for a programming
    error (a KeyError, say) does not say where it came from. The failed
    attempt's work folder is kept, so the traceback stays beside it. A
    write error only warns: the report still records the failure.
    """
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / TRACEBACK_NAME).write_text(
            traceback.format_exc(), encoding="utf-8",
        )
    except OSError as exc:
        print(f"  WARNING: could not save the traceback in {work_dir}: {exc}")


def _empty_work_dir(work_dir: Path) -> None:
    """Give a clip attempt an empty work/<clip>/ folder.

    A failed attempt leaves its folder behind on purpose, for debugging,
    but nothing in it may reach the next attempt: scene splitting lists
    the chunk files it finds, so a stale chunk would be encoded onto the
    new clip. A file that cannot be removed fails this attempt.
    """
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)


def _encode_single_clip(
    clip: Path, config: PipelineConfig, output_base: Path, timeout: float,
) -> ClipReport:
    """Reframe, encode and package one clip; always return its report.

    Both reports go in output/<clip>/ as the last step, and report.json
    is what marks the clip finished. `timeout` is the clip's budget (see
    _clip_budget). It covers conform and encoding only: the deadline
    starts after reframing, which has its own REFRAME_TIMEOUT. Inside it
    every ffmpeg and ffprobe call is stopped when the time runs out and
    no new one starts, so a hung tool fails this clip instead of stalling
    the batch. A successful clip's work folder is deleted; a failed one
    is kept for debugging.
    A finished package from an earlier run is set aside first and put
    back if this attempt fails; the failed attempt's reports then go in
    work/<clip>/ instead.
    """
    if not _is_safe_clip_id(clip.stem):
        raise ValueError(f"unusable clip id {clip.stem!r}")
    clip_dir = output_base / clip.stem
    work_dir = paths.work_dir() / clip.stem
    previous: Optional[Path] = None
    report_dir = work_dir
    state = _ClipState(source=clip)
    started = time.time()
    try:
        _empty_work_dir(work_dir)
        clip_dir, previous = _prepare_output_dir(output_base, clip.stem)
        if previous is None:
            report_dir = clip_dir
        _prepare_source(clip, config, work_dir, clip_dir, state)
        with clip_deadline(timeout):
            _conform_if_needed(state, work_dir)
            report = _encode_stages(
                clip, config, work_dir, clip_dir, state, started,
            )
        _write_clip_reports(report, clip_dir)
    except Exception as exc:
        report = _failure_report(clip, config, state, exc, time.time() - started)
        _save_traceback(work_dir)
        _write_failure_reports(report, report_dir)
        _settle_previous(clip_dir, previous, succeeded=False)
        return report
    _settle_previous(clip_dir, previous, succeeded=True)
    shutil.rmtree(work_dir, ignore_errors=True)
    return report


# --- The batch ----------------------------------------------------------------

def _clip_budget(config: PipelineConfig, duration: float) -> float:
    """Seconds allowed for encoding one clip of `duration` seconds.

    A fixed allowance plus a share per second of video, so a hang on a
    short clip is noticed quickly while a long clip on a slower card
    still has room.
    """
    return config.clip_timeout + config.clip_timeout_per_sec * duration


def _relaxed(config: PipelineConfig) -> PipelineConfig:
    """Settings for the second and later retry passes."""
    return dataclasses.replace(
        config,
        vmaf_targets={
            res: target - config.retry_vmaf_relaxation
            for res, target in config.vmaf_targets.items()
        },
        clip_timeout=config.clip_timeout * config.retry_timeout_multiplier,
        clip_timeout_per_sec=(
            config.clip_timeout_per_sec * config.retry_timeout_multiplier
        ),
        reframe_timeout=(
            config.reframe_timeout * config.retry_timeout_multiplier
        ),
    )


def _encode_pass(
    clips: list[Path], config: PipelineConfig, output_base: Path,
    durations: dict[Path, float],
) -> list[ClipReport]:
    """Phase 2: every clip once, in order."""
    reports: list[ClipReport] = []
    failures: list[dict] = []
    started = time.time()
    for index, clip in enumerate(clips, start=1):
        print(f"\n[{index}/{len(clips)}] {clip.name}")
        print("-" * 40)
        _update_progress(
            len(clips), len(reports) - len(failures), len(failures),
            clip.name, started, failures,
        )
        report = _encode_single_clip(
            clip, config, output_base,
            timeout=_clip_budget(config, durations[clip]),
        )
        reports.append(report)
        if report.status != "failed":
            print(f"  OK ({report.encode_time_sec:.1f}s)")
            continue
        failures.append({"clip": clip.name, "error": report.error})
        print(f"  FAILED: {report.error}")
        if config.batch_on_error == "stop":
            print("  BATCH_ON_ERROR=stop: halting the batch.")
            break
    return reports


def _index_of(reports: list[ClipReport], video_id: str) -> int:
    return next(i for i, r in enumerate(reports) if r.video_id == video_id)


def _retry_passes(
    reports: list[ClipReport], clips_by_id: dict[str, Path],
    config: PipelineConfig, output_base: Path,
    durations: dict[Path, float],
) -> dict:
    """Phase 3: up to RETRY_MAX more attempts for every failed clip.

    Every pass gets the time limits, REFRAME_TIMEOUT included, multiplied
    once by RETRY_TIMEOUT_MULTIPLIER; later passes get no more than pass
    1. Pass 1 keeps the quality targets, because most failures are
    transient and a clip that ran out of time only needs the longer
    limit. Later passes also lower the targets by RETRY_VMAF_RELAXATION.
    """
    summary: dict = {}
    failed = [r for r in reports if r.status == "failed"]
    relaxed = _relaxed(config)
    longer = dataclasses.replace(
        config,
        clip_timeout=relaxed.clip_timeout,
        clip_timeout_per_sec=relaxed.clip_timeout_per_sec,
        reframe_timeout=relaxed.reframe_timeout,
    )
    for retry_pass in range(1, config.retry_max + 1):
        if not failed:
            break
        print(f"\n  Retry pass {retry_pass}: {len(failed)} clip(s)")
        pass_config = longer if retry_pass == 1 else relaxed
        recovered = 0
        still_failed: list[ClipReport] = []
        for old in failed:
            clip = clips_by_id[old.video_id]
            print(f"    Retrying: {clip.name}")
            new = _encode_single_clip(
                clip, pass_config, output_base,
                timeout=_clip_budget(pass_config, durations[clip]),
            )
            reports[_index_of(reports, old.video_id)] = new
            if new.status == "failed":
                still_failed.append(new)
                print(f"    STILL FAILED: {new.error}")
            else:
                recovered += 1
                print(f"    RECOVERED ({new.encode_time_sec:.1f}s)")
        summary[f"retry_{retry_pass}"] = {
            "attempted": len(failed), "recovered": recovered,
        }
        failed = still_failed
    return summary


@dataclass(frozen=True)
class RunResult:
    """What one run did. exit_code is 0 unless a clip still failed or a
    file was not attempted: unreadable, a name that cannot be an output
    folder, too long, a clash with a finished folder, or never reached
    after a BATCH_ON_ERROR=stop halt.

    not_attempted holds file names for the files that could not be used
    at all (unreadable or an unusable name) and clip ids for the rest.
    """

    outputs: tuple[Path, ...] = ()
    failed: tuple[str, ...] = ()
    already_done: tuple[str, ...] = ()
    batch_report: Optional[Path] = None
    not_attempted: tuple[str, ...] = ()

    @property
    def exit_code(self) -> int:
        return 1 if self.failed or self.not_attempted else 0


def _print_phase(number: int, title: str) -> None:
    print()
    print("=" * 60)
    print(f"PHASE {number}: {title}")
    print("=" * 60)


def _resolve_input(source: Path) -> tuple[list[Path], tuple[str, ...]]:
    """The usable video files, and the names of those that are not.

    A file whose name cannot become its own output folder is never
    attempted, and like every other file left out it counts against the
    run's exit code.
    """
    if not source.exists():
        raise PipelineError(
            f"{source} does not exist. Put videos in {paths.input_dir()}, "
            "or name a video file or folder."
        )
    found = _video_files(source)
    usable = _drop_unusable(found)
    return usable, tuple(p.name for p in found if p not in usable)


def _require_programs() -> None:
    """Pre-scan runs ffprobe, so ffmpeg and ffprobe must exist first."""
    problems = preflight._missing_programs()
    if problems:
        raise PipelineError(
            "this machine cannot run the encoder yet:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )


def _startup_checks(config: PipelineConfig) -> None:
    """Check the machine once there is work, before any clip is touched.

    Called from _run_locked only when the to-do list is not empty, so a
    rerun with nothing to encode skips the NVENC trials (bare and with
    the real encode options) and the reframe stage's verify_env.
    """
    problems = preflight.check_toolchain(config)
    if problems:
        raise PipelineError(
            "this machine cannot run the encoder yet:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )
    if (config.film_grain or config.film_grain_bw) and (
        not nvenc_supports_film_grain()
    ):
        print(
            "  WARNING: this FFmpeg's av1_nvenc has no -film_grain option, "
            "so FILM_GRAIN and FILM_GRAIN_BW are not applied; clip reports "
            "record grain 0."
        )
    print(f"  Auto-reframe: {'ON' if config.reframe_enabled else 'OFF'}")
    if not config.reframe_enabled:
        return
    try:
        python = _resolve_reframe_python(config)
        if config.reframe_keep_crops:
            # Read once here, or every clip fails after its reframe.
            _setting_path("REFRAME_CROPS_DIR", config.reframe_crops_dir)
    except RuntimeError as exc:
        raise PipelineError(
            f"{str(exc).rstrip('.')}. You can also run with --no-reframe.",
        ) from None
    _verify_reframe_env(python)


def _summarise(
    reports: list[ClipReport], result: RunResult, wall: float,
) -> None:
    passed = sum(1 for r in reports if r.status == "passed")
    warned = sum(1 for r in reports if r.status == "warning")
    print(
        f"  Clips encoded this run: {len(reports)} (passed {passed}, "
        f"warning {warned}, failed {len(result.failed)})"
    )
    if result.already_done:
        print(f"  Already done before this run: {len(result.already_done)}")
    if result.not_attempted:
        print(
            "  Not attempted (unreadable, an unusable name, too long, a "
            "name clash, or after a BATCH_ON_ERROR=stop halt): "
            + ", ".join(result.not_attempted)
        )
    for report in reports:
        if report.status == "failed":
            print(f"    - {report.video_id}: {report.error}")
    print(f"  Wall time: {wall:.0f}s ({wall / 60:.1f}m)")
    print(f"  Batch report: {result.batch_report}")


def _print_set_aside(clashes: list[Path], states: dict[Path, str]) -> None:
    for clip in clashes:
        if states[clip] == "reframe_results":
            print(
                f"  SKIP: output/{clip.stem}/ holds the results of `ftea "
                f"reframe`, which are never replaced; rename {clip.name}, "
                "or move that folder, to encode it."
            )
            continue
        print(
            f"  SKIP: output/{clip.stem}/ was made from a different "
            f"{clip.name}; rename the file, or delete that folder to "
            "encode it again."
        )


def _print_done_notes(
    done: list[Path], config: PipelineConfig, output_base: Path,
) -> None:
    """Say which finished clips were framed otherwise, and how to encode
    a finished clip again."""
    for clip in done:
        why = _framing_note(output_base / clip.stem, config)
        if why is not None:
            print(
                f"  NOTE: output/{clip.stem}/ {why}. It is kept as it is; "
                f"to encode it again, run `ftea run --force "
                f"{shell_arg(str(clip))}`."
            )
    if done:
        print(
            "  (To encode a finished clip again, run `ftea run --force`, "
            f"or delete its folder in {output_base}.)"
        )


def _run_locked(
    source: Path, candidates: list[Path], config: PipelineConfig,
    output_base: Path, force: bool, unusable: tuple[str, ...] = (),
) -> RunResult:
    durations, too_long = _pre_scan(candidates, config.max_clip_sec)
    # Files ffprobe could not read, or that hold no video stream.
    unreadable = tuple(
        c.name for c in candidates if c not in durations and c not in too_long
    )
    if source.is_file() and too_long:
        raise PipelineError(
            f"{source.name} is longer than MAX_CLIP_SEC "
            f"({config.max_clip_sec}s). ftea is made for short clips: "
            "split it first, or raise MAX_CLIP_SEC in config.env."
        )
    clips = list(durations)
    if source.is_file() and not clips:
        raise PipelineError(
            f"{source.name} has no video stream that ffprobe can read."
        )
    states = {
        c: _resume_state(output_base / c.stem, c, force=force)
        for c in clips
    }
    done = [c for c in clips if states[c] == "done"]
    clashes = [c for c in clips if states[c] in ("clash", "reframe_results")]
    todo = [c for c in clips if states[c] == "todo"]
    already = tuple(c.stem for c in done)
    _print_set_aside(clashes, states)
    # Never attempted this run, so they count against the exit code: the
    # file names of files that could not be used at all, then the ids of
    # clips set aside.
    skipped = unusable + unreadable + tuple(
        c.stem for c in too_long + clashes
    )
    print(
        f"  Videos: {len(clips)}, already done: {len(done)}, "
        f"to encode: {len(todo)}"
    )
    _print_done_notes(done, config, output_base)
    if not todo:
        print("  Nothing to encode.")
        return RunResult(already_done=already, not_attempted=skipped)
    _startup_checks(config)

    started_at = datetime.now(timezone.utc).isoformat()
    batch_start = time.time()
    _print_phase(2, "REFRAME + ENCODE" if config.reframe_enabled else "ENCODE")
    reports = _encode_pass(todo, config, output_base, durations)
    reached = {r.video_id for r in reports}
    # Set aside at pre-scan or resume, plus any clip a halt never reached.
    not_attempted = skipped + tuple(
        c.stem for c in todo if c.stem not in reached
    )
    halted = config.batch_on_error == "stop" and any(
        r.status == "failed" for r in reports
    )

    _print_phase(3, "RETRY")
    if halted:
        # A halt is final: "stop" means stop, and resume makes the rerun
        # cheap, because finished clips are skipped.
        print("  Skipped: BATCH_ON_ERROR=stop halted the batch.")
        retry_summary: dict = {}
    else:
        retry_summary = _retry_passes(
            reports, {c.stem: c for c in todo}, config, output_base,
            durations,
        )
    _manage_failure_dirs(paths.work_dir(), config.max_kept_failures)

    _print_phase(4, "REPORT")
    wall = time.time() - batch_start
    batch = generate_batch_report(
        reports, started_at, datetime.now(timezone.utc).isoformat(), wall,
        retry_summary, not_attempted,
    )
    batch_path = output_base / BATCH_REPORT
    save_batch_report(batch, batch_path)
    result = RunResult(
        outputs=tuple(
            output_base / r.video_id for r in reports if r.status != "failed"
        ),
        failed=tuple(r.video_id for r in reports if r.status == "failed"),
        already_done=already,
        batch_report=batch_path,
        not_attempted=not_attempted,
    )
    _summarise(reports, result, wall)
    return result


def run(
    input_path: Optional[Path] = None,
    *,
    reframe: Optional[bool] = None,
    force: bool = False,
) -> RunResult:
    """Encode every new clip in input_path (default: the input/ folder).

    input_path may be one video file or a folder; a relative path is
    read from the current folder. reframe overrides REFRAME for this run
    (None keeps config.env). force encodes clips again even when their
    output is complete. Raises PipelineError for any problem found before
    the first clip starts: settings, input, tools, the reframe
    environment, or another run holding the lock. Only ffmpeg and ffprobe
    are checked before the lock; the full machine checks wait until
    pre-scan has found something to encode.
    """
    # A clip name the console cannot show must not stop the batch.
    tolerate_unprintable_text()
    try:
        config = load_run_config(reframe)
        output_base = paths.output_dir()
        source = Path(input_path) if input_path is not None else paths.input_dir()
    except (paths.RepoRootError, paths.DataRootError) as exc:
        raise PipelineError(str(exc)) from None

    _print_phase(1, "PRE-SCAN")
    candidates, unusable = _resolve_input(source)
    if not candidates:
        if unusable:
            return RunResult(not_attempted=unusable)
        print(
            f"  No videos found in {source} (looked for "
            f"{', '.join(sorted(VIDEO_EXTENSIONS))})."
        )
        return RunResult()
    _require_programs()
    _acquire_lock()
    try:
        return _run_locked(
            source, candidates, config, output_base, force, unusable,
        )
    finally:
        _release_lock()
        paths.progress_file().unlink(missing_ok=True)


# --- Command line ---------------------------------------------------------------

USAGE = (
    "usage: python -m ftea.encode.main [--reframe | --no-reframe] "
    "[video-or-folder]"
)


def _parse_argv(argv: Sequence[str]) -> tuple[Optional[str], Optional[bool]]:
    """Split the arguments into (input path or None, reframe override).

    The override is None when neither flag is given, and the last flag
    wins. `--` ends option parsing, so an oddly named folder still works.
    """
    reframe: Optional[bool] = None
    positional: list[str] = []
    args = list(argv)
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--":
            positional.extend(args[index + 1:])
            break
        if token == "--reframe":
            reframe = True
        elif token == "--no-reframe":
            reframe = False
        elif token.startswith("-") and token != "-":
            raise UsageError(f"unknown option: {token}")
        else:
            positional.append(token)
        index += 1
    if len(positional) > 1:
        raise UsageError("give at most one video file or folder")
    return (positional[0] if positional else None), reframe


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run from a shell. Returns 0 (ok); 1 when a clip failed, a file was
    not attempted (unreadable, an unusable name, too long, a name clash,
    or never reached after a BATCH_ON_ERROR=stop halt) or the run could
    not start; or 2 when the arguments were not understood."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args in (["-h"], ["--help"]):
        print(USAGE)
        return 0
    try:
        input_arg, reframe = _parse_argv(args)
    except UsageError as exc:
        print(f"ftea: {exc}", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    try:
        result = run(None if input_arg is None else Path(input_arg),
                     reframe=reframe)
    except PipelineError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
