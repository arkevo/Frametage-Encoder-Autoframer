"""Tests for ftea.encode.main, the batch orchestrator."""

from __future__ import annotations

import dataclasses
import io
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import ftea.encode.main as m
from ftea import paths
from ftea.encode.config import PipelineConfig
from ftea.encode.geometry import Geometry
from ftea.encode.report import ClipReport
from ftea.encode.scene import SceneChunk, SourceAnalysis
from ftea.encode.tools import ToolError, ToolTimeout


def _report(video_id: str = "clip", status: str = "passed") -> ClipReport:
    return ClipReport(
        video_id=video_id, source=f"{video_id}.mp4", orientation="portrait",
        fps=24.0, is_bw=False, film_grain=0, has_audio=True, scene_count=1,
        duration_s=5.0, encode_time_sec=1.0, scenes=[], variants=[],
        status=status,
    )


def _finished_child() -> subprocess.Popen:
    """A process that has already exited, with its handle still held.

    Keep the returned object alive while checking its pid. On Windows,
    os.kill(pid, 0) then reports the dead process as running, which is
    exactly the stale-lock bug these tests pin. Without the handle, the
    call raises instead, and the bug would hide.
    """
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child


def _write_config(home: Path, body: str) -> None:
    (home / "config.env").write_text(body, encoding="utf-8")


def _touch(folder: Path, *names: str) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    made = []
    for name in names:
        (folder / name).write_bytes(b"x")
        made.append(folder / name)
    return made


def _finished_folder(
    output: Path, video_id: str, status: str = "passed", master: bool = True,
    **fields,
) -> Path:
    folder = output / video_id
    folder.mkdir(parents=True)
    report = dataclasses.replace(_report(video_id, status), **fields)
    (folder / "report.json").write_text(
        json.dumps(dataclasses.asdict(report)), encoding="utf-8",
    )
    if master:
        (folder / "master.m3u8").write_text("#EXTM3U\n", encoding="utf-8")
    return folder


def _reframe_results(output: Path, name: str) -> Path:
    """A folder as `ftea reframe` leaves it."""
    folder = output / name
    folder.mkdir(parents=True)
    (folder / "ftea-reframe.json").write_text(
        json.dumps({"schema": 1, "source": "/videos/talk.mp4"}),
        encoding="utf-8",
    )
    (folder / "vertical.mp4").write_bytes(b"v")
    (folder / "crops.json").write_text("{}", encoding="utf-8")
    return folder


@pytest.fixture(autouse=True)
def _plain_geometry(monkeypatch):
    """Every clip is an exact 9:16 frame unless a test says otherwise."""
    monkeypatch.setattr(m, "probe_geometry", lambda source: Geometry(1080, 1920))


# --- Settings ---

class TestLoadRunConfig:
    def test_should_use_the_defaults_and_say_so_without_config_env(
        self, capsys,
    ):
        assert m.load_run_config() == PipelineConfig()
        assert "No config.env at" in capsys.readouterr().out

    def test_should_warn_about_a_setting_it_does_not_know(
        self, ftea_home, capsys,
    ):
        _write_config(ftea_home, "VMAF_TARGET_108=95\n")
        m.load_run_config()
        assert "WARNING: config.env setting VMAF_TARGET_108" in (
            capsys.readouterr().out
        )

    def test_should_refuse_an_invalid_value(self, ftea_home):
        _write_config(ftea_home, "BATCH_ON_ERROR=halt\n")
        with pytest.raises(m.PipelineError, match="BATCH_ON_ERROR"):
            m.load_run_config()

    def test_should_refuse_a_file_that_is_not_utf8(self, ftea_home):
        (ftea_home / "config.env").write_text(
            "REFRAME=false\n", encoding="utf-16",
        )
        with pytest.raises(m.PipelineError, match="UTF-8"):
            m.load_run_config()

    def test_should_let_the_caller_override_reframing(self, ftea_home):
        _write_config(ftea_home, "REFRAME=true\n")
        assert m.load_run_config(reframe=False).reframe_enabled is False


# --- The run lock ---

class TestPidAlive:
    def test_should_see_this_process_as_running(self):
        assert m._pid_alive(os.getpid()) is True

    def test_should_see_a_finished_process_as_gone(self):
        """On Windows os.kill(pid, 0) can "succeed" for a dead process.

        Signal 0 is CTRL_C_EVENT there, so a stale lock could stay forever.
        """
        child = _finished_child()
        assert m._pid_alive(child.pid) is False

    @pytest.mark.parametrize("pid", [0, -1])
    def test_should_treat_impossible_ids_as_gone(self, pid):
        assert m._pid_alive(pid) is False


class TestLock:
    def test_should_create_the_lock_with_this_pid_and_remove_it(self):
        m._acquire_lock()
        lock = paths.lock_file()
        assert lock.read_text(encoding="utf-8") == str(os.getpid())
        m._release_lock()
        assert not lock.exists()

    def test_should_create_the_work_folder_when_needed(self):
        assert not paths.work_dir().exists()
        m._acquire_lock()
        assert paths.lock_file().exists()
        m._release_lock()

    def test_should_refuse_when_a_running_process_holds_the_lock(self):
        lock = paths.lock_file()
        lock.parent.mkdir(parents=True)
        holder = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
        )
        try:
            lock.write_text(str(holder.pid), encoding="utf-8")
            with pytest.raises(m.PipelineError, match=str(holder.pid)):
                m._acquire_lock()
        finally:
            holder.kill()
            holder.wait()

    def test_should_take_over_a_lock_left_by_a_finished_process(
        self, capsys,
    ):
        lock = paths.lock_file()
        lock.parent.mkdir(parents=True)
        child = _finished_child()
        lock.write_text(str(child.pid), encoding="utf-8")

        m._acquire_lock()

        assert lock.read_text(encoding="utf-8") == str(os.getpid())
        assert "Removing stale lock" in capsys.readouterr().out
        m._release_lock()

    def test_should_take_over_an_old_lock_without_a_readable_pid(self):
        lock = paths.lock_file()
        lock.parent.mkdir(parents=True)
        lock.write_text("not a pid", encoding="utf-8")
        old = time.time() - m.LOCK_GRACE_SEC - 5
        os.utime(lock, (old, old))

        m._acquire_lock()

        assert lock.read_text(encoding="utf-8") == str(os.getpid())
        m._release_lock()

    @pytest.mark.parametrize("content", ["", "not a pid"])
    def test_should_leave_a_new_lock_without_a_pid_to_its_run(self, content):
        """A run creates the lock first and writes its id just after. A
        second run starting in between used to delete the live lock (or,
        on Windows, crash trying to)."""
        lock = paths.lock_file()
        lock.parent.mkdir(parents=True)
        lock.write_text(content, encoding="utf-8")

        with pytest.raises(m.PipelineError, match="starting"):
            m._acquire_lock()
        assert lock.read_text(encoding="utf-8") == content

    def test_should_stop_cleanly_when_a_stale_lock_cannot_be_removed(
        self, monkeypatch,
    ):
        lock = paths.lock_file()
        lock.parent.mkdir(parents=True)
        child = _finished_child()
        lock.write_text(str(child.pid), encoding="utf-8")

        def in_use(self, missing_ok=False):
            raise PermissionError("[WinError 32] in use")

        monkeypatch.setattr(m.Path, "unlink", in_use)
        with pytest.raises(m.PipelineError, match="could not remove"):
            m._acquire_lock()

    def test_should_leave_another_process_lock_alone_on_release(self):
        lock = paths.lock_file()
        lock.parent.mkdir(parents=True)
        lock.write_text("424242", encoding="utf-8")

        m._release_lock()

        assert lock.exists()


# --- Locating the clips ---

class TestCandidateFiles:
    def test_should_take_a_named_file_whatever_its_extension(self, tmp_path):
        clip, = _touch(tmp_path, "clip.webm")
        assert m._candidate_files(clip) == [clip]

    def test_should_pick_videos_from_a_folder_in_name_order(self, tmp_path):
        _touch(tmp_path, "b.MOV", "a.mp4", "notes.txt", "c.mkv", "d.avi")
        names = [p.name for p in m._candidate_files(tmp_path)]
        assert names == ["a.mp4", "b.MOV", "c.mkv", "d.avi"]

    def test_should_skip_a_second_file_with_the_same_name(
        self, tmp_path, capsys,
    ):
        _touch(tmp_path, "clip.mp4", "Clip.MOV")
        assert [p.name for p in m._candidate_files(tmp_path)] == ["Clip.MOV"]
        assert "same name as Clip.MOV" in capsys.readouterr().out

    def test_should_return_nothing_for_an_empty_folder(self, tmp_path):
        assert m._candidate_files(tmp_path) == []

    @pytest.mark.parametrize("video_id", ["", " ", ".", ".."])
    def test_should_reject_ids_that_are_not_folder_names(self, video_id):
        assert m._is_safe_clip_id(video_id) is False

    def test_should_skip_a_file_whose_name_is_the_output_folder(self, capsys):
        # A file whose stem is "." would write output/./, which is output/
        # itself. Up to Python 3.13 "..mp4" has that stem; 3.14 sees no
        # suffix there, so a stand-in carries the stem on every version.
        dot_file = SimpleNamespace(stem=".", name="..mp4")
        assert m._drop_unusable([dot_file]) == []
        assert "unusable file name" in capsys.readouterr().out


def _ffprobe_answer(streams: list[str], duration: str):
    """A fake run_tool that answers like ffprobe's JSON output."""
    out = json.dumps({
        "streams": [{"codec_type": kind} for kind in streams],
        "format": {"duration": duration},
    })
    return lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, out, "")


class TestPreScan:
    def test_should_keep_files_with_a_video_stream_and_their_length(
        self, monkeypatch, tmp_path,
    ):
        good, bad = _touch(tmp_path, "good.mp4", "bad.mp4")
        monkeypatch.setattr(
            m, "_video_duration", lambda p: 12.5 if p == good else None,
        )
        assert m._pre_scan([good, bad]) == ({good: 12.5}, [])

    def test_should_skip_a_file_ffprobe_cannot_read(
        self, monkeypatch, tmp_path, capsys,
    ):
        clip, = _touch(tmp_path, "slow.mp4")

        def hang(path):
            raise ToolError("ffprobe did not finish within 60s")

        monkeypatch.setattr(m, "_video_duration", hang)
        assert m._pre_scan([clip]) == ({}, [])
        assert "did not finish" in capsys.readouterr().out

    def test_should_set_aside_a_video_longer_than_the_limit(
        self, monkeypatch, tmp_path, capsys,
    ):
        film, short = _touch(tmp_path, "film.mp4", "short.mp4")
        monkeypatch.setattr(
            m, "_video_duration", lambda p: 723.0 if p == film else 30.0,
        )

        assert m._pre_scan([film, short], max_clip_sec=600) == (
            {short: 30.0}, [film],
        )
        assert (
            "SKIP (12:03 long; ftea is made for short clips"
            in capsys.readouterr().out
        )

    def test_should_not_limit_the_length_when_the_limit_is_zero(
        self, monkeypatch, tmp_path,
    ):
        film, = _touch(tmp_path, "film.mp4")
        monkeypatch.setattr(m, "_video_duration", lambda p: 7200.0)
        assert m._pre_scan([film], max_clip_sec=0) == ({film: 7200.0}, [])

    def test_should_read_stream_and_length_in_one_ffprobe_call(
        self, monkeypatch, tmp_path,
    ):
        seen = []
        answer = _ffprobe_answer(["video"], "84.2")

        def fake(cmd, **kwargs):
            seen.append((cmd, kwargs))
            return answer(cmd, **kwargs)

        monkeypatch.setattr(m, "run_tool", fake)
        assert m._video_duration(tmp_path / "c.mp4") == 84.2
        cmd, kwargs = seen[0]
        assert len(seen) == 1 and cmd[0] == "ffprobe"
        assert kwargs["timeout"] == m.FFPROBE_TIMEOUT_SEC

    def test_should_say_none_for_a_file_without_video(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setattr(m, "run_tool", _ffprobe_answer(["audio"], "3.0"))
        assert m._video_duration(tmp_path / "a.m4a") is None

    def test_should_give_ffprobe_s_reason_when_it_fails(
        self, monkeypatch, tmp_path,
    ):
        """A failed ffprobe used to read as "no video stream", and its
        own message (permission denied, a corrupt header) was lost."""
        real_run_tool = m.run_tool
        failing = (
            "import sys; sys.stderr.write('c.mp4: Permission denied\\n'); "
            "sys.exit(1)"
        )

        def ffprobe_failing(cmd, **kwargs):
            return real_run_tool([sys.executable, "-c", failing], **kwargs)

        monkeypatch.setattr(m, "run_tool", ffprobe_failing)
        with pytest.raises(ToolError, match="Permission denied"):
            m._video_duration(tmp_path / "c.mp4")


# --- Resume ---

class TestResume:
    @pytest.mark.parametrize("status", ["passed", "warning"])
    def test_should_count_a_passing_report_beside_a_playlist_as_done(
        self, tmp_path, status,
    ):
        assert m._is_complete(_finished_folder(tmp_path, "a", status)) is True

    def test_should_encode_a_failed_clip_again(self, tmp_path):
        assert m._is_complete(_finished_folder(tmp_path, "a", "failed")) is False

    def test_should_encode_again_without_a_master_playlist(self, tmp_path):
        folder = _finished_folder(tmp_path, "a", master=False)
        assert m._is_complete(folder) is False

    def test_should_encode_again_when_the_report_is_missing(self, tmp_path):
        folder = tmp_path / "a"
        folder.mkdir()
        (folder / "master.m3u8").write_text("#EXTM3U\n", encoding="utf-8")
        assert m._is_complete(folder) is False

    @pytest.mark.parametrize("text", ["{not json", "42", '{"video_id": "a"}'])
    def test_should_treat_an_unusable_report_as_missing(self, tmp_path, text):
        path = tmp_path / "report.json"
        path.write_text(text, encoding="utf-8")
        assert m._load_existing_report(path) is None

    def test_should_ignore_keys_it_does_not_know(self, tmp_path):
        path = tmp_path / "report.json"
        data = {**dataclasses.asdict(_report("a")), "from_a_later_version": 1}
        path.write_text(json.dumps(data), encoding="utf-8")
        assert m._load_existing_report(path).video_id == "a"

    def test_should_match_a_report_made_from_this_very_file(self, tmp_path):
        clip, = _touch(tmp_path, "a.mp4")
        size, digest = m._source_fingerprint(clip)
        report = dataclasses.replace(
            _report("a"), source_size=size, source_digest=digest,
        )
        assert m._same_source(report, clip) is True

    def test_should_see_a_different_file_when_the_size_changed(
        self, tmp_path,
    ):
        clip, = _touch(tmp_path, "a.mp4")
        size, digest = m._source_fingerprint(clip)
        report = dataclasses.replace(
            _report("a"), source_size=size, source_digest=digest,
        )
        clip.write_bytes(b"a longer, different video")
        assert m._same_source(report, clip) is False

    def test_should_count_a_report_without_a_fingerprint_as_a_match(
        self, tmp_path,
    ):
        clip, = _touch(tmp_path, "a.mp4")
        assert m._same_source(_report("a"), clip) is True

    @staticmethod
    def _fingerprinted(clip: Path, data: bytes) -> ClipReport:
        clip.write_bytes(data)
        size, digest = m._source_fingerprint(clip)
        return dataclasses.replace(
            _report(clip.stem), source_size=size, source_digest=digest,
        )

    def test_should_see_a_different_file_of_the_same_size(self, tmp_path):
        clip = tmp_path / "a.mp4"
        report = self._fingerprinted(clip, b"a" * 100)
        clip.write_bytes(b"b" * 100)
        assert m._same_source(report, clip) is False

    def test_should_see_a_change_only_in_the_last_mib_of_a_large_file(
        self, tmp_path,
    ):
        """Two clips from one camera can share size and opening bytes."""
        clip = tmp_path / "a.mp4"
        data = bytearray(b"x" * (3 * m.FINGERPRINT_CHUNK))
        report = self._fingerprinted(clip, bytes(data))
        data[-1:] = b"y"
        clip.write_bytes(bytes(data))
        assert m._same_source(report, clip) is False

    def test_should_ignore_a_change_between_the_first_and_last_mib(
        self, tmp_path,
    ):
        """What the fingerprint reads, and no more: it stays fast."""
        clip = tmp_path / "a.mp4"
        data = bytearray(b"x" * (3 * m.FINGERPRINT_CHUNK))
        report = self._fingerprinted(clip, bytes(data))
        middle = int(1.5 * m.FINGERPRINT_CHUNK)
        data[middle:middle + 1] = b"y"
        clip.write_bytes(bytes(data))
        assert m._same_source(report, clip) is True

    def test_should_give_a_mid_sized_file_the_same_fingerprint_twice(
        self, tmp_path,
    ):
        clip = tmp_path / "a.mp4"
        clip.write_bytes(bytes(range(256)) * (6 * 1024))  # 1.5 MiB
        assert m._source_fingerprint(clip) == m._source_fingerprint(clip)


# --- Progress, failure folders, output folders ---

class TestProgress:
    def test_should_write_progress_for_monitoring(self):
        m._update_progress(
            total=10, completed=5, failed=1, current="clip_6.mp4",
            started=time.time() - 100,
            failures=[{"clip": "clip_3.mp4", "error": "timeout"}],
        )
        data = json.loads(paths.progress_file().read_text(encoding="utf-8"))
        assert (data["total"], data["completed"], data["failed"]) == (10, 5, 1)
        assert data["remaining"] == 4
        assert data["current_clip"] == "clip_6.mp4"

    def test_should_keep_only_the_last_five_failures(self):
        failures = [{"clip": f"clip_{i}.mp4", "error": "e"} for i in range(10)]
        m._update_progress(20, 10, 10, "clip_11.mp4", time.time(), failures)
        data = json.loads(paths.progress_file().read_text(encoding="utf-8"))
        assert [f["clip"] for f in data["failures"]] == [
            f"clip_{i}.mp4" for i in range(5, 10)
        ]


class TestFailureFolders:
    def test_should_remove_the_oldest_beyond_the_limit(self, tmp_path):
        for i in range(7):
            folder = tmp_path / f"failure_{i}"
            folder.mkdir()
            stamp = time.time() - (7 - i)
            os.utime(folder, (stamp, stamp))

        m._manage_failure_dirs(tmp_path, max_kept=5)

        assert sorted(p.name for p in tmp_path.iterdir()) == [
            f"failure_{i}" for i in range(2, 7)
        ]

    def test_should_leave_the_vmaf_log_folder_alone(self, tmp_path):
        (tmp_path / ".vmaf").mkdir()
        for i in range(3):
            (tmp_path / f"f{i}").mkdir()

        m._manage_failure_dirs(tmp_path, max_kept=0)

        assert [p.name for p in tmp_path.iterdir()] == [".vmaf"]

    def test_should_accept_a_missing_work_folder(self, tmp_path):
        m._manage_failure_dirs(tmp_path / "absent", max_kept=5)


class TestPrepareOutputDir:
    def test_should_empty_what_a_previous_attempt_left(self, tmp_path):
        stale = tmp_path / "clip" / "av1_1080p" / "stale.m4s"
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"x")

        folder, previous = m._prepare_output_dir(tmp_path, "clip")

        assert folder == tmp_path / "clip"
        assert list(folder.iterdir()) == []
        assert previous is None

    def test_should_refuse_an_id_that_is_not_a_folder_name(self, tmp_path):
        with pytest.raises(ValueError):
            m._prepare_output_dir(tmp_path, "..")
        assert tmp_path.exists()

    def test_should_keep_a_finished_package_when_the_attempt_fails(
        self, tmp_path,
    ):
        _finished_folder(tmp_path, "clip")

        folder, previous = m._prepare_output_dir(tmp_path, "clip")
        assert list(folder.iterdir()) == []
        assert previous == tmp_path / ".clip.previous"
        (folder / "half.m4s").write_bytes(b"x")
        m._settle_previous(folder, previous, succeeded=False)

        assert m._is_complete(folder)
        assert not (folder / "half.m4s").exists()
        assert not previous.exists()

    def test_should_never_empty_a_folder_of_ftea_reframe_results(
        self, tmp_path,
    ):
        results = _reframe_results(tmp_path, "talk-vertical")

        with pytest.raises(RuntimeError, match="ftea reframe"):
            m._prepare_output_dir(tmp_path, "talk-vertical")

        assert sorted(p.name for p in results.iterdir()) == [
            "crops.json", "ftea-reframe.json", "vertical.mp4",
        ]

    def test_should_remove_the_finished_package_on_success(self, tmp_path):
        _finished_folder(tmp_path, "clip")

        folder, previous = m._prepare_output_dir(tmp_path, "clip")
        (folder / "master.m3u8").write_text("#EXTM3U\n", encoding="utf-8")
        m._settle_previous(folder, previous, succeeded=True)

        assert not previous.exists()
        assert [p.name for p in folder.iterdir()] == ["master.m3u8"]

    def test_should_refuse_at_once_while_the_package_is_in_use(
        self, tmp_path, monkeypatch,
    ):
        _finished_folder(tmp_path, "clip")

        def in_use(self, target):
            raise PermissionError("[WinError 32] in use")

        monkeypatch.setattr(m.Path, "rename", in_use)
        with pytest.raises(m.OutputInUseError, match="Close what is using"):
            m._prepare_output_dir(tmp_path, "clip")
        assert m._is_complete(tmp_path / "clip")

    def test_should_keep_a_package_an_interrupted_run_left_aside(
        self, tmp_path,
    ):
        """A run killed between setting the package aside and putting it
        back leaves the last good copy aside, and a partial clip/."""
        _finished_folder(tmp_path, ".clip.previous")
        _touch(tmp_path / "clip", "half.m4s")

        folder, previous = m._prepare_output_dir(tmp_path, "clip")

        assert previous == tmp_path / ".clip.previous"
        assert m._is_complete(previous)
        assert list(folder.iterdir()) == []
        m._settle_previous(folder, previous, succeeded=False)
        assert m._is_complete(folder) and not previous.exists()

    def test_should_restore_that_package_when_the_next_attempt_fails(
        self, tmp_path, stages,
    ):
        stages.fail.add("c")
        clip, = _touch(tmp_path, "c.mp4")
        output = tmp_path / "o"
        _finished_folder(output, ".c.previous")
        _touch(output / "c", "half.m4s")

        report = m._encode_single_clip(
            clip, PipelineConfig(reframe_enabled=False), output, timeout=60,
        )

        assert report.status == "failed"
        assert m._is_complete(output / "c")
        assert not (output / "c" / "half.m4s").exists()
        assert not (output / ".c.previous").exists()
        saved = json.loads(
            (paths.work_dir() / "c" / "report.json").read_text(
                encoding="utf-8",
            ),
        )
        assert saved["status"] == "failed"

    @pytest.mark.parametrize("succeeded", [True, False])
    def test_should_only_warn_when_the_aside_package_cannot_be_settled(
        self, tmp_path, monkeypatch, capsys, succeeded,
    ):
        """A locked folder must not end the whole batch."""
        _finished_folder(tmp_path, "clip")
        folder, previous = m._prepare_output_dir(tmp_path, "clip")

        def locked(*args, **kwargs):
            raise PermissionError("[WinError 32] in use")

        with monkeypatch.context() as patch:
            if succeeded:
                patch.setattr(m.shutil, "rmtree", locked)
            else:
                patch.setattr(m.Path, "rename", locked)
            m._settle_previous(folder, previous, succeeded=succeeded)

        assert "could not tidy" in capsys.readouterr().out
        assert m._is_complete(previous)


# --- The reframe stage ---

def _completed(returncode: int, payload=None, stderr: str = ""):
    """A fake CompletedProcess for the reframe subprocess."""
    stdout = "" if payload is None else json.dumps(payload)
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


_SKIP_PAYLOAD = {
    "schema": 1, "status": "skipped", "output": None, "crops": None,
    "report": {"applied": False, "reason": "already_vertical",
               "elapsed_sec": 0.0, "source": {}},
    "error": None,
}


def _reframed_payload(output: Path, crops=None) -> dict:
    return {
        "schema": 1, "status": "reframed", "output": str(output),
        "crops": None if crops is None else str(crops),
        "report": {"applied": True, "reason": "landscape"}, "error": None,
    }


def _fake_exe(tmp_path: Path) -> Path:
    exe = tmp_path / "fake-python"
    exe.write_text("", encoding="utf-8")
    return exe


def _run_tool_returning(monkeypatch, completed, calls=None):
    def fake(cmd, **kwargs):
        if calls is not None:
            calls.append((cmd, kwargs))
        return completed

    monkeypatch.setattr(m, "run_tool", fake)


def _refuse_path(value: str):
    """paths.resolve_setting_path refusing a value, as on Windows."""
    raise ValueError(f"{value!r} is not a full Windows path")


class TestResolveReframePython:
    def test_should_use_the_configured_interpreter(self, tmp_path):
        exe = _fake_exe(tmp_path)
        config = PipelineConfig(reframe_python=str(exe))
        assert m._resolve_reframe_python(config) == exe

    def test_should_read_a_relative_setting_from_the_data_folder(
        self, ftea_home,
    ):
        exe = ftea_home / "tools" / "python.exe"
        exe.parent.mkdir()
        exe.write_text("", encoding="utf-8")
        config = PipelineConfig(reframe_python="tools/python.exe")
        assert m._resolve_reframe_python(config).resolve() == exe.resolve()

    def test_should_name_the_setting_when_it_points_nowhere(self, tmp_path):
        config = PipelineConfig(
            reframe_python=str(tmp_path / "missing" / "python.exe"),
        )
        with pytest.raises(RuntimeError, match="REFRAME_PYTHON"):
            m._resolve_reframe_python(config)

    def test_should_name_the_setting_when_its_path_is_refused(
        self, monkeypatch,
    ):
        monkeypatch.setattr(paths, "resolve_setting_path", _refuse_path)
        config = PipelineConfig(reframe_python="/c/venv/python.exe")
        with pytest.raises(RuntimeError, match="^REFRAME_PYTHON: .*full"):
            m._resolve_reframe_python(config)

    @pytest.mark.skipif(os.name != "nt", reason="a Windows path rule")
    def test_should_name_the_setting_for_a_git_bash_path_on_windows(self):
        config = PipelineConfig(reframe_python="/c/venv/python.exe")
        with pytest.raises(RuntimeError, match="^REFRAME_PYTHON: "):
            m._resolve_reframe_python(config)

    def test_should_say_how_to_fix_it_when_nothing_is_installed(
        self, monkeypatch, tmp_path,
    ):
        """Never fall back to the encoder's own Python, which has no torch."""
        monkeypatch.setattr(
            paths, "reframe_venv_dir", lambda: tmp_path / ".venv-reframe",
        )
        with pytest.raises(RuntimeError) as info:
            m._resolve_reframe_python(PipelineConfig())
        assert "install.sh" in str(info.value)
        # `ftea reframe` shares this message and has no --no-reframe;
        # `ftea run` adds it (see TestStartupChecks).
        assert "--no-reframe" not in str(info.value)

    @pytest.mark.parametrize(
        "relative", [("Scripts", "python.exe"), ("bin", "python")],
    )
    def test_should_find_the_interpreter_in_the_reframe_venv(
        self, monkeypatch, tmp_path, relative,
    ):
        venv = tmp_path / ".venv-reframe"
        exe = venv.joinpath(*relative)
        exe.parent.mkdir(parents=True)
        exe.write_text("", encoding="utf-8")
        monkeypatch.setattr(paths, "reframe_venv_dir", lambda: venv)
        assert m._resolve_reframe_python(PipelineConfig()) == exe


class TestVerifyReframeEnv:
    def test_should_run_the_stage_check_from_the_checkout(
        self, monkeypatch, tmp_path,
    ):
        calls = []
        _run_tool_returning(
            monkeypatch,
            subprocess.CompletedProcess([], 0, "all checks passed\n", ""),
            calls,
        )

        m._verify_reframe_env(tmp_path / "python.exe")

        cmd, kwargs = calls[0]
        assert cmd == [
            str(tmp_path / "python.exe"), "-m", paths.VERIFY_ENV_MODULE,
        ]
        assert kwargs["cwd"] == paths.source_root()
        assert kwargs["env"]["PYTHONPATH"] == str(paths.source_root())
        assert kwargs["check"] is False
        assert kwargs["kill_tree"] is True

    def test_should_stop_the_run_with_the_failing_checks(
        self, monkeypatch, tmp_path,
    ):
        output = "".join(f"  ok check {i}\n" for i in range(20))
        output += "  x face model: missing\n"
        _run_tool_returning(
            monkeypatch, subprocess.CompletedProcess([], 1, output, ""),
        )

        with pytest.raises(m.PipelineError) as info:
            m._verify_reframe_env(tmp_path / "python.exe")

        message = str(info.value)
        assert "face model: missing" in message
        assert "ok check 0" not in message
        assert "install.sh" in message and "--no-reframe" in message

    def test_should_stop_the_run_when_the_check_cannot_start(
        self, monkeypatch, tmp_path,
    ):
        def broken(cmd, **kwargs):
            raise ToolError("python.exe could not be started: no such file")

        monkeypatch.setattr(m, "run_tool", broken)
        with pytest.raises(m.PipelineError, match="could not run"):
            m._verify_reframe_env(tmp_path / "python.exe")


class TestRunReframe:
    @pytest.fixture
    def config(self, tmp_path):
        return PipelineConfig(
            reframe_python=str(_fake_exe(tmp_path)), reframe_timeout=777,
        )

    def test_should_return_the_original_clip_when_skipped(
        self, monkeypatch, tmp_path, config,
    ):
        clip, = _touch(tmp_path, "v.mp4")
        _run_tool_returning(monkeypatch, _completed(2, _SKIP_PAYLOAD))

        outcome = m._run_reframe(clip, config, tmp_path / "w")

        assert outcome.source == clip
        assert outcome.entry["applied"] is False
        assert outcome.crops is None

    def test_should_return_the_intermediate_when_reframed(
        self, monkeypatch, tmp_path, config,
    ):
        clip, vertical = _touch(tmp_path, "l.mp4", "vertical.mp4")
        crops = tmp_path / "crops.json"
        _run_tool_returning(
            monkeypatch, _completed(0, _reframed_payload(vertical, crops)),
        )
        monkeypatch.setattr(m, "detect_has_audio", lambda path: False)

        outcome = m._run_reframe(clip, config, tmp_path / "w")

        assert outcome.source == vertical
        assert outcome.entry["applied"] is True
        assert outcome.crops == crops

    def test_should_launch_the_stage_as_the_path_contract_says(
        self, monkeypatch, tmp_path, config,
    ):
        clip, = _touch(tmp_path, "v.mp4")
        calls = []
        _run_tool_returning(monkeypatch, _completed(2, _SKIP_PAYLOAD), calls)

        m._run_reframe(clip, config, tmp_path / "w")

        cmd, kwargs = calls[0]
        assert cmd[:3] == [config.reframe_python, "-m", paths.REFRAME_MODULE]
        assert cmd[cmd.index("--source") + 1] == str(clip.resolve())
        assert Path(cmd[cmd.index("--work-dir") + 1]).is_absolute()
        assert cmd[cmd.index("--config") + 1] == str(paths.config_path())
        assert kwargs["cwd"] == paths.source_root()
        assert kwargs["env"]["PYTHONPATH"] == str(paths.source_root())
        assert kwargs["env"]["PYTHONIOENCODING"] == "utf-8"
        assert kwargs["timeout"] == 777
        assert kwargs["check"] is False
        # The stage runs ffmpeg itself; a timeout must stop those too.
        assert kwargs["kill_tree"] is True

    def test_should_raise_when_the_stage_runs_out_of_time(
        self, monkeypatch, tmp_path,
    ):
        clip, = _touch(tmp_path, "v.mp4")
        config = PipelineConfig(
            reframe_python=str(_fake_exe(tmp_path)), reframe_timeout=42,
        )

        def slow(cmd, **kwargs):
            raise ToolTimeout("fake-python did not finish within 42s")

        monkeypatch.setattr(m, "run_tool", slow)
        with pytest.raises(RuntimeError, match="reframe timed out after 42s"):
            m._run_reframe(clip, config, tmp_path / "w")

    def test_should_raise_when_the_interpreter_cannot_start(
        self, monkeypatch, tmp_path, config,
    ):
        clip, = _touch(tmp_path, "v.mp4")

        def broken(cmd, **kwargs):
            raise ToolError("fake-python could not be started: denied")

        monkeypatch.setattr(m, "run_tool", broken)
        with pytest.raises(RuntimeError, match="could not be started"):
            m._run_reframe(clip, config, tmp_path / "w")

    def test_should_show_stderr_when_stdout_is_not_json(
        self, monkeypatch, tmp_path, config,
    ):
        clip, = _touch(tmp_path, "v.mp4")
        _run_tool_returning(monkeypatch, _completed(
            1, None, stderr="ImportError: No module named 'torch'",
        ))
        with pytest.raises(RuntimeError, match="No module named"):
            m._run_reframe(clip, config, tmp_path / "w")

    def test_should_keep_the_stage_diagnostic_when_it_fails(
        self, monkeypatch, tmp_path, config,
    ):
        clip, = _touch(tmp_path, "v.mp4")
        payload = {
            "schema": 1, "status": "failed", "output": None, "crops": None,
            "report": {"applied": False, "reason": "landscape"},
            "error": "ReframeError: detector exploded",
        }
        _run_tool_returning(monkeypatch, _completed(1, payload))

        with pytest.raises(m.ReframeStageError, match="detector exploded") as info:
            m._run_reframe(clip, config, tmp_path / "w")

        assert info.value.entry["reason"] == "landscape"

    def test_should_refuse_an_exit_code_that_contradicts_the_status(
        self, monkeypatch, tmp_path, config,
    ):
        clip, = _touch(tmp_path, "v.mp4")
        _run_tool_returning(monkeypatch, _completed(0, _SKIP_PAYLOAD))
        with pytest.raises(RuntimeError, match="exited 0 but reported"):
            m._run_reframe(clip, config, tmp_path / "w")

    def test_should_refuse_a_result_of_another_schema(
        self, monkeypatch, tmp_path, config,
    ):
        """`ftea reframe` refuses it; `ftea run` must not accept it."""
        clip, = _touch(tmp_path, "v.mp4")
        payload = dict(_SKIP_PAYLOAD, schema=2)
        _run_tool_returning(monkeypatch, _completed(2, payload))
        with pytest.raises(RuntimeError, match="schema 2"):
            m._run_reframe(clip, config, tmp_path / "w")

    def test_should_refuse_a_path_that_is_not_text(
        self, monkeypatch, tmp_path, config,
    ):
        clip, = _touch(tmp_path, "v.mp4")
        payload = dict(_SKIP_PAYLOAD, crops=["crops.json"])
        _run_tool_returning(monkeypatch, _completed(2, payload))
        with pytest.raises(RuntimeError, match="crops that is not text"):
            m._run_reframe(clip, config, tmp_path / "w")

    @pytest.mark.parametrize("payload", [[], "ok", 3])
    def test_should_refuse_a_result_that_is_not_an_object(
        self, monkeypatch, tmp_path, config, payload,
    ):
        clip, = _touch(tmp_path, "v.mp4")
        _run_tool_returning(monkeypatch, _completed(0, payload))
        with pytest.raises(RuntimeError, match="unexpected result"):
            m._run_reframe(clip, config, tmp_path / "w")

    @pytest.mark.parametrize("status", ["skipped", "failed"])
    @pytest.mark.parametrize("entry", [["applied"], "landscape", 3])
    def test_should_refuse_a_report_entry_that_is_not_an_object(
        self, monkeypatch, tmp_path, config, status, entry,
    ):
        """It would reach the batch report, which reads it as a mapping,
        and crash it after the whole batch had encoded."""
        clip, = _touch(tmp_path, "v.mp4")
        payload = dict(_SKIP_PAYLOAD, status=status, report=entry)
        exit_code = m.REFRAME_STATUS_EXIT[status]
        _run_tool_returning(monkeypatch, _completed(exit_code, payload))
        with pytest.raises(RuntimeError, match="report that is not an obj"):
            m._run_reframe(clip, config, tmp_path / "w")

    def test_should_raise_when_the_intermediate_is_missing(
        self, monkeypatch, tmp_path, config,
    ):
        clip, = _touch(tmp_path, "l.mp4")
        _run_tool_returning(
            monkeypatch, _completed(0, _reframed_payload(tmp_path / "gone.mp4")),
        )
        with pytest.raises(RuntimeError, match="does not exist"):
            m._run_reframe(clip, config, tmp_path / "w")

    def test_should_refuse_an_intermediate_that_lost_the_sound(
        self, monkeypatch, tmp_path, config,
    ):
        clip, vertical = _touch(tmp_path, "l.mp4", "vertical.mp4")
        _run_tool_returning(monkeypatch, _completed(0, _reframed_payload(vertical)))
        answers = iter([True, False])
        monkeypatch.setattr(m, "detect_has_audio", lambda path: next(answers))

        with pytest.raises(RuntimeError, match="silent intermediate"):
            m._run_reframe(clip, config, tmp_path / "w")

    def test_should_accept_a_silent_intermediate_from_a_silent_clip(
        self, monkeypatch, tmp_path, config,
    ):
        clip, vertical = _touch(tmp_path, "l.mp4", "vertical.mp4")
        _run_tool_returning(monkeypatch, _completed(0, _reframed_payload(vertical)))
        monkeypatch.setattr(m, "detect_has_audio", lambda path: False)

        assert m._run_reframe(clip, config, tmp_path / "w").source == vertical


class TestRetainCrops:
    @staticmethod
    def _crops(tmp_path: Path) -> Path:
        crops = tmp_path / "w" / "clip1" / "reframe" / "crops.json"
        crops.parent.mkdir(parents=True)
        crops.write_text('[{"f":0}]', encoding="utf-8")
        return crops

    def test_should_keep_the_crop_track_in_the_clip_folder(self, tmp_path):
        clip_dir = tmp_path / "out" / "clip1"
        clip_dir.mkdir(parents=True)

        m._retain_crops("clip1", self._crops(tmp_path), PipelineConfig(), clip_dir)

        assert (clip_dir / "crops.json").read_text(encoding="utf-8") == (
            '[{"f":0}]'
        )

    def test_should_use_the_configured_folder(self, tmp_path):
        dest = tmp_path / "kept"
        config = PipelineConfig(reframe_crops_dir=str(dest))

        m._retain_crops("clip1", self._crops(tmp_path), config, tmp_path / "o")

        assert (dest / "clip1.crops.json").is_file()

    def test_should_read_a_relative_folder_from_the_data_root(
        self, ftea_home, tmp_path,
    ):
        config = PipelineConfig(reframe_crops_dir="crop-tracks")

        m._retain_crops("clip1", self._crops(tmp_path), config, tmp_path / "o")

        assert (ftea_home / "crop-tracks" / "clip1.crops.json").is_file()

    def test_should_keep_nothing_when_disabled(self, tmp_path):
        clip_dir = tmp_path / "out"
        clip_dir.mkdir()
        config = PipelineConfig(reframe_keep_crops=False)

        m._retain_crops("clip1", self._crops(tmp_path), config, clip_dir)

        assert list(clip_dir.iterdir()) == []

    def test_should_do_nothing_without_a_crop_track(self, tmp_path):
        clip_dir = tmp_path / "out"
        clip_dir.mkdir()
        m._retain_crops("clip1", None, PipelineConfig(), clip_dir)
        assert list(clip_dir.iterdir()) == []

    def test_should_raise_when_the_reported_track_is_missing(self, tmp_path):
        with pytest.raises(RuntimeError, match="missing"):
            m._retain_crops(
                "clip1", tmp_path / "gone.json", PipelineConfig(), tmp_path,
            )

    def test_should_name_the_setting_when_its_path_is_refused(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setattr(paths, "resolve_setting_path", _refuse_path)
        config = PipelineConfig(reframe_crops_dir="/d/crops")
        with pytest.raises(RuntimeError, match="^REFRAME_CROPS_DIR: .*full"):
            m._retain_crops("clip1", self._crops(tmp_path), config, tmp_path)


class TestStartupChecks:
    @pytest.fixture
    def ready(self, monkeypatch):
        """A machine that passes every check; records the reframe one."""
        checked = []
        monkeypatch.setattr(m.preflight, "check_toolchain", lambda c: [])
        monkeypatch.setattr(m, "nvenc_supports_film_grain", lambda: True)
        monkeypatch.setattr(
            m, "_resolve_reframe_python", lambda config: Path("python"),
        )
        monkeypatch.setattr(m, "_verify_reframe_env", checked.append)
        return checked

    def test_should_offer_no_reframe_when_no_interpreter_is_found(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setattr(m.preflight, "check_toolchain", lambda c: [])
        monkeypatch.setattr(m, "nvenc_supports_film_grain", lambda: True)
        monkeypatch.setattr(
            paths, "reframe_venv_dir", lambda: tmp_path / ".venv-reframe",
        )
        with pytest.raises(m.PipelineError) as info:
            m._startup_checks(PipelineConfig())
        assert "install.sh" in str(info.value)
        assert "--no-reframe" in str(info.value)

    def test_should_refuse_a_crops_folder_before_any_clip(
        self, monkeypatch, ready,
    ):
        """Otherwise every clip would fail after its reframe, three times."""
        monkeypatch.setattr(paths, "resolve_setting_path", _refuse_path)
        config = PipelineConfig(reframe_crops_dir="/d/crops")

        with pytest.raises(m.PipelineError, match="^REFRAME_CROPS_DIR: "):
            m._startup_checks(config)
        assert ready == []

    @pytest.mark.parametrize("config", [
        PipelineConfig(reframe_crops_dir="/d/crops", reframe_keep_crops=False),
        PipelineConfig(reframe_crops_dir="/d/crops", reframe_enabled=False),
    ])
    def test_should_ignore_the_crops_folder_when_nothing_is_kept(
        self, monkeypatch, ready, config,
    ):
        monkeypatch.setattr(paths, "resolve_setting_path", _refuse_path)
        m._startup_checks(config)

    def test_should_say_once_when_film_grain_cannot_be_applied(
        self, monkeypatch, ready, capsys,
    ):
        monkeypatch.setattr(m, "nvenc_supports_film_grain", lambda: False)
        m._startup_checks(PipelineConfig(reframe_enabled=False))
        assert "FILM_GRAIN_BW are not applied" in capsys.readouterr().out

    @pytest.mark.parametrize(("supported", "config"), [
        (True, PipelineConfig(reframe_enabled=False)),
        (False, PipelineConfig(
            reframe_enabled=False, film_grain=0, film_grain_bw=0,
        )),
    ], ids=["supported", "no grain configured"])
    def test_should_not_warn_about_film_grain_otherwise(
        self, monkeypatch, ready, capsys, supported, config,
    ):
        monkeypatch.setattr(m, "nvenc_supports_film_grain", lambda: supported)
        m._startup_checks(config)
        assert "film_grain" not in capsys.readouterr().out.lower()


# --- One clip, and the whole run ---

class _Stages:
    """Fast fakes for every stage that needs ffmpeg or a GPU."""

    def __init__(self) -> None:
        self.encoded: list[str] = []
        self.configs: dict[str, list[PipelineConfig]] = {}
        self.fail: set[str] = set()
        self.fail_once: set[str] = set()

    def analyze(self, source, config, work_dir):
        stem = Path(source).stem
        self.encoded.append(stem)
        self.configs.setdefault(stem, []).append(config)
        if stem in self.fail:
            raise RuntimeError(f"boom {stem}")
        if stem in self.fail_once:
            self.fail_once.discard(stem)
            raise RuntimeError(f"transient {stem}")
        return SourceAnalysis(
            orientation="portrait", fps=24.0, keyframe_interval=48,
            is_bw=False, film_grain=0, has_audio=True,
            scenes=[SceneChunk(
                index=0, path=work_dir / "c.mp4", start_time=0.0,
                duration=2.0,
            )],
        )

    @staticmethod
    def package(*, variant_outputs, source, source_analysis, config,
                output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "master.m3u8").write_text("#EXTM3U\n", encoding="utf-8")


@pytest.fixture
def stages(monkeypatch):
    fake = _Stages()
    monkeypatch.setattr(m.preflight, "_missing_programs", lambda: [])
    monkeypatch.setattr(m.preflight, "check_toolchain", lambda config: [])
    monkeypatch.setattr(m, "nvenc_supports_film_grain", lambda: True)
    monkeypatch.setattr(m, "_video_duration", lambda path: 10.0)
    monkeypatch.setattr(m, "analyze_and_split", fake.analyze)
    monkeypatch.setattr(m, "run_probing", lambda **kw: [])
    monkeypatch.setattr(m, "encode_and_concat", lambda **kw: [])
    monkeypatch.setattr(m, "package_hls", fake.package)
    return fake


def _inputs(home: Path, *names: str) -> list[Path]:
    return _touch(home / "input", *names)


def _batch(home: Path) -> dict:
    return json.loads(
        (paths.output_dir() / "batch_report.json").read_text(encoding="utf-8"),
    )


def _stop_here(source, config, work_dir):
    raise RuntimeError("stop here")


class TestEncodeSingleClip:
    def test_should_not_reframe_when_disabled(
        self, monkeypatch, tmp_path, stages,
    ):
        clip, = _touch(tmp_path, "c.mp4")
        calls = []
        monkeypatch.setattr(m, "_run_reframe", lambda *a: calls.append(a))

        report = m._encode_single_clip(
            clip, PipelineConfig(reframe_enabled=False), tmp_path / "o",
            timeout=60,
        )

        assert calls == []
        assert report.reframe is None
        assert report.status == "passed"

    def test_should_hand_on_the_original_path_when_reframing_is_off(
        self, monkeypatch, tmp_path,
    ):
        clip, = _touch(tmp_path, "c.mp4")
        seen = {}

        def capture(source, config, work_dir):
            seen["source"] = source
            raise RuntimeError("stop here")

        monkeypatch.setattr(m, "analyze_and_split", capture)
        m._encode_single_clip(
            clip, PipelineConfig(reframe_enabled=False), tmp_path / "o",
            timeout=60,
        )
        assert seen["source"] is clip

    def test_should_encode_the_intermediate_when_reframed(
        self, monkeypatch, tmp_path,
    ):
        clip, vertical = _touch(tmp_path, "c.mp4", "vertical.mp4")
        entry = {"applied": True, "reason": "landscape"}
        seen = {}

        def capture(source, config, work_dir):
            seen["source"] = source
            raise RuntimeError("stop here")

        monkeypatch.setattr(
            m, "_run_reframe",
            lambda clip, config, work_dir: m.ReframeOutcome(
                source=vertical, entry=entry, crops=None,
            ),
        )
        monkeypatch.setattr(m, "analyze_and_split", capture)

        report = m._encode_single_clip(
            clip, PipelineConfig(), tmp_path / "o", timeout=60,
        )

        assert seen["source"] == vertical
        assert report.reframe == entry
        assert report.source == "c.mp4"

    def test_should_report_a_reframe_failure_without_stopping(
        self, monkeypatch, tmp_path,
    ):
        clip, = _touch(tmp_path, "c.mp4")

        def fail(clip, config, work_dir):
            raise RuntimeError("reframe timed out after 600s")

        monkeypatch.setattr(m, "_run_reframe", fail)
        report = m._encode_single_clip(
            clip, PipelineConfig(), tmp_path / "o", timeout=60,
        )
        assert report.status == "failed"
        assert "reframe timed out" in report.error
        assert report.video_id == "c"
        # Counted as a reframe failure in the batch report.
        assert report.reframe == {"applied": False, "outcome": "failed"}

    def test_should_count_a_stage_that_reported_no_entry_as_failed(
        self, monkeypatch, tmp_path,
    ):
        clip, = _touch(tmp_path, "c.mp4")

        def fail(clip, config, work_dir):
            raise m.ReframeStageError("could not read the video", entry=None)

        monkeypatch.setattr(m, "_run_reframe", fail)
        report = m._encode_single_clip(
            clip, PipelineConfig(), tmp_path / "o", timeout=60,
        )
        assert report.reframe == {"applied": False, "outcome": "failed"}

    def test_should_keep_local_folders_out_of_the_report(
        self, monkeypatch, tmp_path,
    ):
        """The output folder, report.json included, is meant to be
        published, and the bug report form asks for report.json; a local
        path often holds the user's account name."""
        clip, = _touch(tmp_path, "c.mp4")
        data = paths.data_root()
        checkout = paths.source_root()
        home = Path.home()

        def fail(clip, config, work_dir):
            raise RuntimeError(
                f"reframe output {data / 'work' / 'c' / 'v.mp4'} does not "
                f"exist; {checkout / '.venv-reframe'} and "
                f"{(home / 'clips').as_posix()} and {data.as_posix()}/x"
            )

        monkeypatch.setattr(m, "_run_reframe", fail)
        report = m._encode_single_clip(
            clip, PipelineConfig(), tmp_path / "o", timeout=60,
        )

        for folder in (data, checkout, home):
            assert str(folder) not in report.error
            assert folder.as_posix() not in report.error
        sep = os.sep
        assert report.error == (
            f"RuntimeError: reframe output <data>{sep}work{sep}c{sep}v.mp4 "
            f"does not exist; <checkout>{sep}.venv-reframe and ~/clips and "
            "<data>/x"
        )

    def test_should_keep_the_stage_diagnostic_when_reframing_fails(
        self, monkeypatch, tmp_path,
    ):
        clip, = _touch(tmp_path, "c.mp4")

        def fail(clip, config, work_dir):
            raise m.ReframeStageError(
                "reframe failed", entry={"applied": False, "reason": "landscape"},
            )

        monkeypatch.setattr(m, "_run_reframe", fail)
        report = m._encode_single_clip(
            clip, PipelineConfig(), tmp_path / "o", timeout=60,
        )
        assert report.reframe["reason"] == "landscape"
        assert report.status == "failed"

    def test_should_keep_the_crop_track_when_the_encode_fails_later(
        self, monkeypatch, tmp_path,
    ):
        clip, vertical, crops = _touch(
            tmp_path, "c.mp4", "vertical.mp4", "crops.json",
        )
        monkeypatch.setattr(
            m, "_run_reframe",
            lambda clip, config, work_dir: m.ReframeOutcome(
                source=vertical, entry={"applied": True}, crops=crops,
            ),
        )
        monkeypatch.setattr(m, "analyze_and_split", _stop_here)

        m._encode_single_clip(clip, PipelineConfig(), tmp_path / "o", timeout=60)

        assert (tmp_path / "o" / "c" / "crops.json").is_file()

    def test_should_not_charge_reframe_time_to_the_encode_limit(
        self, monkeypatch, tmp_path,
    ):
        clip, = _touch(tmp_path, "c.mp4")

        def slow_reframe(clip, config, work_dir):
            time.sleep(1.5)
            return m.ReframeOutcome(
                source=clip, entry={"applied": False}, crops=None,
            )

        monkeypatch.setattr(m, "_run_reframe", slow_reframe)
        monkeypatch.setattr(m, "analyze_and_split", _stop_here)

        report = m._encode_single_clip(
            clip, PipelineConfig(), tmp_path / "o", timeout=1,
        )

        assert "stop here" in report.error
        assert "timed out" not in report.error.lower()

    def test_should_fail_a_clip_that_runs_past_its_limit(
        self, monkeypatch, tmp_path,
    ):
        clip, = _touch(tmp_path, "slow.mp4")
        # A hung ffmpeg: a real child that sleeps far past the limit. A
        # plain time.sleep here could not be interrupted.
        monkeypatch.setattr(
            m, "analyze_and_split",
            lambda source, config, work_dir: m.run_tool(
                [sys.executable, "-c", "import time; time.sleep(30)"],
            ),
        )
        started = time.monotonic()
        report = m._encode_single_clip(
            clip, PipelineConfig(reframe_enabled=False), tmp_path / "o",
            timeout=1,
        )
        assert report.status == "failed"
        assert "timed out" in report.error.lower()
        assert time.monotonic() - started < 10

    def test_should_write_both_reports_and_clear_work_on_success(
        self, tmp_path, stages,
    ):
        clip, = _touch(tmp_path, "c.mp4")

        report = m._encode_single_clip(
            clip, PipelineConfig(reframe_enabled=False), tmp_path / "o",
            timeout=60,
        )

        folder = tmp_path / "o" / "c"
        saved = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        assert saved["status"] == report.status == "passed"
        assert (folder / "report.md").is_file()
        assert not (paths.work_dir() / "c").exists()
        assert m._is_complete(folder)

    def test_should_write_a_failed_report_and_keep_the_work_folder(
        self, tmp_path, stages,
    ):
        stages.fail.add("c")
        clip, = _touch(tmp_path, "c.mp4")

        m._encode_single_clip(
            clip, PipelineConfig(reframe_enabled=False), tmp_path / "o",
            timeout=60,
        )

        folder = tmp_path / "o" / "c"
        saved = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        assert saved["status"] == "failed"
        assert "boom c" in saved["error"]
        assert (paths.work_dir() / "c").is_dir()
        assert not m._is_complete(folder)

    def test_should_start_each_attempt_with_an_empty_work_folder(
        self, monkeypatch, tmp_path,
    ):
        """A failed attempt keeps its work folder for debugging.

        The next attempt must not see any of it: a stale scene chunk
        would be encoded onto the new clip.
        """
        clip, = _touch(tmp_path, "c.mp4")
        _touch(paths.work_dir() / "c" / "scenes", "chunk_007.mp4")
        _touch(paths.work_dir() / "c", "report.json")
        seen = {}

        def capture(source, config, work_dir):
            seen["left"] = list(work_dir.rglob("*"))
            raise RuntimeError("stop here")

        monkeypatch.setattr(m, "analyze_and_split", capture)
        m._encode_single_clip(
            clip, PipelineConfig(reframe_enabled=False), tmp_path / "o",
            timeout=60,
        )

        assert seen["left"] == []

    def test_should_keep_the_traceback_of_a_failed_attempt(
        self, monkeypatch, tmp_path,
    ):
        """The report keeps only "KeyError: 'missing'"; the traceback in
        the kept work folder says which line raised it."""
        clip, = _touch(tmp_path, "c.mp4")

        def broken(source, config, work_dir):
            return {}["missing"]

        monkeypatch.setattr(m, "analyze_and_split", broken)
        report = m._encode_single_clip(
            clip, PipelineConfig(reframe_enabled=False), tmp_path / "o",
            timeout=60,
        )

        assert report.error == "KeyError: 'missing'"
        text = (paths.work_dir() / "c" / m.TRACEBACK_NAME).read_text(
            encoding="utf-8",
        )
        assert text.startswith("Traceback") and "in broken" in text

    def test_should_write_report_json_after_report_md(
        self, monkeypatch, tmp_path,
    ):
        order = []
        monkeypatch.setattr(
            m, "generate_markdown_report", lambda r, p: order.append(p.name),
        )
        monkeypatch.setattr(
            m, "save_clip_report", lambda r, p: order.append(p.name),
        )

        m._write_clip_reports(_report("c"), tmp_path)

        assert order == ["report.md", "report.json"]

    def test_should_return_the_failed_report_when_it_cannot_be_written(
        self, monkeypatch, tmp_path, stages, capsys,
    ):
        stages.fail.add("c")
        clip, = _touch(tmp_path, "c.mp4")

        def disk_full(report, path):
            raise OSError("disk full")

        monkeypatch.setattr(m, "save_clip_report", disk_full)
        report = m._encode_single_clip(
            clip, PipelineConfig(reframe_enabled=False), tmp_path / "o",
            timeout=60,
        )

        assert report.status == "failed" and "boom c" in report.error
        assert "could not write the report" in capsys.readouterr().out


class TestRun:
    def test_should_encode_every_new_clip_and_write_the_batch_report(
        self, ftea_home, stages,
    ):
        _inputs(ftea_home, "a.mp4", "b.mov")

        result = m.run(reframe=False)

        out = paths.output_dir()
        assert result.outputs == (out / "a", out / "b")
        assert result.failed == () and result.exit_code == 0
        assert result.batch_report == out / "batch_report.json"
        assert _batch(ftea_home)["total"] == 2
        assert not paths.lock_file().exists()
        assert not paths.progress_file().exists()

    def test_should_print_the_four_phases(self, ftea_home, stages, capsys):
        _inputs(ftea_home, "a.mp4")
        m.run(reframe=False)
        printed = capsys.readouterr().out
        for title in ("PHASE 1: PRE-SCAN", "PHASE 2: ENCODE",
                      "PHASE 3: RETRY", "PHASE 4: REPORT"):
            assert title in printed

    def test_should_skip_finished_clips_and_encode_failed_ones_again(
        self, ftea_home, stages,
    ):
        _inputs(ftea_home, "a.mp4", "b.mp4")
        stages.fail.add("b")
        assert m.run(reframe=False).failed == ("b",)

        stages.fail.clear()
        stages.encoded.clear()
        result = m.run(reframe=False)

        assert stages.encoded == ["b"]
        assert result.already_done == ("a",)
        assert result.exit_code == 0

    def test_should_encode_finished_clips_again_when_forced(
        self, ftea_home, stages,
    ):
        _inputs(ftea_home, "a.mp4")
        m.run(reframe=False)
        stages.encoded.clear()

        m.run(reframe=False, force=True)

        assert stages.encoded == ["a"]
        assert not (paths.output_dir() / ".a.previous").exists()

    def test_should_keep_the_finished_package_when_a_forced_encode_fails(
        self, ftea_home, stages,
    ):
        _write_config(ftea_home, "RETRY_MAX=0\n")
        _inputs(ftea_home, "a.mp4")
        m.run(reframe=False)
        stages.fail.add("a")

        result = m.run(reframe=False, force=True)

        assert result.failed == ("a",)
        assert m._is_complete(paths.output_dir() / "a")
        saved = json.loads(
            (paths.work_dir() / "a" / "report.json").read_text(
                encoding="utf-8",
            ),
        )
        assert saved["status"] == "failed"

    def test_should_leave_a_package_made_from_a_different_file_alone(
        self, ftea_home, stages, capsys,
    ):
        clip, = _inputs(ftea_home, "a.mp4")
        m.run(reframe=False)
        clip.write_bytes(b"a different video with the same name")
        stages.encoded.clear()
        capsys.readouterr()

        result = m.run(reframe=False)

        assert stages.encoded == []
        assert result.not_attempted == ("a",) and result.exit_code == 1
        assert "was made from a different a.mp4" in capsys.readouterr().out
        assert (paths.output_dir() / "a" / "master.m3u8").is_file()

    def test_should_encode_a_different_file_again_when_forced(
        self, ftea_home, stages,
    ):
        clip, = _inputs(ftea_home, "a.mp4")
        m.run(reframe=False)
        clip.write_bytes(b"a different video with the same name")
        stages.encoded.clear()

        result = m.run(reframe=False, force=True)

        assert stages.encoded == ["a"] and result.exit_code == 0

    @pytest.mark.parametrize("force", [False, True])
    def test_should_leave_ftea_reframe_results_of_the_same_name_alone(
        self, ftea_home, stages, capsys, force,
    ):
        """`ftea reframe talk.mp4` writes output/talk-vertical/, which is
        where `ftea run` puts a clip named talk-vertical.mp4."""
        _inputs(ftea_home, "talk-vertical.mp4")
        results = _reframe_results(paths.output_dir(), "talk-vertical")

        result = m.run(reframe=False, force=force)

        assert stages.encoded == []
        assert result.not_attempted == ("talk-vertical",)
        assert result.exit_code == 1
        assert "holds the results of `ftea reframe`" in (
            capsys.readouterr().out
        )
        assert sorted(p.name for p in results.iterdir()) == [
            "crops.json", "ftea-reframe.json", "vertical.mp4",
        ]

    @pytest.mark.parametrize(("orientation", "entry", "reframe", "why"), [
        ("landscape", None, True, "was encoded without reframing"),
        ("portrait", {"applied": True, "outcome": "reframed"}, False,
         "was reframed"),
    ], ids=["reframing now on", "reframing now off"])
    def test_should_say_when_a_finished_clip_was_framed_otherwise(
        self, ftea_home, stages, capsys, orientation, entry, reframe, why,
    ):
        """Resume keeps the finished package, but a run that asks for the
        other framing must not pass it over in silence."""
        _inputs(ftea_home, "a.mp4")
        _finished_folder(
            paths.output_dir(), "a", orientation=orientation, reframe=entry,
        )

        result = m.run(reframe=reframe)

        printed = capsys.readouterr().out
        assert result.already_done == ("a",) and stages.encoded == []
        assert f"output/a/ {why}" in printed
        assert "ftea run --force" in printed

    @pytest.mark.parametrize(("orientation", "entry", "reframe"), [
        ("landscape", None, False),
        ("portrait", None, True),
        ("portrait", {"applied": True, "outcome": "reframed"}, True),
        ("portrait", {"applied": False, "outcome": "skipped"}, False),
    ])
    def test_should_not_mention_framing_that_matches_this_run(
        self, ftea_home, stages, capsys, orientation, entry, reframe,
    ):
        _inputs(ftea_home, "a.mp4")
        _finished_folder(
            paths.output_dir(), "a", orientation=orientation, reframe=entry,
        )

        m.run(reframe=reframe)

        assert "NOTE" not in capsys.readouterr().out

    def test_should_say_how_to_encode_a_finished_clip_again(
        self, ftea_home, stages, capsys,
    ):
        _inputs(ftea_home, "a.mp4")
        m.run(reframe=False)
        capsys.readouterr()

        m.run(reframe=False)

        assert "ftea run --force" in capsys.readouterr().out

    def test_should_give_each_clip_a_limit_that_grows_with_its_length(
        self, ftea_home, stages, monkeypatch,
    ):
        _inputs(ftea_home, "a.mp4")
        stages.fail.add("a")
        limits = []
        encode_one = m._encode_single_clip

        def spy(clip, config, output_base, timeout):
            limits.append(timeout)
            return encode_one(clip, config, output_base, timeout=timeout)

        monkeypatch.setattr(m, "_encode_single_clip", spy)
        m.run(reframe=False)

        assert m._clip_budget(PipelineConfig(), 84.0) == 300 + 30 * 84.0
        # The stages fixture reports every clip as 10 s long. Both retry
        # passes multiply both parts by 2: pass 1 already gets the longer
        # limit.
        assert limits == [
            300 + 30 * 10.0,
            2 * (300 + 30 * 10.0),
            2 * (300 + 30 * 10.0),
        ]

    def test_should_recover_a_transient_failure_on_the_first_retry(
        self, ftea_home, stages,
    ):
        _inputs(ftea_home, "a.mp4")
        stages.fail_once.add("a")

        result = m.run(reframe=False)

        assert result.failed == ()
        assert _batch(ftea_home)["retry_summary"] == {
            "retry_1": {"attempted": 1, "recovered": 1},
        }

    def test_should_relax_the_targets_only_from_the_second_retry(
        self, ftea_home, stages,
    ):
        _inputs(ftea_home, "a.mp4")
        stages.fail.add("a")

        m.run(reframe=False)

        first, retry_1, retry_2 = stages.configs["a"]
        assert first.vmaf_targets == retry_1.vmaf_targets == {
            "1080": 93, "720": 93, "360": 88,
        }
        assert retry_2.vmaf_targets == {"1080": 91, "720": 91, "360": 86}
        assert first.clip_timeout == 300
        assert (retry_1.clip_timeout, retry_2.clip_timeout) == (600, 600)
        assert (
            retry_1.clip_timeout_per_sec, retry_2.clip_timeout_per_sec,
        ) == (60, 60)
        # A clip whose reframe ran out of time gets more time too.
        assert first.reframe_timeout == 600
        assert (retry_1.reframe_timeout, retry_2.reframe_timeout) == (
            1200, 1200,
        )

    def test_should_halt_the_batch_when_told_to_stop(self, ftea_home, stages):
        _write_config(ftea_home, "BATCH_ON_ERROR=stop\nRETRY_MAX=0\n")
        _inputs(ftea_home, "a.mp4", "b.mp4")
        stages.fail.add("a")

        result = m.run(reframe=False)

        assert stages.encoded == ["a"]
        assert result.failed == ("a",) and result.exit_code == 1
        assert result.not_attempted == ("b",)
        batch = _batch(ftea_home)
        assert (batch["total"], batch["not_attempted"]) == (2, ["b"])

    def test_should_not_retry_or_claim_success_after_a_halt(
        self, ftea_home, stages, capsys,
    ):
        _write_config(ftea_home, "BATCH_ON_ERROR=stop\n")
        _inputs(ftea_home, "a.mp4", "b.mp4")
        stages.fail_once.add("a")

        result = m.run(reframe=False)

        assert stages.encoded == ["a"]
        assert result.not_attempted == ("b",) and result.exit_code == 1
        assert "Not attempted" in capsys.readouterr().out

    def test_should_record_a_clip_that_still_fails(self, ftea_home, stages):
        _write_config(ftea_home, "RETRY_MAX=0\n")
        _inputs(ftea_home, "a.mp4", "b.mp4")
        stages.fail.add("a")

        result = m.run(reframe=False)

        batch = _batch(ftea_home)
        assert result.failed == ("a",)
        assert result.outputs == (paths.output_dir() / "b",)
        assert (batch["total"], batch["encoded"], batch["failed"]) == (2, 1, 1)
        assert "gate_check" not in batch

    def test_should_use_the_input_folder_by_default(self, ftea_home, stages):
        _inputs(ftea_home, "a.mp4")
        assert m.run(reframe=False).outputs == (paths.output_dir() / "a",)

    def test_should_accept_a_single_named_file(
        self, ftea_home, stages, tmp_path,
    ):
        clip, = _touch(tmp_path / "elsewhere", "one clip.mp4")
        result = m.run(clip, reframe=False)
        assert result.outputs == (paths.output_dir() / "one clip",)

    def test_should_refuse_a_path_that_does_not_exist(
        self, ftea_home, stages, tmp_path,
    ):
        with pytest.raises(m.PipelineError, match="does not exist"):
            m.run(tmp_path / "nowhere", reframe=False)

    def test_should_say_so_when_there_are_no_videos(
        self, ftea_home, monkeypatch, capsys,
    ):
        (ftea_home / "input").mkdir()
        checked = []
        monkeypatch.setattr(
            m.preflight, "check_toolchain",
            lambda config: checked.append(1) or [],
        )

        result = m.run(reframe=False)

        assert result == m.RunResult() and result.exit_code == 0
        assert "No videos found" in capsys.readouterr().out
        assert checked == []
        assert not paths.lock_file().exists()

    def test_should_count_files_it_could_not_read_against_the_run(
        self, ftea_home, stages, monkeypatch,
    ):
        """A skipped file used to leave only a console line: the run
        exited 0 and the batch report never mentioned it."""
        _inputs(ftea_home, "a.mp4", "bad.mp4", "slow.mp4")

        def duration(path):
            if path.stem == "bad":
                return None
            if path.stem == "slow":
                raise ToolError("ffprobe did not finish within 60s")
            return 10.0

        monkeypatch.setattr(m, "_video_duration", duration)

        result = m.run(reframe=False)

        assert stages.encoded == ["a"]
        assert result.not_attempted == ("bad.mp4", "slow.mp4")
        assert result.exit_code == 1
        batch = _batch(ftea_home)
        assert batch["not_attempted"] == ["bad.mp4", "slow.mp4"]
        assert batch["total"] == 3

    def test_should_count_a_name_it_cannot_use_against_the_run(
        self, ftea_home, stages,
    ):
        _inputs(ftea_home, "clip.mp4", "Clip.MOV")

        result = m.run(reframe=False)

        assert stages.encoded == ["Clip"]
        assert result.not_attempted == ("clip.mp4",)
        assert result.exit_code == 1

    def test_should_fail_a_run_whose_every_file_has_an_unusable_name(
        self, ftea_home, stages, monkeypatch, capsys,
    ):
        _inputs(ftea_home, "bad.mp4")
        monkeypatch.setattr(m, "_is_safe_clip_id", lambda video_id: False)

        result = m.run(reframe=False)

        assert stages.encoded == []
        assert result.not_attempted == ("bad.mp4",)
        assert result.exit_code == 1
        assert "No videos found" not in capsys.readouterr().out

    def test_should_refuse_a_named_file_without_video(
        self, ftea_home, stages, monkeypatch, tmp_path,
    ):
        clip, = _touch(tmp_path, "notes.mp4")
        monkeypatch.setattr(m, "_video_duration", lambda path: None)
        with pytest.raises(m.PipelineError, match="no video stream"):
            m.run(clip, reframe=False)
        assert not paths.lock_file().exists()

    def test_should_skip_a_video_longer_than_the_limit(
        self, ftea_home, stages, monkeypatch, capsys,
    ):
        _inputs(ftea_home, "film.mp4", "short.mp4")
        monkeypatch.setattr(
            m, "_video_duration",
            lambda path: 723.0 if path.stem == "film" else 30.0,
        )

        result = m.run(reframe=False)

        assert stages.encoded == ["short"]
        assert result.not_attempted == ("film",) and result.exit_code == 1
        assert "SKIP (12:03 long" in capsys.readouterr().out

    def test_should_refuse_a_named_file_longer_than_the_limit(
        self, ftea_home, stages, monkeypatch, tmp_path,
    ):
        clip, = _touch(tmp_path, "film.mp4")
        monkeypatch.setattr(m, "_video_duration", lambda path: 723.0)
        with pytest.raises(m.PipelineError, match="MAX_CLIP_SEC"):
            m.run(clip, reframe=False)
        assert stages.encoded == []
        assert not paths.lock_file().exists()

    def test_should_stop_before_any_clip_when_the_toolchain_is_missing(
        self, ftea_home, stages, monkeypatch,
    ):
        _inputs(ftea_home, "a.mp4")
        monkeypatch.setattr(
            m.preflight, "check_toolchain",
            lambda config: ["this FFmpeg build has no libvmaf filter."],
        )
        with pytest.raises(m.PipelineError, match="no libvmaf filter"):
            m.run(reframe=False)
        assert not paths.lock_file().exists()
        assert stages.encoded == []

    def test_should_stop_before_pre_scan_when_ffprobe_is_missing(
        self, ftea_home, stages, monkeypatch,
    ):
        _inputs(ftea_home, "a.mp4")
        monkeypatch.setattr(
            m.preflight, "_missing_programs",
            lambda: ["ffprobe is not on PATH."],
        )
        with pytest.raises(m.PipelineError, match="ffprobe is not on PATH"):
            m.run(reframe=False)
        assert not paths.lock_file().exists()
        assert stages.encoded == []

    def test_should_stop_before_any_clip_when_reframing_cannot_start(
        self, ftea_home, stages, monkeypatch,
    ):
        _inputs(ftea_home, "a.mp4")

        def missing(config):
            raise RuntimeError("no reframe interpreter found; run install.sh")

        monkeypatch.setattr(m, "_resolve_reframe_python", missing)
        with pytest.raises(m.PipelineError, match="install.sh"):
            m.run(reframe=True)
        assert not paths.lock_file().exists()
        assert stages.encoded == []

    def test_should_not_check_the_machine_when_every_clip_is_done(
        self, ftea_home, stages, monkeypatch,
    ):
        _inputs(ftea_home, "a.mp4")
        m.run(reframe=False)
        checked = []
        monkeypatch.setattr(
            m.preflight, "check_toolchain",
            lambda config: checked.append("toolchain") or [],
        )
        monkeypatch.setattr(
            m, "_resolve_reframe_python", lambda config: Path("python"),
        )
        monkeypatch.setattr(
            m, "_verify_reframe_env", lambda python: checked.append("reframe"),
        )

        result = m.run(reframe=True)

        assert result.already_done == ("a",) and result.exit_code == 0
        assert checked == []

    def test_should_check_the_reframe_environment_once_per_run(
        self, ftea_home, stages, monkeypatch,
    ):
        _inputs(ftea_home, "a.mp4", "b.mp4")
        checks = []
        monkeypatch.setattr(
            m, "_resolve_reframe_python", lambda config: Path("python"),
        )
        monkeypatch.setattr(m, "_verify_reframe_env", checks.append)
        monkeypatch.setattr(
            m, "_run_reframe",
            lambda clip, config, work_dir: m.ReframeOutcome(
                source=clip,
                entry={"applied": False, "reason": "already_vertical"},
                crops=None,
            ),
        )

        result = m.run(reframe=True)

        assert checks == [Path("python")] and result.exit_code == 0
        saved = json.loads(
            (paths.output_dir() / "a" / "report.json").read_text(
                encoding="utf-8",
            ),
        )
        assert saved["reframe"]["reason"] == "already_vertical"

    def test_should_release_the_lock_when_a_phase_crashes(
        self, ftea_home, stages, monkeypatch,
    ):
        _inputs(ftea_home, "a.mp4")

        def crash(*args):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(m, "_encode_pass", crash)
        with pytest.raises(RuntimeError, match="unexpected"):
            m.run(reframe=False)
        assert not paths.lock_file().exists()

    def test_should_finish_when_the_console_cannot_show_a_clip_name(
        self, ftea_home, stages, monkeypatch,
    ):
        """Windows writes piped or redirected output in the ANSI code page.

        That is the case under Git Bash and for `ftea run > log.txt`. A
        clip name outside that code page must not end the batch.
        """
        raw = io.BytesIO()
        console = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
        monkeypatch.setattr(sys, "stdout", console)
        _inputs(ftea_home, "東京.mp4")

        result = m.run(reframe=False)

        console.flush()
        assert result.exit_code == 0
        assert "\\u6771\\u4eac.mp4" in raw.getvalue().decode("cp1252")


class TestParseArgv:
    @pytest.mark.parametrize("argv, expected", [
        ([], (None, None)),
        (["input"], ("input", None)),
        (["--reframe", "input"], ("input", True)),
        (["input", "--reframe"], ("input", True)),
        (["input", "--no-reframe"], ("input", False)),
        (["--reframe", "--no-reframe", "i"], ("i", False)),
        (["--no-reframe", "--reframe", "i"], ("i", True)),
        (["--", "--weird-dir"], ("--weird-dir", None)),
    ])
    def test_should_split_the_path_from_the_flags(self, argv, expected):
        assert m._parse_argv(argv) == expected

    @pytest.mark.parametrize("argv", [["--wat", "input"], ["a", "b"]])
    def test_should_reject_what_it_cannot_understand(self, argv):
        with pytest.raises(m.UsageError):
            m._parse_argv(argv)


class TestMain:
    def test_should_exit_2_with_the_usage_for_bad_arguments(self, capsys):
        assert m.main(["--wat"]) == 2
        err = capsys.readouterr().err
        assert "unknown option: --wat" in err and m.USAGE in err

    def test_should_print_help_and_exit_0(self, capsys):
        assert m.main(["--help"]) == 0
        assert m.USAGE in capsys.readouterr().out

    def test_should_exit_1_with_the_reason_when_a_run_cannot_start(
        self, tmp_path, capsys,
    ):
        assert m.main([str(tmp_path / "nowhere")]) == 1
        assert "Error:" in capsys.readouterr().err

    @pytest.mark.parametrize("make", ["missing", "file"])
    def test_should_explain_a_data_folder_that_cannot_be_used(
        self, monkeypatch, tmp_path, capsys, make,
    ):
        home = tmp_path / "ftea-data"
        if make == "file":
            home.write_text("", encoding="utf-8")
        monkeypatch.setenv(paths.ENV_HOME, str(home))

        assert m.main([]) == 1
        assert "FTEA_HOME" in capsys.readouterr().err

    def test_should_pass_the_path_and_flag_to_run(self, monkeypatch):
        seen = {}

        def fake_run(input_path, *, reframe):
            seen.update(input_path=input_path, reframe=reframe)
            return m.RunResult(failed=("x",))

        monkeypatch.setattr(m, "run", fake_run)
        assert m.main(["--no-reframe", "clips"]) == 1
        assert seen == {"input_path": Path("clips"), "reframe": False}


# --- Conforming odd shapes ---

class TestConform:
    def test_should_pad_a_square_clip_and_record_it(
        self, monkeypatch, tmp_path, stages,
    ):
        clip, padded = _touch(tmp_path, "sq.mp4", "conformed.mp4")
        monkeypatch.setattr(m, "probe_geometry", lambda source: Geometry(1080, 1080))
        made = []

        def fake_conform(source, work_dir, geometry):
            made.append(source)
            return padded

        monkeypatch.setattr(m, "conform", fake_conform)
        monkeypatch.setattr(m, "detect_has_audio", lambda path: False)

        report = m._encode_single_clip(
            clip, PipelineConfig(reframe_enabled=False), tmp_path / "o",
            timeout=60,
        )

        assert made == [clip]
        assert stages.encoded == ["conformed"]
        assert report.conform == {
            "applied": True, "source": "1080x1080", "rotation": 0,
            "frame": "1080x1920",
        }
        assert report.source == "sq.mp4"

    def test_should_leave_an_exact_frame_alone(
        self, monkeypatch, tmp_path, stages,
    ):
        clip, = _touch(tmp_path, "v.mp4")
        monkeypatch.setattr(
            m, "conform", lambda *a, **k: pytest.fail("conform must not run"),
        )

        report = m._encode_single_clip(
            clip, PipelineConfig(reframe_enabled=False), tmp_path / "o",
            timeout=60,
        )

        assert report.conform is None
        assert stages.encoded == ["v"]

    def test_should_check_the_shape_of_what_reframing_produced(
        self, monkeypatch, tmp_path, stages,
    ):
        clip, vertical = _touch(tmp_path, "l.mp4", "vertical.mp4")
        monkeypatch.setattr(
            m, "_run_reframe",
            lambda clip, config, work_dir: m.ReframeOutcome(
                source=vertical, entry={"applied": True}, crops=None,
            ),
        )
        probed = []
        monkeypatch.setattr(
            m, "probe_geometry",
            lambda source: probed.append(source) or Geometry(1080, 1920),
        )

        m._encode_single_clip(clip, PipelineConfig(), tmp_path / "o", timeout=60)

        assert probed == [vertical]

    def test_should_refuse_a_conform_that_lost_the_sound(
        self, monkeypatch, tmp_path,
    ):
        clip, padded = _touch(tmp_path, "sq.mp4", "conformed.mp4")
        monkeypatch.setattr(m, "probe_geometry", lambda source: Geometry(1080, 1080))
        monkeypatch.setattr(m, "conform", lambda source, work_dir, geometry: padded)
        answers = iter([True, False])
        monkeypatch.setattr(m, "detect_has_audio", lambda path: next(answers))

        report = m._encode_single_clip(
            clip, PipelineConfig(reframe_enabled=False), tmp_path / "o",
            timeout=60,
        )

        assert report.status == "failed"
        assert "silent intermediate" in report.error


# --- No network ---

class TestNoNetwork:
    def test_should_finish_a_run_without_opening_a_connection(
        self, ftea_home, stages, monkeypatch,
    ):
        def refuse(*args, **kwargs):
            raise AssertionError("the encoder tried to open a connection")

        monkeypatch.setattr(socket.socket, "connect", refuse)
        monkeypatch.setattr(socket, "create_connection", refuse)
        monkeypatch.setattr(socket, "getaddrinfo", refuse)
        _inputs(ftea_home, "a.mp4")

        assert m.run(reframe=False).exit_code == 0


class TestWithoutLocalFolders:
    def test_should_replace_a_folder_only_as_a_whole_name(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setattr(m.Path, "home", lambda: tmp_path / "al")
        other = tmp_path / "alice" / "v.mp4"

        text = m._without_local_folders(
            f"{other} and '{tmp_path / 'al' / 'v.mp4'}'",
        )

        assert text == f"{other} and '~{os.sep}v.mp4'"

    def test_should_leave_a_home_folder_that_is_a_root_alone(
        self, monkeypatch,
    ):
        """HOME=/ is common in containers; replacing it would turn every
        path into a mess of ~."""
        root = Path(Path.cwd().anchor)
        monkeypatch.setattr(m.Path, "home", lambda: root)

        assert m._without_local_folders(f"{root} is full") == (
            f"{root} is full"
        )
