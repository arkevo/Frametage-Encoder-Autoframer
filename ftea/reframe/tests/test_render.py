"""Unit tests for ftea/reframe/render.py."""

import dataclasses
import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from ftea.reframe import render
from ftea.reframe.camera import plan_camera
from ftea.reframe.errors import ReframeError
from ftea.reframe.probe import SourceMeta
from ftea.reframe.settings import ReframeSettings, load_settings
from ftea.reframe.shots import Shot
from ftea.reframe.subject import (
    KIND_CENTER, KIND_FACE, NO_TRACK, ShotSubject, SubjectSample,
)


def make_meta(width=1920, height=1080, *, fps=24.0, duration_s=8.0,
              rotation=0, has_audio=True, name="clip.mp4"):
    """A SourceMeta with no ffprobe involved."""
    return SourceMeta(
        path=Path("/src") / name, width=width, height=height,
        rotation=rotation, fps=fps, duration_s=duration_s,
        has_audio=has_audio,
    )


def make_subject(index, start, end, fps, xs, *, stride=2):
    """One ShotSubject whose samples walk cx across `xs`."""
    frames = tuple(range(start, end, stride))
    samples = tuple(
        SubjectSample(
            frame=f, cx=xs[min(i, len(xs) - 1)], cy=0.5,
            x1=0.0, y1=0.0, x2=0.1, y2=0.9, score=1.0,
            observed=True, kind=KIND_FACE, track_id=1,
        )
        for i, f in enumerate(frames)
    )
    return ShotSubject(
        shot_index=index, start_frame=start, end_frame=end, fps=fps,
        stride=stride, sampled_frames=frames, samples=samples,
        has_subject=True, track_count=1, switch_frames=(),
    )


def make_empty_subject(index, start, end, fps, *, stride=2):
    """One ShotSubject with nobody in it: unobserved centre samples."""
    frames = tuple(range(start, end, stride))
    samples = tuple(
        SubjectSample(
            frame=f, cx=0.5, cy=0.5, x1=0.0, y1=0.0, x2=0.0, y2=0.0,
            score=0.0, observed=False, kind=KIND_CENTER,
            track_id=NO_TRACK,
        )
        for f in frames
    )
    return ShotSubject(
        shot_index=index, start_frame=start, end_frame=end, fps=fps,
        stride=stride, sampled_frames=frames, samples=samples,
        has_subject=False, track_count=0, switch_frames=(),
    )


def make_track(width=1920, height=1080, *, fps=24.0, frames=192,
               xs=(0.2, 0.5, 0.8)):
    """A real CropTrack: one shot, a subject that walks left to right."""
    subject = make_subject(0, 0, frames, fps, xs)
    return plan_camera([subject], make_meta(width, height, fps=fps,
                                            duration_s=frames / fps))


def make_shots(frames=192, fps=24.0):
    return [Shot(index=0, start_frame=0, end_frame=frames, fps=fps)]


def make_settings(**overrides):
    """Default ReframeSettings with named fields overridden."""
    base = load_settings(Path("does-not-exist.env"))
    return dataclasses.replace(base, **overrides)


def probe_json(width=608, height=1080, frames=192, *, audio=True,
               rotation=None, codec="hevc", pix_fmt="yuv420p",
               video_start=0.0, audio_start=0.0, audio_codec="aac"):
    """A canned `ffprobe -of json` payload for the fake runner.

    `audio_codec` is also what the fake source-audio-codec probe
    inside render_vertical parses this same payload for, since FakeRunner
    answers every ffprobe call with the one payload it was given.
    """
    video = {
        "index": 0, "codec_type": "video", "codec_name": codec,
        "width": width, "height": height, "pix_fmt": pix_fmt,
        "nb_read_packets": str(frames), "start_time": str(video_start),
    }
    if rotation is not None:
        video["side_data_list"] = [{"rotation": rotation}]
    streams = [video]
    if audio:
        streams.append({"index": 1, "codec_type": "audio",
                        "codec_name": audio_codec,
                        "nb_read_packets": "400",
                        "start_time": str(audio_start)})
    return json.dumps({"streams": streams})


class FakeRunner:
    """Records argv, writes what ffmpeg would have written, answers
    ffprobe.

    A plain object, not a Mock: it has behaviour (it creates the partial
    file) that a Mock cannot express without a side_effect lambda.
    """

    def __init__(self, *, probe_payload=None, fail_on=None):
        self.calls = []                       # list[(argv, cwd)]
        self.probe_payload = probe_payload or probe_json()
        self.fail_on = fail_on                # "ffmpeg" | "ffprobe" | None

    def __call__(self, argv, *, cwd=None):
        argv = list(argv)
        self.calls.append((argv, cwd))
        binary = Path(argv[0]).stem.lower()
        if self.fail_on and self.fail_on in binary:
            raise render.RenderError(f"{binary} failed (exit 1): boom")
        if binary.startswith("ffprobe"):
            return self.probe_payload
        Path(argv[-1]).write_bytes(b"fake video bytes")
        return ""

    @property
    def ffmpeg_argv(self):
        for argv, _cwd in self.calls:
            if Path(argv[0]).stem.lower().startswith("ffmpeg"):
                return argv
        raise AssertionError("ffmpeg was never invoked")

    @property
    def ffmpeg_cwd(self):
        for argv, cwd in self.calls:
            if Path(argv[0]).stem.lower().startswith("ffmpeg"):
                return cwd
        raise AssertionError("ffmpeg was never invoked")


class TestPartial:
    @pytest.mark.parametrize("name,expected", [
        ("vertical.mp4", "vertical.partial.mp4"),
        ("vertical.mkv", "vertical.partial.mkv"),
        ("crops.json", "crops.partial.json"),
    ])
    def test_should_keep_the_real_extension_when_naming_a_partial(
        self, tmp_path, name, expected,
    ):
        assert render._partial(tmp_path / name).name == expected

    def test_should_match_the_entry_points_partial_helper(self, tmp_path):
        """Two definitions, one behaviour, proven here."""
        from ftea.reframe import main

        for name in ("vertical.mp4", "vertical.mkv", "crops.json"):
            assert render._partial(tmp_path / name) == \
                main._partial(tmp_path / name)


class TestFfprobeFor:
    def test_should_return_a_bare_name_when_ffmpeg_is_empty(self):
        assert render.ffprobe_for("") == "ffprobe"

    def test_should_swap_the_name_on_a_posix_path(self):
        assert render.ffprobe_for("/usr/bin/ffmpeg") == "/usr/bin/ffprobe"

    def test_should_keep_the_extension_on_a_windows_path(self):
        got = render.ffprobe_for(r"C:\tools\ffmpeg.EXE")

        assert got == r"C:\tools\ffprobe.EXE"

    def test_should_accept_a_mixed_case_binary_name(self):
        got = render.ffprobe_for(r"C:\tools\FFmpeg.exe")

        assert got == r"C:\tools\ffprobe.exe"

    def test_should_raise_when_the_name_is_not_an_ffmpeg(self):
        with pytest.raises(render.RenderError, match="must name an ffmpeg"):
            render.ffprobe_for("/usr/bin/avconv")


class TestFormatPolicy:
    def test_should_lowercase_and_strip_the_codec_and_container(self):
        settings = make_settings(codec="  HEVC_NVENC ", container=" MP4 ")

        assert render.normalise_format(settings) == ("hevc_nvenc", "mp4")

    def test_should_strip_a_leading_dot_from_the_container(self):
        settings = make_settings(container=".mkv")

        assert render.normalise_format(settings)[1] == "mkv"

    @pytest.mark.parametrize("codec,container", [
        ("hevc_nvenc", "mp4"), ("hevc_nvenc", "mkv"),
        ("av1_nvenc", "mp4"), ("ffv1", "mkv"),
    ])
    def test_should_accept_every_supported_pairing(self, codec, container):
        render.validate_output_format(codec, container)

    def test_should_raise_when_ffv1_targets_mp4(self):
        with pytest.raises(render.RenderError) as excinfo:
            render.validate_output_format("ffv1", "mp4")

        message = str(excinfo.value)
        assert "ffv1 cannot be muxed into mp4" in message
        assert "REFRAME_CONTAINER=mkv" in message

    def test_should_raise_when_the_codec_is_unsupported(self):
        with pytest.raises(render.RenderError, match="unsupported codec"):
            render.validate_output_format("libx265", "mp4")

    def test_should_name_the_allowed_containers_in_the_error(self):
        with pytest.raises(render.RenderError) as excinfo:
            render.validate_output_format("av1_nvenc", "mov")

        assert "mp4, mkv" in str(excinfo.value)

    def test_should_raise_a_reframe_error_subclass(self):
        assert issubclass(render.RenderError, ReframeError)


class TestAudioPassthrough:
    """-c:a copy into a container the source's audio can't enter."""

    @pytest.mark.parametrize("codec,container", [
        ("aac", "mp4"), ("flac", "mp4"), ("pcm_s16le", "mov"),
        ("pcm_s16le", "mkv"), ("anything_at_all", "mkv"),
    ])
    def test_should_accept_every_supported_pairing(self, codec, container):
        render.validate_audio_passthrough(codec, container)

    def test_should_raise_when_pcm_targets_mp4(self):
        with pytest.raises(render.RenderError) as excinfo:
            render.validate_audio_passthrough("pcm_s16le", "mp4")

        message = str(excinfo.value)
        assert "pcm_s16le" in message and "mp4" in message
        assert "REFRAME_CONTAINER=mkv" in message

    def test_should_raise_a_reframe_error_subclass(self):
        with pytest.raises(ReframeError):
            render.validate_audio_passthrough("pcm_s16le", "mp4")


class TestSendcmdLines:
    def test_should_start_at_time_zero_on_the_first_frame(self):
        lines = render.sendcmd_lines(make_track())

        assert lines[0].startswith("0.000000 crop x ")
        assert lines[0].endswith(";")

    def test_should_use_the_first_frames_crop_value_first(self):
        track = make_track()

        first = int(render.sendcmd_lines(track)[0].split()[-1][:-1])

        assert first == track.flatten()[0]

    def test_should_aim_half_a_frame_before_the_target_frame(self):
        track = make_track(fps=24.0)
        changes = render.crop_changes(track)
        frame, value = changes[1]
        expected = "{:.6f}".format((frame - 0.5) / 24.0)

        lines = render.sendcmd_lines(track)

        assert lines[1] == f"{expected} crop x {value};"

    def test_should_emit_one_line_per_change_not_per_frame(self):
        track = make_track(frames=192)

        lines = render.sendcmd_lines(track)

        assert len(lines) == len(render.crop_changes(track))
        assert len(lines) <= track.total_frames

    def test_should_emit_a_single_line_for_a_static_track(self):
        subject = make_empty_subject(0, 0, 48, 24.0)
        track = plan_camera([subject], make_meta(duration_s=2.0))

        assert len(render.sendcmd_lines(track)) == 1

    def test_should_format_every_time_to_six_decimal_places(self):
        for line in render.sendcmd_lines(make_track(fps=30000 / 1001)):
            stamp = line.split(" ", 1)[0]
            assert len(stamp.split(".")[1]) == 6

    def test_should_emit_strictly_increasing_timestamps(self):
        stamps = [float(line.split(" ", 1)[0])
                  for line in render.sendcmd_lines(make_track())]

        assert stamps == sorted(stamps)
        assert len(set(stamps)) == len(stamps)

    def test_should_emit_ascii_only(self):
        for line in render.sendcmd_lines(make_track()):
            line.encode("ascii")

    def test_should_cross_the_ten_second_boundary_without_colliding(self):
        """Regression: the collision guard used to compare the FORMATTED
        strings, so "10.020833" read as <= "9.979167" and every clip
        longer than ten seconds failed at the first change past 10s.
        Found by an end-to-end run on a 58s clip.
        """
        # A subject that keeps moving for the whole 20 seconds, so the
        # crop is still changing well past the boundary.
        walking = tuple(0.3 + 0.4 * (i % 2) for i in range(240))
        track = make_track(fps=24.0, frames=480, xs=walking)

        lines = render.sendcmd_lines(track)

        stamps = [float(line.split(" ", 1)[0]) for line in lines]
        assert stamps[-1] > 10.0
        assert stamps == sorted(stamps)
        assert len(set(stamps)) == len(stamps)

    def test_should_raise_when_two_commands_collide_in_time(self):
        """A pathological fps would round two frames to one timestamp."""
        track = make_track(fps=24.0)
        with patch.object(render, "crop_changes",
                          return_value=[(0, 0), (0, 2)]):
            with pytest.raises(render.RenderError, match="collide"):
                render.sendcmd_lines(track)


class TestWriteSendcmd:
    def test_should_return_the_number_of_command_lines(self, tmp_path):
        track = make_track()
        path = tmp_path / render.SENDCMD_NAME

        count = render.write_sendcmd(track, path)

        assert count == len(render.sendcmd_lines(track))

    def test_should_write_a_header_comment(self, tmp_path):
        path = tmp_path / render.SENDCMD_NAME

        render.write_sendcmd(make_track(), path)

        head = path.read_text(encoding="ascii").splitlines()[0]
        assert head.startswith("# ftea/reframe/render.py:")
        assert "608x1080" in head

    def test_should_write_lf_line_endings_only(self, tmp_path):
        path = tmp_path / render.SENDCMD_NAME

        render.write_sendcmd(make_track(), path)

        assert b"\r\n" not in path.read_bytes()
        assert path.read_bytes().endswith(b";\n")

    def test_should_write_every_command_line(self, tmp_path):
        track = make_track()
        path = tmp_path / render.SENDCMD_NAME

        render.write_sendcmd(track, path)

        body = path.read_text(encoding="ascii").splitlines()[1:]
        assert body == list(render.sendcmd_lines(track))

    def test_should_overwrite_a_stale_script(self, tmp_path):
        path = tmp_path / render.SENDCMD_NAME
        path.write_text("0.000000 crop x 999999;\n", encoding="ascii")

        render.write_sendcmd(make_track(), path)

        assert "999999" not in path.read_text(encoding="ascii")


class TestFiltergraph:
    def test_should_build_the_graph_for_a_1080p_track(self):
        track = make_track(1920, 1080)

        graph = render.filtergraph(track)

        assert graph == (
            "[0:v:0]setpts=PTS-STARTPTS,sendcmd=f=crop.sendcmd,"
            f"crop=608:1080:{track.flatten()[0]}:0[v]"
        )

    @pytest.mark.parametrize("width,height,crop_w", [
        (1280, 720, 406), (2560, 1440, 810), (3840, 2160, 1216),
        (2560, 1080, 608), (1920, 816, 460),
    ])
    def test_should_use_the_windows_geometry_at_every_tier(
        self, width, height, crop_w,
    ):
        track = make_track(width, height)

        assert f"crop={crop_w}:{height}:" in render.filtergraph(track)

    def test_should_select_the_first_video_stream_explicitly(self):
        assert render.filtergraph(make_track()).startswith("[0:v:0]")

    def test_should_normalise_the_start_timestamp_first(self):
        graph = render.filtergraph(make_track())

        assert graph.index("setpts") < graph.index("sendcmd")

    def test_should_seed_the_static_x_with_frame_zeros_value(self):
        track = make_track()

        graph = render.filtergraph(track)

        assert f":{track.flatten()[0]}:0[v]" in graph

    def test_should_carry_only_a_bare_script_name(self):
        graph = render.filtergraph(make_track())

        assert "sendcmd=f=crop.sendcmd," in graph
        assert ":" not in graph.split("sendcmd=f=")[1].split(",")[0]


class TestVideoArgs:
    def test_should_build_nvenc_args_for_the_default_codec(self):
        settings = make_settings(codec="hevc_nvenc", cq=10, preset="p7")

        assert render.video_args("hevc_nvenc", settings) == [
            "-c:v", "hevc_nvenc", "-preset", "p7",
            "-rc", "constqp", "-qp", "10",
        ]

    def test_should_honour_a_custom_cq(self):
        settings = make_settings(cq=4)

        assert "4" in render.video_args("hevc_nvenc", settings)

    def test_should_build_archival_args_for_ffv1(self):
        settings = make_settings(codec="ffv1", container="mkv")

        assert render.video_args("ffv1", settings) == [
            "-c:v", "ffv1", "-level", "3", "-g", "1",
            "-slices", "24", "-slicecrc", "1",
        ]

    def test_should_ignore_cq_and_preset_for_ffv1(self):
        settings = make_settings(codec="ffv1", cq=40, preset="p1")

        args = render.video_args("ffv1", settings)

        assert "40" not in args and "p1" not in args

    def test_should_support_every_nvenc_codec(self):
        for codec in sorted(render.NVENC_CODECS):
            assert render.video_args(codec, make_settings())[1] == codec

    def test_should_raise_for_an_unsupported_codec(self):
        with pytest.raises(render.RenderError, match="unsupported codec"):
            render.video_args("libx265", make_settings())


class TestBuildCommand:
    def test_should_place_the_source_and_output_as_absolute_argv(
        self, tmp_path,
    ):
        source = tmp_path / "It's a source.mp4"
        out = tmp_path / "deep" / "vertical.partial.mp4"

        argv = render.build_command(source, out, make_track(),
                                    make_settings())

        assert argv[argv.index("-i") + 1] == str(source)
        assert argv[-1] == str(out)

    def test_should_absolutise_a_relative_source_and_output(
        self, tmp_path,
    ):
        """Regression: ffmpeg runs with cwd = the output directory,
        so a relative source -- what main.py is handed on the command
        line -- resolved under the work dir and failed with "No such
        file or directory". Found by the end-to-end smoke run.
        """
        argv = render.build_command(
            Path("input/clip.mp4"),
            Path("work/x/vertical.partial.mp4"),
            make_track(), make_settings())

        assert Path(argv[argv.index("-i") + 1]).is_absolute()
        assert Path(argv[-1]).is_absolute()

    def test_should_put_the_filtergraph_after_the_input(self, tmp_path):
        argv = render.build_command(tmp_path / "s.mp4",
                                    tmp_path / "v.partial.mp4",
                                    make_track(), make_settings())

        assert argv.index("-i") < argv.index("-filter_complex")

    def test_should_map_the_filtered_video_and_optional_audio(
        self, tmp_path,
    ):
        argv = render.build_command(tmp_path / "s.mp4",
                                    tmp_path / "v.partial.mp4",
                                    make_track(), make_settings())

        assert argv[argv.index("-map") + 1] == "[v]"
        assert "0:a?" in argv

    def test_should_copy_audio_rather_than_re_encode(self, tmp_path):
        argv = render.build_command(tmp_path / "s.mp4",
                                    tmp_path / "v.partial.mp4",
                                    make_track(), make_settings())

        assert argv[argv.index("-c:a") + 1] == "copy"

    def test_should_pass_frames_through_one_for_one(self, tmp_path):
        argv = render.build_command(tmp_path / "s.mp4",
                                    tmp_path / "v.partial.mp4",
                                    make_track(), make_settings())

        assert argv[argv.index("-fps_mode") + 1] == "passthrough"

    def test_should_not_force_a_pixel_format(self, tmp_path):
        argv = render.build_command(tmp_path / "s.mp4",
                                    tmp_path / "v.partial.mp4",
                                    make_track(), make_settings())

        assert "-pix_fmt" not in argv

    def test_should_drop_subtitle_and_data_streams(self, tmp_path):
        argv = render.build_command(tmp_path / "s.mp4",
                                    tmp_path / "v.partial.mp4",
                                    make_track(), make_settings())

        assert "-sn" in argv and "-dn" in argv

    def test_should_use_the_configured_ffmpeg_binary(self, tmp_path):
        argv = render.build_command(tmp_path / "s.mp4",
                                    tmp_path / "v.partial.mp4",
                                    make_track(), make_settings(),
                                    ffmpeg=r"C:\tools\ffmpeg.EXE")

        assert argv[0] == r"C:\tools\ffmpeg.EXE"


class TestParseProbeJson:
    def test_should_read_the_first_video_stream(self):
        info = render.parse_probe_json(probe_json(608, 1080, 192))

        assert (info.width, info.height, info.frames) == (608, 1080, 192)
        assert info.codec == "hevc" and info.pix_fmt == "yuv420p"

    def test_should_report_audio_when_a_stream_is_present(self):
        assert render.parse_probe_json(probe_json(audio=True)).has_audio

    def test_should_report_no_audio_when_none_is_present(self):
        assert not render.parse_probe_json(
            probe_json(audio=False)).has_audio

    def test_should_report_rotation_when_side_data_carries_it(self):
        info = render.parse_probe_json(probe_json(rotation=-180))

        assert info.rotation == -180

    def test_should_report_no_rotation_when_side_data_is_absent(self):
        assert render.parse_probe_json(probe_json()).rotation is None

    def test_should_raise_when_there_is_no_video_stream(self):
        payload = json.dumps({"streams": [
            {"index": 0, "codec_type": "audio", "codec_name": "aac"}]})

        with pytest.raises(render.RenderError, match="no video stream"):
            render.parse_probe_json(payload)

    def test_should_raise_when_the_payload_is_not_json(self):
        with pytest.raises(render.RenderError, match="ffprobe"):
            render.parse_probe_json("not json at all")


class TestVerifyOutput:
    def _info(self, **kw):
        return render.parse_probe_json(probe_json(**kw))

    def test_should_accept_a_correct_render(self):
        track = make_track(frames=192)

        render.verify_output(self._info(frames=track.total_frames),
                             track, make_meta())

    def test_should_raise_when_the_width_is_wrong(self):
        track = make_track()

        with pytest.raises(render.RenderError, match="geometry"):
            render.verify_output(
                self._info(width=606, frames=track.total_frames),
                track, make_meta())

    def test_should_raise_when_the_height_is_wrong(self):
        track = make_track()

        with pytest.raises(render.RenderError, match="geometry"):
            render.verify_output(
                self._info(height=1078, frames=track.total_frames),
                track, make_meta())

    @pytest.mark.parametrize("delta", [-2, -1, 0, 1, 2])
    def test_should_tolerate_a_two_frame_disagreement(self, delta):
        track = make_track()

        render.verify_output(
            self._info(frames=track.total_frames + delta),
            track, make_meta())

    @pytest.mark.parametrize("delta", [-3, 3, -50])
    def test_should_raise_on_a_larger_frame_disagreement(self, delta):
        track = make_track()

        with pytest.raises(render.RenderError, match="frame count"):
            render.verify_output(
                self._info(frames=track.total_frames + delta),
                track, make_meta())

    def test_should_raise_when_the_output_carries_rotation(self):
        track = make_track()

        with pytest.raises(render.RenderError, match="rotation"):
            render.verify_output(
                self._info(frames=track.total_frames, rotation=90),
                track, make_meta())

    def test_should_raise_when_audio_was_lost(self):
        track = make_track()

        with pytest.raises(render.RenderError, match="silent"):
            render.verify_output(
                self._info(frames=track.total_frames, audio=False),
                track, make_meta(has_audio=True))

    def test_should_accept_no_audio_when_the_source_had_none(self):
        track = make_track()

        render.verify_output(
            self._info(frames=track.total_frames, audio=False),
            track, make_meta(has_audio=False))

    def test_should_raise_when_audio_and_video_start_disagree(self):
        track = make_track()

        with pytest.raises(render.RenderError, match="start offset"):
            render.verify_output(
                self._info(frames=track.total_frames,
                           video_start=0.5, audio_start=0.0),
                track, make_meta(has_audio=True))

    def test_should_tolerate_a_sub_frame_start_offset(self):
        track = make_track()

        render.verify_output(
            self._info(frames=track.total_frames,
                       video_start=1.0 / track.fps / 2, audio_start=0.0),
            track, make_meta(has_audio=True))


class TestDefaultRunner:
    def test_should_return_stdout_on_success(self, tmp_path):
        completed = subprocess.CompletedProcess(
            ["ffprobe"], 0, stdout="payload", stderr="")

        with patch("subprocess.run", return_value=completed):
            assert render._default_runner(["ffprobe"]) == "payload"

    def test_should_raise_when_the_binary_is_missing(self):
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            with pytest.raises(render.RenderError, match="not found"):
                render._default_runner(["ffmpeg", "-i", "x"])

    def test_should_raise_with_the_stderr_tail_on_a_non_zero_exit(self):
        error = subprocess.CalledProcessError(
            1, ["ffmpeg"], output="",
            stderr="line one\nline two\nInvalid argument\n")

        with patch("subprocess.run", side_effect=error):
            with pytest.raises(render.RenderError) as excinfo:
                render._default_runner(["ffmpeg"])

        assert "Invalid argument" in str(excinfo.value)
        assert "exit 1" in str(excinfo.value)

    def test_should_pass_the_cwd_through(self, tmp_path):
        completed = subprocess.CompletedProcess(["ffmpeg"], 0,
                                                stdout="", stderr="")

        with patch("subprocess.run", return_value=completed) as run:
            render._default_runner(["ffmpeg"], cwd=tmp_path)

        assert run.call_args.kwargs["cwd"] == str(tmp_path)


class TestCropsDocument:
    def _doc(self, **kw):
        return render.crops_document(
            kw.pop("meta", make_meta()), kw.pop("track", make_track()),
            kw.pop("shots", make_shots()),
            kw.pop("out_path", Path("/w/vertical.mp4")),
            kw.pop("settings", make_settings()), **kw)

    def test_should_stamp_the_schema_version(self):
        assert self._doc()["schema"] == render.CROPS_SCHEMA_VERSION

    def test_should_record_the_source_by_name_not_by_path(self):
        doc = self._doc(meta=make_meta(name="It's a clip.mp4"))

        assert doc["source"]["name"] == "It's a clip.mp4"
        assert "/" not in doc["source"]["name"]

    def test_should_flatten_every_source_meta_field(self):
        doc = self._doc()["source"]

        assert set(doc) == {
            "name", "width", "height", "rotation", "display_width",
            "display_height", "orientation", "fps", "duration_s",
            "has_audio",
        }

    def test_should_describe_the_output(self):
        doc = self._doc()["output"]

        assert doc["name"] == "vertical.mp4"
        assert doc["codec"] == "hevc_nvenc" and doc["container"] == "mp4"
        assert (doc["width"], doc["height"]) == (608, 1080)

    def test_should_record_the_real_frame_count_when_given_one(self):
        doc = self._doc(frames=191)

        assert doc["output"]["frames"] == 191
        assert doc["track"]["total_frames"] == 192

    def test_should_fall_back_to_the_planned_frame_count(self):
        doc = self._doc()

        assert doc["output"]["frames"] == doc["track"]["total_frames"]

    def test_should_record_the_planned_frame_count_and_the_drift(self):
        """Recorded as data so FRAME_TOLERANCE can be tightened on it."""
        doc = self._doc(frames=191)

        assert doc["output"]["planned_frames"] == 192
        assert doc["output"]["frame_drift"] == -1

    def test_should_record_zero_drift_when_the_frame_count_matches(self):
        doc = self._doc()

        assert doc["output"]["frame_drift"] == 0

    def test_should_record_the_encoding_quality_arguments(self):
        """Path-free quality args, not the full argv."""
        doc = self._doc()

        assert doc["output"]["video_args"] == render.video_args(
            "hevc_nvenc", make_settings())

    def test_should_record_the_target_aspect(self):
        assert self._doc()["aspect"] == {"w": 9, "h": 16}

    def test_should_hoist_success_criterion_two(self):
        sc2 = self._doc()["success_criterion_2"]

        assert sc2["threshold"] == 0.95
        assert set(sc2) == {
            "threshold", "fraction", "passes", "measurable",
            "observed_samples", "inside_samples", "clamped_samples",
            "switch_samples",
        }

    def test_should_hoist_measurability(self):
        """A batch that found nobody is not a clean pass."""
        nobody = make_empty_subject(0, 0, 192, 24.0)
        track = plan_camera([nobody], make_meta())
        doc = self._doc(track=track)

        assert doc["success_criterion_2"]["measurable"] is False

    def test_should_agree_with_the_nested_track_payload(self):
        doc = self._doc()
        sc2, track = doc["success_criterion_2"], doc["track"]

        assert sc2["fraction"] == track["dead_zone_fraction"]
        assert sc2["observed_samples"] == track["observed_samples"]
        assert sc2["inside_samples"] == track["inside_samples"]
        assert sc2["clamped_samples"] == track["clamped_samples"]
        assert sc2["switch_samples"] == track["switch_samples"]

    def test_should_carry_per_frame_subject_centres(self):
        """Without these, the dead-zone metric is unmeasurable."""
        shot = self._doc()["track"]["shots"][0]

        assert len(shot["subject_x"]) == len(shot["x"])
        assert len(shot["observed"]) == len(shot["x"])
        assert len(shot["inside"]) == len(shot["x"])

    def test_should_name_the_stride_the_track_was_planned_at(self):
        """The dead-zone metric is stride-sensitive, so the stride rides
        along."""
        assert self._doc()["track"]["stride"] == 2

    def test_should_serialise_to_json_and_back(self):
        doc = self._doc()

        assert json.loads(json.dumps(doc, ensure_ascii=True)) == doc


class TestWriteCrops:
    def test_should_write_two_space_indented_json(self, tmp_path):
        path = tmp_path / "crops.json"

        render.write_crops({"schema": 1, "a": {"b": 2}}, path)

        assert '\n  "a"' in path.read_text(encoding="utf-8")

    def test_should_end_with_a_newline(self, tmp_path):
        path = tmp_path / "crops.json"

        render.write_crops({"schema": 1}, path)

        assert path.read_bytes().endswith(b"\n")

    def test_should_escape_non_ascii_rather_than_emit_it(self, tmp_path):
        path = tmp_path / "crops.json"

        render.write_crops({"name": "café.mp4"}, path)

        raw = path.read_bytes()
        assert b"caf\\u00e9" in raw
        raw.decode("ascii")


class TestRenderVertical:
    def _call(self, tmp_path, runner, **kw):
        out = kw.pop("out_path", tmp_path / "vertical.mp4")
        return render.render_vertical(
            source=kw.pop("source", tmp_path / "source.mp4"),
            meta=kw.pop("meta", make_meta()),
            track=kw.pop("track", make_track()),
            shots=kw.pop("shots", make_shots()),
            out_path=out,
            crops_path=kw.pop("crops_path", tmp_path / "crops.json"),
            settings=kw.pop("settings", make_settings()),
            runner=runner,
        )

    def test_should_write_both_artifacts_at_their_final_names(
        self, tmp_path,
    ):
        result = self._call(tmp_path, FakeRunner())

        assert result.output.exists() and result.crops.exists()
        assert result.output.name == "vertical.mp4"
        assert result.crops.name == "crops.json"

    def test_should_leave_no_partial_behind(self, tmp_path):
        self._call(tmp_path, FakeRunner())

        assert not (tmp_path / "vertical.partial.mp4").exists()
        assert not (tmp_path / "crops.partial.json").exists()

    def test_should_leave_the_sendcmd_script_on_disk(self, tmp_path):
        result = self._call(tmp_path, FakeRunner())

        assert result.script.name == render.SENDCMD_NAME
        assert result.script.exists()

    def test_should_run_ffmpeg_from_the_output_directory(self, tmp_path):
        runner = FakeRunner()

        self._call(tmp_path, runner)

        assert Path(runner.ffmpeg_cwd) == tmp_path
        graph = runner.ffmpeg_argv[
            runner.ffmpeg_argv.index("-filter_complex") + 1]
        assert f"sendcmd=f={render.SENDCMD_NAME}," in graph

    def test_should_survive_an_apostrophe_in_the_work_directory(
        self, tmp_path,
    ):
        work = tmp_path / "It's a Test Clip part 04"
        work.mkdir()
        runner = FakeRunner()

        result = self._call(tmp_path, runner,
                            out_path=work / "vertical.mp4",
                            crops_path=work / "crops.json")

        graph = runner.ffmpeg_argv[
            runner.ffmpeg_argv.index("-filter_complex") + 1]
        assert "'" not in graph
        assert result.output.exists()

    def test_should_remove_a_stale_partial_before_rendering(
        self, tmp_path,
    ):
        stale = tmp_path / "vertical.partial.mp4"
        stale.write_bytes(b"junk from a killed run")
        stale_crops = tmp_path / "crops.partial.json"
        stale_crops.write_text("{}", encoding="ascii")

        # ffmpeg fails, so nothing rewrites either partial: whatever is
        # missing afterwards was removed by render_vertical itself.
        with pytest.raises(render.RenderError):
            self._call(tmp_path, FakeRunner(fail_on="ffmpeg"))

        assert not stale.exists()
        assert not stale_crops.exists()

    def test_should_validate_the_format_before_spawning_ffmpeg(
        self, tmp_path,
    ):
        runner = FakeRunner()
        settings = make_settings(codec="ffv1", container="mp4")

        with pytest.raises(render.RenderError, match="REFRAME_CONTAINER"):
            self._call(tmp_path, runner, settings=settings)

        assert runner.calls == []

    def test_should_not_rename_anything_when_ffmpeg_fails(
        self, tmp_path,
    ):
        with pytest.raises(render.RenderError):
            self._call(tmp_path, FakeRunner(fail_on="ffmpeg"))

        assert not (tmp_path / "vertical.mp4").exists()
        assert not (tmp_path / "crops.json").exists()

    def test_should_not_rename_anything_when_verification_fails(
        self, tmp_path,
    ):
        runner = FakeRunner(probe_payload=probe_json(width=606))

        with pytest.raises(render.RenderError, match="geometry"):
            self._call(tmp_path, runner)

        assert not (tmp_path / "vertical.mp4").exists()
        assert not (tmp_path / "crops.json").exists()
        assert (tmp_path / "vertical.partial.mp4").exists()

    def test_should_write_a_failed_crops_sibling_when_verification_fails(
        self, tmp_path,
    ):
        """The crop track survives a failed render for diagnosis."""
        runner = FakeRunner(probe_payload=probe_json(width=606))

        with pytest.raises(render.RenderError, match="crops.failed.json"):
            self._call(tmp_path, runner)

        failed = json.loads(
            (tmp_path / "crops.failed.json").read_text(encoding="ascii"))
        assert failed["track"]["shots"][0]["subject_x"]

    def test_should_keep_the_render_error_when_the_salvage_write_fails(
        self, tmp_path, monkeypatch,
    ):
        """A full disk breaks both the render and the salvage write. The
        render's reason is what the report needs, so it must survive."""
        runner = FakeRunner(probe_payload=probe_json(width=606))

        def full_disk(document, path):
            raise render.RenderError(f"could not write {path.name}: "
                                     "No space left on device")

        monkeypatch.setattr(render, "write_crops", full_disk)

        with pytest.raises(render.RenderError) as excinfo:
            self._call(tmp_path, runner)

        message = str(excinfo.value)
        assert "geometry" in message
        assert "crop track NOT saved" in message
        assert "No space left on device" in message

    def test_should_validate_audio_passthrough_before_spawning_ffmpeg(
        self, tmp_path,
    ):
        """PCM into mp4 fails outright with -c:a copy."""
        runner = FakeRunner(
            probe_payload=probe_json(audio_codec="pcm_s16le"))

        with pytest.raises(render.RenderError, match="REFRAME_CONTAINER"):
            self._call(tmp_path, runner)

        binaries = [Path(argv[0]).stem.lower()
                    for argv, _cwd in runner.calls]
        assert not any(b.startswith("ffmpeg") for b in binaries)

    def test_should_not_probe_audio_when_the_source_has_none(
        self, tmp_path,
    ):
        runner = FakeRunner()

        self._call(tmp_path, runner, meta=make_meta(has_audio=False))

        assert "a:0" not in " ".join(
            arg for argv, _cwd in runner.calls for arg in argv)

    def test_should_record_the_probed_frame_count_in_crops(
        self, tmp_path,
    ):
        runner = FakeRunner(probe_payload=probe_json(frames=191))

        result = self._call(tmp_path, runner)

        doc = json.loads(result.crops.read_text(encoding="ascii"))
        assert doc["output"]["frames"] == 191
        assert doc["track"]["total_frames"] == 192
        assert result.frames == 191 and result.planned_frames == 192
        assert result.drift == -1

    def test_should_populate_every_result_field(self, tmp_path):
        result = self._call(tmp_path, FakeRunner())

        assert result.codec == "hevc_nvenc" and result.container == "mp4"
        assert (result.width, result.height) == (608, 1080)
        assert result.commands > 0
        assert result.elapsed_s >= 0.0
        assert result.command[0].startswith("ffmpeg")

    def test_should_render_the_lossless_path_to_mkv(self, tmp_path):
        runner = FakeRunner(probe_payload=probe_json(codec="ffv1"))
        settings = make_settings(codec="ffv1", container="mkv")

        result = self._call(tmp_path, runner,
                            out_path=tmp_path / "vertical.mkv",
                            settings=settings)

        assert result.output.name == "vertical.mkv"
        assert "-slicecrc" in runner.ffmpeg_argv

    def test_should_use_the_configured_ffmpeg_and_its_ffprobe(
        self, tmp_path,
    ):
        runner = FakeRunner()
        settings = make_settings(ffmpeg="/opt/bin/ffmpeg")

        self._call(tmp_path, runner, settings=settings)

        binaries = [argv[0] for argv, _cwd in runner.calls]
        assert binaries == ["/opt/bin/ffprobe", "/opt/bin/ffmpeg",
                            "/opt/bin/ffprobe"]

    def test_should_use_the_default_runner_when_none_is_injected(
        self, tmp_path,
    ):
        """An injectable seam hides a disconnected default."""
        fake = FakeRunner()

        with patch.object(render, "_default_runner", fake):
            self._call(tmp_path, None)

        assert fake.calls

    def test_should_raise_when_ffmpeg_exits_zero_but_writes_nothing(
        self, tmp_path,
    ):
        def silent_runner(argv, *, cwd=None):
            if Path(argv[0]).stem.lower().startswith("ffprobe"):
                return probe_json()
            return ""

        with pytest.raises(render.RenderError, match="wrote nothing"):
            self._call(tmp_path, silent_runner)


class TestRenderSummary:
    def _result(self, tmp_path, runner=None):
        return render.render_vertical(
            source=tmp_path / "s.mp4", meta=make_meta(),
            track=make_track(), shots=make_shots(),
            out_path=tmp_path / "vertical.mp4",
            crops_path=tmp_path / "crops.json",
            settings=make_settings(), runner=runner or FakeRunner())

    def test_should_describe_the_render_in_one_line(self, tmp_path):
        line = render.render_summary(self._result(tmp_path))

        assert "608x1080" in line
        assert "vertical.mp4" in line
        assert "hevc_nvenc" in line

    def test_should_be_ascii(self, tmp_path):
        render.render_summary(self._result(tmp_path)).encode("ascii")

    def test_should_flag_a_frame_count_that_drifted(self, tmp_path):
        result = self._result(
            tmp_path,
            runner=FakeRunner(probe_payload=probe_json(frames=191)))

        assert "planned 192" in render.render_summary(result)


_DETECTIONS = object()          # the DetectionSet the fake detect emits


class _Stats:
    """The DetectStats fields run_stages reads, and nothing else."""

    sampled = 96
    decoded = 96
    ms_per_sampled_frame = 31.7


class TestRunStages:
    """The five-stage sequence. Every stage is a plain function: no Mock,
    no patch.multiple -- patching the five module attributes couples
    every test to pipeline.py's import style."""

    def _wire(self, tmp_path, recorder, *, titles=None, track=None):
        """A Stages bundle of plain local functions.

        `titles` overrides the title stage; left unset it
        records the call and reports one None per shot, which is the
        "no title card anywhere" answer.
        """
        from ftea.reframe.pipeline import Stages

        track = track if track is not None else make_track()
        shots = make_shots()
        subjects = [make_subject(0, 0, 192, 24.0, (0.3, 0.6))]
        dset = _DETECTIONS

        def fake_shots(meta, **kw):
            recorder.append(("shots", kw))
            return shots

        def fake_detect(meta, shots_in, settings, **kw):
            recorder.append(("detect", shots_in))
            return dset, _Stats()

        def fake_subjects(shots_in, detections, meta, **kw):
            recorder.append(("subject", detections))
            return subjects

        def fake_camera(subjects_in, meta, **kw):
            recorder.append(("camera", subjects_in))
            return track

        def fake_render(**kw):
            recorder.append(("render", kw))
            kw["out_path"].write_bytes(b"v")
            kw["crops_path"].write_text("{}", encoding="ascii")
            return render.RenderResult(
                output=kw["out_path"], crops=kw["crops_path"],
                script=tmp_path / render.SENDCMD_NAME,
                codec="hevc_nvenc", container="mp4", width=608,
                height=1080, planned_frames=192, frames=192,
                commands=7, elapsed_s=0.5, command=("ffmpeg",))

        def fake_titles(source_in, shots_in, meta, settings):
            recorder.append(("titles", shots_in))
            return (None,) * len(shots_in)

        return Stages(
            shots=fake_shots, detect=fake_detect, subjects=fake_subjects,
            camera=fake_camera, render=fake_render,
            titles=titles if titles is not None else fake_titles)

    def test_should_call_the_five_stages_in_order(self, tmp_path):
        from ftea.reframe.pipeline import _run_stages

        recorder = []
        _run_stages(source=tmp_path / "s.mp4", meta=make_meta(),
                    out_path=tmp_path / "vertical.mp4",
                    crops_path=tmp_path / "crops.json",
                    settings=make_settings(),
                    stages=self._wire(tmp_path, recorder))

        # "titles" sits between shots and detect: it
        # loads a model and decodes the clip, so a missing weights file
        # should fail in a second rather than after the detect pass.
        assert [name for name, _ in recorder] == [
            "shots", "titles", "detect", "subject", "camera", "render"]

    def test_should_thread_the_configured_shot_thresholds(self, tmp_path):
        from ftea.reframe.pipeline import _run_stages

        recorder = []
        settings = make_settings(scene_threshold=0.44, min_shot_sec=0.9)
        _run_stages(source=tmp_path / "s.mp4", meta=make_meta(),
                    out_path=tmp_path / "vertical.mp4",
                    crops_path=tmp_path / "crops.json",
                    settings=settings,
                    stages=self._wire(tmp_path, recorder))

        kwargs = dict(recorder)["shots"]
        assert kwargs["threshold"] == 0.44
        assert kwargs["min_shot_sec"] == 0.9

    SUBJECT_KEYS = (
        "face_weight", "hysteresis_margin", "hysteresis_sec",
        "centrality_weight", "confidence_weight", "persistence_weight",
        "min_confidence_person", "min_confidence_face", "iou_match_min",
        "track_gap_sec",
    )
    CAMERA_KEYS = (
        "aspect_w", "aspect_h", "dead_zone", "dead_zone_release",
        "smoothing_hz", "max_pan_frac_per_sec",
    )

    def test_should_hand_every_subject_and_camera_key_to_its_stage(
        self, tmp_path,
    ):
        """Every value is distinct, so a dropped key or two keys
        swapped in _run_stages changes what a stage receives."""
        from ftea.reframe.pipeline import _run_stages

        settings = make_settings(
            face_weight=3.1, hysteresis_margin=0.21, hysteresis_sec=1.7,
            centrality_weight=0.11, confidence_weight=0.12,
            persistence_weight=0.13, min_confidence_person=0.81,
            min_confidence_face=0.36, iou_match_min=0.31,
            track_gap_sec=0.27, aspect_w=4, aspect_h=5, dead_zone=0.55,
            dead_zone_release=0.44, smoothing_hz=0.77,
            max_pan_frac_per_sec=0.19, title_edge_margin=0.02,
        )
        values = [getattr(settings, key)
                  for key in self.SUBJECT_KEYS + self.CAMERA_KEYS]
        values.append(settings.title_edge_margin)
        assert len(set(values)) == len(values)
        seen = {}
        wired = self._wire(tmp_path, [])

        def subjects(shots_in, detections, meta, **kwargs):
            seen["subject"] = kwargs
            return wired.subjects(shots_in, detections, meta)

        def camera(subjects_in, meta, **kwargs):
            seen["camera"] = kwargs
            return wired.camera(subjects_in, meta)

        _run_stages(source=tmp_path / "s.mp4", meta=make_meta(),
                    out_path=tmp_path / "vertical.mp4",
                    crops_path=tmp_path / "crops.json",
                    settings=settings,
                    stages=dataclasses.replace(
                        wired, subjects=subjects, camera=camera))

        assert seen["subject"] == {
            key: getattr(settings, key) for key in self.SUBJECT_KEYS}
        camera_kwargs = dict(seen["camera"])
        assert camera_kwargs.pop("titles") == (None,)
        assert camera_kwargs == {
            **{key: getattr(settings, key) for key in self.CAMERA_KEYS},
            "edge_margin": settings.title_edge_margin,
        }

    def test_should_hand_the_detections_to_the_subject_stage(
        self, tmp_path,
    ):
        from ftea.reframe.pipeline import _run_stages

        recorder = []
        _run_stages(source=tmp_path / "s.mp4", meta=make_meta(),
                    out_path=tmp_path / "vertical.mp4",
                    crops_path=tmp_path / "crops.json",
                    settings=make_settings(),
                    stages=self._wire(tmp_path, recorder))

        record = dict(recorder)
        assert record["subject"] is _DETECTIONS

    def test_should_hand_render_the_shots_and_the_track(self, tmp_path):
        from ftea.reframe.pipeline import _run_stages

        recorder = []
        _run_stages(source=tmp_path / "s.mp4", meta=make_meta(),
                    out_path=tmp_path / "vertical.mp4",
                    crops_path=tmp_path / "crops.json",
                    settings=make_settings(),
                    stages=self._wire(tmp_path, recorder))

        kwargs = dict(recorder)["render"]
        assert kwargs["shots"] and kwargs["track"] is not None
        assert kwargs["out_path"].name == "vertical.mp4"

    def test_should_leave_both_artifacts_in_place(self, tmp_path):
        from ftea.reframe.pipeline import _run_stages

        _run_stages(source=tmp_path / "s.mp4", meta=make_meta(),
                    out_path=tmp_path / "vertical.mp4",
                    crops_path=tmp_path / "crops.json",
                    settings=make_settings(),
                    stages=self._wire(tmp_path, []))

        assert (tmp_path / "vertical.mp4").exists()
        assert (tmp_path / "crops.json").exists()

    def test_should_print_one_line_for_every_stage_it_owns(
        self, tmp_path, capsys,
    ):
        from ftea.reframe.pipeline import _run_stages

        _run_stages(source=tmp_path / "s.mp4", meta=make_meta(),
                    out_path=tmp_path / "vertical.mp4",
                    crops_path=tmp_path / "crops.json",
                    settings=make_settings(),
                    stages=self._wire(tmp_path, []))

        lines = [line for line in capsys.readouterr().out.splitlines()
                 if line.strip()]
        # shots, titles, subject, camera, render -- detect logs its own
        # richer line from inside detect_from_settings.
        assert len(lines) == 5
        assert all(line.startswith("  reframe: ") for line in lines)

    def test_should_let_an_upstream_error_propagate_untouched(
        self, tmp_path,
    ):
        from ftea.reframe.pipeline import DEFAULT_STAGES, _run_stages

        def boom(meta, **kw):
            raise ReframeError("shot detection failed: truncated file")

        stages = dataclasses.replace(DEFAULT_STAGES, shots=boom)
        with pytest.raises(ReframeError, match="truncated file"):
            _run_stages(source=tmp_path / "s.mp4", meta=make_meta(),
                        out_path=tmp_path / "vertical.mp4",
                        crops_path=tmp_path / "crops.json",
                        settings=make_settings(), stages=stages)

    def test_should_no_longer_raise_not_implemented(self, tmp_path):
        from ftea.reframe.pipeline import _run_stages

        _run_stages(source=tmp_path / "s.mp4", meta=make_meta(),
                    out_path=tmp_path / "vertical.mp4",
                    crops_path=tmp_path / "crops.json",
                    settings=make_settings(),
                    stages=self._wire(tmp_path, []))

    def test_should_wire_the_real_stage_functions_by_default(self):
        """The DEFAULT_STAGES guard: assert the real default is actually
        wired, not just that a fake one works."""
        from ftea.reframe.camera import plan_camera as real_camera
        from ftea.reframe.detect import detect_from_settings
        from ftea.reframe.pipeline import DEFAULT_STAGES
        from ftea.reframe.render import render_vertical
        from ftea.reframe.shots import detect_shots
        from ftea.reframe.subject import select_subjects

        assert DEFAULT_STAGES.shots is detect_shots
        assert DEFAULT_STAGES.detect is detect_from_settings
        assert DEFAULT_STAGES.subjects is select_subjects
        assert DEFAULT_STAGES.camera is real_camera
        assert DEFAULT_STAGES.render is render_vertical

    def test_should_default_the_titles_stage_to_probe_clip(self):
        from ftea.reframe.pipeline import DEFAULT_STAGES
        from ftea.reframe.text import probe_clip

        assert DEFAULT_STAGES.titles is probe_clip

    def test_should_probe_titles_between_shots_and_detection(
        self, tmp_path,
    ):
        """Before detection, so a missing model fails fast rather than
        after the expensive detect pass."""
        from ftea.reframe.pipeline import _run_stages

        recorder = []
        _run_stages(source=tmp_path / "s.mp4", meta=make_meta(),
                    out_path=tmp_path / "vertical.mp4",
                    crops_path=tmp_path / "crops.json",
                    settings=make_settings(),
                    stages=self._wire(tmp_path, recorder))

        assert [name for name, _ in recorder][:3] == [
            "shots", "titles", "detect"]

    def test_should_thread_the_cards_into_plan_camera(self, tmp_path):
        from ftea.reframe.pipeline import _run_stages

        from ftea.reframe.text import TitleCard

        # A real TitleCard, not a sentinel string: _run_stages prints
        # title_summary(titles) on the way past, so the value has to be
        # the thing text.probe_clip actually returns.
        card = TitleCard(x1=462.0, y1=222.0, x2=1458.0, y2=841.0,
                         centre_x=960.0, line_count=5, frames_seen=5,
                         frames_sampled=5)
        seen = {}
        base = self._wire(tmp_path, [],
                          titles=lambda s, sh, m, st: (card,))

        def fake_camera(subjects_in, meta, **kwargs):
            seen.update(kwargs)
            return make_track()

        _run_stages(source=tmp_path / "s.mp4", meta=make_meta(),
                    out_path=tmp_path / "vertical.mp4",
                    crops_path=tmp_path / "crops.json",
                    settings=make_settings(),
                    stages=dataclasses.replace(base, camera=fake_camera))

        assert seen["titles"] == (card,)
        assert seen["titles"][0] is card
        assert seen["edge_margin"] == 0.01

    def test_should_pass_none_through_when_the_probe_is_disabled(
        self, tmp_path,
    ):
        """REFRAME_TITLE_ENABLED=false must reach plan_camera as None --
        not a tuple of Nones -- because None is the exact value that
        reproduces the trajectory without title detection bit-for-bit."""
        from ftea.reframe.pipeline import _run_stages
        from ftea.reframe.text import probe_clip

        seen = {}
        base = self._wire(tmp_path, [], titles=probe_clip)

        def fake_camera(subjects_in, meta, **kwargs):
            seen.update(kwargs)
            return make_track()

        _run_stages(
            source=tmp_path / "s.mp4", meta=make_meta(),
            out_path=tmp_path / "vertical.mp4",
            crops_path=tmp_path / "crops.json",
            settings=make_settings(title_enabled=False),
            stages=dataclasses.replace(base, camera=fake_camera))

        assert seen["titles"] is None

    def test_should_print_the_title_summary(self, tmp_path, capsys):
        from ftea.reframe.pipeline import _run_stages

        _run_stages(source=tmp_path / "s.mp4", meta=make_meta(),
                    out_path=tmp_path / "vertical.mp4",
                    crops_path=tmp_path / "crops.json",
                    settings=make_settings(),
                    stages=self._wire(tmp_path, []))

        assert "title" in capsys.readouterr().out

    def test_should_delegate_to_the_default_stages_impl(self, tmp_path):
        """The published run_stages is a thin wrapper around
        _run_stages."""
        from ftea.reframe import pipeline

        calls = []

        def fake_impl(*args, **kwargs):
            calls.append((args, kwargs))

        with patch.object(pipeline, "_run_stages", fake_impl):
            pipeline.run_stages(
                source=tmp_path / "s.mp4", meta=make_meta(),
                out_path=tmp_path / "vertical.mp4",
                crops_path=tmp_path / "crops.json",
                settings=make_settings())

        assert len(calls) == 1

    def test_should_expose_every_settings_field_run_stages_reads(self):
        """Eleven of these fields were ADDED to the original settings
        dataclass by later stages. A missing one must fail
        here, by name, at pytest time -- not as an AttributeError
        mid-sequence on the first clip that reaches it."""
        sibling_owned = (
            "scene_threshold", "min_shot_sec", "detect_long_side",
            "min_confidence_person", "min_confidence_face",
            "centrality_weight", "confidence_weight",
            "persistence_weight", "iou_match_min", "track_gap_sec",
            "dead_zone_release",
        )
        names = {f.name for f in dataclasses.fields(ReframeSettings)}
        missing = [name for name in sibling_owned if name not in names]

        assert missing == []


@pytest.mark.ffmpeg
@pytest.mark.skipif(shutil.which("ffmpeg") is None,
                    reason="ffmpeg not on PATH")
class TestRealFfmpegRender:
    """The smallest end-to-end proof that the command shape is real.

    Renders through ffv1/mkv, not the NVENC production default, so this
    proof runs on every machine with ffmpeg -- including one with no
    NVIDIA GPU. See TestRealNvencRender below for the NVENC-specific
    proof.

    640x360 keeps it under a second: round_up_even(360 * 9 / 16) =
    round_up_even(202.5) = 204, which also exercises a crop width that is
    NOT one of the three standard tiers -- the "one code path, no
    resolution tiers" rule in miniature.
    """

    FPS = 24
    FRAMES = 48
    CROP_W = 204

    def _source(self, tmp_path):
        source = tmp_path / "It's a source.mp4"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i",
             f"testsrc2=size=640x360:rate={self.FPS}:duration=2",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-shortest", str(source)],
            check=True, capture_output=True)
        return source

    def _render(self, tmp_path, *, settings=None,
                out_name="vertical.mkv"):
        work = tmp_path / "It's a Test Clip part 04"
        work.mkdir()
        meta = make_meta(640, 360, fps=float(self.FPS), duration_s=2.0)
        track = make_track(640, 360, fps=float(self.FPS),
                           frames=self.FRAMES)
        return render.render_vertical(
            source=self._source(tmp_path), meta=meta, track=track,
            shots=make_shots(self.FRAMES, float(self.FPS)),
            out_path=work / out_name,
            crops_path=work / "crops.json",
            settings=settings or make_settings(
                codec="ffv1", container="mkv"))

    def test_should_render_the_computed_crop_width(self, tmp_path):
        result = self._render(tmp_path)

        assert (result.width, result.height) == (self.CROP_W, 360)
        assert abs(result.frames - self.FRAMES) <= render.FRAME_TOLERANCE

    def test_should_carry_the_sources_audio(self, tmp_path):
        result = self._render(tmp_path)

        info = render.probe_rendered(result.output)
        assert info.has_audio

    def test_should_actually_move_the_crop_window(self, tmp_path):
        result = self._render(tmp_path)

        done = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-i", str(result.output), "-f", "framehash",
             "-hash", "md5", "-"],
            check=True, capture_output=True, text=True)
        hashes = {line.rsplit(",", 1)[-1].strip()
                  for line in done.stdout.splitlines()
                  if line and not line.startswith("#")}

        assert len(hashes) > 1

    def test_should_render_both_states_at_a_constant_geometry(
        self, tmp_path,
    ):
        """The smallest end-to-end proof that the canvas graph is real.

        640x360 source, crop 204x360 (round_up_even(202.5) = 204), so
        the canvas is 640 x even_round(640 * 360 / 204) and the middle
        shot letterboxes while the outer two do not. The assertion is
        the one that matters: a CONSTANT output geometry across a
        treatment change, which is what the static scale exists for.
        """
        from ftea.reframe.camera import canvas_for, plan_camera
        from ftea.reframe.text import TitleCard

        work = tmp_path / "It's a Test Clip part 04"
        work.mkdir()
        fps, frames = float(self.FPS), self.FRAMES
        per_shot = frames // 3
        subjects = [
            make_subject(i, i * per_shot, (i + 1) * per_shot, fps,
                         (0.2, 0.5, 0.8))
            for i in range(3)
        ]
        wide = TitleCard(x1=20.0, y1=100.0, x2=620.0, y2=260.0,
                         centre_x=320.0, line_count=5, frames_seen=5,
                         frames_sampled=5)
        meta = make_meta(640, 360, fps=fps, duration_s=frames / fps)
        track = plan_camera(subjects, meta, titles=[None, wide, None])

        assert track.canvas == canvas_for(track.window)
        assert track.shots[1].treatment == "letterbox"

        result = render.render_vertical(
            source=self._source(tmp_path), meta=meta, track=track,
            shots=make_shots(frames, fps),
            out_path=work / "vertical.mkv",
            crops_path=work / "crops.json",
            settings=make_settings(codec="ffv1", container="mkv"),
        )

        assert (result.width, result.height) == (self.CROP_W, 360)
        assert abs(result.frames - frames) <= render.FRAME_TOLERANCE
        doc = json.loads(
            (work / "crops.json").read_text(encoding="ascii"))
        assert doc["schema"] == 2
        assert doc["canvas"]["width"] == 640
        assert [s["treatment"] for s in doc["track"]["shots"]] == \
            ["crop", "letterbox", "crop"]

    def test_should_render_a_title_free_clip_identically(self, tmp_path):
        """The regression contract, in pixels.

        The same source rendered through a track with no canvas must
        produce a byte-identical video stream whether or not this
        feature exists, because the ffmpeg invocation is literally the
        same one. ffv1 is lossless and framemd5 hashes decoded frames,
        so this compares PICTURES rather than container bytes -- two
        encodes of identical input can differ in muxer metadata while
        the pixels match, and it is the pixels that are promised.
        """
        source = self._source(tmp_path)
        hashes = []
        for run in ("a", "b"):
            work = tmp_path / f"run-{run}"
            work.mkdir()
            render.render_vertical(
                source=source, meta=make_meta(640, 360,
                                              fps=float(self.FPS),
                                              duration_s=2.0),
                track=make_track(640, 360, fps=float(self.FPS),
                                 frames=self.FRAMES),
                shots=make_shots(self.FRAMES, float(self.FPS)),
                out_path=work / "vertical.mkv",
                crops_path=work / "crops.json",
                settings=make_settings(codec="ffv1", container="mkv"),
            )
            digest = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error",
                 "-i", str(work / "vertical.mkv"),
                 "-map", "0:v:0", "-f", "framemd5", "-"],
                check=True, capture_output=True, text=True).stdout
            hashes.append([line for line in digest.splitlines()
                           if not line.startswith("#")])

        assert hashes[0] == hashes[1]
        assert len(hashes[0]) == self.FRAMES


class TestProbeNvenc:
    def test_should_report_no_nvenc_when_listed_but_no_session_opens(
        self, monkeypatch,
    ):
        """A static build lists hevc_nvenc even when no driver can open it."""
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ffmpeg")

        def fake_run(argv, **kwargs):
            if "-encoders" in argv:
                return subprocess.CompletedProcess(
                    argv, 0, stdout=" V....D hevc_nvenc", stderr="")
            return subprocess.CompletedProcess(
                argv, 1, stdout="", stderr="Cannot load libcuda.so.1")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert _probe_nvenc() is False


def _probe_nvenc() -> bool:
    """True when the ffmpeg on PATH can actually open an hevc_nvenc session.

    A listed encoder is not a working encoder: a static build lists
    hevc_nvenc whether or not a driver can open it, so encode one frame.
    """
    if shutil.which("ffmpeg") is None:
        return False
    try:
        done = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "lavfi",
             "-i", "color=s=256x144:d=0.04", "-frames:v", "1",
             "-c:v", "hevc_nvenc", "-f", "null", "-"],
            capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return False
    return done.returncode == 0


_HAS_NVENC = _probe_nvenc()


@pytest.mark.ffmpeg
@pytest.mark.skipif(not _HAS_NVENC, reason="no hevc_nvenc encoder")
class TestRealNvencRender(TestRealFfmpegRender):
    """The same proof, through the production default: hevc_nvenc/mp4.

    Needs an NVIDIA GPU, so this class runs only where one is present
    (a Windows or Linux machine with an NVIDIA GPU); TestRealFfmpegRender
    above is what checks the fakes against reality everywhere else.
    """

    def _render(self, tmp_path, *, settings=None,
                out_name="vertical.mp4"):
        return super()._render(
            tmp_path, settings=settings or make_settings(),
            out_name=out_name)


# --- title detection: two graph shapes and the rect command stream --

def make_title_track(width=1920, height=1080, *, fps=24.0, frames=192,
                     treatments=("letterbox",)):
    """A CropTrack whose first shot carries a title card.

    Built through plan_camera, not by hand: a hand-built track would
    let render.py's tests pass against geometry camera.py would never
    produce.
    """
    from ftea.reframe.camera import plan_camera
    from ftea.reframe.text import TitleCard

    per_shot = frames // len(treatments)
    subjects = [
        make_subject(index, index * per_shot, (index + 1) * per_shot,
                     fps, (0.2, 0.5, 0.8))
        for index in range(len(treatments))
    ]
    titles = []
    for treatment in treatments:
        if treatment is None:
            titles.append(None)
            continue
        card_width = 996.0 if treatment == "letterbox" else 400.0
        titles.append(TitleCard(
            x1=960.0 - card_width / 2.0, y1=222.0,
            x2=960.0 + card_width / 2.0, y2=841.0, centre_x=960.0,
            line_count=5, frames_seen=5, frames_sampled=5))
    return plan_camera(
        subjects, make_meta(width, height, fps=fps,
                            duration_s=frames / fps),
        titles=titles)


class TestFiltergraphStaysTheSameWithoutACanvas:
    def test_should_emit_todays_graph_byte_for_byte(self):
        """The common path. Not "an equivalent graph" -- the same
        string, so a clip with no title cannot change at all."""
        track = make_track()

        assert render.filtergraph(track) == (
            "[0:v:0]setpts=PTS-STARTPTS,sendcmd=f=crop.sendcmd,"
            "crop=608:1080:%d:0[v]" % track.flatten()[0]
        )

    def test_should_emit_todays_command_stream_byte_for_byte(self):
        track = make_track()

        lines = render.sendcmd_lines(track)

        assert all(line.startswith(tuple("0123456789"))
                   for line in lines)
        assert all(" crop x " in line for line in lines)
        assert not any(" crop w " in line for line in lines)


class TestFiltergraphWithACanvas:
    def test_should_pad_then_crop_then_scale(self):
        """Exactly the padded graph that was verified end to end."""
        track = make_title_track(frames=192, treatments=("letterbox",))

        assert render.filtergraph(track) == (
            "[0:v:0]setpts=PTS-STARTPTS,"
            "pad=1920:3410:0:1165:black,"
            "sendcmd=f=crop.sendcmd,"
            "crop=1920:3410:0:0,"
            "scale=608:1080[v]"
        )

    def test_should_seed_the_crop_with_frame_zeros_rectangle(self):
        """The graph must already be correct before the first sendcmd
        command fires."""
        track = make_title_track(frames=192,
                                 treatments=(None, "letterbox"))

        graph = render.filtergraph(track)

        assert "crop=608:1080:" in graph
        assert graph.endswith(":1165,scale=608:1080[v]")

    def test_should_keep_the_script_name_a_bare_filename(self):
        """An apostrophe anywhere in a lavfi option value is
        unescapable, and the per-clip work dir is named from the clip
        filename. The only safe value is one this module chose."""
        track = make_title_track()

        assert "f=crop.sendcmd," in render.filtergraph(track)
        assert "/" not in render.filtergraph(track)


class TestRectCommandStream:
    def test_should_command_all_four_components_at_frame_zero(self):
        track = make_title_track(frames=192, treatments=("letterbox",))

        lines = render.sendcmd_lines(track)

        assert lines[0] == (
            "0.000000 crop w 1920, crop h 3410, crop x 0, crop y 0;")

    def test_should_command_all_four_at_a_treatment_change(self):
        track = make_title_track(frames=192,
                                 treatments=("letterbox", None))

        lines = render.sendcmd_lines(track)
        at_cut = [line for line in lines if line.startswith("3.979167")]

        assert at_cut
        assert " crop w 608," in at_cut[0]
        assert " crop h 1080," in at_cut[0]
        assert " crop y 1165;" in at_cut[0]

    def test_should_command_only_x_while_panning(self):
        track = make_title_track(frames=192,
                                 treatments=("letterbox", None))

        lines = render.sendcmd_lines(track)
        panning = [line for line in lines
                   if " crop w " not in line and line != lines[0]]

        assert panning
        assert all(line.count("crop ") == 1 for line in panning)

    def test_should_keep_the_component_order_w_h_x_y(self):
        """The verified script is `crop w, crop h, crop x, crop y` and
        it rendered exit 0. Do not reorder it on the theory that the
        order cannot matter -- it is what was measured."""
        track = make_title_track(frames=192, treatments=("letterbox",))

        line = render.sendcmd_lines(track)[0]
        positions = [line.index(f" crop {k} ") for k in "whxy"]

        assert positions == sorted(positions)

    def test_should_timestamp_half_a_frame_early(self):
        track = make_title_track(frames=192,
                                 treatments=("letterbox", None))

        lines = render.sendcmd_lines(track)

        assert lines[0].startswith("0.000000 ")
        assert any(line.startswith("3.979167") for line in lines)

    def test_should_compare_timestamps_as_floats_not_strings(self):
        """`"10.020833" <= "9.979167"` is TRUE as a string compare, and
        that killed every clip over ten seconds until a test caught it.
        A 20 s clip crossing the ten-second boundary must not raise."""
        track = make_title_track(frames=480, fps=24.0,
                                 treatments=("letterbox", None))

        lines = render.sendcmd_lines(track)

        assert any(line.startswith("9.") for line in lines)
        assert any(line.startswith("10.") for line in lines)

    def test_should_write_a_header_naming_the_canvas(self, tmp_path):
        track = make_title_track(frames=192, treatments=("letterbox",))
        path = tmp_path / "crop.sendcmd"

        render.write_sendcmd(track, path)
        header = path.read_text(encoding="ascii").splitlines()[0]

        assert header.startswith("#")
        assert "canvas 1920x3410" in header

    def test_should_write_ascii_with_lf_endings(self, tmp_path):
        track = make_title_track(frames=192, treatments=("letterbox",))
        path = tmp_path / "crop.sendcmd"

        render.write_sendcmd(track, path)

        assert b"\r\n" not in path.read_bytes()
        assert path.read_bytes().isascii()


class TestCropsDocumentSchema2:
    def test_should_declare_schema_two(self):
        doc = render.crops_document(
            make_meta(), make_track(), make_shots(), Path("v.mp4"),
            make_settings())

        assert doc["schema"] == 2

    def test_should_omit_the_canvas_without_one(self):
        doc = render.crops_document(
            make_meta(), make_track(), make_shots(), Path("v.mp4"),
            make_settings())

        assert "canvas" not in doc

    def test_should_publish_the_canvas_when_a_shot_letterboxes(self):
        track = make_title_track(frames=192, treatments=("letterbox",))

        doc = render.crops_document(
            make_meta(), track, make_shots(192), Path("v.mp4"),
            make_settings())

        assert doc["canvas"] == {
            "width": 1920, "height": 3410, "pad_x": 0, "pad_y": 1165}

    def test_should_publish_a_treatment_on_every_shot(self):
        track = make_title_track(frames=192,
                                 treatments=("letterbox", None))

        doc = render.crops_document(
            make_meta(), track, make_shots(192), Path("v.mp4"),
            make_settings())

        assert [s["treatment"] for s in doc["track"]["shots"]] == \
            ["letterbox", "crop"]

    def test_should_stay_ascii_and_json_serialisable(self, tmp_path):
        track = make_title_track(frames=192, treatments=("letterbox",))
        doc = render.crops_document(
            make_meta(), track, make_shots(192), Path("v.mp4"),
            make_settings())
        path = tmp_path / "crops.json"

        render.write_crops(doc, path)

        assert path.read_bytes().isascii()
        assert json.loads(
            path.read_text(encoding="ascii"))["schema"] == 2


class TestVerifyOutputIsUnchangedByTheCanvas:
    def test_should_still_demand_the_planned_crop_geometry(self):
        """The static scale puts the letterbox state back to
        crop_w x crop_h, so the output geometry check is exactly the
        same check it was before."""
        track = make_title_track(frames=192, treatments=("letterbox",))
        info = render.parse_probe_json(
            probe_json(width=608, height=1080, frames=192))

        render.verify_output(info, track, make_meta())

    def test_should_reject_a_canvas_sized_output(self):
        track = make_title_track(frames=192, treatments=("letterbox",))
        info = render.parse_probe_json(
            probe_json(width=1920, height=3410, frames=192))

        with pytest.raises(ReframeError) as excinfo:
            render.verify_output(info, track, make_meta())

        assert "does not match the planned crop" in str(excinfo.value)


def make_two_shot_track(fps=24.0):
    """A real CropTrack with a cut at frame 107 (4.458 s at 24 fps)."""
    shots = [make_subject(0, 0, 107, fps, (0.3, 0.4)),
             make_subject(1, 107, 192, fps, (0.6, 0.7))]
    return plan_camera(shots, make_meta(fps=fps, duration_s=192 / fps))


class TestKeyframeArgs:
    def test_should_force_a_keyframe_at_every_shot_start_but_the_first(self):
        # 226 / 24 = 9.41667 is written 9.416, at or before the frame:
        # rounded to nearest (9.417) the keyframe lands one frame late.
        assert render.keyframe_args("hevc_nvenc", [0, 107, 226], 24.0) == [
            "-force_key_frames", "4.458,9.416", "-forced-idr", "1",
        ]

    def test_should_add_nothing_for_the_lossless_render(self):
        """FFV1 renders every frame as a keyframe (-g 1) and has no
        -forced-idr option."""
        codec = next(iter(render.LOSSLESS_CODECS))
        assert render.keyframe_args(codec, [0, 107], 24.0) == []

    def test_should_add_nothing_for_a_single_shot(self):
        assert render.keyframe_args("hevc_nvenc", [0], 24.0) == []

    def test_should_skip_forcing_when_the_list_would_be_too_long(
        self, capsys,
    ):
        starts = range(0, (render.MAX_FORCED_KEYFRAMES + 2) * 10, 10)
        assert render.keyframe_args("hevc_nvenc", list(starts), 24.0) == []
        assert "keyframes" in capsys.readouterr().err

    def test_should_put_the_shot_keyframes_in_the_render_command(
        self, tmp_path,
    ):
        argv = render.build_command(
            tmp_path / "s.mp4", tmp_path / "o.mp4", make_two_shot_track(),
            make_settings(codec="hevc_nvenc"),
        )
        flag = argv.index("-force_key_frames")
        assert argv[flag + 1] == "4.458"
        assert flag > argv.index("-c:v")

    def test_should_force_no_keyframe_for_a_one_shot_render(self, tmp_path):
        argv = render.build_command(
            tmp_path / "s.mp4", tmp_path / "o.mp4", make_track(),
            make_settings(codec="hevc_nvenc"),
        )
        assert "-force_key_frames" not in argv
