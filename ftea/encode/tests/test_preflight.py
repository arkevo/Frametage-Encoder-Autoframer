"""Tests for ftea.encode.preflight.check_toolchain."""

from __future__ import annotations

import subprocess

import pytest

from ftea.encode import preflight
from ftea.encode.config import PipelineConfig
from ftea.encode.tools import ToolError

CONFIG = PipelineConfig()
ENCODERS = (
    " V....D h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)\n"
    " V....D av1_nvenc            NVIDIA NVENC av1 encoder (codec av1)\n"
    " V....D hevc_nvenc           NVIDIA NVENC hevc encoder (codec hevc)\n"
    " V....D libwebp              libwebp WebP image (codec webp)\n"
)
DECODERS = (
    " V..... libdav1d             dav1d AV1 decoder by VideoLAN (codec av1)\n"
    " V....D av1                  Alliance for Open Media AV1\n"
)
FILTERS = (
    " ... libvmaf           VV->V      Calculate the VMAF between two "
    "video streams.\n"
)


class FakeFfmpeg:
    """Answers run_tool the way ffmpeg would, and records trial encodes.

    Bare trials are recorded by encoder name. Trials with the real encode
    options (they carry -preset) are recorded as "<encoder> (real)", and
    `rejects` maps an encoder to the stderr its real trial fails with.
    """

    def __init__(
        self, *, encoders=ENCODERS, filters=FILTERS, failing=(),
        rejects=None, decoders=DECODERS,
    ):
        self.encoders = encoders
        self.filters = filters
        self.decoders = decoders
        self.failing = set(failing)
        self.rejects = dict(rejects or {})
        self.trials: list[str] = []
        self.real_commands: list[list[str]] = []
        self.samples: list[str] = []

    def __call__(self, cmd, **kwargs):
        cmd = [str(part) for part in cmd]
        if "-encoders" in cmd:
            return subprocess.CompletedProcess(cmd, 0, self.encoders, "")
        if "-filters" in cmd:
            return subprocess.CompletedProcess(cmd, 0, self.filters, "")
        if "-decoders" in cmd:
            return subprocess.CompletedProcess(cmd, 0, self.decoders, "")
        encoder = cmd[cmd.index("-c:v") + 1]
        if encoder == "ffv1":
            self.samples.append(cmd[-1])
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if "-preset" in cmd:
            self.trials.append(f"{encoder} (real)")
            self.real_commands.append(cmd)
            stderr = self.rejects.get(encoder)
            return subprocess.CompletedProcess(
                cmd, 1 if stderr else 0, "", stderr or "",
            )
        self.trials.append(encoder)
        if encoder in self.failing:
            return subprocess.CompletedProcess(
                cmd, 1, "",
                "OpenEncodeSessionEx failed: unsupported device (2)\n",
            )
        return subprocess.CompletedProcess(cmd, 0, "", "")


@pytest.fixture
def on_path(monkeypatch):
    monkeypatch.setattr(
        preflight.shutil, "which", lambda name: f"/usr/bin/{name}",
    )


def _use(monkeypatch, fake):
    monkeypatch.setattr(preflight, "run_tool", fake)
    return fake


class TestCheckToolchain:
    def test_should_report_nothing_on_a_ready_machine(
        self, monkeypatch, on_path,
    ):
        fake = _use(monkeypatch, FakeFfmpeg())

        assert preflight.check_toolchain(CONFIG) == []
        assert fake.trials == [
            "h264_nvenc", "av1_nvenc", "hevc_nvenc",
            "av1_nvenc (real)", "h264_nvenc (real)",
        ]
        assert len(fake.samples) == 1

    @pytest.mark.parametrize("missing", ["ffmpeg", "ffprobe"])
    def test_should_name_a_missing_program_and_the_guide(
        self, monkeypatch, missing,
    ):
        monkeypatch.setattr(
            preflight.shutil, "which",
            lambda name: None if name == missing else f"/usr/bin/{name}",
        )
        fake = _use(monkeypatch, FakeFfmpeg())

        problems = preflight.check_toolchain(CONFIG)

        assert len(problems) == 1
        assert problems[0].startswith(f"{missing} is not on PATH")
        assert "docs/PROCESS-GUIDE.md" in problems[0]
        assert fake.trials == []

    def test_should_name_an_encoder_the_build_lacks(
        self, monkeypatch, on_path,
    ):
        no_av1 = "".join(
            line + "\n" for line in ENCODERS.splitlines()
            if "av1_nvenc" not in line
        )
        fake = _use(monkeypatch, FakeFfmpeg(encoders=no_av1))

        problems = preflight.check_toolchain(CONFIG)

        assert len(problems) == 1 and "av1_nvenc" in problems[0]
        assert fake.trials == []

    def test_should_not_mistake_a_longer_name_for_the_encoder(
        self, monkeypatch, on_path,
    ):
        lookalike = ENCODERS.replace("av1_nvenc ", "av1_nvenc_x ")
        _use(monkeypatch, FakeFfmpeg(encoders=lookalike))

        assert any("av1_nvenc" in p for p in preflight.check_toolchain(CONFIG))

    def test_should_say_when_libvmaf_is_missing(self, monkeypatch, on_path):
        _use(monkeypatch, FakeFfmpeg(filters=" ... scale  V->V  Scale.\n"))

        problems = preflight.check_toolchain(CONFIG)

        assert len(problems) == 1 and "libvmaf" in problems[0]

    def test_should_say_when_the_webp_thumbnail_cannot_be_made(
        self, monkeypatch, on_path,
    ):
        """Packaging runs last, after every encode of the clip."""
        no_webp = ENCODERS.replace("libwebp ", "libwebp_anim ")
        fake = _use(monkeypatch, FakeFfmpeg(encoders=no_webp))

        problems = preflight.check_toolchain(CONFIG)

        assert len(problems) == 1
        assert "libwebp" in problems[0] and "THUMB_FORMAT" in problems[0]
        assert fake.trials == []

    def test_should_not_need_libwebp_for_a_jpg_thumbnail(
        self, monkeypatch, on_path,
    ):
        no_webp = ENCODERS.replace("libwebp ", "libwebp_anim ")
        _use(monkeypatch, FakeFfmpeg(encoders=no_webp))

        config = PipelineConfig(thumb_format="jpg")
        assert preflight.check_toolchain(config) == []

    def test_should_say_when_no_software_av1_decoder_is_present(
        self, monkeypatch, on_path,
    ):
        """VMAF reads the av1_nvenc probes back, and FFmpeg's own av1
        decoder works only with hardware acceleration."""
        hardware_only = DECODERS.replace("libdav1d ", "av1_cuvid ")
        fake = _use(monkeypatch, FakeFfmpeg(decoders=hardware_only))

        problems = preflight.check_toolchain(CONFIG)

        assert len(problems) == 1 and "AV1 decoder" in problems[0]
        assert fake.trials == []

    def test_should_accept_libaom_as_the_av1_decoder(
        self, monkeypatch, on_path,
    ):
        libaom = DECODERS.replace("libdav1d ", "libaom-av1 ")
        _use(monkeypatch, FakeFfmpeg(decoders=libaom))

        assert preflight.check_toolchain(CONFIG) == []

    def test_should_blame_the_gpu_or_driver_when_h264_cannot_start(
        self, monkeypatch, on_path,
    ):
        fake = _use(monkeypatch, FakeFfmpeg(
            failing={"h264_nvenc", "av1_nvenc", "hevc_nvenc"},
        ))

        problems = preflight.check_toolchain(CONFIG)

        assert len(problems) == 1
        assert "nvidia-smi" in problems[0]
        assert "unsupported device" in problems[0]
        assert fake.trials == ["h264_nvenc"]

    def test_should_explain_that_av1_needs_an_rtx_40_series(
        self, monkeypatch, on_path,
    ):
        _use(monkeypatch, FakeFfmpeg(failing={"av1_nvenc"}))

        problems = preflight.check_toolchain(CONFIG)

        assert len(problems) == 1 and "RTX 40-series" in problems[0]

    def test_should_report_a_failing_hevc_encoder(self, monkeypatch, on_path):
        _use(monkeypatch, FakeFfmpeg(failing={"hevc_nvenc"}))

        problems = preflight.check_toolchain(CONFIG)

        assert len(problems) == 1 and "hevc_nvenc" in problems[0]

    def test_should_turn_a_tool_error_into_a_problem(
        self, monkeypatch, on_path,
    ):
        def broken(cmd, **kwargs):
            raise ToolError("ffmpeg did not finish within 30s")

        _use(monkeypatch, broken)

        assert preflight.check_toolchain(CONFIG) == [
            "could not query ffmpeg: ffmpeg did not finish within 30s",
        ]

    def test_should_turn_a_malformed_setting_into_one_problem(
        self, monkeypatch, on_path,
    ):
        """load_config refuses a malformed cap; a config built in code
        reaches build_variants instead, and must not escape as a raw
        traceback."""
        _use(monkeypatch, FakeFfmpeg())
        config = PipelineConfig(
            h264_maxrates={"1080": "4000k", "720": "2500k", "360": "800"},
        )

        problems = preflight.check_toolchain(config)

        assert len(problems) == 1
        assert problems[0].startswith("config.env: H264_MAXRATE_360 must be")

    def test_should_run_the_real_options_from_the_encode_builder(
        self, monkeypatch, on_path,
    ):
        fake = _use(monkeypatch, FakeFfmpeg())
        built: list[list[str]] = []
        build = preflight.encode._build_encode_cmd

        def spy(*args, **kwargs):
            built.append(build(*args, **kwargs))
            return built[-1]

        monkeypatch.setattr(preflight.encode, "_build_encode_cmd", spy)

        assert preflight.check_toolchain(CONFIG) == []
        assert fake.real_commands == built
        av1, h264 = built
        assert av1[av1.index("-c:v") + 1] == "av1_nvenc"
        assert "uhq" in av1 and "p010le" in av1
        assert h264[h264.index("-c:v") + 1] == "h264_nvenc"
        assert "-profile:v" in h264 and "4.0" in h264

    def test_should_name_the_setting_this_ffmpeg_does_not_know(
        self, monkeypatch, on_path,
    ):
        _use(monkeypatch, FakeFfmpeg(rejects={
            "av1_nvenc": 'Unable to parse option value "uhq"\n'
                         "Error setting option tune to value uhq.\n",
        }))

        problems = preflight.check_toolchain(CONFIG)

        assert problems == [
            "this FFmpeg does not know NVENC_TUNE=uhq; install a newer "
            "build or set NVENC_TUNE=hq.",
        ]

    def test_should_name_an_option_the_build_does_not_recognise(
        self, monkeypatch, on_path,
    ):
        _use(monkeypatch, FakeFfmpeg(rejects={
            "h264_nvenc": "Unrecognized option 'multipass'.\n"
                          "Error splitting the argument list: Option not "
                          "found\n",
        }))

        problems = preflight.check_toolchain(CONFIG)

        assert len(problems) == 1
        assert "NVENC_MULTIPASS=fullres" in problems[0]
        assert "RTX 40-series" not in problems[0]

    def test_should_report_a_real_options_failure_that_is_no_option_error(
        self, monkeypatch, on_path,
    ):
        _use(monkeypatch, FakeFfmpeg(rejects={
            "av1_nvenc": "InitializeEncoder failed: invalid param (8)\n",
        }))

        problems = preflight.check_toolchain(CONFIG)

        assert len(problems) == 1
        assert problems[0].startswith("av1_nvenc starts, but not with")
        assert "invalid param" in problems[0]
