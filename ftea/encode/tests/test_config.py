"""Tests for ftea.encode.config."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from ftea.encode.config import (
    ENCODER_KEYS,
    REFRAME_STAGE_KEYS,
    ConfigValidationError,
    PipelineConfig,
    load_config,
    load_reframe_launch_config,
    unknown_keys,
)


def _write(tmp_path: Path, body: str, encoding: str = "utf-8") -> Path:
    path = tmp_path / "config.env"
    path.write_bytes(body.encode(encoding))
    return path


class TestReframeDefaults:
    def test_should_enable_reframe_when_config_has_no_reframe_keys(
        self, tmp_path,
    ):
        config = load_config(_write(tmp_path, "VMAF_FLOOR=91\n"))

        assert config.reframe_enabled is True
        assert config.reframe_python == ""
        assert config.reframe_timeout == 600
        assert config.reframe_keep_crops is True
        assert config.reframe_crops_dir == ""

    def test_should_default_to_enabled_on_a_bare_dataclass(self):
        assert PipelineConfig().reframe_enabled is True


class TestReframeFileKeys:
    @pytest.mark.parametrize("value", ["false", "0", "no"])
    def test_should_disable_when_the_file_says_so(self, tmp_path, value):
        config = load_config(_write(tmp_path, f"REFRAME={value}\n"))
        assert config.reframe_enabled is False

    @pytest.mark.parametrize("value", ["true", "1", "yes", "TRUE"])
    def test_should_enable_when_the_file_says_so(self, tmp_path, value):
        config = load_config(_write(tmp_path, f"REFRAME={value}\n"))
        assert config.reframe_enabled is True

    @pytest.mark.parametrize(
        ("value", "expected"), [("on", True), ("On", True), ("off", False)],
    )
    def test_should_read_on_and_off(self, tmp_path, value, expected):
        config = load_config(_write(tmp_path, f"REFRAME={value}\n"))
        assert config.reframe_enabled is expected

    @pytest.mark.parametrize("key", ["REFRAME", "REFRAME_KEEP_CROPS"])
    @pytest.mark.parametrize("value", ["ture", "enabled", "y", "2"])
    def test_should_refuse_a_switch_it_cannot_read(
        self, tmp_path, key, value,
    ):
        """A typo must not quietly turn reframing or crop keeping off."""
        with pytest.raises(ConfigValidationError, match=f"{key} must be"):
            load_config(_write(tmp_path, f"{key}={value}\n"))

    def test_should_refuse_an_unreadable_switch_from_the_environment(
        self, tmp_path,
    ):
        with pytest.raises(ConfigValidationError, match="REFRAME must be"):
            load_config(
                _write(tmp_path, "REFRAME=true\n"), env={"REFRAME": "maybe"},
            )

    def test_should_read_python_timeout_and_crops_from_the_file(
        self, tmp_path,
    ):
        config = load_config(_write(
            tmp_path,
            "REFRAME_PYTHON=/opt/venv/bin/python\n"
            "REFRAME_TIMEOUT=900\n"
            "REFRAME_KEEP_CROPS=false\n"
            "REFRAME_CROPS_DIR=/var/crops\n",
        ))

        assert config.reframe_python == "/opt/venv/bin/python"
        assert config.reframe_timeout == 900
        assert config.reframe_keep_crops is False
        assert config.reframe_crops_dir == "/var/crops"


class TestReframeEnvOverlay:
    def test_should_ignore_the_environment_when_env_is_not_passed(
        self, tmp_path,
    ):
        assert load_config(
            _write(tmp_path, "REFRAME=false\n"),
        ).reframe_enabled is False

    def test_should_let_the_environment_override_the_file(self, tmp_path):
        config = load_config(
            _write(tmp_path, "REFRAME=true\n"), env={"REFRAME": "0"},
        )
        assert config.reframe_enabled is False

    def test_should_let_the_environment_supply_an_absent_key(
        self, tmp_path,
    ):
        config = load_config(
            _write(tmp_path, "VMAF_FLOOR=91\n"),
            env={"REFRAME_TIMEOUT": "1200"},
        )
        assert config.reframe_timeout == 1200

    def test_should_not_let_the_environment_change_other_keys(
        self, tmp_path,
    ):
        config = load_config(
            _write(tmp_path, "VMAF_FLOOR=91\nCLIP_TIMEOUT=300\n"),
            env={"VMAF_FLOOR": "60", "CLIP_TIMEOUT": "30",
                 "FILM_GRAIN": "8"},
        )

        assert config.vmaf_floor == 91
        assert config.clip_timeout == 300
        assert config.film_grain == 0

    def test_should_ignore_empty_environment_values(self, tmp_path):
        config = load_config(
            _write(tmp_path, "REFRAME=false\n"), env={"REFRAME": ""},
        )
        assert config.reframe_enabled is False


class TestReframeValidation:
    def test_should_reject_a_timeout_below_thirty(self, tmp_path):
        with pytest.raises(ConfigValidationError, match="REFRAME_TIMEOUT"):
            load_config(_write(tmp_path, "REFRAME_TIMEOUT=10\n"))

    def test_should_reject_a_non_integer_timeout(self, tmp_path):
        with pytest.raises(ConfigValidationError, match="REFRAME_TIMEOUT"):
            load_config(_write(tmp_path, "REFRAME_TIMEOUT=soon\n"))


class TestNotUtf8:
    def test_should_say_the_file_is_not_utf8(self, tmp_path):
        path = _write(tmp_path, "VMAF_FLOOR=91", encoding="utf-16")
        with pytest.raises(ConfigValidationError, match="not UTF-8"):
            load_config(path)
        with pytest.raises(ConfigValidationError, match="not UTF-8"):
            unknown_keys(path)


class TestFileHandling:
    def test_should_raise_when_the_file_is_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "absent.env")

    def test_should_use_the_defaults_when_a_missing_file_is_allowed(
        self, tmp_path,
    ):
        config = load_config(tmp_path / "absent.env", missing_ok=True)
        assert config == PipelineConfig()

    def test_should_still_apply_the_environment_without_a_file(
        self, tmp_path,
    ):
        config = load_config(
            tmp_path / "absent.env", env={"REFRAME": "0"}, missing_ok=True,
        )
        assert config.reframe_enabled is False

    def test_should_read_a_file_saved_with_a_byte_order_mark(self, tmp_path):
        path = _write(
            tmp_path, "VMAF_FLOOR=88\r\nCLIP_TIMEOUT=120\r\n",
            encoding="utf-8-sig",
        )
        config = load_config(path)
        assert (config.vmaf_floor, config.clip_timeout) == (88, 120)

    def test_should_treat_a_blank_value_as_the_default(self, tmp_path):
        config = load_config(
            _write(tmp_path, "THUMB_FORMAT=\nAV1_MAXRATE_1080=\n"),
        )
        assert config.thumb_format == "webp"
        assert config.av1_maxrates["1080"] == "3000k"

    def test_should_strip_quotes_and_skip_comments(self, tmp_path):
        config = load_config(
            _write(tmp_path, '# a comment\nAUDIO_BITRATE="160k"\n'),
        )
        assert config.audio_bitrate == "160k"


class TestUnknownKeys:
    def test_should_report_a_misspelt_setting(self, tmp_path):
        path = _write(tmp_path, "VMAF_TARGET_108=95\nVMAF_FLOOR=91\n")
        assert unknown_keys(path) == ["VMAF_TARGET_108"]

    def test_should_know_the_reframe_stage_keys(self, tmp_path):
        path = _write(
            tmp_path,
            "REFRAME_STRIDE=3\nREFRAME_TITLE_ENABLED=true\nREFRAME_FFMPEG=\n",
        )
        assert unknown_keys(path) == []

    def test_should_report_a_misspelt_reframe_setting(self, tmp_path):
        """Nothing else checks REFRAME_ keys: the stage reads the keys it
        knows and ignores the rest, so a typo would silently keep its
        default. That includes the encoder's own REFRAME_ keys."""
        path = _write(
            tmp_path,
            "REFRAME_STRIDES=1\nREFRAME_TIMOUT=1200\n"
            "REFRAME_KEEP_CROP=false\nREFRAME_STRIDE=3\n",
        )
        assert unknown_keys(path) == [
            "REFRAME_KEEP_CROP", "REFRAME_STRIDES", "REFRAME_TIMOUT",
        ]

    def test_should_keep_the_two_halves_of_the_namespace_apart(self):
        assert not set(REFRAME_STAGE_KEYS) & set(ENCODER_KEYS)

    def test_should_report_nothing_for_a_missing_file(self, tmp_path):
        assert unknown_keys(tmp_path / "absent.env") == []

    def test_should_list_exactly_the_keys_load_config_reads(self):
        source = inspect.getsource(load_config)
        read = set(re.findall(r'"([A-Z][A-Z0-9_]+)"', source))
        assert read == set(ENCODER_KEYS)


class TestThumbFormat:
    """The player and ftea preview know thumbnail.webp and .jpg only;
    any other value gives a package with no poster."""

    @pytest.mark.parametrize(("value", "expected"), [
        ("webp", "webp"), ("JPG", "jpg"), (" Webp ", "webp"),
    ])
    def test_should_accept_webp_or_jpg_in_any_case(
        self, tmp_path, value, expected,
    ):
        config = load_config(_write(tmp_path, f"THUMB_FORMAT={value}\n"))
        assert config.thumb_format == expected

    @pytest.mark.parametrize("value", ["png", "jpeg", "gif"])
    def test_should_refuse_any_other_format(self, tmp_path, value):
        with pytest.raises(ConfigValidationError, match="THUMB_FORMAT"):
            load_config(_write(tmp_path, f"THUMB_FORMAT={value}\n"))


class TestReframeLaunchConfig:
    """`ftea reframe` needs only REFRAME_PYTHON and REFRAME_TIMEOUT."""

    def test_should_ignore_encoder_settings(self, tmp_path):
        config = load_reframe_launch_config(_write(
            tmp_path,
            "VMAF_TARGET_1080=abc\nCQ_BRACKET=0\nREFRAME=maybe\n"
            "REFRAME_PYTHON=/envs/py\nREFRAME_TIMEOUT=90\n",
        ))
        assert (config.reframe_python, config.reframe_timeout) == (
            "/envs/py", 90,
        )

    @pytest.mark.parametrize("value", ["10", "soon"])
    def test_should_still_refuse_a_bad_time_limit(self, tmp_path, value):
        with pytest.raises(ConfigValidationError, match="REFRAME_TIMEOUT"):
            load_reframe_launch_config(
                _write(tmp_path, f"REFRAME_TIMEOUT={value}\n"),
            )

    def test_should_read_the_environment_overrides(self, tmp_path):
        config = load_reframe_launch_config(
            _write(tmp_path, "REFRAME_TIMEOUT=90\n"),
            env={"REFRAME_TIMEOUT": "120"},
        )
        assert config.reframe_timeout == 120

    def test_should_use_the_defaults_without_a_file(self, tmp_path):
        config = load_reframe_launch_config(
            tmp_path / "config.env", missing_ok=True,
        )
        assert config == PipelineConfig()


class TestValidation:
    def test_should_accept_stop_in_any_case(self, tmp_path):
        config = load_config(_write(tmp_path, "BATCH_ON_ERROR=STOP\n"))
        assert config.batch_on_error == "stop"

    def test_should_reject_an_unknown_batch_mode(self, tmp_path):
        with pytest.raises(ConfigValidationError, match="BATCH_ON_ERROR"):
            load_config(_write(tmp_path, "BATCH_ON_ERROR=halt\n"))

    def test_should_reject_a_negative_retry_count(self, tmp_path):
        with pytest.raises(ConfigValidationError, match="RETRY_MAX"):
            load_config(_write(tmp_path, "RETRY_MAX=-1\n"))

    def test_should_reject_a_vmaf_target_out_of_range(self, tmp_path):
        with pytest.raises(ConfigValidationError, match="VMAF target"):
            load_config(_write(tmp_path, "VMAF_TARGET_720=120\n"))

    def test_should_read_the_per_second_allowance_and_the_length_limit(
        self, tmp_path,
    ):
        config = load_config(
            _write(tmp_path, "CLIP_TIMEOUT_PER_SEC=12\nMAX_CLIP_SEC=0\n"),
        )
        assert (config.clip_timeout_per_sec, config.max_clip_sec) == (12, 0)

    @pytest.mark.parametrize("key", ["CLIP_TIMEOUT_PER_SEC", "MAX_CLIP_SEC"])
    def test_should_reject_a_negative_time_setting(self, tmp_path, key):
        with pytest.raises(ConfigValidationError, match=key):
            load_config(_write(tmp_path, f"{key}=-1\n"))

    @pytest.mark.parametrize(("body", "key"), [
        ("MAX_SCENE_SEC=0", "MAX_SCENE_SEC"),
        ("MAX_SCENE_SEC=-5", "MAX_SCENE_SEC"),
        ("MIN_SCENE_SEC=-1", "MIN_SCENE_SEC"),
        ("MIN_SCENE_SEC=12\nMAX_SCENE_SEC=10", "MIN_SCENE_SEC"),
        ("CQ_BRACKET=0", "CQ_BRACKET"),
        ("CQ_BRACKET=-2", "CQ_BRACKET"),
        ("RETRY_TIMEOUT_MULTIPLIER=0", "RETRY_TIMEOUT_MULTIPLIER"),
        ("RETRY_VMAF_RELAXATION=-1", "RETRY_VMAF_RELAXATION"),
        ("RETRY_VMAF_RELAXATION=39", "RETRY_VMAF_RELAXATION"),
        ("BUFSIZE_MULTIPLIER=0", "BUFSIZE_MULTIPLIER"),
        ("AV1_BIT_DEPTH=12", "AV1_BIT_DEPTH"),
        ("H264_BIT_DEPTH=10", "H264_BIT_DEPTH"),
        ("H264_MAXRATE_360=800", "H264_MAXRATE_360"),
        ("AV1_MAXRATE_720=0k", "AV1_MAXRATE_720"),
        ("AV1_MAXRATE_1080=fast", "AV1_MAXRATE_1080"),
    ])
    def test_should_reject_a_value_the_encoder_cannot_use(
        self, tmp_path, body, key,
    ):
        """Each of these used to load, then fail every clip, or make a
        stream that does not match what master.m3u8 says about it."""
        with pytest.raises(ConfigValidationError, match=f"^{key} must be"):
            load_config(_write(tmp_path, body + "\n"))

    @pytest.mark.parametrize("body", [
        "MIN_SCENE_SEC=0",
        "MIN_SCENE_SEC=10\nMAX_SCENE_SEC=10",
        "CQ_BRACKET=1",
        "RETRY_TIMEOUT_MULTIPLIER=1",
        "RETRY_VMAF_RELAXATION=0",
        "RETRY_VMAF_RELAXATION=38",
        "BUFSIZE_MULTIPLIER=1",
        "AV1_BIT_DEPTH=8",
    ])
    def test_should_accept_the_edge_of_each_range(self, tmp_path, body):
        load_config(_write(tmp_path, body + "\n"))

    def test_should_bound_the_relaxation_by_the_lowest_target(
        self, tmp_path,
    ):
        """Relaxed targets must stay inside the 50-100 range as well."""
        config = load_config(_write(
            tmp_path, "VMAF_TARGET_360=95\nRETRY_VMAF_RELAXATION=43\n",
        ))
        assert config.retry_vmaf_relaxation == 43
        with pytest.raises(ConfigValidationError, match="between 0 and 43"):
            load_config(_write(
                tmp_path, "VMAF_TARGET_360=95\nRETRY_VMAF_RELAXATION=44\n",
            ))


class TestVariants:
    def test_should_build_six_vertical_variants(self):
        variants = PipelineConfig().build_variants("portrait")
        assert [
            (v.codec, v.resolution, v.width, v.height) for v in variants
        ] == [
            ("av1", "1080", 1080, 1920), ("av1", "720", 720, 1280),
            ("av1", "360", 360, 640), ("h264", "1080", 1080, 1920),
            ("h264", "720", 720, 1280), ("h264", "360", 360, 640),
        ]

    def test_should_derive_the_buffer_from_the_cap(self):
        first = PipelineConfig().build_variants("landscape")[0]
        assert (first.width, first.height, first.maxrate, first.bufsize) == (
            1920, 1080, "3000k", "6000k",
        )

    def test_should_name_a_malformed_cap_on_a_config_built_in_code(self):
        config = PipelineConfig(
            h264_maxrates={"1080": "4000k", "720": "2500k", "360": "800"},
        )
        with pytest.raises(ConfigValidationError, match="H264_MAXRATE_360"):
            config.build_variants("portrait")

    @pytest.mark.parametrize("orientation", ["unknown", "Portrait", ""])
    def test_should_refuse_an_orientation_it_does_not_know(
        self, orientation,
    ):
        """Anything but "portrait" used to get landscape variants."""
        with pytest.raises(ValueError, match="orientation"):
            PipelineConfig().build_variants(orientation)
