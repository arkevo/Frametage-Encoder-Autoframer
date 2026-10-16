"""Tests for ftea.reframe.settings — the REFRAME_* stage keys."""

import os
from pathlib import Path

import pytest

from ftea import paths
from ftea.reframe.errors import ReframeError
from ftea.reframe.settings import ReframeSettings, load_settings


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.env"
    path.write_text(body)
    return path


class TestDefaults:
    def test_should_use_defaults_when_file_is_empty(self, tmp_path):
        settings = load_settings(_write(tmp_path, ""))

        assert settings.stride == 3
        assert settings.batch_size == 16
        assert settings.device == "cuda"
        assert settings.face_weight == 2.0
        assert settings.hysteresis_margin == 0.2
        assert settings.hysteresis_sec == 1.0
        assert settings.dead_zone == 0.9
        assert settings.smoothing_hz == 0.6
        assert settings.max_pan_frac_per_sec == 0.20
        assert settings.aspect_w == 9
        assert settings.aspect_h == 16
        assert settings.codec == "hevc_nvenc"
        assert settings.cq == 10
        assert settings.preset == "p7"
        assert settings.container == "mp4"
        assert settings.person_model == ""
        assert settings.face_model == ""
        assert settings.models_dir == paths.models_dir()

    def test_should_use_defaults_when_file_is_missing(self, tmp_path):
        """A missing config.env is not an error — every key has a default."""
        settings = load_settings(tmp_path / "absent.env")

        assert settings.stride == 3
        assert settings.container == "mp4"

    def test_should_ignore_unrelated_encoder_keys(self, tmp_path):
        settings = load_settings(_write(
            tmp_path, "VMAF_FLOOR=91\nNVENC_PRESET=p1\nCLIP_TIMEOUT=300\n",
        ))
        assert settings.preset == "p7"

    def test_should_be_frozen(self, tmp_path):
        settings = load_settings(_write(tmp_path, ""))
        with pytest.raises(Exception):
            settings.stride = 1


class TestNotUtf8:
    def test_should_say_the_file_is_not_utf8(self, tmp_path):
        path = tmp_path / "config.env"
        path.write_text("REFRAME_STRIDE=3", encoding="utf-16")
        with pytest.raises(ReframeError, match="not UTF-8"):
            load_settings(path)


class TestFileValues:
    def test_should_read_every_stage_key_from_the_file(self, tmp_path):
        models = tmp_path / "models"
        settings = load_settings(_write(
            tmp_path,
            f"REFRAME_MODELS_DIR={models}\n"
            f"REFRAME_PERSON_MODEL={models / 'person.pt'}\n"
            f"REFRAME_FACE_MODEL={models / 'face.onnx'}\n"
            "REFRAME_STRIDE=2\n"
            "REFRAME_BATCH_SIZE=8\n"
            "REFRAME_DEVICE=cpu\n"
            "REFRAME_FACE_WEIGHT=3.0\n"
            "REFRAME_HYSTERESIS_MARGIN=0.25\n"
            "REFRAME_HYSTERESIS_SEC=1.5\n"
            "REFRAME_DEAD_ZONE=0.4\n"
            "REFRAME_SMOOTHING_HZ=2.0\n"
            "REFRAME_MAX_PAN_FRAC_PER_SEC=0.125\n"
            "REFRAME_ASPECT_W=4\n"
            "REFRAME_ASPECT_H=5\n"
            "REFRAME_CODEC=ffv1\n"
            "REFRAME_CQ=0\n"
            "REFRAME_PRESET=p5\n"
            "REFRAME_CONTAINER=mkv\n",
        ))

        assert settings.models_dir == models
        assert settings.person_model == str(models / "person.pt")
        assert settings.face_model == str(models / "face.onnx")
        assert settings.stride == 2
        assert settings.batch_size == 8
        assert settings.device == "cpu"
        assert settings.face_weight == 3.0
        assert settings.hysteresis_margin == 0.25
        assert settings.hysteresis_sec == 1.5
        assert settings.dead_zone == 0.4
        assert settings.smoothing_hz == 2.0
        assert settings.max_pan_frac_per_sec == 0.125
        assert settings.aspect_w == 4
        assert settings.aspect_h == 5
        assert settings.codec == "ffv1"
        assert settings.cq == 0
        assert settings.preset == "p5"
        assert settings.container == "mkv"

    def test_should_strip_quotes_like_the_encoder_loader(self, tmp_path):
        settings = load_settings(_write(tmp_path, 'REFRAME_DEVICE="cpu"\n'))
        assert settings.device == "cpu"

    def test_should_ignore_comments_and_blank_lines(self, tmp_path):
        settings = load_settings(_write(
            tmp_path, "\n# REFRAME_STRIDE=99\n\nREFRAME_STRIDE=5\n",
        ))
        assert settings.stride == 5


class TestEnvOverlay:
    def test_should_let_env_override_the_file(self, tmp_path):
        settings = load_settings(
            _write(tmp_path, "REFRAME_STRIDE=3\n"),
            env={"REFRAME_STRIDE": "7"},
        )
        assert settings.stride == 7

    def test_should_not_let_env_override_non_reframe_keys(self, tmp_path):
        settings = load_settings(
            _write(tmp_path, "REFRAME_PRESET=p7\n"),
            env={"NVENC_PRESET": "p1", "PRESET": "p1"},
        )
        assert settings.preset == "p7"


class TestValidation:
    @pytest.mark.parametrize("body,needle", [
        ("REFRAME_STRIDE=0\n", "REFRAME_STRIDE"),
        ("REFRAME_STRIDE=-1\n", "REFRAME_STRIDE"),
        ("REFRAME_BATCH_SIZE=0\n", "REFRAME_BATCH_SIZE"),
        ("REFRAME_FACE_WEIGHT=0.5\n", "REFRAME_FACE_WEIGHT"),
        ("REFRAME_HYSTERESIS_MARGIN=1.0\n", "REFRAME_HYSTERESIS_MARGIN"),
        ("REFRAME_HYSTERESIS_SEC=-1\n", "REFRAME_HYSTERESIS_SEC"),
        ("REFRAME_DEAD_ZONE=0\n", "REFRAME_DEAD_ZONE"),
        ("REFRAME_DEAD_ZONE=1\n", "REFRAME_DEAD_ZONE"),
        ("REFRAME_SMOOTHING_HZ=0\n", "REFRAME_SMOOTHING_HZ"),
        ("REFRAME_MAX_PAN_FRAC_PER_SEC=0\n", "REFRAME_MAX_PAN_FRAC_PER_SEC"),
        ("REFRAME_ASPECT_W=0\n", "REFRAME_ASPECT_W"),
        ("REFRAME_ASPECT_H=0\n", "REFRAME_ASPECT_H"),
        ("REFRAME_CQ=52\n", "REFRAME_CQ"),
        ("REFRAME_CQ=-1\n", "REFRAME_CQ"),
        ("REFRAME_CONTAINER=.mp4\n", "REFRAME_CONTAINER"),
        ("REFRAME_STRIDE=many\n", "REFRAME_STRIDE"),
        ("REFRAME_DEAD_ZONE=wide\n", "REFRAME_DEAD_ZONE"),
    ])
    def test_should_reject_out_of_range_values(self, tmp_path, body, needle):
        with pytest.raises(ReframeError, match=needle):
            load_settings(_write(tmp_path, body))


class TestDetectFloorsAndInputSize:
    def test_should_default_the_person_floor_to_zero_point_eight(
        self, tmp_path,
    ):
        """Fabric scores ~0.60 on our test footage, real people 0.98-1.00."""
        settings = load_settings(_write(tmp_path, ""))
        assert settings.min_confidence_person == 0.8

    def test_should_default_the_face_floor_to_zero_point_three_five(
        self, tmp_path,
    ):
        """Face scores never exceed 0.7462 on our test footage."""
        settings = load_settings(_write(tmp_path, ""))
        assert settings.min_confidence_face == 0.35

    def test_should_default_detect_long_side_to_native(self, tmp_path):
        assert load_settings(_write(tmp_path, "")).detect_long_side == 0

    def test_should_read_the_two_floors_independently(self, tmp_path):
        settings = load_settings(_write(tmp_path, (
            'REFRAME_MIN_CONFIDENCE_PERSON="0.9"\n'
            'REFRAME_MIN_CONFIDENCE_FACE="0.2"\n'
        )))
        assert settings.min_confidence_person == 0.9
        assert settings.min_confidence_face == 0.2

    def test_should_read_a_configured_detect_long_side(self, tmp_path):
        settings = load_settings(
            _write(tmp_path, 'REFRAME_DETECT_LONG_SIDE="1920"\n'))
        assert settings.detect_long_side == 1920

    @pytest.mark.parametrize("body,needle", [
        ('REFRAME_MIN_CONFIDENCE_PERSON="0"\n',
         "REFRAME_MIN_CONFIDENCE_PERSON"),
        ('REFRAME_MIN_CONFIDENCE_PERSON="1"\n',
         "REFRAME_MIN_CONFIDENCE_PERSON"),
        ('REFRAME_MIN_CONFIDENCE_FACE="1.5"\n',
         "REFRAME_MIN_CONFIDENCE_FACE"),
        ('REFRAME_DETECT_LONG_SIDE="200"\n', "REFRAME_DETECT_LONG_SIDE"),
        ('REFRAME_DETECT_LONG_SIDE="-1"\n', "REFRAME_DETECT_LONG_SIDE"),
    ])
    def test_should_reject_out_of_range_detect_values(
        self, tmp_path, body, needle,
    ):
        with pytest.raises(ReframeError, match=needle):
            load_settings(_write(tmp_path, body))

    def test_should_no_longer_expose_a_shared_min_confidence(
        self, tmp_path,
    ):
        """One shared floor cannot serve both classes, so it is dropped."""
        settings = load_settings(_write(tmp_path, ""))
        assert not hasattr(settings, "min_confidence")

    def test_should_warn_but_succeed_when_the_dropped_key_is_present(
        self, tmp_path, capsys,
    ):
        """A stale REFRAME_MIN_CONFIDENCE warns, not fails."""
        settings = load_settings(_write(
            tmp_path, 'REFRAME_MIN_CONFIDENCE="0.25"\n'))
        assert settings.min_confidence_person == 0.8
        assert "REFRAME_MIN_CONFIDENCE was removed" in \
            capsys.readouterr().err


class TestSubjectStageKeys:
    def test_should_default_the_subject_weights(self, tmp_path):
        settings = load_settings(tmp_path / "absent.env")

        assert settings.centrality_weight == 0.5
        assert settings.confidence_weight == 0.3
        assert settings.persistence_weight == 0.5
        assert settings.iou_match_min == 0.3
        assert settings.track_gap_sec == 0.25

    def test_should_read_the_subject_weights_from_the_config(self, tmp_path):
        config = tmp_path / "config.env"
        config.write_text(
            "REFRAME_CENTRALITY_WEIGHT=0.2\n"
            "REFRAME_CONFIDENCE_WEIGHT=0.4\n"
            "REFRAME_PERSISTENCE_WEIGHT=0.6\n"
            "REFRAME_IOU_MATCH_MIN=0.45\n"
            "REFRAME_TRACK_GAP_SEC=0.5\n",
            encoding="utf-8",
        )

        settings = load_settings(config)

        assert settings.centrality_weight == 0.2
        assert settings.confidence_weight == 0.4
        assert settings.persistence_weight == 0.6
        assert settings.iou_match_min == 0.45
        assert settings.track_gap_sec == 0.5

    def test_should_reject_a_centrality_weight_of_one(self, tmp_path):
        config = tmp_path / "config.env"
        config.write_text("REFRAME_CENTRALITY_WEIGHT=1.0\n",
                          encoding="utf-8")

        with pytest.raises(ReframeError, match="REFRAME_CENTRALITY_WEIGHT"):
            load_settings(config)

    def test_should_reject_a_negative_confidence_weight(self, tmp_path):
        config = tmp_path / "config.env"
        config.write_text("REFRAME_CONFIDENCE_WEIGHT=-0.1\n",
                          encoding="utf-8")

        with pytest.raises(ReframeError, match="REFRAME_CONFIDENCE_WEIGHT"):
            load_settings(config)

    def test_should_reject_a_persistence_weight_of_one(self, tmp_path):
        config = tmp_path / "config.env"
        config.write_text("REFRAME_PERSISTENCE_WEIGHT=1.0\n",
                          encoding="utf-8")

        with pytest.raises(ReframeError, match="REFRAME_PERSISTENCE_WEIGHT"):
            load_settings(config)

    def test_should_reject_a_zero_iou_match_min(self, tmp_path):
        config = tmp_path / "config.env"
        config.write_text("REFRAME_IOU_MATCH_MIN=0.0\n", encoding="utf-8")

        with pytest.raises(ReframeError, match="REFRAME_IOU_MATCH_MIN"):
            load_settings(config)

    def test_should_reject_a_negative_track_gap_sec(self, tmp_path):
        config = tmp_path / "config.env"
        config.write_text("REFRAME_TRACK_GAP_SEC=-0.1\n", encoding="utf-8")

        with pytest.raises(ReframeError, match="REFRAME_TRACK_GAP_SEC"):
            load_settings(config)


class TestCameraStageKeys:
    def test_should_default_the_dead_zone_release(self, tmp_path):
        settings = load_settings(tmp_path / "absent.env")

        assert settings.dead_zone_release == 0.6

    def test_should_read_the_dead_zone_release_from_the_config(
        self, tmp_path,
    ):
        config = tmp_path / "config.env"
        config.write_text(
            "REFRAME_DEAD_ZONE_RELEASE=0.4\n", encoding="utf-8")

        settings = load_settings(config)

        assert settings.dead_zone_release == 0.4

    def test_should_accept_a_dead_zone_release_of_zero(self, tmp_path):
        config = tmp_path / "config.env"
        config.write_text(
            "REFRAME_DEAD_ZONE_RELEASE=0\n", encoding="utf-8")

        settings = load_settings(config)

        assert settings.dead_zone_release == 0.0

    def test_should_reject_a_dead_zone_release_of_one(self, tmp_path):
        config = tmp_path / "config.env"
        config.write_text(
            "REFRAME_DEAD_ZONE_RELEASE=1.0\n", encoding="utf-8")

        with pytest.raises(ReframeError) as excinfo:
            load_settings(config)

        assert "REFRAME_DEAD_ZONE_RELEASE" in str(excinfo.value)

    def test_should_reject_a_negative_dead_zone_release(self, tmp_path):
        config = tmp_path / "config.env"
        config.write_text(
            "REFRAME_DEAD_ZONE_RELEASE=-0.1\n", encoding="utf-8")

        with pytest.raises(ReframeError):
            load_settings(config)


class TestSettingsIsPlainData:
    def test_should_expose_every_documented_field(self, tmp_path):
        load_settings(_write(tmp_path, ""))
        expected = {
            "models_dir", "person_model", "face_model", "stride",
            "batch_size", "device", "face_weight",
            "hysteresis_margin", "hysteresis_sec", "dead_zone",
            "smoothing_hz", "max_pan_frac_per_sec", "aspect_w", "aspect_h",
            "codec", "cq", "preset", "container",
            "min_confidence_person", "min_confidence_face",
            "detect_long_side",
            "centrality_weight", "confidence_weight",
            "persistence_weight", "iou_match_min", "track_gap_sec",
            "dead_zone_release",
            "ffmpeg", "scene_threshold", "min_shot_sec",
            # --- title detection (text.py). Appended at the END.
            "title_enabled", "title_model", "title_long_side",
            "title_probe_frames", "title_min_height", "title_min_lines",
            "title_centre_tol", "title_max_skew", "title_edge_margin",
        }
        assert {f.name for f in ReframeSettings.__dataclass_fields__.values()} \
            == expected


class TestShippedTemplate:
    def test_should_parse_the_example_config_with_documented_defaults(self):
        """The committed template must round-trip through this loader."""
        example = paths.config_example_path()
        if not example.is_file():
            pytest.skip("config.env.example arrives with the encoder port")
        settings = load_settings(example)

        assert settings.stride == 3
        assert settings.dead_zone == 0.9
        assert settings.codec == "hevc_nvenc"
        assert settings.container == "mp4"
        assert settings.models_dir == paths.models_dir()
        assert settings.min_confidence_person == 0.8
        assert settings.min_confidence_face == 0.35
        assert settings.detect_long_side == 0
        assert settings.title_enabled is True
        assert settings.title_long_side == 640
        assert settings.title_min_height == 0.04


class TestFfmpegKey:
    """REFRAME_FFMPEG -- the one key the render stage owns."""

    def test_should_default_the_ffmpeg_path_to_empty(self, tmp_path):
        settings = load_settings(tmp_path / "absent.env")

        assert settings.ffmpeg == ""

    def test_should_read_the_ffmpeg_path_from_the_environment(
        self, tmp_path,
    ):
        settings = load_settings(
            tmp_path / "absent.env",
            env={"REFRAME_FFMPEG": r"C:\tools\ffmpeg.EXE"},
        )

        assert settings.ffmpeg == r"C:\tools\ffmpeg.EXE"


class TestShotThresholdKeys:
    """REFRAME_SCENE_THRESHOLD / REFRAME_MIN_SHOT_SEC.

    run_stages cannot thread a configured value it cannot read, so these
    live in ReframeSettings under the same append-only protocol, with
    defaults of 0.3 / 0.4, mirroring shots.py's constants.
    """

    def test_should_default_the_scene_threshold(self, tmp_path):
        settings = load_settings(tmp_path / "absent.env")

        assert settings.scene_threshold == 0.3

    def test_should_default_the_minimum_shot_length(self, tmp_path):
        settings = load_settings(tmp_path / "absent.env")

        assert settings.min_shot_sec == 0.4

    def test_should_read_both_from_the_environment(self, tmp_path):
        settings = load_settings(
            tmp_path / "absent.env",
            env={"REFRAME_SCENE_THRESHOLD": "0.44",
                 "REFRAME_MIN_SHOT_SEC": "0.9"},
        )

        assert settings.scene_threshold == 0.44
        assert settings.min_shot_sec == 0.9

    def test_should_reject_a_scene_threshold_out_of_range(self, tmp_path):
        with pytest.raises(ReframeError, match="REFRAME_SCENE_THRESHOLD"):
            load_settings(_write(tmp_path, "REFRAME_SCENE_THRESHOLD=0"))

    def test_should_reject_a_negative_minimum_shot_length(self, tmp_path):
        with pytest.raises(ReframeError, match="REFRAME_MIN_SHOT_SEC"):
            load_settings(_write(tmp_path, "REFRAME_MIN_SHOT_SEC=-1"))


class TestTitleKeys:
    """REFRAME_TITLE_* -- the nine keys title detection owns.

    The majority rule ("present in at least ceil(sampled / 2) frames")
    is deliberately NOT here: it is fixed, and one more knob on a rule
    already carrying five is not worth the surface.
    """

    def test_should_default_every_title_key(self, tmp_path):
        settings = load_settings(tmp_path / "absent.env")

        assert settings.title_enabled is True
        assert settings.title_model == ""
        assert settings.title_long_side == 640
        assert settings.title_probe_frames == 5
        assert settings.title_min_height == 0.04
        assert settings.title_min_lines == 2
        assert settings.title_centre_tol == 0.05
        assert settings.title_max_skew == 10.0
        assert settings.title_edge_margin == 0.01

    def test_should_read_every_title_key_from_the_environment(
        self, tmp_path,
    ):
        settings = load_settings(
            tmp_path / "absent.env",
            env={
                "REFRAME_TITLE_ENABLED": "false",
                "REFRAME_TITLE_MODEL": str(tmp_path / "db.onnx"),
                "REFRAME_TITLE_LONG_SIDE": "960",
                "REFRAME_TITLE_PROBE_FRAMES": "9",
                "REFRAME_TITLE_MIN_HEIGHT": "0.06",
                "REFRAME_TITLE_MIN_LINES": "3",
                "REFRAME_TITLE_CENTRE_TOL": "0.02",
                "REFRAME_TITLE_MAX_SKEW": "5.0",
                "REFRAME_TITLE_EDGE_MARGIN": "0.03",
            },
        )

        assert settings.title_enabled is False
        assert settings.title_model == str(tmp_path / "db.onnx")
        assert settings.title_long_side == 960
        assert settings.title_probe_frames == 9
        assert settings.title_min_height == 0.06
        assert settings.title_min_lines == 3
        assert settings.title_centre_tol == 0.02
        assert settings.title_max_skew == 5.0
        assert settings.title_edge_margin == 0.03

    @pytest.mark.parametrize("key,value,rule", [
        ("REFRAME_TITLE_LONG_SIDE", "128", ">= 320"),
        ("REFRAME_TITLE_PROBE_FRAMES", "0", ">= 1"),
        ("REFRAME_TITLE_MIN_HEIGHT", "0", "in (0, 1)"),
        ("REFRAME_TITLE_MIN_HEIGHT", "1.0", "in (0, 1)"),
        ("REFRAME_TITLE_MIN_LINES", "1", ">= 2"),
        ("REFRAME_TITLE_CENTRE_TOL", "0", "in (0, 1)"),
        ("REFRAME_TITLE_MAX_SKEW", "-1", "in [0, 45]"),
        ("REFRAME_TITLE_MAX_SKEW", "46", "in [0, 45]"),
        ("REFRAME_TITLE_EDGE_MARGIN", "-0.1", "in [0, 0.25]"),
        ("REFRAME_TITLE_EDGE_MARGIN", "0.3", "in [0, 0.25]"),
    ])
    def test_should_reject_an_out_of_range_title_value(
        self, tmp_path, key, value, rule,
    ):
        with pytest.raises(ReframeError) as excinfo:
            load_settings(tmp_path / "absent.env", env={key: value})

        assert key in str(excinfo.value)
        assert rule in str(excinfo.value)

    @pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "Off"])
    def test_should_read_a_false_word_as_disabled(self, tmp_path, raw):
        """The switch grammar is the encoder's (ftea/encode/config.py):
        true/yes/on/1 and false/no/off/0, in any case."""
        settings = load_settings(
            tmp_path / "absent.env", env={"REFRAME_TITLE_ENABLED": raw},
        )
        assert settings.title_enabled is False

    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "On"])
    def test_should_read_a_true_word_as_enabled(self, tmp_path, raw):
        settings = load_settings(
            tmp_path / "absent.env", env={"REFRAME_TITLE_ENABLED": raw},
        )
        assert settings.title_enabled is True

    @pytest.mark.parametrize("raw", ["ture", "enabled", "y", "maybe"])
    def test_should_refuse_a_value_that_is_neither_true_nor_false(
        self, tmp_path, raw,
    ):
        """A typo must not quietly switch title detection off: the
        default is on, so reading a typo as false changes the output."""
        with pytest.raises(ReframeError) as excinfo:
            load_settings(
                tmp_path / "absent.env",
                env={"REFRAME_TITLE_ENABLED": raw},
            )

        message = str(excinfo.value)
        assert "REFRAME_TITLE_ENABLED" in message
        assert raw in message


class TestRenderAndDeviceKeys:
    """REFRAME_CODEC, _CONTAINER, _PRESET and _DEVICE are checked when
    config.env loads, so a bad value stops the clip in milliseconds
    instead of after shot, title and detection passes."""

    def test_should_use_the_default_when_a_text_key_is_blank(
        self, tmp_path,
    ):
        settings = load_settings(_write(
            tmp_path,
            "REFRAME_CODEC=\nREFRAME_PRESET=\nREFRAME_DEVICE=\n"
            "REFRAME_CONTAINER=\n",
        ))

        assert settings.codec == "hevc_nvenc"
        assert settings.preset == "p7"
        assert settings.device == "cuda"
        assert settings.container == "mp4"

    def test_should_refuse_a_codec_the_render_cannot_use(self, tmp_path):
        with pytest.raises(ReframeError, match="REFRAME_CODEC"):
            load_settings(_write(tmp_path, "REFRAME_CODEC=libx265\n"))

    def test_should_refuse_ffv1_in_the_default_mp4_container(
        self, tmp_path,
    ):
        with pytest.raises(ReframeError) as excinfo:
            load_settings(_write(tmp_path, "REFRAME_CODEC=ffv1\n"))

        assert "REFRAME_CONTAINER=mkv" in str(excinfo.value)

    def test_should_refuse_a_container_no_codec_can_use(self, tmp_path):
        with pytest.raises(ReframeError, match="REFRAME_CONTAINER"):
            load_settings(_write(tmp_path, "REFRAME_CONTAINER=avi\n"))

    def test_should_accept_any_letter_case_for_codec_and_container(
        self, tmp_path,
    ):
        settings = load_settings(_write(
            tmp_path, "REFRAME_CODEC=HEVC_NVENC\nREFRAME_CONTAINER=MKV\n",
        ))

        assert settings.codec == "HEVC_NVENC"
        assert settings.container == "MKV"

    @pytest.mark.parametrize("device", ["gpu", "cuda:x", "cuda0", "tpu"])
    def test_should_refuse_a_device_torch_cannot_use(
        self, tmp_path, device,
    ):
        with pytest.raises(ReframeError, match="REFRAME_DEVICE"):
            load_settings(_write(tmp_path, f"REFRAME_DEVICE={device}\n"))

    @pytest.mark.parametrize("raw,expected", [
        ("cpu", "cpu"), ("cuda", "cuda"), ("cuda:1", "cuda:1"),
        ("CUDA", "cuda"), ("Cpu", "cpu"),
    ])
    def test_should_accept_cpu_cuda_and_a_numbered_cuda_device(
        self, tmp_path, raw, expected,
    ):
        settings = load_settings(_write(tmp_path, f"REFRAME_DEVICE={raw}\n"))

        assert settings.device == expected

    @pytest.mark.parametrize("preset", ["p0", "p8", "fastest", "7"])
    def test_should_refuse_a_preset_nvenc_does_not_have(
        self, tmp_path, preset,
    ):
        with pytest.raises(ReframeError, match="REFRAME_PRESET"):
            load_settings(_write(tmp_path, f"REFRAME_PRESET={preset}\n"))

    @pytest.mark.parametrize("raw,expected", [
        ("p1", "p1"), ("p4", "p4"), ("P7", "p7"), ("slow", "slow"),
        ("hq", "hq"),
    ])
    def test_should_accept_every_preset_nvenc_has(
        self, tmp_path, raw, expected,
    ):
        """Lower-cased on the way in: ffmpeg matches preset names
        exactly, so P7 would otherwise fail at render time."""
        settings = load_settings(_write(tmp_path, f"REFRAME_PRESET={raw}\n"))

        assert settings.preset == expected

    def test_should_ignore_the_preset_for_ffv1(self, tmp_path):
        """FFV1 has no presets and the render never passes one."""
        settings = load_settings(_write(
            tmp_path,
            "REFRAME_CODEC=ffv1\nREFRAME_CONTAINER=mkv\n"
            "REFRAME_PRESET=anything\n",
        ))

        assert settings.codec == "ffv1"


class TestPathContract:
    """Path-valued keys resolve against the data root, never the cwd."""

    def test_should_default_models_dir_to_the_data_root(self, tmp_path):
        settings = load_settings(tmp_path / "absent.env")

        assert settings.models_dir == paths.models_dir()
        assert settings.models_dir.is_absolute()

    def test_should_anchor_a_relative_models_dir_at_the_data_root(
        self, tmp_path, monkeypatch,
    ):
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        settings = load_settings(_write(tmp_path, "REFRAME_MODELS_DIR=w\n"))

        assert settings.models_dir == paths.data_root() / "w"

    def test_should_keep_an_absolute_models_dir(self, tmp_path):
        target = tmp_path / "my models"
        settings = load_settings(
            _write(tmp_path, f"REFRAME_MODELS_DIR={target}\n"))

        assert settings.models_dir == target

    def test_should_anchor_relative_model_files_at_the_data_root(
        self, tmp_path,
    ):
        settings = load_settings(_write(
            tmp_path,
            "REFRAME_PERSON_MODEL=w/person.pth\n"
            "REFRAME_FACE_MODEL=w/face.onnx\n"
            "REFRAME_TITLE_MODEL=w/text.onnx\n",
        ))
        root = paths.data_root()

        assert settings.person_model == str(root / "w" / "person.pth")
        assert settings.face_model == str(root / "w" / "face.onnx")
        assert settings.title_model == str(root / "w" / "text.onnx")

    def test_should_expand_the_home_folder_in_model_paths(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))

        settings = load_settings(
            _write(tmp_path, "REFRAME_FACE_MODEL=~/face.onnx\n"))

        assert settings.face_model == str(tmp_path / "face.onnx")

    def test_should_leave_blank_model_paths_blank(self, tmp_path):
        settings = load_settings(_write(
            tmp_path, "REFRAME_PERSON_MODEL=\nREFRAME_FACE_MODEL=   \n"))

        assert settings.person_model == ""
        assert settings.face_model == ""
        assert settings.title_model == ""

    def test_should_anchor_an_environment_override_too(self, tmp_path):
        settings = load_settings(
            tmp_path / "absent.env", env={"REFRAME_MODELS_DIR": "from-env"})

        assert settings.models_dir == paths.data_root() / "from-env"

    def test_should_leave_the_ffmpeg_command_unresolved(self, tmp_path):
        """REFRAME_FFMPEG may be a bare command found on PATH."""
        settings = load_settings(
            tmp_path / "absent.env", env={"REFRAME_FFMPEG": "ffmpeg"})

        assert settings.ffmpeg == "ffmpeg"

    @pytest.mark.skipif(os.name != "nt", reason="Windows path forms only")
    @pytest.mark.parametrize("key", [
        "REFRAME_MODELS_DIR", "REFRAME_PERSON_MODEL",
        "REFRAME_FACE_MODEL", "REFRAME_TITLE_MODEL",
    ])
    def test_should_name_the_key_when_a_path_is_not_a_full_windows_path(
        self, tmp_path, key,
    ):
        """Git Bash's /c/... form is refused, and the message says where."""
        with pytest.raises(ReframeError, match=key):
            load_settings(_write(tmp_path, f"{key}=/c/models/x.onnx\n"))


class TestByteOrderMark:
    def test_should_read_the_first_key_of_a_file_saved_with_a_bom(
        self, tmp_path,
    ):
        """Windows Notepad saves UTF-8 with a byte-order mark. The encoder
        drops it when it reads config.env, so the stage must too, or the
        first key in the file is silently ignored."""
        path = tmp_path / "config.env"
        path.write_bytes(b"\xef\xbb\xbfREFRAME_STRIDE=2\n")

        assert load_settings(path).stride == 2
