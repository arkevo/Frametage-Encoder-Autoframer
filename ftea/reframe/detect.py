"""Strided person + face detection over a landscape source.

Decodes the source ONCE, forward, sampling a precomputed set of absolute
source frame indices; runs a batched GPU person detector and a
concurrent CPU face detector over those frames; emits one frozen
DetectionSet of display-space boxes.

The design choices below were measured on real footage: the decode
mechanism, the decode/detect schedule, the record shape, separate
confidence floors per class, box clamping, and the deliberate absence
of face-to-body association. Each is explained where it is implemented.

torch, torchvision and cv2 are imported LAZILY inside the loader
functions: pytest runs on a different interpreter from the reframe
virtualenv, and this module must import cleanly on both.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import (
    AbstractSet,
    Any,
    Callable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

from .. import paths
from .errors import ReframeError
from .shots import Shot

# Windows hands Python a cp1252 stdout and the first non-ASCII print
# kills the script mid-run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass


class DetectError(ReframeError):
    """Raised when detection cannot produce a trustworthy DetectionSet."""


CLS_PERSON = "person"
CLS_FACE = "face"

DEFAULT_STRIDE = 3
DEFAULT_BATCH_SIZE = 16
DEFAULT_DEVICE = "cuda"
# Person boxes on lace and curtain texture score ~0.60 while real
# people score 0.98-1.00, so 0.8 is what separates a subject from a
# phantom. Face scores NEVER exceed 0.7462 on our test footage, so the
# same number would erase the face class entirely.
DEFAULT_MIN_CONFIDENCE_PERSON = 0.8
DEFAULT_MIN_CONFIDENCE_FACE = 0.35
# 0 == submit the frame at its native display resolution. The only
# setting backed by evidence; measure recall at a new scale before
# changing it.
DEFAULT_DETECT_LONG_SIDE = 0
DEFAULT_MISSING_TOLERANCE = 2
DEFAULT_FACE_MODEL_NAME = "face_detection_yunet_2023mar.onnx"

MIN_DETECT_LONG_SIDE = 320
MIN_BOX_PX = 1.0
FACE_NMS_THRESHOLD = 0.3
FACE_TOP_K = 5000
YUNET_SCORE_COLUMN = 14
COCO_PERSON_LABEL = 1
COCO_NUM_CLASSES = 91
QUEUE_BYTES_BUDGET = 256 * 1024 * 1024

FrameArray = Any
Box = Tuple[float, float, float, float, float]
DecoderFn = Callable[[Path, AbstractSet[int]],
                     Iterator[Tuple[int, FrameArray]]]
BatchDetectorFn = Callable[[Sequence[FrameArray]], List[List[Box]]]


def _log(message: str) -> None:
    """Human-readable progress. stdout is reserved for JSON."""
    print(f"  reframe: {message}", file=sys.stderr)


@dataclass(frozen=True)
class Detection:
    """One detected box on one sampled frame, display-space source px."""

    frame: int
    cls: str
    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


@dataclass(frozen=True)
class DetectionSet:
    """Everything detection knows about one clip.

    A frame IN sampled_frames with no Detection means the detectors ran
    and found nothing. A frame ABSENT from sampled_frames was never
    decoded. Keeping those two apart is the whole reason this record
    publishes the sampled set.
    """

    stride: int
    sampled_frames: Tuple[int, ...]
    detections: Tuple[Detection, ...]

    def for_frame(self, frame: int) -> Tuple[Detection, ...]:
        return tuple(d for d in self.detections if d.frame == frame)

    def of_class(self, cls: str) -> Tuple[Detection, ...]:
        return tuple(d for d in self.detections if d.cls == cls)

    def in_range(self, start_frame: int,
                 end_frame: int) -> Tuple[Detection, ...]:
        """Half-open [start_frame, end_frame), matching Shot exactly."""
        return tuple(
            d for d in self.detections
            if start_frame <= d.frame < end_frame
        )

    def sampled_in_range(self, start_frame: int,
                         end_frame: int) -> Tuple[int, ...]:
        """Half-open [start_frame, end_frame), matching Shot exactly."""
        return tuple(
            f for f in self.sampled_frames
            if start_frame <= f < end_frame
        )

    @property
    def frames_with_detections(self) -> int:
        return len({d.frame for d in self.detections})


def sample_frames(shots: Sequence[Shot], stride: int) -> Tuple[int, ...]:
    """Absolute source frame indices to decode, strictly increasing.

    The stride phase RESETS at every cut: each shot contributes
    range(start_frame, end_frame, stride). Because shots.py guarantees
    half-open contiguous tiling ranges, the per-shot ranges are disjoint
    and their concatenation is already sorted -- so a violation of that
    guarantee is a bug worth raising on, not something to sort away.
    """
    if stride < 1:
        raise DetectError(f"stride must be >= 1, got {stride}")
    if not shots:
        raise DetectError(
            "shots must be non-empty; a single-take clip is one Shot, "
            "never []"
        )
    wanted: List[int] = []
    previous_end = shots[0].start_frame
    for shot in shots:
        if shot.end_frame <= shot.start_frame:
            raise DetectError(
                f"shot {shot.index} is empty: "
                f"[{shot.start_frame}, {shot.end_frame})"
            )
        if shot.start_frame != previous_end:
            raise DetectError(
                f"shots are not contiguous: shot {shot.index} starts at "
                f"{shot.start_frame}, previous ended at {previous_end}"
            )
        wanted.extend(range(shot.start_frame, shot.end_frame, stride))
        previous_end = shot.end_frame
    return tuple(wanted)


def detector_input_size(display_w: int, display_h: int,
                        long_side: int) -> Tuple[int, int]:
    """Frame size actually submitted to the detectors.

    `long_side` is REFRAME_DETECT_LONG_SIDE. 0 -- the default -- means
    native display resolution, the only setting with evidence behind
    it. A value at or above the source's long side is ignored rather
    than upscaling: upscaling costs detector time and invents no
    detail. Plain rounding, NOT camera.py's round_up_even -- the even
    rule exists because yuv420p rejects odd dimensions in render.py, and
    a detector input is an in-memory RGB array with no chroma
    subsampling.
    """
    if display_w < 1 or display_h < 1:
        raise DetectError(
            f"display geometry must be positive, got "
            f"{display_w}x{display_h}"
        )
    if long_side < 0:
        raise DetectError(
            f"detect_long_side must be 0 (native) or "
            f">= {MIN_DETECT_LONG_SIDE}, got {long_side}"
        )
    source_long = max(display_w, display_h)
    if long_side == 0 or long_side >= source_long:
        return display_w, display_h
    if long_side < MIN_DETECT_LONG_SIDE:
        raise DetectError(
            f"detect_long_side must be 0 (native) or "
            f">= {MIN_DETECT_LONG_SIDE}, got {long_side}"
        )
    scale = long_side / float(source_long)
    width = max(1, int(round(display_w * scale)))
    height = max(1, int(round(display_h * scale)))
    return width, height


def scale_box(box: Box, *, from_size: Tuple[int, int],
              to_size: Tuple[int, int]) -> Box:
    """Map a box between two frame sizes, per axis.

    Used to bring a detector-space box back to display-space source
    pixels when REFRAME_DETECT_LONG_SIDE downscaled the input. The two
    axes get independent ratios because the rounded detector size is not
    exactly proportional on both.
    """
    if from_size == to_size:
        return box
    from_w, from_h = from_size
    to_w, to_h = to_size
    if from_w < 1 or from_h < 1:
        raise DetectError(f"cannot scale from {from_size}")
    ratio_x = to_w / float(from_w)
    ratio_y = to_h / float(from_h)
    x1, y1, x2, y2, score = box
    return (x1 * ratio_x, y1 * ratio_y, x2 * ratio_x, y2 * ratio_y, score)


def clamp_box(box: Box, width: int, height: int) -> Optional[Box]:
    """Clamp into [0, width] x [0, height]; None if nothing survives.

    On our test footage 3912 boxes extended outside the frame. A box
    entirely outside clamps to zero area, and a zero-area "subject"
    would drag camera.py's crop to an edge -- so it is dropped rather
    than emitted.
    """
    x1, y1, x2, y2, score = (float(v) for v in box)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    limit_w = float(width)
    limit_h = float(height)
    x1 = min(max(x1, 0.0), limit_w)
    x2 = min(max(x2, 0.0), limit_w)
    y1 = min(max(y1, 0.0), limit_h)
    y2 = min(max(y2, 0.0), limit_h)
    if (x2 - x1) < MIN_BOX_PX or (y2 - y1) < MIN_BOX_PX:
        return None
    return (x1, y1, x2, y2, score)


def _records_for_class(
    frame_index: int,
    boxes: Sequence[Box],
    cls: str,
    floor: float,
    source_size: Tuple[int, int],
    detect_size: Tuple[int, int],
) -> List[Detection]:
    """Filter by score, map to source pixels, clamp, then label.

    Order matters. The score filter is scale-independent, so it runs
    first and the arithmetic is only paid for boxes that survive. The
    clamp runs LAST, because a box legal in detector space can exceed
    the source frame once it is scaled up.
    """
    out: List[Detection] = []
    width, height = source_size
    for box in boxes:
        if float(box[4]) < floor:
            continue
        scaled = scale_box(box, from_size=detect_size, to_size=source_size)
        clamped = clamp_box(scaled, width, height)
        if clamped is None:
            continue
        x1, y1, x2, y2, score = clamped
        out.append(Detection(frame=frame_index, cls=cls, x1=x1, y1=y1,
                             x2=x2, y2=y2, score=score))
    return out


def records_for_frame(
    frame_index: int,
    person_boxes: Sequence[Box],
    face_boxes: Sequence[Box],
    *,
    source_size: Tuple[int, int],
    detect_size: Tuple[int, int],
    min_confidence_person: float = DEFAULT_MIN_CONFIDENCE_PERSON,
    min_confidence_face: float = DEFAULT_MIN_CONFIDENCE_FACE,
) -> List[Detection]:
    """Turn one frame's raw detector output into Detection records.

    SEPARATE floors per class, and that is not a style choice: on our
    test footage person scores reach 0.9856 while face scores never
    exceed 0.7462, so no single number can serve both. The floors are
    applied HERE as well as inside the detectors so an injected fake
    exercises the same rule the production path uses.

    Faces are NOT matched to bodies -- association belongs in
    subject.py, which needs a face-without-body branch for the 34.8% of
    faces that have no containing person box.
    """
    records = _records_for_class(
        frame_index, person_boxes, CLS_PERSON, min_confidence_person,
        source_size, detect_size,
    )
    records.extend(_records_for_class(
        frame_index, face_boxes, CLS_FACE, min_confidence_face,
        source_size, detect_size,
    ))
    return records


def decode_frames(
    source: Path,
    wanted: AbstractSet[int],
) -> Iterator[Tuple[int, FrameArray]]:
    """Decode ONE forward pass, retrieving only the wanted frames.

    grab() on every frame, retrieve() only on members of `wanted`.
    This measured 3.482 ms per decoded source frame at 4K stride 2
    against 32.209 for an ffmpeg rawvideo pipe: grab() skips colour
    conversion entirely and only retrieve() pays it. Never seeks -- a
    long-GOP seek re-decodes from the preceding keyframe anyway -- and
    never uses an ffmpeg select prefilter, which cannot report an
    absolute source frame index and so cannot support per-shot striding.

    cv2.VideoCapture takes a bare filename string; filenames with
    apostrophes and spaces were tested and open correctly.
    """
    if not wanted:
        return
    import cv2

    cap = cv2.VideoCapture(str(source))
    # VideoCapture fails SILENTLY on a missing or unreadable file: no
    # exception, isOpened() False, CAP_PROP_FRAME_COUNT -1.0. Without
    # this check a broken source looks exactly like a clip with nobody
    # in it, and subject.py would centre-crop it without complaint.
    if not cap.isOpened():
        raise DetectError(f"cannot open for decoding: {source}")
    try:
        last = max(wanted)
        index = 0
        while index <= last:
            if not cap.grab():
                return
            if index in wanted:
                ok, frame = cap.retrieve()
                if not ok:
                    return
                yield index, frame
            index += 1
    finally:
        cap.release()


def torch_home(models_dir: Path) -> Path:
    """The torch hub cache for the person detector: <models dir>/torch."""
    return Path(models_dir) / "torch"


def _models_dir_or_default(models_dir: Optional[Path]) -> Path:
    """The given models directory, or <data root>/models when None."""
    return paths.models_dir() if models_dir is None else Path(models_dir)


def face_model_path(models_dir: Path, face_model: str) -> Path:
    """Resolve REFRAME_FACE_MODEL, or the default YuNet file in the
    models directory (fetched there, not shipped in the repository)."""
    if face_model:
        return Path(face_model)
    return Path(models_dir) / DEFAULT_FACE_MODEL_NAME


def load_person_detector(
    *,
    device: str = DEFAULT_DEVICE,
    models_dir: Optional[Path] = None,
    weights_path: str = "",
    score_floor: float = DEFAULT_MIN_CONFIDENCE_PERSON,
) -> BatchDetectorFn:
    """torchvision Faster R-CNN MobileNetV3-L FPN, COCO_V1, BSD-3.

    Note what this model does NOT do: it does not see native 4K.
    GeneralizedRCNNTransform resizes every input to a 1333-px long side
    (min_size=800, max_size=1333), so a 16:9 source of ANY tier reaches
    the backbone at 1333x749. Person recall is therefore
    resolution-invariant by construction, and the only detector whose
    cost and recall move with source resolution is YuNet.
    """
    import os

    # Enforced, not setdefault: a TORCH_HOME already set on the machine
    # would otherwise send the weights into the user profile instead of
    # the models directory. The models directory (REFRAME_MODELS_DIR, or
    # <data root>/models) is the one control over where weights live.
    target = str(torch_home(_models_dir_or_default(models_dir)))
    existing = os.environ.get("TORCH_HOME")
    if existing is not None and existing != target:
        print(f"  reframe: TORCH_HOME overridden to {target} "
              f"(was {existing})", file=sys.stderr)
    os.environ["TORCH_HOME"] = target
    import torch
    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_fpn,
    )

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise DetectError(
            "REFRAME_DEVICE requests CUDA but torch reports no CUDA "
            "device; run 'python -m ftea.reframe.verify_env' with the "
            "reframe venv interpreter to diagnose, or set "
            "REFRAME_DEVICE=cpu to run person detection on the CPU "
            "(much slower; the render still needs NVENC)"
        )
    torch_device = torch.device(device)

    if weights_path:
        path = Path(weights_path)
        if not path.exists():
            raise DetectError(f"REFRAME_PERSON_MODEL not found: {path}")
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=None, weights_backbone=None,
            num_classes=COCO_NUM_CLASSES, box_score_thresh=score_floor,
        )
        model.load_state_dict(torch.load(str(path), map_location="cpu"))
    else:
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1,
            box_score_thresh=score_floor,
        )
    model = model.eval().to(torch_device)

    def run(frames: Sequence[FrameArray]) -> List[List[Box]]:
        if not frames:
            return []
        with torch.inference_mode():
            # uint8 goes to the device FIRST, so the host->device copy
            # is 24.9 MB per 4K frame rather than the 99.5 MB a
            # host-side float32 conversion would move.
            batch = [
                torch.from_numpy(frame)
                .to(torch_device, non_blocking=True)
                .permute(2, 0, 1)      # HWC -> CHW
                .flip(0)               # BGR -> RGB
                .to(torch.float32)
                .div_(255.0)
                for frame in frames
            ]
            outputs = model(batch)
        results: List[List[Box]] = []
        for out in outputs:
            keep = out["labels"] == COCO_PERSON_LABEL
            boxes = out["boxes"][keep].detach().to("cpu").tolist()
            scores = out["scores"][keep].detach().to("cpu").tolist()
            results.append([
                (float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                 float(s))
                for b, s in zip(boxes, scores)
            ])
        return results

    return run


def load_face_detector(
    *,
    model_path: Path,
    input_size: Tuple[int, int],
    score_floor: float = DEFAULT_MIN_CONFIDENCE_FACE,
) -> BatchDetectorFn:
    """OpenCV Zoo YuNet, MIT, on CPU.

    score_threshold, nms_threshold and top_k are passed as NAMED
    arguments. This matters: the positional-default call in
    verify_env.py inherits OpenCV's score_threshold of 0.9, which would
    filter out every face this pipeline can see -- face scores never
    exceed 0.7462 on our test footage.

    The returned callable is only ever invoked from the single-worker
    face pool, so the mutable `current` input size is never shared
    across threads.
    """
    import cv2

    path = Path(model_path)
    if not path.exists():
        raise DetectError(
            f"face model not found: {path} -- run "
            "install.sh to fetch "
            f"{DEFAULT_FACE_MODEL_NAME}"
        )
    detector = cv2.FaceDetectorYN_create(
        str(path), "", (320, 320),
        score_threshold=float(score_floor),
        nms_threshold=FACE_NMS_THRESHOLD,
        top_k=FACE_TOP_K,
    )
    current = (int(input_size[0]), int(input_size[1]))
    detector.setInputSize(current)

    def run(frames: Sequence[FrameArray]) -> List[List[Box]]:
        nonlocal current
        results: List[List[Box]] = []
        for frame in frames:
            size = (int(frame.shape[1]), int(frame.shape[0]))
            if size != current:
                detector.setInputSize(size)
                current = size
            _, faces = detector.detect(frame)
            # YuNet returns None -- not an empty array -- when it finds
            # nothing, and its retval is 1 either way, so retval must
            # NOT be used as a "found something" flag.
            if faces is None:
                results.append([])
                continue
            results.append([
                (float(row[0]), float(row[1]),
                 float(row[0]) + float(row[2]),
                 float(row[1]) + float(row[3]),
                 float(row[YUNET_SCORE_COLUMN]))
                for row in faces
            ])
        return results

    return run


_SENTINEL = object()
_JOIN_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class _Failure:
    """Carries a decoder-thread exception back to the consumer."""

    error: BaseException


@dataclass(frozen=True)
class DetectStats:
    """Measured cost of one detection pass. Never part of DetectionSet.

    face_join_s is the RESIDUAL face time -- what was left to wait for
    after the GPU person pass finished -- not the total face cost. With
    the face pass running alongside the person pass (see run_detection),
    a small residual is the point.
    """

    sampled: int
    decoded: int
    batches: int
    elapsed_s: float
    decode_wait_s: float
    person_s: float
    face_join_s: float
    detect_width: int
    detect_height: int
    clip_seconds: float

    @property
    def ms_per_sampled_frame(self) -> float:
        """Divided by `decoded`, not `sampled` -- the honest denominator.

        `sampled` is what the schedule asked for; `decoded` is what
        actually went through both detectors. They differ only when the
        tolerated tail shortfall fired, and in that case dividing by the
        larger number would flatter the result.
        """
        if self.decoded < 1:
            return 0.0
        return (self.elapsed_s * 1000.0) / float(self.decoded)

    @property
    def realtime_ratio(self) -> float:
        """Clip seconds per wall second. > 1.0 beats real time."""
        if self.elapsed_s <= 0.0 or self.clip_seconds <= 0.0:
            return 0.0
        return self.clip_seconds / self.elapsed_s


def _validate_run_args(batch_size: int, person_floor: float,
                       face_floor: float, missing_tolerance: int) -> None:
    if batch_size < 1:
        raise DetectError(f"batch_size must be >= 1, got {batch_size}")
    if not 0.0 < person_floor < 1.0:
        raise DetectError(
            f"min_confidence_person must be in (0, 1), got {person_floor}")
    if not 0.0 < face_floor < 1.0:
        raise DetectError(
            f"min_confidence_face must be in (0, 1), got {face_floor}")
    if missing_tolerance < 0:
        raise DetectError(
            f"missing_tolerance must be >= 0, got {missing_tolerance}")


def _check_frame(frame: FrameArray, source_size: Tuple[int, int]) -> None:
    """Every decoded frame must be in DISPLAY space.

    OpenCV auto-rotates (CAP_PROP_ORIENTATION_AUTO is 1.0 by default),
    so what arrives is already display space. A mismatch means someone
    upstream handed us meta.width where meta.display_width was meant,
    and every box after this point would be in the wrong coordinate
    system, silently.
    """
    height = int(frame.shape[0])
    width = int(frame.shape[1])
    if (width, height) != source_size:
        raise DetectError(
            f"decoded frame is {width}x{height} but probe reported "
            f"display geometry {source_size[0]}x{source_size[1]}; detect "
            "works in DISPLAY space only"
        )


def _resize_for_detection(frame: FrameArray, detect_size: Tuple[int, int],
                          source_size: Tuple[int, int]) -> FrameArray:
    """Downscale only when REFRAME_DETECT_LONG_SIDE asked for it.

    cv2 is imported here rather than at module scope so the default
    (native) path never needs OpenCV at all -- which is what lets every
    logic test run on an interpreter that has a different cv2 build.
    """
    if detect_size == source_size:
        return frame
    import cv2

    return cv2.resize(frame, detect_size, interpolation=cv2.INTER_AREA)


def _validate_shot_coverage(shots: Sequence[Shot],
                            sampled: Tuple[int, ...]) -> None:
    """Every shot must keep at least one sampled frame.

    A shot with zero samples is indistinguishable, in the published
    contract, from a shot where nobody was found -- and subject.py
    centre-crops the latter. So it raises.
    """
    if not sampled:
        raise DetectError("no frames were decoded at all")
    pointer = 0
    for shot in shots:
        found = False
        while pointer < len(sampled) and sampled[pointer] < shot.end_frame:
            if sampled[pointer] >= shot.start_frame:
                found = True
            pointer += 1
        if not found:
            raise DetectError(
                f"shot {shot.index} [{shot.start_frame}, "
                f"{shot.end_frame}) has no sampled frame; subject.py "
                "cannot tell it apart from a shot where nobody was found"
            )


def _drain(work_queue: "queue.Queue", producer: threading.Thread,
           timeout_s: float = _JOIN_TIMEOUT_S) -> None:
    """Keep emptying the queue until the producer thread has exited.

    A single pass is not enough. When the consumer raises mid-clip the
    producer is typically blocked on a full queue; draining once frees
    it to put the next frame, refill the queue and block again. So drain
    in a loop for as long as it is alive, then take whatever is left.
    Without this the join below waits out its whole timeout on every
    failing test.
    """
    deadline = time.perf_counter() + timeout_s
    while producer.is_alive() and time.perf_counter() < deadline:
        try:
            work_queue.get(timeout=0.05)
        except queue.Empty:
            continue
    while True:
        try:
            work_queue.get_nowait()
        except queue.Empty:
            return


def _detector_error(stage: str, exc: BaseException, n: int,
                    batch_size: int) -> DetectError:
    """Classify a detector-call failure: only an out-of-memory is wrapped.

    An out-of-memory condition is foreseeable and has a config-level
    remedy, so it is named. Anything else -- a TypeError or index error
    in our own batching code -- is a bug, not an operational condition,
    and the traceback is the whole diagnosis, so it is re-raised
    unchanged.

    `n` is the failing batch's length, which is shorter than the setting
    on the tail batch; `batch_size` is the setting the remedy names. The
    person detector runs on REFRAME_DEVICE and the face detector on the
    CPU, so each gets its own remedy.
    """
    is_oom = (
        isinstance(exc, MemoryError)
        or type(exc).__name__ == "OutOfMemoryError"
        or "out of memory" in str(exc).lower()
    )
    if not is_oom:
        raise exc
    if stage == "face":
        remedy = (
            "the face detector runs on the CPU, so this is host memory: "
            f"lower REFRAME_BATCH_SIZE (currently {batch_size}) or set "
            "REFRAME_DETECT_LONG_SIDE to detect on smaller frames"
        )
    else:
        remedy = (
            "if this is a CUDA out-of-memory, lower REFRAME_BATCH_SIZE "
            f"(currently {batch_size}) or set REFRAME_DEVICE=cpu to run "
            "person detection on the CPU (much slower)"
        )
    return DetectError(
        f"{stage} detection failed on a batch of {n} frame(s): {exc}; "
        f"{remedy}"
    )


def run_detection(
    meta: Any,
    shots: Sequence[Shot],
    *,
    stride: int = DEFAULT_STRIDE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = DEFAULT_DEVICE,
    models_dir: Optional[Path] = None,
    person_model: str = "",
    face_model: str = "",
    min_confidence_person: float = DEFAULT_MIN_CONFIDENCE_PERSON,
    min_confidence_face: float = DEFAULT_MIN_CONFIDENCE_FACE,
    detect_long_side: int = DEFAULT_DETECT_LONG_SIDE,
    missing_tolerance: int = DEFAULT_MISSING_TOLERANCE,
    decoder_fn: Optional[DecoderFn] = None,
    person_fn: Optional[BatchDetectorFn] = None,
    face_fn: Optional[BatchDetectorFn] = None,
) -> Tuple[DetectionSet, DetectStats]:
    """Detect people and faces over one clip, and report what it cost.

    The schedule that measured best: the decoder runs on its own thread
    behind a bounded queue, the CPU face pass for each batch runs on a
    single-worker pool, and the GPU person pass for the same batch runs
    in this thread. In our measurements: 146.730 ms/sampled frame at 4K
    against sequential's 188.678, at identical (1.000) face recall --
    the only schedule tried besides sequential that kept face recall at
    95% or better.
    """
    _validate_run_args(batch_size, min_confidence_person,
                       min_confidence_face, missing_tolerance)
    models_dir = _models_dir_or_default(models_dir)
    source_size = (meta.display_width, meta.display_height)
    detect_size = detector_input_size(source_size[0], source_size[1],
                                      detect_long_side)
    wanted = sample_frames(shots, stride)
    wanted_set = frozenset(wanted)

    decoder = decoder_fn or decode_frames
    person = person_fn or load_person_detector(
        device=device, models_dir=models_dir, weights_path=person_model,
        score_floor=min_confidence_person,
    )
    face = face_fn or load_face_detector(
        model_path=face_model_path(models_dir, face_model),
        input_size=detect_size, score_floor=min_confidence_face,
    )

    frame_bytes = detect_size[0] * detect_size[1] * 3
    queue_depth = max(
        2, min(batch_size, QUEUE_BYTES_BUDGET // max(1, frame_bytes)))

    _log(
        f"detect: {len(wanted)} sampled frame(s), stride {stride}, "
        f"batch {batch_size}, detector input "
        f"{detect_size[0]}x{detect_size[1]}, queue depth {queue_depth}"
    )

    work: "queue.Queue" = queue.Queue(maxsize=queue_depth)
    stop = threading.Event()

    def _produce() -> None:
        frames = decoder(meta.path, wanted_set)
        try:
            for index, frame in frames:
                if stop.is_set():
                    break
                _check_frame(frame, source_size)
                work.put(
                    (index, _resize_for_detection(frame, detect_size,
                                                  source_size))
                )
        except BaseException as exc:   # forwarded to the consumer
            work.put(_Failure(exc))
        finally:
            close = getattr(frames, "close", None)
            if close is not None:
                close()
            work.put(_SENTINEL)

    collected: List[Detection] = []
    decoded: List[int] = []
    decode_wait_s = 0.0
    person_s = 0.0
    face_join_s = 0.0
    batches = 0

    def _flush(items: List[Tuple[int, FrameArray]]) -> None:
        nonlocal person_s, face_join_s, batches
        frames = [f for _, f in items]
        future = pool.submit(face, frames)
        mark = time.perf_counter()
        try:
            person_boxes = person(frames)
        except BaseException as exc:
            future.cancel()
            raise _detector_error(
                "person", exc, len(frames), batch_size) from exc
        person_s += time.perf_counter() - mark
        mark = time.perf_counter()
        try:
            face_boxes = future.result()
        except BaseException as exc:
            raise _detector_error(
                "face", exc, len(frames), batch_size) from exc
        face_join_s += time.perf_counter() - mark
        if len(person_boxes) != len(frames) or \
                len(face_boxes) != len(frames):
            raise DetectError(
                f"detector returned {len(person_boxes)} person and "
                f"{len(face_boxes)} face result lists for {len(frames)} "
                "frame(s); one list per frame is required"
            )
        for (index, _), persons, faces in zip(items, person_boxes,
                                              face_boxes):
            decoded.append(index)
            collected.extend(records_for_frame(
                index, persons, faces,
                source_size=source_size, detect_size=detect_size,
                min_confidence_person=min_confidence_person,
                min_confidence_face=min_confidence_face,
            ))
        batches += 1

    producer = threading.Thread(target=_produce, name="reframe-decode",
                                daemon=True)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rf-face")
    started = time.perf_counter()
    producer.start()
    try:
        batch: List[Tuple[int, FrameArray]] = []
        while True:
            mark = time.perf_counter()
            item = work.get()
            decode_wait_s += time.perf_counter() - mark
            if item is _SENTINEL:
                break
            if isinstance(item, _Failure):
                raise DetectError(
                    f"decode failed: {item.error}") from item.error
            batch.append(item)
            if len(batch) >= batch_size:
                _flush(batch)
                batch = []
        if batch:
            _flush(batch)
    finally:
        stop.set()
        _drain(work, producer)
        pool.shutdown(wait=True)
        producer.join(timeout=_JOIN_TIMEOUT_S)
    elapsed_s = time.perf_counter() - started

    sampled = tuple(sorted(decoded))
    seen = set(sampled)
    missing = [f for f in wanted if f not in seen]
    # The missing-frame tolerance is about the tail (shots.py's own
    # frame-count slack); a gap that is NOT a contiguous suffix of
    # `wanted` is a different condition -- a frame failed mid-clip -- and
    # is reported with different text so the two are never confused.
    is_tail = bool(missing) and list(wanted[-len(missing):]) == missing
    if len(missing) > missing_tolerance:
        if is_tail:
            raise DetectError(
                f"source ended early: {len(missing)} of {len(wanted)} "
                f"sampled frame(s) were never decoded (from {missing[0]})"
            )
        raise DetectError(
            f"{len(missing)} sampled frame(s) failed to decode "
            f"mid-clip (first {missing[0]}); tolerance is "
            f"{missing_tolerance}"
        )
    if missing:
        if is_tail:
            _log(
                f"detect: WARNING {len(missing)} sampled frame(s) never "
                f"decoded at the tail: {missing}"
            )
        else:
            _log(
                f"detect: WARNING {len(missing)} sampled frame(s) "
                f"failed to decode mid-clip (tolerated): {missing}"
            )
    _validate_shot_coverage(shots, sampled)

    detections = tuple(sorted(
        collected, key=lambda d: (d.frame, d.cls, -d.score, d.x1)))
    dset = DetectionSet(stride=stride, sampled_frames=sampled,
                        detections=detections)
    stats = DetectStats(
        sampled=len(wanted), decoded=len(sampled), batches=batches,
        elapsed_s=elapsed_s, decode_wait_s=decode_wait_s,
        person_s=person_s, face_join_s=face_join_s,
        detect_width=detect_size[0], detect_height=detect_size[1],
        clip_seconds=meta.duration_s,
    )
    _log(
        f"detect: {len(detections)} detection(s) over {len(sampled)} "
        f"frame(s) in {elapsed_s:.3f}s "
        f"({stats.ms_per_sampled_frame:.1f} ms/frame, "
        f"{stats.realtime_ratio:.3f}x real time)"
    )
    return dset, stats


def detect_video(meta: Any, shots: Sequence[Shot],
                 **kwargs: Any) -> DetectionSet:
    """run_detection without the DetectStats.

    Not used by the stage itself (pipeline.py calls detect_from_settings);
    kept for debugging and tests, so a caller that only wants detections
    never has to unpack a tuple.
    """
    dset, _stats = run_detection(meta, shots, **kwargs)
    return dset


def detections_to_json(dset: DetectionSet, *,
                       stats: Optional[DetectStats] = None) -> dict:
    """A debuggable dump of one detection pass.

    Publishes sampled_frames alongside detections so a reader can tell
    "looked, found nobody" from "never looked" -- the same distinction
    the record itself exists to preserve.
    """
    payload: dict = {
        "stride": dset.stride,
        "sampled_frames": list(dset.sampled_frames),
        "detections": [
            {
                "frame": det.frame,
                "cls": det.cls,
                "x1": round(det.x1, 2),
                "y1": round(det.y1, 2),
                "x2": round(det.x2, 2),
                "y2": round(det.y2, 2),
                "score": round(det.score, 4),
            }
            for det in dset.detections
        ],
    }
    if stats is not None:
        payload["stats"] = {
            "sampled": stats.sampled,
            "decoded": stats.decoded,
            "batches": stats.batches,
            "elapsed_s": round(stats.elapsed_s, 3),
            "decode_wait_s": round(stats.decode_wait_s, 3),
            "person_s": round(stats.person_s, 3),
            "face_join_s": round(stats.face_join_s, 3),
            "detect_width": stats.detect_width,
            "detect_height": stats.detect_height,
            "ms_per_sampled_frame": round(stats.ms_per_sampled_frame, 3),
            "clip_seconds": round(stats.clip_seconds, 3),
            "realtime_ratio": round(stats.realtime_ratio, 3),
        }
    return payload


def detect_from_settings(
    meta: Any, shots: Sequence[Shot], settings: Any, **overrides: Any,
) -> Tuple[DetectionSet, DetectStats]:
    """Unpack ReframeSettings into run_detection's keyword arguments.

    This is the one place the stage's config shape meets this module, so
    pipeline.py's run_stages can call detection in a single line and
    detect.py itself never reads config.env.
    """
    kwargs: dict = {
        "stride": settings.stride,
        "batch_size": settings.batch_size,
        "device": settings.device,
        "models_dir": settings.models_dir,
        "person_model": settings.person_model,
        "face_model": settings.face_model,
        "min_confidence_person": settings.min_confidence_person,
        "min_confidence_face": settings.min_confidence_face,
        "detect_long_side": settings.detect_long_side,
    }
    kwargs.update(overrides)
    return run_detection(meta, shots, **kwargs)


def _pick(*candidates: Any) -> Any:
    """First non-None candidate. CLI flag, then config, then default."""
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None


def main(argv: Optional[List[str]] = None) -> int:
    """Diagnostic CLI: python -m ftea.reframe.detect <clip> [flags].

    Exit codes come from probe.py's constants: 0 ok, 1 error (detection
    has no skip path).

    Settings load from <data root>/config.env by default (or --config),
    overlaid with REFRAME_* from the process environment -- the same
    overlay the stage itself applies. Shots are cut with the configured
    REFRAME_SCENE_THRESHOLD and REFRAME_MIN_SHOT_SEC, as the stage cuts
    them, so the same frames are sampled; the flags below then override
    single detection settings. A missing default config.env is not an
    error, since every key has a default, but a --config path that does
    not exist is.
    """
    import argparse
    import json
    import os

    from .probe import EXIT_ERROR, EXIT_REFRAME, ProbeError, probe_source
    from .settings import load_settings
    from .shots import detect_shots

    parser = argparse.ArgumentParser(
        prog="python -m ftea.reframe.detect",
        description="Strided person + face detection over one clip.",
    )
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--config", type=Path, default=None,
        help="config.env to read REFRAME_* from "
             "(default: <data root>/config.env)")
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--detect-long-side", type=int, default=None)
    parser.add_argument("--person-floor", type=float, default=None)
    parser.add_argument("--face-floor", type=float, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.config is not None and not args.config.is_file():
        print(f"  reframe: --config {args.config} does not exist",
              file=sys.stderr)
        return EXIT_ERROR
    config_path = args.config or paths.config_path()
    try:
        settings = load_settings(config_path, env=os.environ)
        meta = probe_source(args.source)
        shots = detect_shots(
            meta,
            threshold=settings.scene_threshold,
            min_shot_sec=settings.min_shot_sec,
        )
        dset, stats = run_detection(
            meta, shots,
            stride=_pick(args.stride, settings.stride),
            batch_size=_pick(args.batch_size, settings.batch_size),
            device=_pick(args.device, settings.device),
            models_dir=settings.models_dir,
            person_model=settings.person_model,
            face_model=settings.face_model,
            min_confidence_person=_pick(
                args.person_floor, settings.min_confidence_person),
            min_confidence_face=_pick(
                args.face_floor, settings.min_confidence_face),
            detect_long_side=_pick(
                args.detect_long_side, settings.detect_long_side),
        )
    except (ReframeError, ProbeError) as exc:
        print(f"  reframe: detection failed: {exc}", file=sys.stderr)
        return EXIT_ERROR

    text = json.dumps(detections_to_json(dset, stats=stats), indent=2)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return EXIT_REFRAME


if __name__ == "__main__":   # pragma: no cover - CLI entry
    raise SystemExit(main())
