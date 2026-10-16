"""Per-shot subject selection for the auto-reframe stage.

Consumes the shot list from `shots.py` and the sampled detections from
`detect.py`, and emits ONE `ShotSubject` per shot: the point `camera.py`
should follow at every sampled frame, whether that point was actually
observed, and where the followed subject changed.

Everything here is pure logic -- no I/O, no subprocess, no GPU, no video
file. Every coordinate it emits is NORMALIZED to the source's DISPLAY
dimensions, so one set of tuning weights behaves identically at 720p,
1080p, 1440p, 4K, 2560x1080 ultrawide and letterboxed 1920x816.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .detect import Detection, DetectionSet
from .errors import ReframeError
from .probe import SourceMeta
from .shots import Shot


class SubjectError(ReframeError):
    """Raised when subject selection cannot produce a trustworthy track."""


DEFAULT_FACE_WEIGHT = 2.0
DEFAULT_HYSTERESIS_MARGIN = 0.2
DEFAULT_HYSTERESIS_SEC = 1.0
DEFAULT_CENTRALITY_WEIGHT = 0.5
DEFAULT_CONFIDENCE_WEIGHT = 0.3
DEFAULT_PERSISTENCE_WEIGHT = 0.5
DEFAULT_MIN_CONFIDENCE_PERSON = 0.8
DEFAULT_MIN_CONFIDENCE_FACE = 0.35
DEFAULT_IOU_MATCH_MIN = 0.3
DEFAULT_TRACK_GAP_SEC = 0.25

CROSS_KIND_PENALTY = 0.5

CLS_PERSON = "person"
CLS_FACE = "face"

KIND_PERSON = "person"
KIND_FACE = "face"
KIND_BOTH = "person+face"
KIND_HELD = "held"
KIND_CENTER = "center"

NO_TRACK = -1


@dataclass(frozen=True)
class NormBox:
    """An axis-aligned box in normalized display coordinates, [0, 1]."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def extent(self) -> float:
        """Geometric mean of the normalized sides -- a linear size."""
        return math.sqrt(self.width * self.height)


@dataclass(frozen=True)
class Observation:
    """One subject candidate at one sampled frame."""

    frame: int
    person: Optional[NormBox]
    face: Optional[NormBox]
    person_score: float
    face_score: float

    def __post_init__(self) -> None:
        if self.person is None and self.face is None:
            raise SubjectError(
                f"observation at frame {self.frame} has neither a person "
                "nor a face box; it needs at least one")

    @property
    def anchor(self) -> NormBox:
        return self.person if self.person is not None else self.face

    @property
    def kind(self) -> str:
        if self.person is not None and self.face is not None:
            return KIND_BOTH
        if self.person is not None:
            return KIND_PERSON
        return KIND_FACE

    @property
    def point(self) -> Tuple[float, float]:
        """The point camera.py follows: the face when one was found."""
        box = self.face if self.face is not None else self.anchor
        return (box.cx, box.cy)


@dataclass(frozen=True)
class Track:
    """One subject candidate followed across a shot's sampled frames."""

    track_id: int
    observations: Tuple[Observation, ...]

    @property
    def first_frame(self) -> int:
        return self.observations[0].frame

    @property
    def last_frame(self) -> int:
        return self.observations[-1].frame

    def at(self, frame: int) -> Optional[Observation]:
        for obs in self.observations:
            if obs.frame == frame:
                return obs
        return None


@dataclass(frozen=True)
class SubjectSample:
    """The followed point at ONE sampled frame of ONE shot."""

    frame: int
    cx: float
    cy: float
    x1: float
    y1: float
    x2: float
    y2: float
    observed: bool
    kind: str
    score: float
    track_id: int

    @property
    def box(self) -> NormBox:
        return NormBox(self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class ShotSubject:
    """Everything camera.py needs for ONE shot."""

    shot_index: int
    start_frame: int
    end_frame: int
    fps: float
    stride: int
    sampled_frames: Tuple[int, ...]
    samples: Tuple[SubjectSample, ...]
    has_subject: bool
    track_count: int
    switch_frames: Tuple[int, ...]

    @property
    def observed_fraction(self) -> float:
        if not self.samples:
            return 0.0
        hits = sum(1 for s in self.samples if s.observed)
        return round(hits / len(self.samples), 6)


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def normalize_box(
    det: Detection, frame_width: int, frame_height: int
) -> NormBox:
    """Map a display-space pixel box onto the unit square."""
    if frame_width <= 0 or frame_height <= 0:
        raise SubjectError(
            "frame dimensions must be positive, got "
            f"{frame_width}x{frame_height}"
        )
    x1 = clamp01(min(det.x1, det.x2) / frame_width)
    x2 = clamp01(max(det.x1, det.x2) / frame_width)
    y1 = clamp01(min(det.y1, det.y2) / frame_height)
    y2 = clamp01(max(det.y1, det.y2) / frame_height)
    return NormBox(x1, y1, x2, y2)


def intersection_area(a: NormBox, b: NormBox) -> float:
    width = min(a.x2, b.x2) - max(a.x1, b.x1)
    height = min(a.y2, b.y2) - max(a.y1, b.y1)
    if width <= 0.0 or height <= 0.0:
        return 0.0
    return width * height


def iou(a: NormBox, b: NormBox) -> float:
    inter = intersection_area(a, b)
    union = a.area + b.area - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def contains_ratio(small: NormBox, big: NormBox) -> float:
    """How much of `small` lies inside `big`, in [0, 1]."""
    if small.area <= 0.0:
        return 0.0
    return intersection_area(small, big) / small.area


def associate_faces(
    persons: Sequence[Tuple[NormBox, float]],
    faces: Sequence[Tuple[NormBox, float]],
) -> Tuple[Dict[int, int], List[int]]:
    """Attach each face to the tightest person box containing its centre.

    Returns `(face_index -> person_index, orphan_face_indices)`. A person
    keeps at most ONE face, the largest of those assigned to it; the rest
    become orphans, because a second face inside one body box is almost
    always a different person whose body was not detected.
    """
    assigned: Dict[int, int] = {}
    orphans: List[int] = []
    for f_idx, (face, _score) in enumerate(faces):
        best: Optional[int] = None
        best_area = 0.0
        for p_idx, (person, _pscore) in enumerate(persons):
            if not (person.x1 <= face.cx <= person.x2):
                continue
            if not (person.y1 <= face.cy <= person.y2):
                continue
            if best is None or person.area < best_area:
                best = p_idx
                best_area = person.area
        if best is None:
            orphans.append(f_idx)
        else:
            assigned[f_idx] = best

    kept: Dict[int, int] = {}
    for p_idx in {v for v in assigned.values()}:
        members = [f for f, p in assigned.items() if p == p_idx]
        members.sort(key=lambda f: (-faces[f][0].area, f))
        kept[members[0]] = p_idx
        orphans.extend(members[1:])
    orphans.sort()
    return kept, orphans


def observations_for_frame(
    frame: int,
    detections: Sequence[Detection],
    frame_width: int,
    frame_height: int,
    *,
    min_confidence_person: float = DEFAULT_MIN_CONFIDENCE_PERSON,
    min_confidence_face: float = DEFAULT_MIN_CONFIDENCE_FACE,
) -> List[Observation]:
    """Turn one frame's raw detections into subject candidates.

    Detections below their class's own floor are dropped here, before
    association: they are a `detect.py` contract violation, not a
    low-quality candidate to demerit softly downstream.
    """
    persons = [
        (normalize_box(d, frame_width, frame_height), d.score)
        for d in detections
        if d.cls == CLS_PERSON and d.score >= min_confidence_person
    ]
    faces = [
        (normalize_box(d, frame_width, frame_height), d.score)
        for d in detections
        if d.cls == CLS_FACE and d.score >= min_confidence_face
    ]
    kept, orphans = associate_faces(persons, faces)
    face_of: Dict[int, int] = {p: f for f, p in kept.items()}

    result: List[Observation] = []
    for p_idx, (person, pscore) in enumerate(persons):
        f_idx = face_of.get(p_idx)
        face = faces[f_idx][0] if f_idx is not None else None
        fscore = faces[f_idx][1] if f_idx is not None else 0.0
        result.append(
            Observation(frame, person, face, pscore, fscore)
        )
    for f_idx in orphans:
        face, fscore = faces[f_idx]
        result.append(Observation(frame, None, face, 0.0, fscore))
    return result


def link_affinity(previous: Observation, current: Observation) -> float:
    """How strongly `current` continues `previous`, in [0, 1].

    Same-kind pairs are matched by IoU. A person-only observation and a
    face-only one -- the flicker case where the person detector drops
    out of a close-up -- are matched by containment, penalised so a
    genuine same-kind IoU of the same strength always wins.
    """
    if previous.person is not None and current.person is not None:
        return iou(previous.person, current.person)
    if previous.face is not None and current.face is not None:
        return iou(previous.face, current.face)
    person = (previous.person if previous.person is not None
              else current.person)
    face = previous.face if previous.face is not None else current.face
    if person is None or face is None:
        return 0.0
    return CROSS_KIND_PENALTY * contains_ratio(face, person)


def gap_samples(track_gap_sec: float, fps: float, stride: int) -> int:
    """Convert a hold budget in seconds into a count of samples."""
    if fps <= 0.0:
        raise SubjectError(f"fps must be positive, got {fps}")
    if stride < 1:
        raise SubjectError(f"stride must be >= 1, got {stride}")
    if track_gap_sec < 0.0:
        raise SubjectError(
            f"track_gap_sec must be >= 0, got {track_gap_sec}"
        )
    return int(round(track_gap_sec * fps / stride))


def build_tracks(
    sampled_frames: Sequence[int],
    by_frame: Dict[int, List[Observation]],
    *,
    iou_match_min: float = DEFAULT_IOU_MATCH_MIN,
    max_gap: int = 1,
) -> List[Track]:
    """Greedily link observations across consecutive sampled frames."""
    open_tracks: List[Tuple[int, int, List[Observation]]] = []
    closed: List[Tuple[int, List[Observation]]] = []
    next_id = 0

    for position, frame in enumerate(sampled_frames):
        candidates = by_frame.get(frame, [])
        pairs: List[Tuple[float, int, int]] = []
        for t_idx, (_tid, _pos, observations) in enumerate(open_tracks):
            for c_idx, candidate in enumerate(candidates):
                score = link_affinity(observations[-1], candidate)
                if score >= iou_match_min:
                    pairs.append((score, t_idx, c_idx))
        pairs.sort(key=lambda p: (-p[0], p[1], p[2]))

        used_tracks: set = set()
        used_candidates: set = set()
        for score, t_idx, c_idx in pairs:
            if t_idx in used_tracks or c_idx in used_candidates:
                continue
            used_tracks.add(t_idx)
            used_candidates.add(c_idx)
            tid, _pos, observations = open_tracks[t_idx]
            observations.append(candidates[c_idx])
            open_tracks[t_idx] = (tid, position, observations)

        for c_idx, candidate in enumerate(candidates):
            if c_idx in used_candidates:
                continue
            open_tracks.append((next_id, position, [candidate]))
            next_id += 1

        still_open: List[Tuple[int, int, List[Observation]]] = []
        for tid, last_pos, observations in open_tracks:
            if position - last_pos > max_gap:
                closed.append((tid, observations))
            else:
                still_open.append((tid, last_pos, observations))
        open_tracks = still_open

    for tid, _pos, observations in open_tracks:
        closed.append((tid, observations))
    closed.sort(key=lambda item: item[0])
    return [Track(tid, tuple(obs)) for tid, obs in closed]


def _normalised_confidence(score: float, floor: float) -> float:
    span = 1.0 - floor
    if span <= 0.0:
        return 1.0
    return clamp01((score - floor) / span)


def _confidence_quality(
    obs: Observation, floor_person: float, floor_face: float
) -> float:
    qualities: List[float] = []
    if obs.person is not None:
        qualities.append(_normalised_confidence(obs.person_score,
                                                floor_person))
    if obs.face is not None:
        qualities.append(_normalised_confidence(obs.face_score, floor_face))
    return max(qualities) if qualities else 0.0


def observation_score(
    obs: Observation,
    *,
    face_weight: float = DEFAULT_FACE_WEIGHT,
    centrality_weight: float = DEFAULT_CENTRALITY_WEIGHT,
    confidence_weight: float = DEFAULT_CONFIDENCE_WEIGHT,
    min_confidence_person: float = DEFAULT_MIN_CONFIDENCE_PERSON,
    min_confidence_face: float = DEFAULT_MIN_CONFIDENCE_FACE,
) -> float:
    """Instantaneous dominance of ONE observation, in [0, 1]."""
    if face_weight < 1.0:
        raise SubjectError(
            f"face_weight must be >= 1.0, got {face_weight}")
    face_extent = obs.face.extent if obs.face is not None else 0.0
    body_extent = obs.person.extent if obs.person is not None else 0.0
    size = (face_weight * face_extent + body_extent) / (face_weight + 1.0)
    cx, _cy = obs.point
    centrality_q = 1.0 - clamp01(abs(cx - 0.5) / 0.5)
    confidence_q = _confidence_quality(
        obs, min_confidence_person, min_confidence_face
    )
    return (
        size
        * (1.0 - centrality_weight * (1.0 - centrality_q))
        * (1.0 - confidence_weight * (1.0 - confidence_q))
    )


def track_scores(
    track: Track,
    sample_count: int,
    *,
    face_weight: float = DEFAULT_FACE_WEIGHT,
    centrality_weight: float = DEFAULT_CENTRALITY_WEIGHT,
    confidence_weight: float = DEFAULT_CONFIDENCE_WEIGHT,
    persistence_weight: float = DEFAULT_PERSISTENCE_WEIGHT,
    min_confidence_person: float = DEFAULT_MIN_CONFIDENCE_PERSON,
    min_confidence_face: float = DEFAULT_MIN_CONFIDENCE_FACE,
) -> Dict[int, float]:
    """Score every observation of one track, frame -> score."""
    if sample_count <= 0:
        raise SubjectError("sample_count must be positive")
    persistence_q = clamp01(len(track.observations) / sample_count)
    persistence_term = 1.0 - persistence_weight * (1.0 - persistence_q)
    return {
        obs.frame: observation_score(
            obs,
            face_weight=face_weight,
            centrality_weight=centrality_weight,
            confidence_weight=confidence_weight,
            min_confidence_person=min_confidence_person,
            min_confidence_face=min_confidence_face,
        )
        * persistence_term
        for obs in track.observations
    }


def _alive(track: Track, position: int, sampled: Sequence[int],
           max_gap: int) -> bool:
    low = max(0, position - max_gap)
    high = min(len(sampled) - 1, position + max_gap)
    window = set(sampled[low:high + 1])
    return any(obs.frame in window for obs in track.observations)


def _held_score(track: Track, position: int, sampled: Sequence[int],
                scores: Dict[int, Dict[int, float]]) -> float:
    own = scores[track.track_id]
    for index in range(position, -1, -1):
        frame = sampled[index]
        if frame in own:
            return own[frame]
    for index in range(position, len(sampled)):
        frame = sampled[index]
        if frame in own:
            return own[frame]
    return 0.0


def follow(
    sampled: Sequence[int],
    tracks: Sequence[Track],
    scores: Dict[int, Dict[int, float]],
    fps: float,
    *,
    hysteresis_margin: float = DEFAULT_HYSTERESIS_MARGIN,
    hysteresis_sec: float = DEFAULT_HYSTERESIS_SEC,
    max_gap: int = 1,
) -> Tuple[List[Optional[int]], List[int]]:
    """Walk the samples, returning (followed id per sample, switches).

    The incumbent is adopted freely at the first sample and whenever the
    previous incumbent has gone. Replacing a LIVE incumbent needs the
    challenger to beat it by `hysteresis_margin` continuously for
    `hysteresis_sec`.
    """
    if hysteresis_margin < 0.0 or hysteresis_margin >= 1.0:
        raise SubjectError(
            f"hysteresis_margin must be in [0, 1), got {hysteresis_margin}"
        )
    if hysteresis_sec < 0.0:
        raise SubjectError(
            f"hysteresis_sec must be >= 0, got {hysteresis_sec}"
        )
    by_id = {track.track_id: track for track in tracks}
    incumbent: Optional[int] = None
    last_real: Optional[int] = None
    challenger: Optional[int] = None
    clock = 0.0
    followed: List[Optional[int]] = []
    switches: List[int] = []

    for position, frame in enumerate(sampled):
        present = [t for t in tracks if frame in scores[t.track_id]]
        leader: Optional[int] = None
        if present:
            leader = max(
                present,
                key=lambda t: (scores[t.track_id][frame], -t.track_id),
            ).track_id

        alive = (
            incumbent is not None
            and _alive(by_id[incumbent], position, sampled, max_gap)
        )
        if not alive:
            changed = (
                leader is not None
                and last_real is not None
                and leader != last_real
            )
            if changed:
                switches.append(frame)
            incumbent = leader
            if leader is not None:
                last_real = leader
            challenger = None
            clock = 0.0
        elif leader is not None and leader != incumbent:
            reference = _held_score(
                by_id[incumbent], position, sampled, scores
            )
            beats = (
                scores[leader][frame]
                >= reference * (1.0 + hysteresis_margin)
            )
            if beats:
                if challenger == leader:
                    clock += (frame - sampled[position - 1]) / fps
                else:
                    challenger = leader
                    clock = 0.0
                if clock >= hysteresis_sec:
                    incumbent = leader
                    last_real = leader
                    switches.append(frame)
                    challenger = None
                    clock = 0.0
            else:
                challenger = None
                clock = 0.0
        else:
            challenger = None
            clock = 0.0
        followed.append(incumbent)
    return followed, switches


def _shot_samples(shot: Shot, dets: DetectionSet) -> Tuple[int, ...]:
    return tuple(f for f in dets.sampled_frames if shot.contains(f))


def _validate_inputs(shots: Sequence[Shot], dets: DetectionSet) -> None:
    if not shots:
        raise SubjectError("shots must not be empty")
    frames = dets.sampled_frames
    for earlier, later in zip(frames, frames[1:]):
        if later <= earlier:
            raise SubjectError(
                "DetectionSet.sampled_frames must be strictly increasing, "
                f"saw {earlier} then {later}"
            )
    sampled = set(frames)
    for det in dets.detections:
        if det.frame not in sampled:
            raise SubjectError(
                f"detection at frame {det.frame} is not in sampled_frames"
            )
        if det.cls not in (CLS_PERSON, CLS_FACE):
            raise SubjectError(
                f"unknown detection class {det.cls!r} at frame {det.frame}"
            )
    end = shots[-1].end_frame
    for frame in frames:
        if frame < 0 or frame >= end:
            raise SubjectError(
                f"sampled frame {frame} falls outside the shot list "
                f"[0, {end})"
            )


def _validate_params(
    face_weight: float,
    hysteresis_margin: float,
    hysteresis_sec: float,
    centrality_weight: float,
    confidence_weight: float,
    persistence_weight: float,
    min_confidence_person: float,
    min_confidence_face: float,
    iou_match_min: float,
    track_gap_sec: float,
) -> None:
    """Range-check all ten tunables, unconditionally on the data."""
    if face_weight < 1.0:
        raise SubjectError(
            f"face_weight must be >= 1.0, got {face_weight}")
    unit_open = (
        ("centrality_weight", centrality_weight),
        ("confidence_weight", confidence_weight),
        ("persistence_weight", persistence_weight),
        ("hysteresis_margin", hysteresis_margin),
    )
    for name, value in unit_open:
        if not 0.0 <= value < 1.0:
            raise SubjectError(f"{name} must be in [0, 1), got {value}")
    if not 0.0 < iou_match_min <= 1.0:
        raise SubjectError(
            f"iou_match_min must be in (0, 1], got {iou_match_min}"
        )
    if hysteresis_sec < 0.0:
        raise SubjectError(
            f"hysteresis_sec must be >= 0, got {hysteresis_sec}"
        )
    if track_gap_sec < 0.0:
        raise SubjectError(
            f"track_gap_sec must be >= 0, got {track_gap_sec}"
        )
    open_unit = (
        ("min_confidence_person", min_confidence_person),
        ("min_confidence_face", min_confidence_face),
    )
    for name, value in open_unit:
        if not 0.0 < value < 1.0:
            raise SubjectError(f"{name} must be in (0, 1), got {value}")


def select_subjects(
    shots: Sequence[Shot],
    detections: DetectionSet,
    meta: SourceMeta,
    *,
    face_weight: float = DEFAULT_FACE_WEIGHT,
    hysteresis_margin: float = DEFAULT_HYSTERESIS_MARGIN,
    hysteresis_sec: float = DEFAULT_HYSTERESIS_SEC,
    centrality_weight: float = DEFAULT_CENTRALITY_WEIGHT,
    confidence_weight: float = DEFAULT_CONFIDENCE_WEIGHT,
    persistence_weight: float = DEFAULT_PERSISTENCE_WEIGHT,
    min_confidence_person: float = DEFAULT_MIN_CONFIDENCE_PERSON,
    min_confidence_face: float = DEFAULT_MIN_CONFIDENCE_FACE,
    iou_match_min: float = DEFAULT_IOU_MATCH_MIN,
    track_gap_sec: float = DEFAULT_TRACK_GAP_SEC,
) -> List[ShotSubject]:
    """Pick and follow one subject per shot. One result per input shot."""
    _validate_params(
        face_weight, hysteresis_margin, hysteresis_sec,
        centrality_weight, confidence_weight, persistence_weight,
        min_confidence_person, min_confidence_face,
        iou_match_min, track_gap_sec,
    )
    _validate_inputs(shots, detections)
    width = meta.display_width
    height = meta.display_height
    by_frame_raw: Dict[int, List[Detection]] = {}
    for det in detections.detections:
        by_frame_raw.setdefault(det.frame, []).append(det)

    results: List[ShotSubject] = []
    for shot in shots:
        sampled = _shot_samples(shot, detections)
        if not sampled:
            raise SubjectError(
                f"shot {shot.index} [{shot.start_frame}, "
                f"{shot.end_frame}) has no sampled frame; detect.py "
                "strides per shot and guarantees a "
                "sample at every shot's start_frame, so the shot list "
                "passed to detect_frames is not the one passed here"
            )
        if shot.start_frame not in detections.sampled_frames:
            nearest = min(
                detections.sampled_frames,
                key=lambda f: abs(f - shot.start_frame),
            )
            raise SubjectError(
                f"shot {shot.index} [{shot.start_frame}, "
                f"{shot.end_frame}) has no sample at its start_frame "
                f"{shot.start_frame}; nearest sampled frame is "
                f"{nearest}. detect.py strides per shot "
                "and guarantees a sample at every "
                "shot's start_frame, so the shot list passed to "
                "detect_frames is not the one passed here"
            )
        by_frame = {
            frame: observations_for_frame(
                frame, by_frame_raw.get(frame, []), width, height,
                min_confidence_person=min_confidence_person,
                min_confidence_face=min_confidence_face,
            )
            for frame in sampled
        }
        max_gap = gap_samples(track_gap_sec, shot.fps, detections.stride)
        tracks = build_tracks(
            sampled, by_frame,
            iou_match_min=iou_match_min, max_gap=max_gap,
        )
        scores = {
            track.track_id: track_scores(
                track, len(sampled),
                face_weight=face_weight,
                centrality_weight=centrality_weight,
                confidence_weight=confidence_weight,
                persistence_weight=persistence_weight,
                min_confidence_person=min_confidence_person,
                min_confidence_face=min_confidence_face,
            )
            for track in tracks
        }
        followed, switches = follow(
            sampled, tracks, scores, shot.fps,
            hysteresis_margin=hysteresis_margin,
            hysteresis_sec=hysteresis_sec,
            max_gap=max_gap,
        )
        samples = _build_samples(sampled, tracks, scores, followed)
        results.append(
            ShotSubject(
                shot_index=shot.index,
                start_frame=shot.start_frame,
                end_frame=shot.end_frame,
                fps=shot.fps,
                stride=detections.stride,
                sampled_frames=sampled,
                samples=samples,
                has_subject=any(s.observed for s in samples),
                track_count=len(tracks),
                switch_frames=tuple(switches),
            )
        )
    return results


def _build_samples(
    sampled: Sequence[int],
    tracks: Sequence[Track],
    scores: Dict[int, Dict[int, float]],
    followed: Sequence[Optional[int]],
) -> Tuple[SubjectSample, ...]:
    by_id = {track.track_id: track for track in tracks}
    raw: List[Optional[SubjectSample]] = []
    for position, frame in enumerate(sampled):
        track_id = followed[position]
        obs = by_id[track_id].at(frame) if track_id is not None else None
        if obs is None:
            raw.append(None)
            continue
        cx, cy = obs.point
        box = obs.anchor
        raw.append(
            SubjectSample(
                frame=frame, cx=cx, cy=cy,
                x1=box.x1, y1=box.y1, x2=box.x2, y2=box.y2,
                observed=True, kind=obs.kind,
                score=scores[track_id][frame], track_id=track_id,
            )
        )

    observed_positions = [i for i, s in enumerate(raw) if s is not None]
    out: List[SubjectSample] = []
    for position, frame in enumerate(sampled):
        sample = raw[position]
        if sample is not None:
            out.append(sample)
            continue
        source = _nearest_observed(position, observed_positions)
        if source is None:
            out.append(
                SubjectSample(
                    frame=frame, cx=0.5, cy=0.5,
                    x1=0.5, y1=0.5, x2=0.5, y2=0.5,
                    observed=False, kind=KIND_CENTER,
                    score=0.0, track_id=NO_TRACK,
                )
            )
            continue
        anchor = raw[source]
        out.append(
            SubjectSample(
                frame=frame, cx=anchor.cx, cy=anchor.cy,
                x1=anchor.cx, y1=anchor.cy, x2=anchor.cx, y2=anchor.cy,
                observed=False, kind=KIND_HELD,
                score=0.0, track_id=anchor.track_id,
            )
        )
    return tuple(out)


def _nearest_observed(
    position: int, observed_positions: Sequence[int]
) -> Optional[int]:
    """Hold the last observed sample; back-fill the first if none before."""
    earlier = [p for p in observed_positions if p < position]
    if earlier:
        return earlier[-1]
    later = [p for p in observed_positions if p > position]
    if later:
        return later[0]
    return None


def subject_summary(results: Sequence[ShotSubject]) -> str:
    """One progress line for run_stages to print. No I/O here."""
    empty = sum(1 for r in results if not r.has_subject)
    switches = sum(len(r.switch_frames) for r in results)
    samples = sum(len(r.samples) for r in results)
    observed = sum(
        1 for r in results for s in r.samples if s.observed
    )
    share = (observed / samples) if samples else 0.0
    return (
        f"subject: {len(results)} shot(s), {empty} with no subject, "
        f"{switches} switch(es), {share:.1%} of samples observed"
    )


def subject_to_json(results: Sequence[ShotSubject]) -> List[dict]:
    """Serialise for debugging and tests, six decimal places.

    Not written to crops.json: render.crops_document carries the camera
    track, which already holds each frame's followed subject centre.
    """
    return [
        {
            "shot_index": result.shot_index,
            "start_frame": result.start_frame,
            "end_frame": result.end_frame,
            "stride": result.stride,
            "has_subject": result.has_subject,
            "track_count": result.track_count,
            "observed_fraction": result.observed_fraction,
            "switch_frames": list(result.switch_frames),
            "samples": [
                {
                    "frame": s.frame,
                    "cx": round(s.cx, 6),
                    "cy": round(s.cy, 6),
                    "box": [
                        round(s.x1, 6), round(s.y1, 6),
                        round(s.x2, 6), round(s.y2, 6),
                    ],
                    "observed": s.observed,
                    "kind": s.kind,
                    "score": round(s.score, 6),
                    "track_id": s.track_id,
                }
                for s in result.samples
            ],
        }
        for result in results
    ]
