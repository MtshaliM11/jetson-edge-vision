"""Original, dependency-free class-aware greedy IoU tracking (not SORT/ByteTrack)."""
from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class TrackingConfig:
    iou_threshold: float = 0.3
    max_missed: int = 5
    min_hits: int = 2

    def __post_init__(self):
        if (isinstance(self.iou_threshold, bool)
                or not isinstance(self.iou_threshold, (int, float))
                or not math.isfinite(self.iou_threshold) or not 0 < self.iou_threshold <= 1):
            raise ValueError("track-iou must be a finite number in (0, 1]")
        if type(self.max_missed) is not int or self.max_missed < 0:
            raise ValueError("track-max-missed must be an integer >= 0")
        if type(self.min_hits) is not int or self.min_hits < 1:
            raise ValueError("track-min-hits must be an integer >= 1")


@dataclass(frozen=True)
class Track:
    track_id: int
    class_id: int
    class_name: str
    box_xyxy: tuple[float, float, float, float]
    confidence: float
    first_frame: int
    last_frame: int
    hits: int = 1
    consecutive_hits: int = 1
    missed: int = 0
    confirmed: bool = False


def box_iou(first, second) -> float:
    """Continuous xyxy coordinates, without an inclusive-pixel '+1'."""
    intersection = (max(0., min(first[2], second[2]) - max(first[0], second[0]))
                    * max(0., min(first[3], second[3]) - max(first[1], second[1])))
    area_first = max(0., first[2] - first[0]) * max(0., first[3] - first[1])
    area_second = max(0., second[2] - second[0]) * max(0., second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.


def _detection_record(record: dict) -> dict:
    """Copy and validate before changing any tracker state."""
    try:
        class_id, name = record["class_id"], record["class_name"]
        box, score = record["box_xyxy"], record["confidence"]
        if type(class_id) is not int or class_id <= 0 or not isinstance(name, str) or not name:
            raise ValueError("invalid category")
        if len(box) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
            raise ValueError("invalid box")
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("box must have positive area")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("invalid confidence")
        return {**record, "box_xyxy": tuple(float(v) for v in box), "confidence": float(score)}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid tracking detection: {exc}") from exc


class IoUTracker:
    """One instance per video; retains last observed boxes, never predicted boxes.

    Candidates sort by descending IoU, then ascending track ID and input index.
    Greedy matches consume each track/detection at most once. This is not global
    optimal assignment and does not identify people or infer biometric identity.
    """

    def __init__(self, config: TrackingConfig | None = None):
        if config is not None and not isinstance(config, TrackingConfig):
            raise ValueError("config must be a TrackingConfig")
        self.config = config if config is not None else TrackingConfig()
        self.reset()

    def reset(self) -> None:
        self._tracks: dict[int, Track] = {}
        self._next_id = 1
        self.frame_index = -1
        self.tracks_created = 0
        self.tracks_expired = 0
        self.matched_detections = 0

    @property
    def tracks(self) -> tuple[Track, ...]:
        """Immutable snapshot, including retained but currently missed tracks."""
        return tuple(self._tracks[key] for key in sorted(self._tracks))

    def update(self, detections: list[dict]) -> list[dict]:
        records = [_detection_record(record) for record in detections]
        self.frame_index += 1
        candidates = []
        for track_id, track in self._tracks.items():
            for index, record in enumerate(records):
                if (track.class_id, track.class_name) != (record["class_id"], record["class_name"]):
                    continue
                overlap = box_iou(track.box_xyxy, record["box_xyxy"])
                if overlap >= self.config.iou_threshold:
                    candidates.append((-overlap, track_id, index))
        used_tracks, assignments = set(), {}
        for _, track_id, index in sorted(candidates):
            if track_id not in used_tracks and index not in assignments:
                assignments[index] = track_id
                used_tracks.add(track_id)

        updated = {}
        for index, track_id in assignments.items():
            track, record = self._tracks[track_id], records[index]
            streak = track.consecutive_hits + 1 if track.missed == 0 else 1
            updated[track_id] = replace(
                track, box_xyxy=record["box_xyxy"], confidence=record["confidence"],
                last_frame=self.frame_index, hits=track.hits + 1, consecutive_hits=streak,
                missed=0, confirmed=track.confirmed or streak >= self.config.min_hits,
            )
        self.matched_detections += len(assignments)
        for track_id, track in self._tracks.items():
            if track_id in used_tracks:
                continue
            missed = track.missed + 1
            if missed > self.config.max_missed:
                self.tracks_expired += 1
            else:
                updated[track_id] = replace(track, missed=missed, consecutive_hits=0)

        for index, record in enumerate(records):
            if index in assignments:
                continue
            track_id = self._next_id
            self._next_id += 1
            self.tracks_created += 1
            updated[track_id] = Track(
                track_id, record["class_id"], record["class_name"], record["box_xyxy"],
                record["confidence"], self.frame_index, self.frame_index,
                confirmed=self.config.min_hits == 1,
            )
            assignments[index] = track_id
        self._tracks = updated
        # Keep detection order and return only current observations, not stale boxes.
        return [{**record, "track_id": assignments[index],
                 "track_confirmed": updated[assignments[index]].confirmed}
                for index, record in enumerate(records)]
