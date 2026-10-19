"""Dependency-free, session-local counting over existing immutable Track snapshots."""
from collections import Counter
from dataclasses import asdict, dataclass
import math

from .tracking import Track


LIMITATIONS = [
    "IoU ID switches, fragmentation, missed detections and class errors can cause count errors.",
    "Only confirmed consecutive observations count; gaps break motion continuity.",
    "Centers and straight segments between observations approximate motion; no physical identity or accuracy guarantee.",
    "Video timestamps are frame-index/source-FPS estimates, not wall-clock or variable-frame-rate timestamps.",
    "Events and scalar timing samples grow with session duration; frames/GPU tensors are not retained.",
]


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True)
class CountingConfig:
    line: tuple[float, float, float, float] = (0.5, 0.1, 0.5, 0.9)
    hysteresis_px: float = 2.0
    line_id: str = "line-1"

    def __post_init__(self):
        if (not isinstance(self.line, (tuple, list)) or len(self.line) != 4
                or any(not _finite(v) or not 0 <= v <= 1 for v in self.line)):
            raise ValueError("count-line requires four finite normalized coordinates in [0, 1]")
        if tuple(self.line[:2]) == tuple(self.line[2:]):
            raise ValueError("count-line endpoints must define a nonzero-length line")
        if not _finite(self.hysteresis_px) or self.hysteresis_px < 0:
            raise ValueError("count-hysteresis must be a finite pixel distance >= 0")
        if not isinstance(self.line_id, str) or not self.line_id.strip():
            raise ValueError("Counting line identifier must be nonempty")
        object.__setattr__(self, "line", tuple(float(v) for v in self.line))

    def pixels(self, size):
        if (not isinstance(size, (tuple, list)) or len(size) != 2
                or any(type(v) is not int or v < 2 for v in size)):
            raise ValueError("Frame dimensions must be integer width/height >= 2")
        w, h = size
        return tuple(v * (w - 1 if i % 2 == 0 else h - 1) for i, v in enumerate(self.line))


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _subtract(a, b):
    return a[0] - b[0], a[1] - b[1]


def _intersects(p, q, a, b):
    """Closed finite segments; include endpoint touches, exclude collinear motion."""
    r, s = _subtract(q, p), _subtract(b, a)
    denominator = _cross(r, s)
    if denominator == 0:
        return False
    t = _cross(_subtract(a, p), s) / denominator
    u = _cross(_subtract(a, p), r) / denominator
    return 0 <= t <= 1 and 0 <= u <= 1


@dataclass(frozen=True)
class CrossingEvent:
    event_id: str
    track_id: int
    class_id: int
    class_name: str
    frame_index: int
    direction: str
    line_id: str
    position: tuple[float, float]
    video_timestamp_seconds: float | None


@dataclass
class _State:
    identity: tuple
    point: tuple[float, float]
    side: int
    touched: bool = False


class LineCounter:
    """A=negative, B=positive signed cross(line vector, center-start).

    Image y increases downwards: a top-to-bottom vertical line has A on its
    right and B on its left. Reversing endpoints reverses directions. A crossing
    is emitted at the observation reaching the opposite stable side, after an
    actual consecutive center-motion segment intersects the finite line.
    """

    def __init__(self, size: tuple[int, int], config: CountingConfig | None = None):
        if config is not None and not isinstance(config, CountingConfig):
            raise ValueError("config must be a CountingConfig")
        self.config = config or CountingConfig()
        self.pixel_line = self.config.pixels(size)
        self._a, self._b = self.pixel_line[:2], self.pixel_line[2:]
        self._vector = _subtract(self._b, self._a)
        self._length = math.hypot(*self._vector)
        self.reset()

    def reset(self):
        """Reset alongside the tracker when starting/restarting a video session."""
        self._states = {}
        self._events = []
        self._directions = Counter({"A->B": 0, "B->A": 0})
        self._categories = {}
        self.frame_index = -1

    @property
    def events(self):
        return tuple(self._events)

    def update(self, tracks, frame_index: int, video_timestamp_seconds=None):
        """Full active snapshot, including missed tracks, once per increasing frame.

        Missing/unconfirmed/stale observations clear their motion history. IDs
        absent from the snapshot expire immediately. Invalid inputs fail before
        mutation; duplicate/out-of-order frames raise rather than count twice.
        """
        if type(frame_index) is not int or frame_index < 0 or frame_index <= self.frame_index:
            raise ValueError("Counting frame index must be nonnegative and strictly increasing")
        if video_timestamp_seconds is not None and (not _finite(video_timestamp_seconds) or video_timestamp_seconds < 0):
            raise ValueError("Video timestamp must be finite seconds >= 0 or None")
        tracks = tuple(tracks)
        ids = set()
        for t in tracks:
            if (not isinstance(t, Track) or type(t.track_id) is not int or t.track_id < 1
                    or t.track_id in ids or type(t.class_id) is not int or t.class_id < 1
                    or not isinstance(t.class_name, str) or not t.class_name
                    or len(t.box_xyxy) != 4 or any(not _finite(v) for v in t.box_xyxy)
                    or t.box_xyxy[2] <= t.box_xyxy[0] or t.box_xyxy[3] <= t.box_xyxy[1]):
                raise ValueError("Invalid or duplicate counting track")
            ids.add(t.track_id)
        # A skipped counter update is also a gap in observed motion.
        if frame_index != self.frame_index + 1:
            self._states.clear()
        self.frame_index = frame_index
        self._states = {key: state for key, state in self._states.items() if key in ids}
        emitted = []
        for t in sorted(tracks, key=lambda item: item.track_id):
            if not t.confirmed or t.missed != 0 or t.last_frame != frame_index:
                self._states.pop(t.track_id, None)
                continue
            point = ((t.box_xyxy[0] + t.box_xyxy[2]) / 2,
                     (t.box_xyxy[1] + t.box_xyxy[3]) / 2)
            distance = _cross(self._vector, _subtract(point, self._a)) / self._length
            side = 1 if distance > self.config.hysteresis_px else -1 if distance < -self.config.hysteresis_px else 0
            identity = (t.class_id, t.class_name, t.first_frame)
            state = self._states.get(t.track_id)
            if state is None or state.identity != identity:
                self._states[t.track_id] = _State(identity, point, side)
                continue
            state.touched |= _intersects(state.point, point, self._a, self._b)
            if side:
                if state.side and side != state.side and state.touched:
                    direction = "A->B" if state.side == -1 else "B->A"
                    event = CrossingEvent(f"{self.config.line_id}:{len(self._events) + 1:06d}",
                                          t.track_id, t.class_id, t.class_name, frame_index,
                                          direction, self.config.line_id, point, video_timestamp_seconds)
                    self._events.append(event)
                    emitted.append(event)
                    self._directions[direction] += 1
                    category = self._categories.setdefault(t.class_id, {
                        "class_id": t.class_id, "class_name": t.class_name,
                        "total_crossings": 0, "by_direction": {"A->B": 0, "B->A": 0}})
                    category["total_crossings"] += 1
                    category["by_direction"][direction] += 1
                state.side = side
                state.touched = False
            state.point = point
        return tuple(emitted)

    def totals(self):
        return {"total_crossings": len(self._events), "by_direction": dict(self._directions),
                "by_category": {str(k): {**v, "by_direction": dict(v["by_direction"])}
                                for k, v in sorted(self._categories.items())}}

    def report(self):
        return {"enabled": True, "line": {"id": self.config.line_id,
                "normalized_xyxy": self.config.line, "pixel_xyxy": self.pixel_line},
                "hysteresis_px": self.config.hysteresis_px, **self.totals(),
                "events": [asdict(e) for e in self._events], "known_limitations": list(LIMITATIONS)}
