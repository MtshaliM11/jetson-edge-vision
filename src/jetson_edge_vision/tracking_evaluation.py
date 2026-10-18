"""Reproducible CPU tracking diagnostics, separate from production inference.

These explicitly defined diagnostic counts are not MOTChallenge/IDF1/HOTA metrics.
Unlabelled traces never produce identity-accuracy metrics.
"""
import argparse
from dataclasses import asdict
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
import statistics
import sys
import time

from .evaluation_scenarios import synthetic_scenarios
from .tracking import IoUTracker, TrackingConfig


DETECTION_FIELDS = ("class_id", "class_name", "confidence", "box_xyxy")
IDENTITY_METRICS = ("id_switches", "identity_links", "id_continuations", "persistence_ratio",
                    "extra_identity_tracks", "identity_merges", "missed_object_frames",
                    "category_errors", "false_positive_detections", "false_tracks_created",
                    "false_tracks_confirmed", "false_positive_matches")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_grid() -> list[TrackingConfig]:
    return [TrackingConfig(iou, missed, hits)
            for iou, missed, hits in product((0.1, 0.3, 0.5), (0, 5, 10), (1, 2, 3))]


def validate_frames(frames: list[dict], labelled: bool) -> list[dict]:
    if not isinstance(frames, list) or not frames:
        raise ValueError("Evaluation requires a nonempty frame list")
    normalized = []
    for index, row in enumerate(frames):
        try:
            if type(row["frame_index"]) is not int or row["frame_index"] != index:
                raise ValueError("frame indices must be contiguous, zero-based, and ordered")
            if type(row.get("reset_before", False)) is not bool:
                raise ValueError("reset_before must be a boolean")
            if not isinstance(row["detections"], list):
                raise ValueError("detections must be a list")
            # Ground truth and saved tracker IDs must never enter the tracker.
            detections = [{key: detection[key] for key in DETECTION_FIELDS} for detection in row["detections"]]
            result = {"frame_index": index, "detections": detections,
                      "reset_before": row.get("reset_before", False)}
            if labelled:
                truth_ids, visible = row["truth_ids"], row["visible_objects"]
                if not isinstance(truth_ids, list) or len(truth_ids) != len(detections) or not isinstance(visible, dict):
                    raise ValueError("truth_ids must align with detections; visible_objects must be an object")
                for identity, category in visible.items():
                    if (not isinstance(identity, str) or not identity or type(category["class_id"]) is not int
                            or category["class_id"] <= 0 or not isinstance(category["class_name"], str)
                            or not category["class_name"]):
                        raise ValueError("Visible objects require a nonempty ID and valid true category")
                linked = [identity for identity in truth_ids if identity is not None]
                if any(not isinstance(identity, str) or identity not in visible for identity in linked):
                    raise ValueError("Every non-null truth ID must identify a visible object")
                if len(set(linked)) != len(linked):
                    raise ValueError("Each visible object may label at most one detection per frame; mark duplicate detections null")
                result.update(truth_ids=truth_ids[:], visible_objects=visible.copy())
            normalized.append(result)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid evaluation frame {index}: {exc}") from exc
    return normalized


def evaluate_sequence(frames: list[dict], config: TrackingConfig, *, labelled: bool,
                      repeats: int = 10) -> dict:
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be an integer >= 1")
    frames = validate_frames(frames, labelled)
    samples = []
    started = time.perf_counter_ns()
    first_metrics = first_observations = None
    for repeat in range(repeats):
        tracker = IoUTracker(config)
        metrics = {key: 0 for key in IDENTITY_METRICS if key != "persistence_ratio"}
        metrics.update(tracks_created=0, tracks_expired=0, active_tracks=0,
                       peak_active_tracks=0, resets=0, detection_observations=0,
                       confirmed_observations=0)
        last_id, identity_tracks, track_identity = {}, {}, {}
        known_tracks, false_born, false_confirmed = set(), set(), set()
        observations = {}
        segment = 0
        expired_before = 0
        for row in frames:
            if row["reset_before"]:
                tracker.reset()
                expired_before = 0
                segment += 1
                metrics["resets"] += 1
            update_started = time.perf_counter_ns()
            tracked = tracker.update(row["detections"])
            samples.append((time.perf_counter_ns() - update_started) / 1_000_000)
            metrics["tracks_expired"] += tracker.tracks_expired - expired_before
            expired_before = tracker.tracks_expired
            metrics["active_tracks"] = len(tracker.tracks)
            metrics["peak_active_tracks"] = max(metrics["peak_active_tracks"], metrics["active_tracks"])
            metrics["detection_observations"] += len(tracked)
            if labelled:
                metrics["missed_object_frames"] += len(set(row["visible_objects"]) - set(row["truth_ids"]))
            for index, record in enumerate(tracked):
                track = (segment, record["track_id"])
                new_track = track not in known_tracks
                known_tracks.add(track)
                metrics["tracks_created"] += int(new_track)
                metrics["confirmed_observations"] += int(record["track_confirmed"])
                summary = observations.setdefault(track, {"segment": segment, "track_id": record["track_id"],
                    "class_name": record["class_name"], "frames": [], "confirmed_frames": []})
                summary["frames"].append(row["frame_index"])
                if record["track_confirmed"]:
                    summary["confirmed_frames"].append(row["frame_index"])
                if not labelled:
                    continue
                truth_id = row["truth_ids"][index]
                if truth_id is None:
                    metrics["false_positive_detections"] += 1
                    if new_track:
                        false_born.add(track)
                    else:
                        metrics["false_positive_matches"] += 1
                else:
                    identity = (segment, truth_id)
                    actual = row["visible_objects"][truth_id]
                    metrics["category_errors"] += int((record["class_id"], record["class_name"])
                                                       != (actual["class_id"], actual["class_name"]))
                    if identity in last_id:
                        metrics["identity_links"] += 1
                        same_id = last_id[identity] == track
                        metrics["id_continuations"] += int(same_id)
                        metrics["id_switches"] += int(not same_id)
                    last_id[identity] = track
                    identity_tracks.setdefault(identity, set()).add(track)
                    if track in track_identity and track_identity[track] != identity:
                        metrics["identity_merges"] += 1
                    track_identity[track] = identity
                if track in false_born and record["track_confirmed"]:
                    false_confirmed.add(track)
        metrics["false_tracks_created"] = len(false_born)
        metrics["false_tracks_confirmed"] = len(false_confirmed)
        metrics["extra_identity_tracks"] = sum(len(ids) - 1 for ids in identity_tracks.values())
        links = metrics["identity_links"]
        metrics["persistence_ratio"] = metrics["id_continuations"] / links if links else None
        if not labelled:
            metrics.update({key: None for key in IDENTITY_METRICS})
        if repeat == 0:
            first_metrics = metrics
            first_observations = list(observations.values())
        elif metrics != first_metrics or list(observations.values()) != first_observations:
            raise RuntimeError("Tracker produced nondeterministic assignments/counts across repeats")
    wall_ms = (time.perf_counter_ns() - started) / 1_000_000
    total_ms = sum(samples)
    return {"labelled": labelled, "frame_count": len(frames), "repeats": repeats,
            "metrics": first_metrics, "observations": first_observations,
            "timing": {"calls": len(samples), "update_mean_ms": statistics.mean(samples),
                       "update_median_ms": statistics.median(samples), "update_max_ms": max(samples),
                       "update_total_ms": total_ms, "update_only_fps": len(samples) * 1000 / total_ms if total_ms > 0 else None,
                       "evaluation_wall_ms": wall_ms,
                       "replay_processing_fps": len(samples) * 1000 / wall_ms if wall_ms > 0 else None}}


def read_trace(path: Path) -> list[dict]:
    frames = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
            if "message" in row:
                message = row["message"]
                if not isinstance(message, str):
                    raise ValueError("log message must be text")
                if not message.startswith("Tracking frame: "):
                    continue
                row = json.loads(message.removeprefix("Tracking frame: "))
            frames.append(row)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid trace JSON at line {number}: {exc}") from exc
    return validate_frames(frames, labelled=False)


def apply_annotations(frames: list[dict], path: Path, trace_path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
            or data["schema_version"] != 1 or data.get("reviewed") is not True
            or data.get("trace_sha256") != sha256_file(trace_path)):
        raise ValueError("Annotations require schema_version=1, reviewed=true, and the exact source trace_sha256")
    annotations = data.get("frames")
    if not isinstance(annotations, list) or len(annotations) != len(frames):
        raise ValueError("Annotations must cover every trace frame, including empty frames")
    combined = []
    for frame, annotation in zip(frames, annotations):
        if not isinstance(annotation, dict) or annotation.get("frame_index") != frame["frame_index"]:
            raise ValueError("Annotation frame indices must align exactly with the trace")
        combined.append({**annotation, "detections": frame["detections"]})
    return validate_frames(combined, labelled=True)


def load_trace_source(trace_path: Path, annotations_path: Path | None, video_report_path: Path | None):
    frames = read_trace(trace_path)
    canonical = json.dumps(frames, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    detections_sha256 = hashlib.sha256(canonical).hexdigest()
    labelled = annotations_path is not None
    if labelled:
        frames = apply_annotations(frames, annotations_path, trace_path)
    source = {"trace_sha256": sha256_file(trace_path),
              "detections_sha256": detections_sha256,
              "annotations_sha256": sha256_file(annotations_path) if labelled else None,
              "video_report": None}
    if video_report_path is not None:
        evidence = json.loads(video_report_path.read_text(encoding="utf-8"))
        if (not isinstance(evidence, dict) or evidence.get("frames_processed") != len(frames)
                or evidence.get("total_detections") != sum(len(frame["detections"]) for frame in frames)):
            raise ValueError("Video report frame/detection counts do not match the source trace")
        source["video_report"] = {"sha256": sha256_file(video_report_path), "report": evidence}
    return frames, labelled, source


def build_report(*, repeats: int = 10, baseline_only: bool = False, trace_path: Path | None = None,
                 annotations_path: Path | None = None, trace_only: bool = False,
                 video_report_path: Path | None = None) -> dict:
    if trace_path is None and (annotations_path is not None or trace_only or video_report_path is not None):
        raise ValueError("Annotations, video-report, and trace-only options require --trace")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be an integer >= 1")
    trace_source = load_trace_source(trace_path, annotations_path, video_report_path) if trace_path is not None else None
    configs = [TrackingConfig()] if baseline_only else parameter_grid()
    report = {"schema_version": 1, "algorithm": "unchanged_class_aware_greedy_iou",
              "tracker_sha256": sha256_file(Path(__file__).with_name("tracking.py")),
              "evaluator_sha256": sha256_file(Path(__file__)),
              "scenario_source_sha256": sha256_file(Path(__file__).with_name("evaluation_scenarios.py")),
              "runtime": {"python": platform.python_version(), "architecture": platform.machine()},
              "metric_scope": "Custom diagnostic counts, not MOTChallenge/IDF1/HOTA; replay FPS is not video FPS",
              "synthetic": [], "trace": None}
    if not trace_only:
        scenarios = synthetic_scenarios()
        for config in configs:
            cases = []
            for case in scenarios:
                evaluated = evaluate_sequence(case["frames"], config, labelled=True, repeats=repeats)
                if config == TrackingConfig():
                    for key, expected in case["expected_default"].items():
                        if evaluated["metrics"][key] != expected:
                            raise RuntimeError(f"Default scenario {case['name']} violated expected {key}={expected}")
                cases.append({"name": case["name"], "description": case["description"],
                              "expected_default": case["expected_default"], **evaluated})
            report["synthetic"].append({"config": asdict(config), "scenarios": cases})
    if trace_path is not None:
        frames, labelled, source = trace_source
        report["trace"] = {"source": source, "labelled": labelled,
                           "configurations": [{"config": asdict(config), **evaluate_sequence(
                               frames, config, labelled=labelled, repeats=repeats)} for config in configs]}
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CPU evaluation of the unchanged IoU tracker; no downloads/inference")
    parser.add_argument("--output", type=Path, help="New JSON report path; stdout if omitted")
    parser.add_argument("--trace", type=Path, help="Existing contiguous DEBUG tracking JSONL or frame JSONL")
    parser.add_argument("--annotations", type=Path, help="Optional reviewed labels bound to the exact trace SHA256")
    parser.add_argument("--video-report", type=Path, help="Existing pipeline JSON report; throughput is provenance, not replay timing")
    parser.add_argument("--trace-only", action="store_true", help="Skip synthetic suite; requires --trace")
    parser.add_argument("--baseline-only", action="store_true", help="Evaluate only unchanged default parameters")
    parser.add_argument("--repeats", type=int, default=10, help="Fresh-tracker replays per scenario/config (default: 10)")
    args = parser.parse_args(argv)
    try:
        if args.output is not None and (args.output.exists() or args.output.is_symlink()):
            raise ValueError(f"Output already exists: {args.output}")
        report = build_report(repeats=args.repeats, baseline_only=args.baseline_only, trace_path=args.trace,
                              annotations_path=args.annotations, trace_only=args.trace_only,
                              video_report_path=args.video_report)
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            print(encoded, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
        return 0
    except (ValueError, OSError, TypeError, RuntimeError) as exc:
        print(f"Tracking evaluation failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
