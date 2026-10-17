"""Deterministic synthetic tracker tests and optional actual-CUDA integration."""
import contextlib
import copy
from dataclasses import FrozenInstanceError
import io
import json
import logging
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jetson_edge_vision.cli import main
from jetson_edge_vision.tracking import IoUTracker, TrackingConfig, box_iou
from jetson_edge_vision import video_pipeline as video
from jetson_edge_vision.image_pipeline import annotate_image


def detection(x=10, y=30, *, category=1, name="person", score=0.9):
    return {"class_id": category, "class_name": name, "confidence": score,
            "box_xyxy": [x, y, x + 30, y + 30]}


class TrackerTests(unittest.TestCase):
    def test_iou_geometry(self):
        self.assertEqual(box_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.)
        self.assertAlmostEqual(box_iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)
        self.assertEqual(box_iou((0, 0, 10, 10), (10, 0, 20, 10)), 0.)
        self.assertEqual(box_iou((0, 0, 0, 0), (0, 0, 0, 0)), 0.)

    def test_configuration_validation(self):
        for kwargs in ({"iou_threshold": 0}, {"iou_threshold": 1.1}, {"iou_threshold": float("nan")},
                       {"iou_threshold": float("inf")}, {"iou_threshold": True}, {"iou_threshold": "0.3"},
                       {"max_missed": -1}, {"max_missed": 1.5}, {"max_missed": True},
                       {"min_hits": 0}, {"min_hits": 1.5}, {"min_hits": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrackingConfig(**kwargs)
        self.assertEqual(TrackingConfig(iou_threshold=1, max_missed=0, min_hits=1).min_hits, 1)
        with self.assertRaises(ValueError):
            IoUTracker("bad")

    def test_track_creation_does_not_mutate_input(self):
        records = [detection(), detection(70)]
        original = copy.deepcopy(records)
        tracker = IoUTracker()
        result = tracker.update(records)
        self.assertEqual([r["track_id"] for r in result], [1, 2])
        self.assertFalse(any(r["track_confirmed"] for r in result))
        self.assertEqual(records, original)
        result[0]["box_xyxy"] = (999, 999, 1000, 1000)
        self.assertEqual(tracker.tracks[0].box_xyxy, (10., 30., 40., 60.))
        with self.assertRaises(FrozenInstanceError):
            tracker.tracks[0].missed = 100

    def test_id_persistence_and_box_category_score_update(self):
        tracker = IoUTracker()
        tracker.update([detection()])
        result = tracker.update([detection(12, score=0.8)])
        self.assertEqual(result[0]["track_id"], 1)
        self.assertTrue(result[0]["track_confirmed"])
        track = tracker.tracks[0]
        self.assertEqual((track.first_frame, track.last_frame, track.hits), (0, 1, 2))
        self.assertEqual(track.box_xyxy, (12., 30., 42., 60.))
        self.assertEqual((track.class_name, track.confidence), ("person", 0.8))

    def test_two_objects_moving_with_detection_order_changes(self):
        tracker = IoUTracker()
        for frame in range(7):
            records = [detection(10 + frame * 2), detection(80 - frame * 2)]
            if frame % 2:
                records.reverse()
            tracked = tracker.update(records)
            self.assertEqual([r["track_id"] for r in tracked], [2, 1] if frame % 2 else [1, 2])
            self.assertEqual(len({r["track_id"] for r in tracked}), 2)
        self.assertEqual(tracker.tracks_created, 2)
        self.assertEqual(tracker.matched_detections, 12)

    def test_missed_detections_retained_without_stale_output(self):
        tracker = IoUTracker(TrackingConfig(max_missed=2))
        tracker.update([detection()])
        self.assertEqual(tracker.update([]), [])
        self.assertEqual(tracker.update([]), [])
        self.assertEqual(tracker.tracks[0].missed, 2)
        result = tracker.update([detection(11)])
        self.assertEqual(result[0]["track_id"], 1)
        self.assertEqual(tracker.tracks[0].missed, 0)

    def test_expiration_boundary_and_ids_not_reused(self):
        tracker = IoUTracker(TrackingConfig(max_missed=1))
        tracker.update([detection()])
        tracker.update([])
        self.assertEqual(len(tracker.tracks), 1)
        tracker.update([])
        self.assertEqual(tracker.tracks, ())
        self.assertEqual(tracker.tracks_expired, 1)
        self.assertEqual(tracker.update([detection()])[0]["track_id"], 2)

    def test_zero_missed_frames(self):
        tracker = IoUTracker(TrackingConfig(max_missed=0, min_hits=1))
        self.assertTrue(tracker.update([detection()])[0]["track_confirmed"])
        tracker.update([])
        self.assertEqual(tracker.tracks, ())

    def test_confirmation_requires_consecutive_hits(self):
        tracker = IoUTracker(TrackingConfig(min_hits=3))
        tracker.update([detection()])
        tracker.update([detection()])
        tracker.update([])
        self.assertFalse(tracker.update([detection()])[0]["track_confirmed"])
        self.assertFalse(tracker.update([detection()])[0]["track_confirmed"])
        self.assertTrue(tracker.update([detection()])[0]["track_confirmed"])
        tracker.update([])
        self.assertTrue(tracker.update([detection()])[0]["track_confirmed"])

    def test_different_categories_never_associate(self):
        tracker = IoUTracker()
        tracker.update([detection()])
        result = tracker.update([detection(category=3, name="car")])
        self.assertEqual(result[0]["track_id"], 2)
        self.assertEqual(len(tracker.tracks), 2)

    def test_overlapping_objects_have_one_to_one_deterministic_matches(self):
        for _ in range(3):
            tracker = IoUTracker()
            tracker.update([detection(), detection()])
            result = tracker.update([detection(), detection()])
            self.assertEqual([r["track_id"] for r in result], [1, 2])
            result = tracker.update([detection()])
            self.assertEqual(result[0]["track_id"], 1)
            self.assertEqual(tracker.tracks[1].missed, 1)

    def test_highest_iou_wins_before_track_id(self):
        tracker = IoUTracker()
        tracker.update([detection(10), detection(15)])
        result = tracker.update([detection(15)])
        self.assertEqual(result[0]["track_id"], 2)

    def test_association_threshold_is_inclusive(self):
        tracker = IoUTracker(TrackingConfig(iou_threshold=1))
        tracker.update([detection()])
        self.assertEqual(tracker.update([detection()])[0]["track_id"], 1)
        self.assertEqual(tracker.update([detection(11)])[0]["track_id"], 2)

    def test_no_overlap_creates_new_track(self):
        tracker = IoUTracker()
        tracker.update([detection()])
        self.assertEqual(tracker.update([detection(100)])[0]["track_id"], 2)

    def test_crossing_ambiguity_is_an_explicit_baseline_limit(self):
        tracker = IoUTracker()
        for left, right in ((10, 70), (20, 60), (30, 50), (40, 40)):
            tracker.update([detection(left), detection(right)])
        # Physical A now moves right to 50, B left to 30. Detection order changes.
        # Identical previous boxes contain no appearance/motion evidence to resolve this.
        result = tracker.update([detection(30), detection(50)])
        self.assertEqual([r["track_id"] for r in result], [1, 2])
        self.assertEqual(result[1]["track_id"], 2)  # A's original ID was 1: documented switch.

    def test_reset_and_independent_sessions(self):
        tracker = IoUTracker()
        tracker.update([detection(), detection(70)])
        other = IoUTracker()
        self.assertEqual(other.update([detection()])[0]["track_id"], 1)
        self.assertEqual(len(tracker.tracks), 2)
        tracker.reset()
        self.assertEqual(tracker.tracks, ())
        self.assertEqual((tracker.frame_index, tracker.tracks_created, tracker.tracks_expired), (-1, 0, 0))
        self.assertEqual(tracker.update([detection()])[0]["track_id"], 1)

    def test_empty_session(self):
        tracker = IoUTracker()
        for _ in range(5):
            self.assertEqual(tracker.update([]), [])
        self.assertEqual(tracker.tracks, ())

    def test_invalid_detection_leaves_state_unchanged(self):
        tracker = IoUTracker()
        tracker.update([detection()])
        before = tracker.tracks
        for updates in ({"box_xyxy": [1, 1, 1, 2]}, {"box_xyxy": [float("nan"), 1, 2, 3]},
                        {"box_xyxy": [1, 2, 3]}, {"class_id": 0}, {"confidence": 2.},
                        {"confidence": float("inf")}, {"class_name": ""}):
            with self.subTest(updates=updates), self.assertRaisesRegex(ValueError, "Invalid tracking detection"):
                tracker.update([detection(), detection() | updates])
            self.assertEqual(tracker.tracks, before)
            self.assertEqual(tracker.frame_index, 0)


class TrackingVideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise unittest.SkipTest(str(exc))
        cls.cv2, cls.np = cv2, np

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source, self.output = self.root / "in.avi", self.root / "out.avi"
        self.frame = self.np.zeros((100, 160, 3), dtype=self.np.uint8)
        writer = self.cv2.VideoWriter(str(self.source), self.cv2.VideoWriter_fourcc(*"MJPG"), 12., (160, 100))
        self.assertTrue(writer.isOpened())
        try:
            for _ in range(5):
                writer.write(self.frame)
        finally:
            writer.release()

    def fake_detector(self):
        factory = patch.object(video, "SSDLiteDetector")
        mocked = factory.start()
        self.addCleanup(factory.stop)
        mocked.return_value.runtime = {"model_device": "cuda:0"}
        mocked.return_value.weights_sha256 = "unit-test"
        mocked.return_value.detect.return_value = {
            "detections": [detection()], "inference": {"mean_ms": 10.}}
        return mocked

    def test_video_pipeline_persistence_metrics_and_session_reset(self):
        factory = self.fake_detector()
        seen = []
        annotate = annotate_image

        def capture(frame, records):
            seen.append(copy.deepcopy(records))
            return annotate(frame, records)

        with patch.object(video, "annotate_image", side_effect=capture):
            report = video.detect_video(self.source, self.output, warmup=0, tracking=True)
            second = video.detect_video(self.source, self.root / "second.avi", warmup=0, tracking=True)
        self.assertEqual(factory.call_count, 2)  # Once per session, not frame.
        self.assertEqual([frame[0]["track_id"] for frame in seen], [1] * 10)
        self.assertFalse(seen[0][0]["track_confirmed"])
        self.assertTrue(seen[1][0]["track_confirmed"])
        for result in (report, second):
            self.assertEqual(result["tracking"]["active_tracks"], 1)
            self.assertEqual(result["tracking"]["tracks_created"], 1)
            self.assertEqual(result["tracking"]["matched_detections"], 4)
            self.assertEqual(result["tracking"]["association"]["measured_frames"], 5)
            self.assertGreater(result["tracking"]["association"]["mean_ms"], 0)
            self.assertEqual(result["average_detections_per_frame"], 1)
            self.assertEqual(result["output_verification"]["decoded_frames"], 5)
            json.dumps(result)

    def test_tracking_disabled_does_not_create_tracker_or_ids(self):
        self.fake_detector()
        with patch.object(video, "IoUTracker") as tracker, patch.object(video, "annotate_image", wraps=annotate_image) as annotate:
            report = video.detect_video(self.source, self.output, warmup=0)
        tracker.assert_not_called()
        for call in annotate.call_args_list:
            self.assertNotIn("track_id", call.args[1][0])
        self.assertFalse(report["tracking"]["enabled"])
        self.assertEqual(report["tracking"]["association"]["total_ms"], 0.)
        self.assertEqual(report["tracking"]["association"]["measured_frames"], 0)

    def test_missed_frames_not_drawn_and_expire(self):
        factory = self.fake_detector()
        factory.return_value.detect.side_effect = [
            {"detections": [detection()], "inference": {"mean_ms": 10.}},
            *[{"detections": [], "inference": {"mean_ms": 10.}} for _ in range(4)],
        ]
        report = video.detect_video(self.source, self.output, warmup=0, tracking=True,
                                    tracking_config=TrackingConfig(max_missed=1))
        self.assertEqual(report["tracking"]["tracks_expired"], 1)
        self.assertEqual(report["tracking"]["active_tracks"], 0)
        self.assertEqual(report["tracking"]["visible_tracks_last_frame"], 0)

    def test_id_text_confirmation_and_color(self):
        record = detection() | {"track_id": 2, "track_confirmed": False}
        with patch.object(self.cv2, "putText", wraps=self.cv2.putText) as text:
            frame = annotate_image(self.frame, [record])
            self.assertEqual(text.call_args.args[1], "person #002? 0.90")
            record["track_confirmed"] = True
            annotate_image(self.frame, [record])
            self.assertEqual(text.call_args.args[1], "person #002 0.90")
        self.assertTrue(frame.any())
        self.assertFalse(self.frame.any())
        self.assertEqual(tuple(frame[40, 10]), (255, 190, 0))

    def test_tracking_api_validation(self):
        for kwargs in ({"tracking": "yes"}, {"tracking": True, "tracking_config": "bad"},
                       {"tracking_config": TrackingConfig()}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                video.detect_video(self.source, self.output, **kwargs)

    def test_debug_trace_contains_visible_ids(self):
        self.fake_detector()
        with self.assertLogs("jetson_edge_vision", level=logging.DEBUG) as logs:
            video.detect_video(self.source, self.output, warmup=0, tracking=True)
        messages = [record.getMessage() for record in logs.records]
        trace = [json.loads(message.removeprefix("Tracking frame: ")) for message in messages]
        self.assertEqual([row["frame_index"] for row in trace], list(range(5)))
        self.assertEqual([row["detections"][0]["track_id"] for row in trace], [1] * 5)


class TrackingCliTests(unittest.TestCase):
    def test_enabled_and_disabled(self):
        for flags, enabled in ((["--track", "--track-iou", "0.4", "--track-max-missed", "2", "--track-min-hits", "3"], True),
                               (["--no-track"], False)):
            with self.subTest(flags=flags):
                with patch.object(video, "detect_video", return_value={"frames_processed": 1, "processing_fps": 1.}) as detect:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        self.assertEqual(main(["--video", "in.mp4", "--output", "out.mp4", *flags]), 0)
                if enabled:
                    self.assertTrue(detect.call_args.kwargs["tracking"])
                    self.assertEqual(detect.call_args.kwargs["tracking_config"], TrackingConfig(0.4, 2, 3))
                else:
                    self.assertNotIn("tracking_config", detect.call_args.kwargs)
                    self.assertFalse(detect.call_args.kwargs.get("tracking", False))

    def test_invalid_flag_combinations(self):
        for flags in (["--track"], ["--no-track"], ["--track-iou", "0.3"],
                      ["--video", "x.mp4", "--output", "y.mp4", "--track", "--no-track"],
                      ["--video", "x.mp4", "--output", "y.mp4", "--track-min-hits", "2"],
                      ["--image", "x.png", "--output", "y.png", "--track"]):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(flags)
                self.assertEqual(error.exception.code, 2)

    def test_invalid_settings_fail_before_video_io(self):
        for flags in (["--track-iou", "nan"], ["--track-iou", "0"],
                      ["--track-max-missed", "-1"], ["--track-min-hits", "0"]):
            with self.subTest(flags=flags), patch.object(video, "detect_video") as detect:
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(main(["--video", "in.mp4", "--output", "out.mp4", "--track", *flags]), 1)
                detect.assert_not_called()


@unittest.skipUnless(os.environ.get("JEV_GPU_TESTS") == "1", "Set JEV_GPU_TESTS=1 for CUDA tracking integration")
class TrackingCudaIntegrationTests(unittest.TestCase):
    def test_real_cuda_video_tracking_and_id_persistence(self):
        path = Path(os.environ.get("JEV_TRACK_VIDEO", "data/sintel-sample.mp4"))
        weights = Path(os.environ.get("JEV_WEIGHTS", str(video.DEFAULT_WEIGHTS)))
        # Explicit opt-in requires local media/weights; never downloads or skips missing CUDA.
        with tempfile.TemporaryDirectory() as directory:
            with self.assertLogs("jetson_edge_vision", level=logging.DEBUG) as logs:
                report = video.detect_video(path, Path(directory) / "tracked.avi", weights_path=weights,
                                            max_frames=3, warmup=1, tracking=True)
            trace = [json.loads(r.getMessage().removeprefix("Tracking frame: ")) for r in logs.records]
            self.assertEqual(report["runtime"]["model_device"], "cuda:0")
            self.assertEqual(report["output_verification"]["decoded_frames"], 3)
            self.assertEqual(report["tracking"]["association"]["measured_frames"], 3)
            self.assertGreater(report["tracking"]["matched_detections"], 0)
            first = {r["track_id"] for r in trace[0]["detections"]}
            second = {r["track_id"] for r in trace[1]["detections"]}
            self.assertTrue(first & second)


if __name__ == "__main__":
    unittest.main()
