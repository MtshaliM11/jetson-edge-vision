"""Known-label scoring checks, replay integrity, and opt-in CUDA trace contract."""
import contextlib
import copy
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jetson_edge_vision.evaluation_scenarios import synthetic_scenarios
from jetson_edge_vision.tracking import IoUTracker, TrackingConfig
from jetson_edge_vision import tracking_evaluation as evaluation


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.cases = {case["name"]: case for case in synthetic_scenarios()}

    def run_case(self, name, config=None, repeats=1):
        return evaluation.evaluate_sequence(self.cases[name]["frames"], config or TrackingConfig(),
                                            labelled=True, repeats=repeats)

    def test_all_default_expectations(self):
        self.assertEqual(len(self.cases), 17)
        for name, case in self.cases.items():
            with self.subTest(scenario=name):
                result = self.run_case(name)
                for key, expected in case["expected_default"].items():
                    self.assertEqual(result["metrics"][key], expected, key)

    def test_persistence_numerator_and_denominator(self):
        metrics = self.run_case("one_moving")["metrics"]
        self.assertEqual(metrics["identity_links"], 7)
        self.assertEqual(metrics["id_continuations"], 7)
        self.assertEqual(metrics["persistence_ratio"], 1.)
        result = self.run_case("sudden_jump")["metrics"]
        self.assertEqual(result["identity_links"], 2)
        self.assertEqual(result["id_continuations"], 0)
        self.assertEqual(result["persistence_ratio"], 0.)

    def test_switches_and_distinct_track_fragmentation_are_different(self):
        metrics = self.run_case("category_flip")["metrics"]
        self.assertEqual(metrics["id_switches"], 2)  # 1 -> 2 -> 1
        self.assertEqual(metrics["extra_identity_tracks"], 1)  # Two distinct IDs for A.
        self.assertEqual(metrics["category_errors"], 1)  # Detector error, not three bad detections.

    def test_crossing_errors_have_perfect_detection_labels(self):
        metrics = self.run_case("crossing")["metrics"]
        self.assertEqual(metrics["id_switches"], 2)
        self.assertEqual(metrics["identity_merges"], 2)
        for key in ("category_errors", "missed_object_frames", "false_positive_detections"):
            self.assertEqual(metrics[key], 0)

    def test_visible_misses_distinguished_from_true_absence(self):
        self.assertEqual(self.run_case("temporary_misses")["metrics"]["missed_object_frames"], 2)
        self.assertEqual(self.run_case("reappears_after_gap")["metrics"]["missed_object_frames"], 0)

    def test_max_missed_tradeoff_recovery_versus_stale_identity(self):
        longer = TrackingConfig(max_missed=10)
        self.assertEqual(self.run_case("reappears_after_gap")["metrics"]["id_switches"], 1)
        self.assertEqual(self.run_case("reappears_after_gap", longer)["metrics"]["id_switches"], 0)
        self.assertEqual(self.run_case("stale_identity_after_gap")["metrics"]["identity_merges"], 0)
        self.assertEqual(self.run_case("stale_identity_after_gap", longer)["metrics"]["identity_merges"], 1)
        self.assertEqual(self.run_case("temporary_misses", TrackingConfig(max_missed=0))["metrics"]["id_switches"], 1)

    def test_iou_tradeoffs(self):
        self.assertEqual(self.run_case("fast_motion")["metrics"]["id_switches"], 3)
        self.assertEqual(self.run_case("fast_motion", TrackingConfig(iou_threshold=0.1))["metrics"]["id_switches"], 0)
        self.assertEqual(self.run_case("box_jitter", TrackingConfig(iou_threshold=0.5))["metrics"]["id_switches"], 3)
        self.assertEqual(self.run_case("sudden_jump", TrackingConfig(iou_threshold=0.1))["metrics"]["id_switches"], 2)

    def test_confirmation_does_not_prevent_false_track_birth(self):
        for hits in (1, 2, 3):
            with self.subTest(min_hits=hits):
                metrics = self.run_case("false_positive_burst", TrackingConfig(min_hits=hits))["metrics"]
                self.assertEqual(metrics["false_tracks_created"], 1)
                self.assertEqual(metrics["false_tracks_confirmed"], int(hits <= 2))
                persistent = self.run_case("persistent_false_positive", TrackingConfig(min_hits=hits))["metrics"]
                self.assertEqual(persistent["false_tracks_confirmed"], 1)

    def test_scene_reset_scopes_numeric_ids(self):
        metrics = self.run_case("explicit_scene_reset")["metrics"]
        self.assertEqual(metrics["tracks_created"], 2)
        self.assertEqual(metrics["identity_merges"], 0)
        self.assertEqual(metrics["resets"], 1)
        self.assertEqual(self.run_case("scene_change_without_reset")["metrics"]["identity_merges"], 1)

    def test_no_identity_opportunities_returns_null_ratio(self):
        frames = copy.deepcopy(self.cases["one_moving"]["frames"][:1])
        result = evaluation.evaluate_sequence(frames, TrackingConfig(), labelled=True, repeats=1)
        self.assertEqual(result["metrics"]["identity_links"], 0)
        self.assertIsNone(result["metrics"]["persistence_ratio"])

    def test_unlabelled_data_never_yields_accuracy_metrics(self):
        frames = self.cases["crossing"]["frames"]
        result = evaluation.evaluate_sequence(frames, TrackingConfig(), labelled=False, repeats=2)
        for key in evaluation.IDENTITY_METRICS:
            self.assertIsNone(result["metrics"][key], key)
        self.assertEqual(result["metrics"]["tracks_created"], 2)

    def test_ground_truth_and_saved_ids_never_reach_tracker(self):
        frames = copy.deepcopy(self.cases["one_moving"]["frames"])
        frames[0]["detections"][0].update(track_id=999, truth_id="leak")
        original = IoUTracker.update

        def inspect(tracker, records):
            for record in records:
                self.assertEqual(set(record), set(evaluation.DETECTION_FIELDS))
            return original(tracker, records)

        with patch.object(IoUTracker, "update", inspect):
            evaluation.evaluate_sequence(frames, TrackingConfig(), labelled=True, repeats=1)
        self.assertEqual(frames[0]["detections"][0]["track_id"], 999)  # Caller data unchanged.

    def test_repeat_timing_counts_without_multiplying_metrics(self):
        result = self.run_case("one_moving", repeats=3)
        self.assertEqual(result["timing"]["calls"], 24)
        self.assertEqual(result["metrics"]["identity_links"], 7)
        self.assertGreater(result["timing"]["update_mean_ms"], 0)
        self.assertGreater(result["timing"]["replay_processing_fps"], 0)
        json.dumps(result, allow_nan=False)

    def test_invalid_labels_and_frame_coverage(self):
        original = self.cases["one_moving"]["frames"][:1]
        invalid = [{"frame_index": 1}, {"frame_index": True}, {"truth_ids": []},
                   {"truth_ids": ["unknown"]}, {"reset_before": "yes"},
                   {"visible_objects": {"A": {"class_id": 0, "class_name": "person"}}},
                   {"detections": [None]}]
        for changes in invalid:
            with self.subTest(changes=changes):
                frames = [copy.deepcopy(original[0]) | changes]
                with self.assertRaises(ValueError):
                    evaluation.evaluate_sequence(frames, TrackingConfig(), labelled=True, repeats=1)
        duplicate = copy.deepcopy(original[0])
        duplicate["detections"] *= 2
        duplicate["truth_ids"] *= 2
        with self.assertRaisesRegex(ValueError, "at most one"):
            evaluation.validate_frames([duplicate], True)

    def test_invalid_repeats(self):
        for repeats in (0, -1, True, 1.5):
            with self.subTest(repeats=repeats), self.assertRaises(ValueError):
                self.run_case("one_moving", repeats=repeats)

    def test_grid_preserves_defaults_and_is_complete(self):
        grid = evaluation.parameter_grid()
        self.assertEqual(len(grid), 27)
        self.assertEqual(len(set(grid)), 27)
        self.assertIn(TrackingConfig(), grid)
        self.assertEqual(asdict(TrackingConfig()), {"iou_threshold": 0.3, "max_missed": 5, "min_hits": 2})


class EvaluationFileTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.trace = self.root / "trace.jsonl"
        self.frames = synthetic_scenarios()[0]["frames"]
        rows = [{"frame_index": row["frame_index"], "detections": row["detections"]} for row in self.frames]
        self.trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        self.annotations = self.root / "annotations.json"
        self.annotation_data = {"schema_version": 1, "reviewed": True,
                                "trace_sha256": evaluation.sha256_file(self.trace),
                                "frames": [{k: v for k, v in row.items() if k != "detections"} for row in self.frames]}
        self.annotations.write_text(json.dumps(self.annotation_data))

    def test_trace_accepts_structured_logs_and_discards_old_ids(self):
        _, _, before = evaluation.load_trace_source(self.trace, None, None)
        rows = []
        for row in self.frames:
            event = {"frame_index": row["frame_index"], "detections": [dict(record, track_id=999) for record in row["detections"]]}
            rows.append(json.dumps({"message": "Tracking frame: " + json.dumps(event)}))
        rows.append(json.dumps({"message": "Video detection completed"}))
        self.trace.write_text("\n".join(rows))
        loaded = evaluation.read_trace(self.trace)
        self.assertEqual(len(loaded), 8)
        self.assertNotIn("track_id", loaded[0]["detections"][0])
        _, _, after = evaluation.load_trace_source(self.trace, None, None)
        self.assertNotEqual(before["trace_sha256"], after["trace_sha256"])
        self.assertEqual(before["detections_sha256"], after["detections_sha256"])

    def test_trace_rejects_corruption_empty_and_gaps(self):
        for content in ("not json", "", '{"message": "just an info record"}',
                        '{"frame_index":1,"detections":[]}',
                        '{"frame_index":0,"detections":[]}\n{"frame_index":2,"detections":[]}'):
            with self.subTest(content=content):
                self.trace.write_text(content)
                with self.assertRaises(ValueError):
                    evaluation.read_trace(self.trace)

    def test_manual_annotations_enable_labelled_scoring(self):
        frames = evaluation.read_trace(self.trace)
        labelled = evaluation.apply_annotations(frames, self.annotations, self.trace)
        result = evaluation.evaluate_sequence(labelled, TrackingConfig(), labelled=True, repeats=1)
        self.assertEqual(result["metrics"]["id_switches"], 0)
        self.assertEqual(result["metrics"]["identity_links"], 7)

    def test_annotations_require_review_sha_and_full_coverage(self):
        for changes in ({"reviewed": False}, {"trace_sha256": "wrong"}, {"schema_version": True}, {"frames": []}):
            with self.subTest(changes=changes):
                self.annotations.write_text(json.dumps(self.annotation_data | changes))
                with self.assertRaises(ValueError):
                    evaluation.apply_annotations(evaluation.read_trace(self.trace), self.annotations, self.trace)

    def test_build_trace_report_does_not_claim_video_fps(self):
        report = evaluation.build_report(repeats=1, baseline_only=True, trace_path=self.trace, trace_only=True)
        self.assertEqual(report["synthetic"], [])
        self.assertFalse(report["trace"]["labelled"])
        self.assertEqual(len(report["tracker_sha256"]), 64)
        self.assertIsNone(report["trace"]["configurations"][0]["metrics"]["id_switches"])
        self.assertIn("not video FPS", report["metric_scope"])

    def test_pipeline_report_count_validation(self):
        path = self.root / "video-report.json"
        path.write_text(json.dumps({"frames_processed": 99, "total_detections": 8}))
        with self.assertRaisesRegex(ValueError, "do not match"):
            evaluation.build_report(repeats=1, baseline_only=True, trace_path=self.trace,
                                     trace_only=True, video_report_path=path)

    def test_trace_required_for_trace_options(self):
        for kwargs in ({"trace_only": True}, {"annotations_path": self.annotations},
                       {"video_report_path": self.root / "missing.json"}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "require --trace"):
                evaluation.build_report(repeats=1, **kwargs)

    def test_invalid_trace_fails_before_synthetic_sweep(self):
        with patch.object(evaluation, "synthetic_scenarios") as scenarios:
            with self.assertRaises(OSError):
                evaluation.build_report(trace_path=self.root / "missing.jsonl")
            scenarios.assert_not_called()

    def test_cli_writes_new_report_and_refuses_overwrite(self):
        output = self.root / "nested" / "report.json"
        args = ["--output", str(output), "--trace", str(self.trace), "--trace-only", "--baseline-only", "--repeats", "1"]
        self.assertEqual(evaluation.main(args), 0)
        self.assertEqual(json.loads(output.read_text())["schema_version"], 1)
        before = output.read_bytes()
        with patch.object(evaluation, "build_report") as build, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(evaluation.main(args), 1)
            build.assert_not_called()
        self.assertEqual(output.read_bytes(), before)

    def test_evaluator_does_not_import_hardware_libraries(self):
        import builtins
        original = builtins.__import__

        def guard(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "torchvision", "cv2", "numpy"):
                raise AssertionError(f"Unexpected import: {name}")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guard):
            evaluation.build_report(repeats=1, baseline_only=True)

    def test_cli_reports_bad_repeats_and_missing_trace(self):
        for argv in (["--repeats", "0"], ["--trace", str(self.root / "missing")], ["--trace-only"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(evaluation.main(argv), 1)


@unittest.skipUnless(os.environ.get("JEV_GPU_TESTS") == "1", "Set JEV_GPU_TESTS=1 for actual CUDA trace evaluation")
class EvaluationCudaIntegrationTests(unittest.TestCase):
    def test_actual_gpu_trace_is_replayable_without_ground_truth(self):
        from jetson_edge_vision.detection import DEFAULT_WEIGHTS
        from jetson_edge_vision.video_pipeline import detect_video

        source = Path(os.environ.get("JEV_TRACK_VIDEO", "data/sintel-sample.mp4"))
        weights = Path(os.environ.get("JEV_WEIGHTS", str(DEFAULT_WEIGHTS)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertLogs("jetson_edge_vision", level="DEBUG") as logs:
                pipeline = detect_video(source, root / "tracked.avi", weights_path=weights,
                                         max_frames=3, warmup=1, tracking=True)
            self.assertEqual(pipeline["runtime"]["model_device"], "cuda:0")
            trace = root / "trace.jsonl"
            trace.write_text("\n".join(json.dumps({"message": record.getMessage()}) for record in logs.records))
            report = evaluation.build_report(repeats=2, baseline_only=True, trace_path=trace, trace_only=True)
            result = report["trace"]["configurations"][0]
            self.assertEqual(result["frame_count"], 3)
            self.assertGreater(result["metrics"]["detection_observations"], 0)
            self.assertIsNone(result["metrics"]["id_switches"])
            self.assertIsNone(result["metrics"]["false_tracks_created"])


if __name__ == "__main__":
    unittest.main()
