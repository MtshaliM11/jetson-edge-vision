"""Known center trajectories: synthetic functional tests, not field accuracy."""
from dataclasses import replace
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jetson_edge_vision.counting import CountingConfig, LineCounter
from jetson_edge_vision.tracking import IoUTracker, Track, TrackingConfig
from jetson_edge_vision.cli import main


def track(identity, x, frame, category=3, name="car", y=50, **kwargs):
    return Track(identity, category, name, (x-10, y-10, x+10, y+10), .9,
                 0, frame, confirmed=True, **kwargs)


class CountingTests(unittest.TestCase):
    def setUp(self):
        # Pixels: x=50, y=10..90 exactly.
        self.counter = LineCounter((101, 101))

    def move(self, objects, positions=(70, 60, 40, 30)):
        for frame, x in enumerate(positions):
            events = self.counter.update([track(i, x, frame, cid, name)
                                          for i, cid, name in objects], frame)
        return self.counter.report()

    def test_one_car(self):
        report = self.move([(1, 3, "car")])
        self.assertEqual(report["total_crossings"], 1)
        self.assertEqual(report["by_direction"], {"A->B": 1, "B->A": 0})

    def test_one_person(self):
        report = self.move([(1, 1, "person")])
        self.assertEqual(report["by_category"]["1"]["total_crossings"], 1)

    def test_multiple_cars(self):
        self.assertEqual(self.move([(i, 3, "car") for i in range(1, 5)])["total_crossings"], 4)

    def test_multiple_people(self):
        self.assertEqual(self.move([(i, 1, "person") for i in range(1, 5)])["total_crossings"], 4)

    def test_simultaneous_mixed_categories(self):
        report = self.move([(1, 3, "car"), (2, 1, "person")])
        self.assertEqual([e["track_id"] for e in report["events"]], [1, 2])
        self.assertEqual([e["frame_index"] for e in report["events"]], [2, 2])

    def test_two_cars_same_frame(self):
        report = self.move([(2, 3, "car"), (1, 3, "car")])
        self.assertEqual(report["by_category"]["3"]["by_direction"]["A->B"], 2)
        self.assertEqual(len({e["event_id"] for e in report["events"]}), 2)

    def test_three_cars_two_people(self):
        report = self.move([(1, 3, "car"), (2, 3, "car"), (3, 3, "car"),
                            (4, 1, "person"), (5, 1, "person")])
        self.assertEqual(report["total_crossings"], 5)
        self.assertEqual([report["by_category"][k]["total_crossings"] for k in ("3", "1")], [3, 2])
        self.assertEqual([e["frame_index"] for e in report["events"]], [2]*5)

    def test_opposite_directions_and_category_totals(self):
        for f, x in enumerate((70, 60, 40, 30)):
            self.counter.update([track(1, x, f), track(2, 100-x, f, 1, "person")], f)
        report = self.counter.report()
        self.assertEqual(report["by_direction"], {"A->B": 1, "B->A": 1})
        self.assertEqual(report["by_category"]["1"]["by_direction"], {"A->B": 0, "B->A": 1})
        self.assertEqual(report["by_category"]["3"]["by_direction"], {"A->B": 1, "B->A": 0})

    def test_no_crossing(self):
        self.assertEqual(self.move([(1, 3, "car")], (70, 65, 60, 55))["total_crossings"], 0)

    def test_stationary(self):
        self.assertEqual(self.move([(1, 3, "car")], (50,)*10)["total_crossings"], 0)

    def test_jitter_and_no_repeat(self):
        self.assertEqual(self.move([(1, 3, "car")], (60, 51, 49, 51, 49, 50, 40, 40, 49, 51, 40))["total_crossings"], 1)

    def test_gap_is_not_inferred_crossing(self):
        self.counter.update([track(1, 60, 0)], 0)
        self.counter.update([replace(track(1, 60, 0), missed=1)], 1)
        self.counter.update([track(1, 40, 2)], 2)
        self.assertEqual(len(self.counter.events), 0)
        self.counter.update([track(1, 60, 3)], 3)
        self.assertEqual(self.counter.events[0].direction, "B->A")

    def test_expired_id_removes_state(self):
        self.counter.update([track(1, 60, 0)], 0)
        self.counter.update([], 1)
        self.counter.update([track(1, 40, 2)], 2)
        self.assertEqual(self.counter.events, ())

    def test_id_switch_does_not_transfer_history(self):
        self.counter.update([track(1, 60, 0)], 0)
        self.counter.update([track(2, 40, 1)], 1)
        self.assertEqual(self.counter.events, ())
        self.counter.update([track(2, 60, 2)], 2)
        self.assertEqual(self.counter.events[0].track_id, 2)

    def test_category_change_resets_history(self):
        self.counter.update([track(1, 60, 0)], 0)
        self.counter.update([track(1, 40, 1, 1, "person")], 1)
        self.assertEqual(self.counter.events, ())

    def test_tentative_tracks(self):
        for f, x in enumerate((60, 40)):
            self.counter.update([replace(track(1, x, f), confirmed=False)], f)
        self.assertEqual(self.counter.events, ())

    def test_confirmation_starts_new_history(self):
        self.counter.update([replace(track(1, 60, 0), confirmed=False)], 0)
        self.counter.update([track(1, 40, 1)], 1)
        self.assertEqual(self.counter.events, ())
        self.counter.update([track(1, 60, 2)], 2)
        self.assertEqual(len(self.counter.events), 1)

    def test_reverse_crossing_rearms_on_opposite_stable_side(self):
        report = self.move([(1, 3, "car")], (60, 50, 40, 50, 60))
        self.assertEqual(report["by_direction"], {"A->B": 1, "B->A": 1})

    def test_overlapping_objects_count_independently(self):
        report = self.move([(i, 3, "car") for i in range(1, 21)])
        self.assertEqual(report["total_crossings"], 20)
        self.assertEqual(len({e["track_id"] for e in report["events"]}), 20)

    def test_empty_frames(self):
        for f in range(4):
            self.assertEqual(self.counter.update([], f), ())
        self.assertEqual(self.counter.report()["total_crossings"], 0)

    def test_invalid_lines(self):
        for line in ((0, 0, 0, 0), (0, 0, 2, 1), (-.1, 0, 1, 1),
                     (0, 0, float("nan"), 1), (0, 0, float("inf"), 1),
                     (False, 0, 1, 1), (0, 1), "oops", None):
            with self.subTest(line=line), self.assertRaises(ValueError):
                CountingConfig(line=line)

    def test_invalid_dimensions_and_hysteresis(self):
        for size in ((0, 100), (1, 100), (100.5, 100), (True, 100), (100,), None):
            with self.subTest(size=size), self.assertRaises(ValueError):
                LineCounter(size)
        for h in (-1, True, float("nan"), "2"):
            with self.assertRaises(ValueError):
                CountingConfig(hysteresis_px=h)

    def test_outside_finite_line(self):
        for f, x in enumerate((60, 40)):
            self.counter.update([track(1, x, f, y=5)], f)
        self.assertEqual(self.counter.events, ())

    def test_deadband_path_around_endpoint_is_not_chord_crossing(self):
        for f, (x, y) in enumerate(((60, 50), (51, 50), (51, 0), (49, 0), (49, 50), (40, 50))):
            self.counter.update([track(1, x, f, y=y)], f)
        self.assertEqual(self.counter.events, ())

    def test_endpoint_intersection(self):
        for f, x in enumerate((60, 40)):
            self.counter.update([track(1, x, f, y=10)], f)
        self.assertEqual(len(self.counter.events), 1)

    def test_counter_reset(self):
        self.move([(1, 3, "car")])
        self.counter.reset()
        self.assertEqual(self.counter.report()["total_crossings"], 0)
        self.assertEqual(self.move([(1, 3, "car")])["events"][0]["event_id"], "line-1:000001")

    def test_tracker_reset_with_counter(self):
        tracker = IoUTracker(TrackingConfig(min_hits=1))
        for f, x in enumerate((60, 54, 46)):
            tracker.update([{"class_id": 3, "class_name": "car", "confidence": .9,
                             "box_xyxy": (x-10, 40, x+10, 60)}])
            self.counter.update(tracker.tracks, f)
        self.assertEqual(len(self.counter.events), 1)
        tracker.reset()
        self.counter.reset()
        tracker.update([])
        self.assertEqual(self.counter.update(tracker.tracks, 0), ())
        self.assertEqual(self.counter.report()["total_crossings"], 0)

    def test_report_consistency_and_timestamp(self):
        self.counter.update([track(1, 60, 0)], 0, 0.)
        self.counter.update([track(1, 40, 1)], 1, .04)
        report = json.loads(json.dumps(self.counter.report()))
        self.assertEqual(report["total_crossings"], len(report["events"]))
        self.assertEqual(sum(report["by_direction"].values()), 1)
        self.assertEqual(report["events"][0]["position"], [40, 50])
        self.assertEqual(report["events"][0]["video_timestamp_seconds"], .04)
        report["by_category"]["3"]["by_direction"]["A->B"] = 99
        self.assertEqual(self.counter.totals()["by_category"]["3"]["by_direction"]["A->B"], 1)

    def test_duplicate_and_out_of_order_frame(self):
        self.counter.update([track(1, 60, 0)], 0)
        with self.assertRaises(ValueError):
            self.counter.update([track(1, 40, 0)], 0)
        self.counter.update([track(1, 40, 1)], 1)
        with self.assertRaises(ValueError):
            self.counter.update([], 0)
        self.assertEqual(len(self.counter.events), 1)

    def test_skipped_update_breaks_continuity(self):
        self.counter.update([track(1, 60, 0)], 0)
        self.counter.update([track(1, 40, 2)], 2)
        self.assertEqual(self.counter.events, ())

    def test_resolution_independence(self):
        for w, h in ((101, 101), (641, 481), (1921, 1081)):
            counter = LineCounter((w, h))
            for f, fraction in enumerate((.7, .3)):
                counter.update([track(1, fraction*(w-1), f, y=(h-1)*.5)], f)
            self.assertEqual(len(counter.events), 1)

    def test_horizontal_diagonal_and_endpoint_order(self):
        for line, points, direction in (((.1, .5, .9, .5), ((50, 30), (50, 70)), "A->B"),
                                        ((0, 0, 1, 1), ((70, 30), (30, 70)), "A->B"),
                                        ((.5, .9, .5, .1), ((70, 50), (30, 50)), "B->A")):
            counter = LineCounter((101, 101), CountingConfig(line=line))
            for f, (x, y) in enumerate(points):
                counter.update([track(1, x, f, y=y)], f)
            self.assertEqual(counter.events[0].direction, direction)

    def test_unlisted_category_supported(self):
        report = self.move([(1, 72, "tv")])
        self.assertEqual(report["by_category"]["72"]["total_crossings"], 1)

    def test_invalid_update_does_not_mutate_state(self):
        self.counter.update([track(1, 60, 0)], 0)
        for records, timestamp in (([track(1, 40, 1)]*2, None),
                                   ([track(1, float("nan"), 1)], None),
                                   ([track(1, 40, 1)], float("nan"))):
            with self.assertRaises(ValueError):
                self.counter.update(records, 1, timestamp)
        self.assertEqual(len(self.counter.update([track(1, 40, 1)], 1)), 1)


class CountingCliTests(unittest.TestCase):
    def test_counting_flags_require_video_and_tracking(self):
        for args in (["--count"], ["--video", "in.avi", "--output", "out.avi", "--count"],
                     ["--video", "in.avi", "--output", "out.avi", "--no-track", "--count-line", ".5,.1,.5,.9"],
                     ["--count-hysteresis", "2"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                main(args)
            self.assertEqual(exc.exception.code, 2)

    def test_invalid_cli_geometry(self):
        prefix = ["--video", "in.avi", "--output", "out.avi", "--track", "--count"]
        for args in (["--count-line", "bad"], ["--count-line", "0,0,0,0"],
                     ["--count-line", "0,0,1,inf"], ["--count-hysteresis", "-1"],
                     ["--count-line", "0,0,2,1"], ["--count-line", "0,0,1"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(prefix + args)

    def test_cli_passes_valid_config(self):
        for flags in (["--count"], ["--count-line", ".2,.1,.2,.9", "--count-hysteresis", "4"]):
            with patch("jetson_edge_vision.video_pipeline.detect_video", return_value={"frames_processed": 2, "processing_fps": 1.}) as detect:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(["--video", "in.avi", "--output", "out.avi", "--track"] + flags), 0)
                self.assertTrue(detect.call_args.kwargs["tracking"])
                config = detect.call_args.kwargs["counting_config"]
                self.assertEqual(config.hysteresis_px, 2 if flags == ["--count"] else 4)


class CountingVideoTests(unittest.TestCase):
    def test_modes_report_and_real_codec_roundtrip(self):
        import cv2
        import numpy as np
        from jetson_edge_vision.video_pipeline import detect_video
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.avi"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 12., (320, 240))
            self.assertTrue(writer.isOpened())
            for _ in range(5):
                writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
            writer.release()
            for mode in ("detection", "tracking", "counting"):
                outputs = []
                # 3 cars and 2 people; 10 px motion with 40 px boxes preserves IoU > .3.
                for f, x in enumerate((180, 170, 160, 150, 140)):
                    outputs.append({"detections": [{"class_id": 3 if i < 3 else 1,
                        "class_name": "car" if i < 3 else "person", "confidence": .9,
                        "box_xyxy": [x-20, 20+i*38, x+20, 50+i*38]} for i in range(5)],
                        "inference": {"mean_ms": 1.}})
                with patch("jetson_edge_vision.video_pipeline.SSDLiteDetector") as factory:
                    factory.return_value.detect.side_effect = outputs
                    factory.return_value.runtime = {"model_device": "cuda:0", "gpu": "mock"}
                    factory.return_value.weights_sha256 = "mock"
                    report = detect_video(source, root / (mode+".avi"), warmup=0,
                                          tracking=mode != "detection",
                                          counting_config=CountingConfig() if mode == "counting" else None)
                self.assertEqual(report["output_verification"]["decoded_frames"], 5)
                self.assertEqual(report["total_detections"], 25)
                self.assertEqual(report["counting"]["enabled"], mode == "counting")
                if mode == "counting":
                    counts = report["counting"]
                    self.assertEqual(counts["total_crossings"], 5)
                    self.assertEqual([e["frame_index"] for e in counts["events"]], [3]*5)
                    self.assertEqual(counts["by_category"]["3"]["total_crossings"], 3)
                    self.assertEqual(counts["by_category"]["1"]["total_crossings"], 2)
                    self.assertEqual(counts["processing"]["measured_frames"], 5)
                    self.assertGreater(counts["processing"]["mean_ms"], 0)
                    self.assertEqual(report["tracking"]["tracks_created"], 5)
                    json.dumps(report)
                else:
                    self.assertEqual(report["counting"]["processing"]["measured_frames"], 0)

    def test_counting_requires_tracking_before_model_load(self):
        from jetson_edge_vision.video_pipeline import detect_video
        with patch("jetson_edge_vision.video_pipeline.SSDLiteDetector") as factory:
            with self.assertRaisesRegex(ValueError, "requires tracking"):
                detect_video(Path("missing.avi"), Path("out.avi"), counting_config=CountingConfig())
            factory.assert_not_called()

    def test_overlay_matches_event_totals_and_bounds_category_rows(self):
        import numpy as np
        from jetson_edge_vision.counting_annotation import annotate_counting
        counter = LineCounter((640, 480))
        for f, x in enumerate((400, 200)):
            counter.update([track(i, x, f, i, f"category{i}", y=240) for i in range(1, 31)], f)
        with patch("cv2.putText") as put:
            annotate_counting(np.zeros((480, 640, 3), dtype=np.uint8), counter)
        rows = [call.args[1] for call in put.call_args_list]
        self.assertEqual(rows[:2], ["Crossings: 30", "A->B: 30  B->A: 0"])
        self.assertIn("categories in JSON", rows[-1])
        self.assertLess(len(rows), 15)


@unittest.skipUnless(os.environ.get("JEV_GPU_TESTS") == "1", "Set JEV_GPU_TESTS=1 for CUDA counting integration")
class CountingCudaTests(unittest.TestCase):
    def test_real_cuda_counting_session(self):
        from jetson_edge_vision.video_pipeline import detect_video
        with tempfile.TemporaryDirectory() as directory:
            report = detect_video(Path(os.environ.get("JEV_TRACK_VIDEO", "data/sintel-sample.mp4")),
                                  Path(directory)/"counted.avi", max_frames=3, warmup=1,
                                  tracking=True, counting_config=CountingConfig())
        self.assertEqual(report["runtime"]["model_device"], "cuda:0")
        self.assertEqual(report["output_verification"]["decoded_frames"], 3)
        self.assertEqual(report["counting"]["processing"]["measured_frames"], 3)
        self.assertEqual(report["counting"]["total_crossings"], len(report["counting"]["events"]))
