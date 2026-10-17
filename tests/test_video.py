"""Sequential video tests with synthetic media; opt-in real CUDA integration."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from jetson_edge_vision.cli import main
from jetson_edge_vision import video_pipeline as video


class VideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise unittest.SkipTest(f"Existing OpenCV/NumPy required: {exc}")
        cls.cv2, cls.np = cv2, np

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "source.avi"
        self.output = self.root / "nested" / "annotated.avi"
        self.frame = self.np.zeros((64, 96, 3), dtype=self.np.uint8)
        writer = self.cv2.VideoWriter(str(self.source), self.cv2.VideoWriter_fourcc(*"MJPG"), 12., (96, 64))
        self.assertTrue(writer.isOpened(), "The verified MJPG/AVI codec must work in this environment")
        try:
            for index in range(3):
                frame = self.frame.copy()
                frame[5:15, index * 5: index * 5 + 10] = 255
                writer.write(frame)
        finally:
            writer.release()
        self.factory_patch = patch.object(video, "SSDLiteDetector")
        self.factory = self.factory_patch.start()
        self.addCleanup(self.factory_patch.stop)
        self.detector = self.factory.return_value
        self.detector.runtime = {"model_device": "cuda:0", "gpu": "mock"}
        self.detector.weights_sha256 = "unit-test-checksum"
        self.detector.detect.return_value = {
            "detections": [{"class_id": 1, "class_name": "person", "confidence": 0.9,
                            "box_xyxy": [10., 30., 60., 55.]}],
            "inference": {"mean_ms": 12.},
        }

    def metadata_override(self, **updates):
        original = video.read_metadata
        return patch.object(video, "read_metadata", side_effect=lambda cap: original(cap) | updates)

    def assert_no_partial_output(self):
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.glob(".jev-video-*")), [])

    def test_sequential_frames_annotation_and_output(self):
        original = self.source.read_bytes()
        report = video.detect_video(self.source, self.output, confidence=0.7, warmup=3)
        self.factory.assert_called_once_with(device="cuda", weights_path=video.DEFAULT_WEIGHTS)
        self.assertEqual(self.detector.detect.call_count, 4)  # One warm-up call, then three frames.
        self.assertEqual(self.detector.detect.call_args_list[0].kwargs,
                         {"confidence": 0.7, "warmup": 2, "runs": 1})
        for call in self.detector.detect.call_args_list[1:]:
            self.assertEqual(call.kwargs, {"confidence": 0.7, "warmup": 0, "runs": 1})
        self.assertEqual(report["frames_processed"], 3)
        self.assertEqual(report["total_detections"], 3)
        self.assertEqual(report["detections_by_class"], {"person": 3})
        self.assertEqual(report["source"]["fps"], 12.)
        self.assertEqual(report["source"]["reported_frame_count"], 3)
        self.assertEqual(report["source"]["decoded_width"], 96)
        self.assertEqual(report["output_fps"], 12.)
        self.assertEqual(report["stop_reason"], "decoder_end")
        self.assertEqual(report["output_verification"]["decoded_frames"], 3)
        self.assertEqual(report["inference"]["mean_ms"], 12.)
        self.assertEqual(report["inference"]["median_ms"], 12.)
        self.assertGreater(report["total_processing_ms"], report["stream_processing_ms"])
        self.assertAlmostEqual(report["processing_fps"], 3000 / report["stream_processing_ms"])
        self.assertAlmostEqual(report["end_to_end_fps"], 3000 / report["total_processing_ms"])
        json.dumps(report)
        cap = self.cv2.VideoCapture(str(self.output))
        try:
            ok, frame = cap.read()
            self.assertTrue(ok)
            self.assertEqual(frame.shape, self.frame.shape)
            self.assertGreater(int(frame[40, 10, 1]), 150)  # Green box survived MJPEG encoding.
            self.assertTrue(frame[15:30].any())  # Class/score annotation above the box.
        finally:
            cap.release()
        self.assertEqual(self.source.read_bytes(), original)

    def test_frame_limit(self):
        report = video.detect_video(self.source, self.output, max_frames=2, warmup=0)
        self.assertEqual(report["frames_processed"], 2)
        self.assertEqual(report["output_verification"]["decoded_frames"], 2)
        self.assertEqual(report["stop_reason"], "frame_limit")
        self.assertEqual(self.detector.detect.call_count, 2)

    def test_mp4_and_mov_codec_round_trips(self):
        for suffix in (".mp4", ".mov"):
            with self.subTest(container=suffix):
                encoded = self.root / ("container-test" + suffix)
                report = video.detect_video(self.source, encoded, warmup=0)
                self.assertEqual(report["codec"], "mp4v")
                self.assertEqual(report["output_verification"]["decoded_frames"], 3)
                readback = self.root / ("readback-" + suffix[1:] + ".avi")
                report = video.detect_video(encoded, readback, warmup=0)
                self.assertEqual(report["frames_processed"], 3)

    def test_empty_detections_are_valid(self):
        self.detector.detect.return_value["detections"] = []
        report = video.detect_video(self.source, self.output, warmup=0)
        self.assertEqual(report["total_detections"], 0)
        self.assertEqual(report["detections_by_class"], {})
        self.assertEqual(report["frames_processed"], 3)

    def test_invalid_paths_and_options_before_inference(self):
        cases = [({"video_path": self.root / "missing.mp4"}, "not found"),
                 ({"video_path": self.source.with_suffix(".mkv")}, "not found"),
                 ({"output_path": self.source}, "overwrite"),
                 ({"output_path": self.root / "out.gif"}, "extension"),
                 ({"confidence": float("nan")}, "confidence"),
                 ({"confidence": 1.1}, "confidence"),
                 ({"max_frames": 0}, "max-frames"), ({"max_frames": True}, "max-frames"),
                 ({"output_fps": 0}, "output-fps"), ({"output_fps": float("inf")}, "output-fps"),
                 ({"codec": "bad"}, "codec"), ({"warmup": -1}, "warmup")]
        unsupported = self.root / "unsupported.mkv"
        unsupported.write_bytes(b"not a video")
        cases.append(({"video_path": unsupported}, "extension"))
        for updates, message in cases:
            with self.subTest(updates=updates):
                kwargs = {"video_path": self.source, "output_path": self.output} | updates
                with self.assertRaisesRegex(ValueError, message):
                    video.detect_video(**kwargs)
        self.factory.assert_not_called()

    def test_existing_output_preserved(self):
        self.output.parent.mkdir()
        self.output.write_bytes(b"existing")
        with self.assertRaisesRegex(ValueError, "already exists"):
            video.detect_video(self.source, self.output)
        self.assertEqual(self.output.read_bytes(), b"existing")
        self.factory.assert_not_called()

    def test_corrupt_input(self):
        broken = self.root / "broken.mp4"
        broken.write_bytes(b"not a video")
        with self.assertRaisesRegex(ValueError, "Cannot open"):
            video.detect_video(broken, self.output)
        self.factory.assert_not_called()

    def test_metadata_unknown_dimensions_and_frame_count(self):
        cap = Mock()
        values = {self.cv2.CAP_PROP_FPS: float("nan"),
                  self.cv2.CAP_PROP_FRAME_COUNT: float("inf"),
                  self.cv2.CAP_PROP_FRAME_WIDTH: 0., self.cv2.CAP_PROP_FRAME_HEIGHT: -1.}
        cap.get.side_effect = lambda prop: values.get(prop, 0.)
        cap.getBackendName.return_value = "fake"
        metadata = video.read_metadata(cap)
        self.assertIsNone(metadata["fps"])
        self.assertIsNone(metadata["reported_frame_count"])
        self.assertIsNone(metadata["reported_width"])
        self.assertIsNone(metadata["reported_height"])
        with self.metadata_override(reported_width=None, reported_height=None, reported_frame_count=None):
            report = video.detect_video(self.source, self.output, warmup=0)
        self.assertEqual(report["source"]["decoded_width"], 96)
        self.assertIn("cannot distinguish", report["warnings"][0])

    def test_unknown_source_fps_requires_explicit_override(self):
        with self.metadata_override(fps=None):
            with self.assertRaisesRegex(ValueError, "--output-fps"):
                video.detect_video(self.source, self.output)
            self.factory.assert_not_called()
            report = video.detect_video(self.source, self.output, output_fps=15., warmup=0)
        self.assertEqual(report["output_fps"], 15.)
        self.assertIsNone(report["source"]["fps"])
        self.assertTrue(report["warnings"])

    def test_explicit_fps_override_warns_about_duration(self):
        report = video.detect_video(self.source, self.output, output_fps=24., warmup=0)
        self.assertEqual(report["output_verification"]["fps"], 24.)
        self.assertIn("duration", report["warnings"][0])

    def test_rotation_without_backend_support_is_rejected(self):
        with self.metadata_override(rotation_degrees=90., orientation_auto=False):
            with self.assertRaisesRegex(ValueError, "rotation"):
                video.detect_video(self.source, self.output)
        self.factory.assert_not_called()

    def test_non_square_pixels_rejected(self):
        with self.metadata_override(sample_aspect_ratio=2.):
            with self.assertRaisesRegex(ValueError, "aspect ratio"):
                video.detect_video(self.source, self.output)

    def test_reported_geometry_does_not_override_decoded_frames(self):
        with self.metadata_override(reported_width=64, reported_height=96, rotation_degrees=90., orientation_auto=True):
            report = video.detect_video(self.source, self.output, warmup=0)
        self.assertEqual(report["source"]["decoded_width"], 96)
        self.assertEqual(report["source"]["decoded_height"], 64)

    def test_empty_input_releases_capture(self):
        cap = Mock()
        cap.open.return_value = True
        cap.get.return_value = 0.
        cap.read.return_value = (False, None)
        with patch.object(self.cv2, "VideoCapture", return_value=cap):
            with self.assertRaisesRegex(ValueError, "empty"):
                video.detect_video(self.source, self.output, output_fps=12.)
        cap.release.assert_called()
        self.factory.assert_not_called()

    def test_corrupt_frame_and_geometry_change(self):
        for second in (None, self.frame[:, :94]):
            with self.subTest(second_is_none=second is None):
                cap = Mock()
                cap.open.return_value = True
                cap.get.return_value = 0.
                cap.read.side_effect = [(True, self.frame), (True, second)]
                with patch.object(self.cv2, "VideoCapture", return_value=cap):
                    with self.assertRaisesRegex(ValueError, "frame 1|Frame 1"):
                        video.detect_video(self.source, self.output, output_fps=12., warmup=0)
                cap.release.assert_called()
                self.assert_no_partial_output()

    def test_odd_dimensions_rejected_without_silent_crop(self):
        cap = Mock()
        cap.open.return_value = True
        cap.get.return_value = 0.
        cap.read.return_value = (True, self.frame[:, :95])
        with patch.object(self.cv2, "VideoCapture", return_value=cap):
            with self.assertRaisesRegex(ValueError, "even"):
                video.detect_video(self.source, self.output, output_fps=12.)
        self.factory.assert_not_called()
        cap.release.assert_called()

    def test_early_decoder_stop_is_reported(self):
        with self.metadata_override(reported_frame_count=10):
            with self.assertRaisesRegex(ValueError, "Decoder stopped after 3"):
                video.detect_video(self.source, self.output, warmup=0)
        self.assert_no_partial_output()

    def test_writer_unavailable_proposes_alternative(self):
        writer = Mock()
        writer.open.return_value = False
        with patch.object(self.cv2, "VideoWriter", return_value=writer):
            with self.assertRaisesRegex(OSError, "MJPG"):
                video.detect_video(self.source, self.output)
        writer.release.assert_called()
        self.factory.assert_not_called()
        self.assert_no_partial_output()

    def test_writer_failure_releases_resources(self):
        cap = Mock(wraps=self.cv2.VideoCapture())
        writer = Mock()
        writer.open.return_value = True
        writer.write.side_effect = self.cv2.error("write failed")
        with patch.object(self.cv2, "VideoCapture", return_value=cap), patch.object(self.cv2, "VideoWriter", return_value=writer):
            with self.assertRaisesRegex(RuntimeError, "write failed"):
                video.detect_video(self.source, self.output, warmup=0)
        writer.release.assert_called()
        cap.release.assert_called()
        self.assert_no_partial_output()

    def test_cuda_failure_discards_staged_output(self):
        self.factory.side_effect = RuntimeError("CUDA unavailable; no CPU fallback")
        with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
            video.detect_video(self.source, self.output)
        self.factory.assert_called_once()
        self.assert_no_partial_output()

    def test_interruption_releases_resources_and_discards_output(self):
        for warmup in (0, 3):
            with self.subTest(warmup=warmup):
                cap = Mock(wraps=self.cv2.VideoCapture())
                writer = Mock(wraps=self.cv2.VideoWriter())
                self.detector.detect.side_effect = KeyboardInterrupt
                with patch.object(self.cv2, "VideoCapture", return_value=cap), patch.object(self.cv2, "VideoWriter", return_value=writer):
                    with self.assertRaises(KeyboardInterrupt):
                        video.detect_video(self.source, self.output, warmup=warmup)
                writer.release.assert_called()
                cap.release.assert_called()
                self.assert_no_partial_output()

    def test_silent_encoder_failure_is_not_published(self):
        with patch.object(video, "verify_output", side_effect=OSError("Encoded output decoded 1 frames; expected 3")):
            with self.assertRaisesRegex(OSError, "expected 3"):
                video.detect_video(self.source, self.output, warmup=0)
        self.assert_no_partial_output()

    def test_file_appearing_during_processing_is_not_overwritten(self):
        verify = video.verify_output

        def race(*args):
            result = verify(*args)
            self.output.write_bytes(b"another process created this")
            return result

        with patch.object(video, "verify_output", side_effect=race):
            with self.assertRaises(FileExistsError):
                video.detect_video(self.source, self.output, warmup=0)
        self.assertEqual(self.output.read_bytes(), b"another process created this")

    def test_output_verifier_checks_frame_count(self):
        with self.assertRaisesRegex(OSError, "expected 4"):
            video.verify_output(self.source, 4, (96, 64), 12.)


class VideoCliTests(unittest.TestCase):
    def test_video_routing_and_json(self):
        with patch.object(video, "detect_video", return_value={"frames_processed": 2, "processing_fps": 5.}) as detect:
            with contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()):
                code = main(["--video", "in.mp4", "--output", "out.avi", "--confidence", "0.6",
                             "--max-frames", "2", "--codec", "MJPG", "--output-fps", "24"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["frames_processed"], 2)
        self.assertEqual(detect.call_args.args, (Path("in.mp4"), Path("out.avi")))
        self.assertEqual(detect.call_args.kwargs["device"], "cuda")
        self.assertEqual(detect.call_args.kwargs["max_frames"], 2)
        self.assertEqual(detect.call_args.kwargs["confidence"], 0.6)

    def test_cli_argument_combinations(self):
        for argv in (["--video", "x.mp4"], ["--video", "x.mp4", "--image", "x.png"],
                     ["--video", "x.mp4", "--check-environment"],
                     ["--video", "x.mp4", "--output", "y.mp4", "--runs", "1"],
                     ["--max-frames", "2"], ["--codec", "MJPG"], ["--output-fps", "24"],
                     ["--image", "x.png", "--output", "y.png", "--max-frames", "2"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    main(argv)
                self.assertEqual(exc.exception.code, 2)

    def test_cli_invalid_frame_limit(self):
        with contextlib.redirect_stderr(io.StringIO()) as error:
            self.assertEqual(main(["--video", "x.mp4", "--output", "y.mp4", "--max-frames", "0"]), 1)
        self.assertIn("max-frames", error.getvalue())

    def test_cli_video_error_and_interrupt(self):
        for exception, code in ((RuntimeError("CUDA unavailable"), 1), (KeyboardInterrupt(), 130)):
            with self.subTest(code=code), patch.object(video, "detect_video", side_effect=exception):
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(main(["--video", "x.mp4", "--output", "y.mp4"]), code)


@unittest.skipUnless(os.environ.get("JEV_GPU_TESTS") == "1", "Set JEV_GPU_TESTS=1 for CUDA video integration")
class VideoCudaIntegrationTests(unittest.TestCase):
    def test_actual_cuda_video_inference_and_playable_output(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "synthetic.avi"
            output = Path(directory) / "annotated.avi"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 12., (320, 240))
            self.assertTrue(writer.isOpened())
            try:
                for _ in range(3):
                    writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
            finally:
                writer.release()
            path = Path(os.environ.get("JEV_TEST_VIDEO", str(source)))
            weights = Path(os.environ.get("JEV_WEIGHTS", str(video.DEFAULT_WEIGHTS)))
            report = video.detect_video(path, output, weights_path=weights, max_frames=3, warmup=1)
            self.assertEqual(report["runtime"]["model_device"], "cuda:0")
            self.assertEqual(report["frames_processed"], 3)
            self.assertEqual(report["output_verification"]["decoded_frames"], 3)
            self.assertEqual(report["inference"]["measured_frames"], 3)
            self.assertGreater(report["inference"]["mean_ms"], 0)
            self.assertGreater(report["end_to_end_fps"], 0)
            expected = os.environ.get("JEV_VIDEO_EXPECT_CLASS")
            if expected:
                self.assertIn(expected, report["detections_by_class"])


if __name__ == "__main__":
    unittest.main()
