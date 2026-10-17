"""Offline unit tests; the opt-in CUDA integration test never downloads assets."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jetson_edge_vision.cli import main
from jetson_edge_vision.detection import (
    DEFAULT_WEIGHTS, SSDLiteDetector, download_weights, latency_summary,
    select_detections, validate_options, verify_weights,
)
from jetson_edge_vision.image_pipeline import annotate_image, detect_image


class DetectionTests(unittest.TestCase):
    def test_invalid_options(self):
        for confidence, warmup, runs in ((-0.1, 0, 1), (1.1, 0, 1), (float("nan"), 0, 1),
                                         (float("inf"), 0, 1), (True, 0, 1),
                                         (0.5, -1, 1), (0.5, 0.5, 1), (0.5, 0, 0)):
            with self.subTest(options=(confidence, warmup, runs)):
                with self.assertRaises(ValueError):
                    validate_options(confidence, warmup, runs)
        validate_options(0, 0, 1)
        validate_options(1, 3, 10)

    def test_threshold_and_sparse_coco_labels(self):
        categories = ["__background__", "person", "N/A", "car"]
        result = select_detections(
            [[1, 2, 30, 40]] * 8, [1, 3, 3, 2, 0, 99, 1, 1],
            [0.8, 0.5, 0.49, 0.9, 0.9, 0.9, float("nan"), float("inf")], categories, 0.5,
        )
        self.assertEqual([item.class_name for item in result], ["person", "car"])
        self.assertEqual(result[1].class_id, 3)
        self.assertEqual(result[1].confidence, 0.5)
        self.assertEqual(result[0].box_xyxy, (1., 2., 30., 40.))
        self.assertEqual(select_detections([], [], [], categories, 0.5), [])

    def test_latency_statistics(self):
        result = latency_summary([10., 30., 20.])
        self.assertEqual(result, {"samples_ms": [10., 30., 20.], "mean_ms": 20.,
                                  "median_ms": 20., "min_ms": 10., "max_ms": 30.})

    def test_missing_and_corrupt_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pth"
            with self.assertRaisesRegex(ValueError, "--download-weights"):
                verify_weights(path)
            path.write_bytes(b"not the approved checkpoint")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                verify_weights(path)


class ImagePipelineTests(unittest.TestCase):
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
        self.input = self.root / "input.png"
        self.output = self.root / "nested" / "annotated.png"
        self.frame = self.np.zeros((80, 120, 3), dtype=self.np.uint8)
        self.cv2.imwrite(str(self.input), self.frame)
        self.detection = {"class_id": 1, "class_name": "person", "confidence": 0.9,
                          "box_xyxy": [10., 20., 90., 70.]}

    def test_annotation_preserves_source_and_draws_boxes_and_labels(self):
        annotated = annotate_image(self.frame, [self.detection])
        self.assertEqual(annotated.shape, self.frame.shape)
        self.assertFalse(self.frame.any())
        self.assertTrue(annotated[20, 10].any())
        self.assertTrue(annotated[:20].any())
        self.np.testing.assert_array_equal(annotate_image(self.frame, []), self.frame)

    def test_pipeline_writes_image_and_reports_results(self):
        original = self.input.read_bytes()
        with patch("jetson_edge_vision.image_pipeline.SSDLiteDetector") as factory:
            detector = factory.return_value
            detector.weights_sha256 = "example-checksum"
            detector.runtime = {"model_device": "cuda:0"}
            detector.detect.return_value = {"detections": [self.detection], "inference": {"mean_ms": 12.}}
            result = detect_image(self.input, self.output, confidence=0.75, warmup=2, runs=4)
            factory.assert_called_once_with(device="cuda", weights_path=DEFAULT_WEIGHTS)
            self.assertEqual(detector.detect.call_args.kwargs, {"confidence": 0.75, "warmup": 2, "runs": 4})
        self.assertEqual(result["image_size"], {"width": 120, "height": 80})
        self.assertEqual(result["detections"], [self.detection])
        self.assertGreater(result["total_processing_ms"], 0)
        self.assertEqual(self.cv2.imread(str(self.output)).shape, self.frame.shape)
        self.assertEqual(self.input.read_bytes(), original)
        json.dumps(result)

    def test_invalid_paths_fail_before_model_load(self):
        corrupt = self.root / "corrupt.png"
        corrupt.write_bytes(b"bad image")
        cases = [(self.root / "missing.png", self.output, "not found"),
                 (corrupt, self.output, "Cannot decode"),
                 (self.input, self.input, "overwrite"),
                 (self.input, self.root / "output.mp4", "extension")]
        with patch("jetson_edge_vision.image_pipeline.SSDLiteDetector") as factory:
            for source, output, message in cases:
                with self.subTest(source=source, output=output):
                    with self.assertRaisesRegex(ValueError, message):
                        detect_image(source, output)
            factory.assert_not_called()

    def test_existing_output_is_preserved(self):
        self.output.parent.mkdir()
        self.output.write_bytes(b"keep existing output")
        with self.assertRaisesRegex(ValueError, "already exists"):
            detect_image(self.input, self.output)
        self.assertEqual(self.output.read_bytes(), b"keep existing output")

    def test_write_failure_is_reported(self):
        with patch("jetson_edge_vision.image_pipeline.SSDLiteDetector") as factory:
            factory.return_value.detect.return_value = {"detections": []}
            with patch.object(self.cv2, "imwrite", return_value=False):
                with self.assertRaisesRegex(OSError, "Cannot write"):
                    detect_image(self.input, self.output)

    def test_codec_exception_is_reported(self):
        with patch("jetson_edge_vision.image_pipeline.SSDLiteDetector") as factory:
            factory.return_value.detect.return_value = {"detections": []}
            with patch.object(self.cv2, "imwrite", side_effect=self.cv2.error("codec unavailable")):
                with self.assertRaisesRegex(OSError, "Cannot write"):
                    detect_image(self.input, self.output)


class DetectorRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import torchvision
            import cv2
            import numpy as np
        except ImportError as exc:
            raise unittest.SkipTest(f"Existing AI environment required: {exc}")
        cls.torch, cls.np = torch, np

    def test_cuda_unavailable_fails_without_cpu_fallback(self):
        with patch.object(self.torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
                SSDLiteDetector("cuda")

    def test_missing_weights_never_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(self.torch.hub, "download_url_to_file") as download:
                with self.assertRaisesRegex(ValueError, "--download-weights"):
                    SSDLiteDetector("cpu", Path(directory) / "missing.pth")
                download.assert_not_called()

    def test_existing_corrupt_weights_are_not_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pth"
            path.write_bytes(b"corrupt")
            with patch.object(self.torch.hub, "download_url_to_file") as download:
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    download_weights(path)
                download.assert_not_called()
            self.assertEqual(path.read_bytes(), b"corrupt")

    def test_bgr_to_rgb_normalization_and_inference_mode(self):
        torch = self.torch
        detector = SSDLiteDetector.__new__(SSDLiteDetector)
        detector.torch, detector.device = torch, torch.device("cpu")
        detector.categories = ["__background__", "person"]
        captured = []

        def model(inputs):
            self.assertFalse(torch.is_grad_enabled())
            captured.append(inputs[0].clone())
            return [{"boxes": torch.tensor([[0., 0., 1., 1.]]),
                     "labels": torch.tensor([1]), "scores": torch.tensor([0.9])}]

        detector.model = model
        frame = self.np.array([[[0, 128, 255]]], dtype=self.np.uint8)
        result = detector.detect(frame, warmup=2, runs=3)
        self.assertEqual(len(captured), 5)
        torch.testing.assert_close(captured[0][:, 0, 0], torch.tensor([1., 128 / 255, 0.]))
        self.assertEqual(result["detections"][0]["class_name"], "person")
        self.assertEqual(len(result["inference"]["samples_ms"]), 3)
        with self.assertRaisesRegex(ValueError, "BGR uint8"):
            detector.detect(frame.astype(self.np.float32))


class DetectionCliTests(unittest.TestCase):
    def test_cli_routes_image_and_prints_json(self):
        with patch("jetson_edge_vision.image_pipeline.detect_image", return_value={"detections": []}) as detect:
            with contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()):
                result = main(["--image", "input.jpg", "--output", "output.jpg", "--confidence", "0.7",
                               "--warmup", "1", "--runs", "2"])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue()), {"detections": []})
        self.assertEqual(detect.call_args.kwargs["device"], "cuda")
        self.assertEqual(detect.call_args.kwargs["confidence"], 0.7)

    def test_invalid_input_does_not_download(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("jetson_edge_vision.detection.download_weights") as download:
                with contextlib.redirect_stderr(io.StringIO()):
                    result = main(["--image", str(Path(directory) / "missing.png"), "--output", "out.png"])
                self.assertEqual(result, 1)
                download.assert_not_called()

    def test_startup_keeps_hardware_imports_lazy(self):
        import builtins
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "torchvision", "cv2", "numpy"):
                raise AssertionError(f"Unexpected hardware import: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([]), 0)

    def test_cli_usage_errors(self):
        for argv in (["--image", "x.png"], ["--output", "x.png"],
                     ["--image", "x.png", "--check-environment"],
                     ["--image", "x.png", "--download-weights"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    main(argv)
                self.assertEqual(exc.exception.code, 2)

    def test_cuda_failure_is_reported_without_fallback(self):
        with patch("jetson_edge_vision.image_pipeline.detect_image", side_effect=RuntimeError("CUDA unavailable")) as detect:
            with contextlib.redirect_stderr(io.StringIO()) as error:
                self.assertEqual(main(["--image", "x.png", "--output", "y.png"]), 1)
        self.assertIn("CUDA unavailable", error.getvalue())
        detect.assert_called_once()
        self.assertEqual(detect.call_args.kwargs["device"], "cuda")


@unittest.skipUnless(os.environ.get("JEV_GPU_TESTS") == "1", "Set JEV_GPU_TESTS=1 for actual CUDA inference")
class CudaIntegrationTests(unittest.TestCase):
    def test_pretrained_cuda_image_inference(self):
        import cv2
        import numpy as np
        from jetson_edge_vision.detection import SSDLiteDetector

        # Opt-in means missing CUDA/weights is a failure, never a silent skip or download.
        weights = Path(os.environ.get("JEV_WEIGHTS", str(DEFAULT_WEIGHTS)))
        detector = SSDLiteDetector("cuda", weights)
        self.assertEqual(next(detector.model.parameters()).device.type, "cuda")
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = detector.detect(frame, confidence=0.99, warmup=1, runs=2)
        self.assertEqual(result["detections"], [])
        self.assertEqual(len(result["inference"]["samples_ms"]), 2)
        self.assertGreater(result["inference"]["mean_ms"], 0)
        # A real public/example image can additionally be tested for semantic detection.
        image = os.environ.get("JEV_TEST_IMAGE")
        if image:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "annotated.png"
                report = detect_image(Path(image), output, weights_path=weights, warmup=1, runs=2)
                self.assertIsNotNone(cv2.imread(str(output)))
                self.assertTrue(report["detections"])
                expected = os.environ.get("JEV_EXPECT_CLASS")
                if expected:
                    self.assertIn(expected, {d["class_name"] for d in report["detections"]})


if __name__ == "__main__":
    unittest.main()
