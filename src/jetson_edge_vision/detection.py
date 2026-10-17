"""Approved SSDLite model, explicit weight acquisition, and frame detection.

Importing this module does not import hardware libraries or download anything.
"""
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import statistics
import time


WEIGHT_FILENAME = "ssdlite320_mobilenet_v3_large_coco-a79551df.pth"
DEFAULT_WEIGHTS = Path("models") / WEIGHT_FILENAME
WEIGHT_HASH_PREFIX = "a79551df"
MODEL_NAME = "ssdlite320_mobilenet_v3_large"


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]


def validate_options(confidence: float, warmup: int, runs: int) -> None:
    if isinstance(confidence, bool) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be a finite number between 0 and 1")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be an integer >= 0")
    if type(runs) is not int or runs < 1:
        raise ValueError("runs must be an integer >= 1")


def verify_weights(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"Weights not found: {path}; run --download-weights first")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    checksum = digest.hexdigest()
    if not checksum.startswith(WEIGHT_HASH_PREFIX):
        raise ValueError(f"Weights SHA256 does not match the official COCO_V1 checkpoint: {path}")
    return checksum


def download_weights(path: Path = DEFAULT_WEIGHTS) -> dict:
    """Download only the approved checkpoint; never replace an existing file."""
    import torch
    from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights

    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(weights.url, str(path), hash_prefix=WEIGHT_HASH_PREFIX)
    return {"weights": str(path.resolve()), "sha256": verify_weights(path), "url": weights.url}


def select_detections(boxes, labels, scores, categories, confidence: float) -> list[Detection]:
    """Convert model output to plain records; labels use COCO's non-contiguous IDs."""
    selected = []
    for box, label, score in zip(boxes, labels, scores):
        label, score = int(label), float(score)
        if score < confidence or not math.isfinite(score):
            continue
        if label <= 0 or label >= len(categories) or categories[label] == "N/A":
            continue
        selected.append(Detection(label, categories[label], score, tuple(float(v) for v in box)))
    return selected


def latency_summary(samples_ms: list[float]) -> dict:
    return {
        "samples_ms": samples_ms,
        "mean_ms": statistics.mean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


class SSDLiteDetector:
    """A reusable FP32 detector; the caller owns image I/O and annotation."""

    def __init__(self, device: str = "cuda", weights_path: Path = DEFAULT_WEIGHTS):
        import torch
        import torchvision
        from torchvision.models.detection import (
            SSDLite320_MobileNet_V3_Large_Weights,
            ssdlite320_mobilenet_v3_large,
        )

        if device not in ("cuda", "cpu"):
            raise ValueError("device must be cuda or cpu")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; verify NVIDIA device access and the existing AI environment; no CPU fallback")
        self.torch = torch
        self.device = torch.device(device)
        self.weights_sha256 = verify_weights(weights_path)
        weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        self.categories = weights.meta["categories"]
        # Both None arguments are essential: the backbone otherwise downloads ImageNet weights.
        self.model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        self.model.eval().to(self.device)
        self.synchronize()
        self.runtime = {
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(self.device) if device == "cuda" else None,
            "model_device": str(next(self.model.parameters()).device),
            "precision": "float32",
        }

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def detect(self, frame, confidence: float = 0.5, warmup: int = 3, runs: int = 10) -> dict:
        """Accept an OpenCV BGR uint8 image; return original-image pixel boxes."""
        import cv2
        import numpy as np

        validate_options(confidence, warmup, runs)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
            raise ValueError("Expected a nonempty HxWx3 BGR uint8 image")
        torch = self.torch
        started = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(device=self.device, dtype=torch.float32) / 255.0
        self.synchronize()
        preprocessing_ms = (time.perf_counter() - started) * 1000
        samples = []
        with torch.inference_mode():
            for _ in range(warmup):
                self.model([tensor])
            self.synchronize()
            for _ in range(runs):
                self.synchronize()
                started = time.perf_counter()
                prediction = self.model([tensor])[0]
                self.synchronize()
                samples.append((time.perf_counter() - started) * 1000)
            # Transfer once, outside inference timing; Torchvision already applied NMS.
            detections = select_detections(
                prediction["boxes"].cpu().tolist(), prediction["labels"].cpu().tolist(),
                prediction["scores"].cpu().tolist(), self.categories, confidence,
            )
        return {
            "detections": [asdict(item) for item in detections],
            "preprocessing_ms": preprocessing_ms,
            "inference": latency_summary(samples),
            "warmup_runs": warmup, "measured_runs": runs,
        }
