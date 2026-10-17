"""Image-file orchestration and OpenCV annotation, separate from the detector."""
from pathlib import Path
import time

from .detection import DEFAULT_WEIGHTS, MODEL_NAME, SSDLiteDetector, validate_options


TRACK_COLORS = ((0, 255, 0), (255, 190, 0), (0, 220, 255),
                (255, 100, 255), (255, 255, 0), (160, 255, 120))


def annotate_image(frame, detections: list[dict]):
    import cv2

    annotated = frame.copy()
    height, width = frame.shape[:2]
    for detection in detections:
        x1, y1, x2, y2 = detection["box_xyxy"]
        x1, x2 = (max(0, min(width - 1, round(x))) for x in (x1, x2))
        y1, y2 = (max(0, min(height - 1, round(y))) for y in (y1, y2))
        color = TRACK_COLORS[(detection["track_id"] - 1) % len(TRACK_COLORS)] if "track_id" in detection else (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"{detection['class_name']} {detection['confidence']:.2f}"
        if "track_id" in detection:
            suffix = "" if detection["track_confirmed"] else "?"
            text = f"{detection['class_name']} #{detection['track_id']:03d}{suffix} {detection['confidence']:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx = max(0, min(x1, width - tw - 1))
        ty = min(height - 1 - baseline, max(th + 2, y1 - 5))
        cv2.rectangle(annotated, (tx, max(0, ty - th - 2)),
                      (min(width - 1, tx + tw), min(height - 1, ty + baseline)), color, -1)
        cv2.putText(annotated, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return annotated


def detect_image(image_path: Path, output_path: Path, *, device: str = "cuda",
                 confidence: float = 0.5, weights_path: Path = DEFAULT_WEIGHTS,
                 warmup: int = 3, runs: int = 10) -> dict:
    """Total time includes imports, loading, warm-up, all runs, and output writing."""
    started = time.perf_counter()
    validate_options(confidence, warmup, runs)
    if not image_path.is_file():
        raise ValueError(f"Input image not found: {image_path}")
    if output_path.resolve() in (image_path.resolve(), weights_path.resolve()):
        raise ValueError("Output must not overwrite the input image or model weights")
    if output_path.exists():
        raise ValueError(f"Output already exists; choose a new output path: {output_path}")
    if output_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"):
        raise ValueError("Output must have a supported image extension (.jpg, .png, .bmp, .tif, .webp)")

    import cv2

    try:
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ValueError(f"Cannot decode input image: {image_path}: {exc}") from exc
    if frame is None:
        raise ValueError(f"Cannot decode input image: {image_path}")
    loading_started = time.perf_counter()
    detector = SSDLiteDetector(device=device, weights_path=weights_path)
    model_load_ms = (time.perf_counter() - loading_started) * 1000
    result = detector.detect(frame, confidence=confidence, warmup=warmup, runs=runs)
    annotated = annotate_image(frame, result["detections"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        written = cv2.imwrite(str(output_path), annotated)
    except cv2.error as exc:
        raise OSError(f"Cannot write annotated image: {output_path}: {exc}") from exc
    if not written:
        raise OSError(f"Cannot write annotated image: {output_path}")
    return {
        "model": MODEL_NAME, "weights_variant": "COCO_V1",
        "weights_path": str(weights_path.resolve()), "weights_sha256": detector.weights_sha256,
        "device": device, "runtime": detector.runtime,
        "input": str(image_path.resolve()), "output": str(output_path.resolve()),
        "image_size": {"width": frame.shape[1], "height": frame.shape[0]},
        "confidence_threshold": confidence, **result,
        "model_load_ms": model_load_ms,
        "total_processing_ms": (time.perf_counter() - started) * 1000,
    }
