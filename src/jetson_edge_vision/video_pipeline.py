"""Sequential local-video detection using the unchanged Phase 2A detector."""
from array import array
from collections import Counter
from contextlib import ExitStack
import json
import logging
import math
import os
from pathlib import Path
import statistics
import tempfile
import time

from .detection import DEFAULT_WEIGHTS, MODEL_NAME, SSDLiteDetector, validate_options
from .image_pipeline import annotate_image
from .tracking import IoUTracker, TrackingConfig


DEFAULT_CODECS = {".mp4": "mp4v", ".mov": "mp4v", ".avi": "MJPG"}


def _positive_number(value):
    return float(value) if math.isfinite(value) and value > 0 else None


def read_metadata(capture) -> dict:
    """Metadata is advisory; decoded frames determine the output dimensions."""
    import cv2

    def integer(prop):
        value = _positive_number(capture.get(prop))
        return int(value) if value is not None and value >= 1 else None

    rotation = capture.get(cv2.CAP_PROP_ORIENTATION_META)
    orientation_auto = capture.get(cv2.CAP_PROP_ORIENTATION_AUTO)
    sar_num = _positive_number(capture.get(cv2.CAP_PROP_SAR_NUM))
    sar_den = _positive_number(capture.get(cv2.CAP_PROP_SAR_DEN))
    return {
        "reported_width": integer(cv2.CAP_PROP_FRAME_WIDTH),
        "reported_height": integer(cv2.CAP_PROP_FRAME_HEIGHT),
        "fps": _positive_number(capture.get(cv2.CAP_PROP_FPS)),
        "reported_frame_count": integer(cv2.CAP_PROP_FRAME_COUNT),
        "rotation_degrees": float(rotation) if math.isfinite(rotation) else None,
        "orientation_auto": math.isfinite(orientation_auto) and orientation_auto > 0,
        "sample_aspect_ratio": sar_num / sar_den if sar_num and sar_den else None,
        "backend": capture.getBackendName(),
    }


def _frame_size(frame, index: int, expected=None) -> tuple[int, int]:
    import numpy as np

    if (frame is None or frame.dtype != np.uint8 or frame.ndim != 3
            or frame.shape[2] != 3 or not frame.size):
        raise ValueError(f"Invalid/corrupted decoded frame {index}; expected nonempty BGR uint8")
    size = (frame.shape[1], frame.shape[0])
    if expected is not None and size != expected:
        raise ValueError(f"Frame {index} changed dimensions from {expected} to {size}; cannot preserve video geometry")
    return size


def verify_output(path: Path, expected_frames: int, size: tuple[int, int], fps: float) -> dict:
    """Decode the finalized file completely to catch silent writer failures."""
    import cv2

    capture = cv2.VideoCapture()
    try:
        if not capture.open(str(path)):
            raise OSError("Encoded output cannot be opened; try an AVI output with --codec MJPG")
        output_fps = _positive_number(capture.get(cv2.CAP_PROP_FPS))
        if output_fps is None or not math.isclose(output_fps, fps, rel_tol=0.001, abs_tol=0.001):
            raise OSError(f"Encoded output FPS {output_fps} differs from requested FPS {fps}")
        decoded = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            _frame_size(frame, decoded, size)
            decoded += 1
        if decoded != expected_frames:
            raise OSError(f"Encoded output decoded {decoded} frames; expected {expected_frames}; check disk space/codec")
        return {"decoded_frames": decoded, "fps": output_fps, "backend": capture.getBackendName()}
    finally:
        capture.release()


def detect_video(video_path: Path, output_path: Path, *, device: str = "cuda",
                 confidence: float = 0.5, weights_path: Path = DEFAULT_WEIGHTS,
                 warmup: int = 3, max_frames: int | None = None,
                 codec: str | None = None, output_fps: float | None = None,
                 tracking: bool = False, tracking_config: TrackingConfig | None = None) -> dict:
    """Process one frame at a time. Only scalar latency samples accumulate.

    Output is staged next to its destination and published without replacement
    only after decoding every encoded frame. Exceptions/interrupts discard it.
    """
    started = time.perf_counter()
    validate_options(confidence, warmup, 1)
    if type(tracking) is not bool:
        raise ValueError("tracking must be enabled or disabled with a boolean")
    if tracking_config is not None and not isinstance(tracking_config, TrackingConfig):
        raise ValueError("tracking_config must be a TrackingConfig")
    if not tracking and tracking_config is not None:
        raise ValueError("Tracking options require tracking to be enabled")
    tracker = IoUTracker(tracking_config) if tracking else None
    logger = logging.getLogger("jetson_edge_vision")
    if max_frames is not None and (type(max_frames) is not int or max_frames < 1):
        raise ValueError("max-frames must be an integer >= 1")
    if output_fps is not None and (isinstance(output_fps, bool) or _positive_number(output_fps) is None):
        raise ValueError("output-fps must be a finite number > 0")
    if not video_path.is_file():
        raise ValueError(f"Input video not found: {video_path}")
    if video_path.suffix.lower() not in DEFAULT_CODECS:
        raise ValueError("Input video extension must be .mp4, .avi, or .mov; codec availability depends on OpenCV")
    if output_path.resolve() in (video_path.resolve(), weights_path.resolve()):
        raise ValueError("Output must not overwrite the input video or model weights")
    if output_path.exists() or output_path.is_symlink():
        raise ValueError(f"Output already exists; choose a new output path: {output_path}")
    if output_path.suffix.lower() not in DEFAULT_CODECS:
        raise ValueError("Output video extension must be .mp4, .avi, or .mov")
    codec = codec if codec is not None else DEFAULT_CODECS[output_path.suffix.lower()]
    if len(codec) != 4 or not codec.isascii() or not codec.isalnum():
        raise ValueError("codec must be a four-character ASCII alphanumeric FourCC, such as mp4v or MJPG")

    import cv2

    capture, writer = cv2.VideoCapture(), cv2.VideoWriter()
    try:
        if not capture.open(str(video_path)):
            raise ValueError(f"Cannot open input video {video_path}; file may be damaged or its codec unavailable")
        # Some backends ignore this property; inspect both metadata and decoded geometry.
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        source = read_metadata(capture)
        rotation = source["rotation_degrees"]
        if rotation is not None and rotation % 360 and not source["orientation_auto"]:
            raise ValueError("Video has rotation metadata but this backend cannot auto-rotate; normalize orientation before processing")
        sar = source["sample_aspect_ratio"]
        if sar is not None and not math.isclose(sar, 1.0):
            raise ValueError("Non-square source pixels are unsupported; normalize pixel aspect ratio before processing")
        fps = output_fps if output_fps is not None else source["fps"]
        if fps is None:
            raise ValueError("Source FPS is unavailable/invalid; specify --output-fps explicitly")
        ok, frame = capture.read()
        if not ok:
            raise ValueError("Video is empty or the first frame cannot be decoded")
        size = _frame_size(frame, 0)
        if size[0] % 2 or size[1] % 2:
            raise ValueError("Video dimensions must be even; refusing codec truncation that would change the input geometry")
        source["decoded_width"], source["decoded_height"] = size
        warnings = []
        if source["fps"] is None:
            warnings.append("Source FPS unavailable; using the explicit output FPS")
        elif not math.isclose(fps, source["fps"]):
            warnings.append("Output FPS differs from source FPS; playback duration will change")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".jev-video-", dir=output_path.parent) as directory, ExitStack() as cleanup:
            # Close even when model initialization/warm-up fails, before deleting staging files.
            cleanup.callback(writer.release)
            staged = Path(directory) / output_path.name
            if not writer.open(str(staged), cv2.VideoWriter_fourcc(*codec), fps, size):
                raise OSError(f"Cannot open video writer for {output_path} with codec {codec}; check permissions/codec support; try a new .avi output with --codec MJPG")
            writer_backend = writer.getBackendName()
            loading_started = time.perf_counter()
            detector = SSDLiteDetector(device=device, weights_path=weights_path)
            model_load_ms = (time.perf_counter() - loading_started) * 1000
            warmup_started = time.perf_counter()
            if warmup:
                # Exactly warmup calls, through the existing inference_mode/CUDA path.
                detector.detect(frame, confidence=confidence, warmup=warmup - 1, runs=1)
            warmup_ms = (time.perf_counter() - warmup_started) * 1000
            samples = array("d")
            association_samples = array("d")
            peak_active_tracks = 0
            visible_tracks = 0
            categories = Counter()
            processed = 0
            stream_started = time.perf_counter()
            try:
                while True:
                    _frame_size(frame, processed, size)
                    result = detector.detect(frame, confidence=confidence, warmup=0, runs=1)
                    if tracker is not None:
                        association_started = time.perf_counter()
                        result["detections"] = tracker.update(result["detections"])
                        association_samples.append((time.perf_counter() - association_started) * 1000)
                        peak_active_tracks = max(peak_active_tracks, len(tracker.tracks))
                        visible_tracks = len(result["detections"])
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug("Tracking frame: %s", json.dumps({
                                "frame_index": processed, "detections": result["detections"],
                            }))
                    annotated = annotate_image(frame, result["detections"])
                    if writer.write(annotated) is False:
                        raise OSError(f"Video writer failed at frame {processed}")
                    samples.append(result["inference"]["mean_ms"])
                    categories.update(item["class_name"] for item in result["detections"])
                    processed += 1
                    # No detection records or frame tensors are retained between iterations.
                    del result, annotated, frame
                    if max_frames is not None and processed >= max_frames:
                        stop_reason = "frame_limit"
                        break
                    ok, frame = capture.read()
                    if not ok:
                        count = source["reported_frame_count"]
                        if count is not None and processed < count:
                            raise ValueError(f"Decoder stopped after {processed} frames; source reports {count}. Input may be truncated/corrupt or frame-count metadata inaccurate")
                        stop_reason = "decoder_end"
                        break
            finally:
                writer.release()
            stream_ms = (time.perf_counter() - stream_started) * 1000
            capture.release()
            verification_started = time.perf_counter()
            verified = verify_output(staged, processed, size, fps)
            verification_ms = (time.perf_counter() - verification_started) * 1000
            # Same-filesystem hard link publishes atomically and fails if a file appeared meanwhile.
            os.link(staged, output_path)

        if stop_reason == "decoder_end" and source["reported_frame_count"] is None:
            warnings.append("Frame count unavailable; OpenCV cannot distinguish ordinary EOF from a decoder failure")
        total_ms = (time.perf_counter() - started) * 1000
        return {
            "model": MODEL_NAME, "weights_variant": "COCO_V1", "device": device,
            "runtime": detector.runtime, "weights_sha256": detector.weights_sha256,
            "input": str(video_path.resolve()), "output": str(output_path.resolve()),
            "source": source, "output_fps": fps, "codec": codec, "writer_backend": writer_backend,
            "confidence_threshold": confidence, "warmup_runs": warmup,
            "max_frames": max_frames, "stop_reason": stop_reason,
            "frames_processed": processed, "total_detections": sum(categories.values()),
            "average_detections_per_frame": sum(categories.values()) / processed,
            "detections_by_class": dict(sorted(categories.items())),
            "tracking": {
                "enabled": tracking, "algorithm": "class_aware_greedy_iou" if tracker else None,
                "config": {"iou_threshold": tracker.config.iou_threshold,
                           "max_missed": tracker.config.max_missed,
                           "min_hits": tracker.config.min_hits} if tracker else None,
                "active_tracks": len(tracker.tracks) if tracker else 0,
                "confirmed_active_tracks": sum(t.confirmed for t in tracker.tracks) if tracker else 0,
                "visible_tracks_last_frame": visible_tracks,
                "peak_active_tracks": peak_active_tracks,
                "tracks_created": tracker.tracks_created if tracker else 0,
                "tracks_expired": tracker.tracks_expired if tracker else 0,
                "matched_detections": tracker.matched_detections if tracker else 0,
                "association": {
                    "measured_frames": len(association_samples),
                    "total_ms": sum(association_samples),
                    "mean_ms": statistics.mean(association_samples) if association_samples else 0.,
                    "median_ms": statistics.median(association_samples) if association_samples else 0.,
                    "max_ms": max(association_samples) if association_samples else 0.,
                },
            },
            "inference": {"measured_frames": len(samples), "mean_ms": statistics.mean(samples),
                          "median_ms": statistics.median(samples), "min_ms": min(samples), "max_ms": max(samples)},
            "model_load_ms": model_load_ms, "warmup_ms": warmup_ms,
            "stream_processing_ms": stream_ms, "processing_fps": processed * 1000 / stream_ms,
            "output_verification_ms": verification_ms,
            "total_processing_ms": total_ms, "end_to_end_fps": processed * 1000 / total_ms,
            "output_verification": verified, "warnings": warnings,
        }
    except cv2.error as exc:
        raise RuntimeError(f"OpenCV video processing failed: {exc}; check the input, output location, and codec; AVI with MJPG is an alternative") from exc
    finally:
        try:
            writer.release()
        finally:
            capture.release()
