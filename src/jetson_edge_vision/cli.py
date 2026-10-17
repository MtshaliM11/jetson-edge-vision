"""Startup, diagnostics, and opt-in image/video detection."""
import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import load_config
from .logging_utils import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Jetson Edge Vision image/video detection")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", type=Path, help="Path to a JSON configuration")
    commands = parser.add_mutually_exclusive_group()
    commands.add_argument("--check-environment", action="store_true", help="Run tensor, NMS and OpenCV checks")
    commands.add_argument("--download-weights", action="store_true", help="Download the approved official COCO_V1 checkpoint, then exit")
    commands.add_argument("--image", type=Path, help="Input image file; inference never downloads weights")
    commands.add_argument("--video", type=Path, help="Local MP4, AVI, or MOV file; headless sequential detection")
    parser.add_argument("--output", type=Path, help="New annotated image/video path (required with --image or --video)")
    parser.add_argument("--weights", type=Path, help="COCO_V1 checkpoint path (default: models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth)")
    parser.add_argument("--confidence", type=float, default=0.5, help="Minimum score, 0 to 1 (default: 0.5)")
    parser.add_argument("--warmup", type=int, default=3, help="Unmeasured warm-up passes; once per video (default: 3)")
    parser.add_argument("--runs", type=int, default=None, help="Image-only measured passes (default: 10); video always runs once per frame")
    parser.add_argument("--max-frames", type=int, help="Video-only processing limit, >= 1")
    parser.add_argument("--codec", help="Video output FourCC (defaults: mp4v for MP4/MOV, MJPG for AVI)")
    parser.add_argument("--output-fps", type=float, help="Video output FPS override; required if source FPS is unknown")
    tracking_mode = parser.add_mutually_exclusive_group()
    tracking_mode.add_argument("--track", action="store_true", help="Enable class-aware IoU video tracking")
    tracking_mode.add_argument("--no-track", action="store_true", help="Disable video tracking (default)")
    parser.add_argument("--track-iou", type=float, help="Tracking association IoU in (0, 1] (default: 0.3)")
    parser.add_argument("--track-max-missed", type=int, help="Retain tracks for this many missed frames (default: 5)")
    parser.add_argument("--track-min-hits", type=int, help="Consecutive observations required for confirmation (default: 2)")
    args = parser.parse_args(argv)
    if (args.image is not None or args.video is not None) and args.output is None:
        parser.error("--image or --video requires --output")
    if args.output is not None and args.image is None and args.video is None:
        parser.error("--output requires --image or --video")
    if args.video is None and any(value is not None for value in (args.max_frames, args.codec, args.output_fps)):
        parser.error("--max-frames, --codec, and --output-fps require --video")
    if args.video is not None and args.runs is not None:
        parser.error("--runs is image-only; video performs exactly one measured inference per frame")
    tracking_options = (args.track_iou, args.track_max_missed, args.track_min_hits)
    if args.video is None and (args.track or args.no_track or any(value is not None for value in tracking_options)):
        parser.error("Tracking flags require --video")
    if not args.track and any(value is not None for value in tracking_options):
        parser.error("--track-iou, --track-max-missed, and --track-min-hits require --track")
    logger = configure_logging()
    try:
        config = load_config(args.config)
        logger = configure_logging(config.log_level)
        if args.download_weights:
            from .detection import DEFAULT_WEIGHTS, download_weights
            print(json.dumps(download_weights(args.weights or DEFAULT_WEIGHTS), indent=2))
        elif args.video is not None:
            from .detection import DEFAULT_WEIGHTS
            from .video_pipeline import detect_video
            from .tracking import TrackingConfig
            tracking_kwargs = {}
            if args.track:
                tracking_kwargs = {"tracking": True, "tracking_config": TrackingConfig(
                    iou_threshold=args.track_iou if args.track_iou is not None else 0.3,
                    max_missed=args.track_max_missed if args.track_max_missed is not None else 5,
                    min_hits=args.track_min_hits if args.track_min_hits is not None else 2,
                )}
            result = detect_video(args.video, args.output, device=config.device,
                                  confidence=args.confidence, weights_path=args.weights or DEFAULT_WEIGHTS,
                                  warmup=args.warmup, max_frames=args.max_frames,
                                  codec=args.codec, output_fps=args.output_fps, **tracking_kwargs)
            print(json.dumps(result, indent=2))
            logger.info("Video detection completed; device=%s; frames=%d; processing_fps=%.2f",
                        config.device, result["frames_processed"], result["processing_fps"])
        elif args.image is not None:
            from .detection import DEFAULT_WEIGHTS
            from .image_pipeline import detect_image
            result = detect_image(args.image, args.output, device=config.device,
                                  confidence=args.confidence, weights_path=args.weights or DEFAULT_WEIGHTS,
                                  warmup=args.warmup, runs=args.runs if args.runs is not None else 10)
            print(json.dumps(result, indent=2))
            logger.info("Image detection completed; device=%s; detections=%d", config.device, len(result["detections"]))
        elif args.check_environment:
            from .diagnostics import check_environment
            print(json.dumps(check_environment(config.device), indent=2))
        else:
            logger.info("Foundation ready; configured device=%s; use --image for detection", config.device)
        return 0
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        logger.error("Startup failed: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted")
        return 130
