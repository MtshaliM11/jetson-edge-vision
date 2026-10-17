"""Startup, explicit environment diagnostics, and opt-in image detection."""
import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import load_config
from .logging_utils import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Jetson Edge Vision image detection")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", type=Path, help="Path to a JSON configuration")
    commands = parser.add_mutually_exclusive_group()
    commands.add_argument("--check-environment", action="store_true", help="Run tensor, NMS and OpenCV checks")
    commands.add_argument("--download-weights", action="store_true", help="Download the approved official COCO_V1 checkpoint, then exit")
    commands.add_argument("--image", type=Path, help="Input image file; inference never downloads weights")
    parser.add_argument("--output", type=Path, help="New annotated image path (required with --image)")
    parser.add_argument("--weights", type=Path, help="COCO_V1 checkpoint path (default: models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth)")
    parser.add_argument("--confidence", type=float, default=0.5, help="Minimum score, 0 to 1 (default: 0.5)")
    parser.add_argument("--warmup", type=int, default=3, help="Unmeasured warm-up passes (default: 3)")
    parser.add_argument("--runs", type=int, default=10, help="Measured inference passes (default: 10)")
    args = parser.parse_args(argv)
    if args.image is not None and args.output is None:
        parser.error("--image requires --output")
    if args.output is not None and args.image is None:
        parser.error("--output requires --image")
    logger = configure_logging()
    try:
        config = load_config(args.config)
        logger = configure_logging(config.log_level)
        if args.download_weights:
            from .detection import DEFAULT_WEIGHTS, download_weights
            print(json.dumps(download_weights(args.weights or DEFAULT_WEIGHTS), indent=2))
        elif args.image is not None:
            from .detection import DEFAULT_WEIGHTS
            from .image_pipeline import detect_image
            result = detect_image(args.image, args.output, device=config.device,
                                  confidence=args.confidence, weights_path=args.weights or DEFAULT_WEIGHTS,
                                  warmup=args.warmup, runs=args.runs)
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
