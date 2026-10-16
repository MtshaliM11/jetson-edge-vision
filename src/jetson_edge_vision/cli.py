"""Phase 1 startup and explicit environment diagnostics."""
import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import load_config
from .logging_utils import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Jetson Edge Vision foundation")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", type=Path, help="Path to a JSON configuration")
    parser.add_argument("--check-environment", action="store_true", help="Run tensor, NMS and OpenCV checks")
    args = parser.parse_args(argv)
    logger = configure_logging()
    try:
        config = load_config(args.config)
        logger = configure_logging(config.log_level)
        if args.check_environment:
            from .diagnostics import check_environment
            print(json.dumps(check_environment(config.device), indent=2))
        else:
            logger.info("Foundation ready; configured device=%s; inference is not implemented", config.device)
        return 0
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        logger.error("Startup failed: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted")
        return 130
