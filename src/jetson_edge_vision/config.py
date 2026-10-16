"""Strict JSON configuration without additional runtime dependencies."""
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    log_level: str = "INFO"
    device: str = "cuda"


def load_config(path: Path | None = None) -> AppConfig:
    if path is None:
        return AppConfig()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read configuration {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a JSON object")
    unknown = data.keys() - {"log_level", "device"}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    level = data.get("log_level", "INFO")
    device = data.get("device", "cuda")
    if level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ValueError("log_level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
    if device not in ("cuda", "cpu"):
        raise ValueError("device must be cuda or cpu")
    return AppConfig(log_level=level, device=device)
