"""
config.py
=========
Central configuration for the AquaGuard pipeline.

Resolution order (later wins):

1. Built-in defaults (:data:`DEFAULTS`).
2. ``configs/pipeline.yaml`` (or the file named by ``AQUAGUARD_CONFIG``).
3. Environment variables (``DFINE_ONNX``, ``POSE_MODEL``, ``SAMPLE_FPS``,
   ``BATCH_SIZE``, ``ENABLE_TTA``, ``STREAM_SOURCE``, ``MAX_UPLOAD_MB``,
   ``HOST``, ``PORT``).

Relative paths are resolved against the project root, so the app works no
matter which directory it is started from.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.yaml"


DEFAULTS: Dict[str, Any] = {
    "detector": {
        "onnx": "models/dfine_s_coco.onnx",
        "pose": "models/pose_landmarker_lite.task",
        "batch_size": 8,
        "tta": False,
        "overlap": 0.18,
        "tile_w": None,
        "tile_h": None,
        "enable_sonar": True,
        "enable_water_mask": True,
        "max_pose_tracks": 8,
        "use_learned_risk_model": True,
    },
    "processing": {
        "sample_fps": 2.0,
        "max_upload_mb": 500,
        "force_underwater": False,
    },
    "stream": {
        "source": "",
        "max_width": 960,
        "min_process_interval": 0.7,
    },
    "geofence": {
        "enabled": None,  # None = use the value in configs/geofence.json
        "config_file": "configs/geofence.json",
    },
    "server": {
        "host": "127.0.0.1",
        "port": 5000,
        "secret_key": "aquaguard-development-key",
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else _as_bool(raw)


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML config merged with defaults and environment overrides."""
    config = copy.deepcopy(DEFAULTS)

    candidate = Path(path) if path else DEFAULT_CONFIG_PATH
    if path is None:
        env_path = os.environ.get("AQUAGUARD_CONFIG")
        if env_path:
            candidate = Path(env_path)

    if candidate.is_file():
        try:
            import yaml

            with candidate.open("r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh) or {}
            if not isinstance(loaded, dict):
                print(f"[config] {candidate} is not a mapping; ignoring.")
            else:
                config = _deep_merge(config, loaded)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[config] Could not read {candidate}: {exc}")
    elif path is not None:
        raise FileNotFoundError(f"Config file not found: {candidate}")

    detector = config.setdefault("detector", {})
    processing = config.setdefault("processing", {})
    stream = config.setdefault("stream", {})
    server = config.setdefault("server", {})

    # Environment overrides.
    detector["onnx"] = os.environ.get("DFINE_ONNX", detector.get("onnx"))
    detector["pose"] = os.environ.get("POSE_MODEL", detector.get("pose"))
    if "BATCH_SIZE" in os.environ:
        detector["batch_size"] = int(os.environ["BATCH_SIZE"])
    detector["tta"] = _env_bool("ENABLE_TTA", bool(detector.get("tta")))

    if "SAMPLE_FPS" in os.environ:
        processing["sample_fps"] = float(os.environ["SAMPLE_FPS"])
    if "MAX_UPLOAD_MB" in os.environ:
        processing["max_upload_mb"] = int(os.environ["MAX_UPLOAD_MB"])

    stream["source"] = os.environ.get("STREAM_SOURCE", stream.get("source", ""))
    server["host"] = os.environ.get("HOST", server.get("host", "127.0.0.1"))
    if "PORT" in os.environ:
        server["port"] = int(os.environ["PORT"])

    # Resolve model paths so a different CWD does not break startup.
    for key in ("onnx", "pose"):
        value = detector.get(key)
        if value and not Path(value).is_absolute():
            detector[key] = str(PROJECT_ROOT / value)

    return config


def pipeline_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a loaded config into ``PipelineConfig`` keyword arguments."""
    from detection_pipeline import PipelineConfig

    detector = config.get("detector", {})
    processing = config.get("processing", {})
    defaults = PipelineConfig()
    return {
        "onnx_path": detector.get("onnx", defaults.onnx_path),
        "pose_model_path": detector.get("pose", defaults.pose_model_path),
        "tile_w": detector.get("tile_w", defaults.tile_w),
        "tile_h": detector.get("tile_h", defaults.tile_h),
        "overlap": detector.get("overlap", defaults.overlap),
        "batch_size": int(detector.get("batch_size", defaults.batch_size)),
        "tta": _as_bool(detector.get("tta", defaults.tta)),
        "enable_sonar": _as_bool(detector.get("enable_sonar", defaults.enable_sonar)),
        "enable_water_mask": _as_bool(
            detector.get("enable_water_mask", defaults.enable_water_mask)
        ),
        "max_pose_tracks": int(
            detector.get("max_pose_tracks", defaults.max_pose_tracks)
        ),
        "use_learned_risk_model": _as_bool(
            detector.get("use_learned_risk_model", defaults.use_learned_risk_model)
        ),
        "force_underwater": _as_bool(
            processing.get("force_underwater", defaults.force_underwater)
        ),
        "geofence": config.get("geofence", {}),
    }
