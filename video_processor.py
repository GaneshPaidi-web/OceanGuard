"""
video_processor.py
==================
End-to-end file processing for the Flask app and the CLI.

``VideoProcessor`` walks a video, runs :class:`detection_pipeline.DetectionPipeline`
on sampled frames, writes an annotated MP4, builds a per-detection CSV and a
de-duplicated alert log, and returns a result dictionary the UI can render.

Everything runs on CPU if necessary; the sampling rate is the main speed
knob.  ``sample_fps=2`` is enough for drowning behaviour, which unfolds over
seconds rather than frames.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2

from detection_pipeline import (
    DetectionPipeline,
    RISK_PRIORITY,
)


# --------------------------------------------------
# FFMPEG DISCOVERY
# --------------------------------------------------

KNOWN_FFMPEG = (
    r"C:\Users\GANESH PAIDI\AppData\Local\Microsoft\WinGet"
    r"\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe"
)


def find_ffmpeg() -> Optional[str]:
    env = os.environ.get("FFMPEG_PATH")
    if env and Path(env).is_file():
        return env

    found = shutil.which("ffmpeg")
    if found:
        return found

    if Path(KNOWN_FFMPEG).is_file():
        return KNOWN_FFMPEG

    return None


# --------------------------------------------------
# RESULT
# --------------------------------------------------

@dataclass
class ProcessingResult:
    success: bool = False
    error: Optional[str] = None
    output_name: Optional[str] = None
    preview_name: Optional[str] = None
    csv_name: Optional[str] = None
    alerts_csv_name: Optional[str] = None
    alerts: List[dict] = field(default_factory=list)
    stats: Dict[str, object] = field(default_factory=dict)
    tracks: List[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "success": self.success,
            "error": self.error,
            "output_name": self.output_name,
            "preview_name": self.preview_name,
            "csv_name": self.csv_name,
            "alerts_csv_name": self.alerts_csv_name,
            "alerts": self.alerts,
            "stats": self.stats,
            "tracks": self.tracks,
        }


ProgressCallback = Callable[[float, str], None]


# --------------------------------------------------
# PROCESSOR
# --------------------------------------------------

class VideoProcessor:
    def __init__(self, pipeline: DetectionPipeline):
        self.pipeline = pipeline

    # --------------------------------------------------
    # VIDEO
    # --------------------------------------------------

    def process(
        self,
        input_path: str,
        output_dir: str,
        sample_fps: float = 2.0,
        max_frames: int = 0,
        force_underwater: Optional[bool] = None,
        progress: Optional[ProgressCallback] = None,
        job_id: Optional[str] = None,
        dump_features: Optional[str] = None,
    ) -> ProcessingResult:
        input_path = str(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        video = cv2.VideoCapture(input_path)
        if not video.isOpened():
            return ProcessingResult(success=False, error="Could not open video.")

        source_fps = float(video.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if source_fps <= 0:
            source_fps = 25.0
        if width <= 0 or height <= 0:
            video.release()
            return ProcessingResult(success=False, error="Invalid video dimensions.")

        duration = total_frames / source_fps if total_frames else 0.0

        frame_step = max(1, int(round(source_fps / max(0.1, sample_fps))))
        output_fps = source_fps / frame_step

        run_id = job_id or uuid.uuid4().hex[:12]
        base = Path(input_path).stem or "video"

        output_name = f"{run_id}_{base}_processed.mp4"
        output_path = output_dir / output_name
        preview_name = f"{run_id}_{base}_preview.jpg"
        preview_path = output_dir / preview_name
        csv_name = f"{run_id}_{base}_detections.csv"
        csv_path = output_dir / csv_name
        alerts_csv_name = f"{run_id}_{base}_alerts.csv"
        alerts_csv_path = output_dir / alerts_csv_name

        temp_name = f"{run_id}_{base}_temp.mp4"
        temp_path = output_dir / temp_name

        writer = cv2.VideoWriter(
            str(temp_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(1.0, output_fps),
            (width, height),
        )

        if not writer.isOpened():
            video.release()
            return ProcessingResult(success=False, error="Could not create output video.")

        self.pipeline.reset()

        alerts: List[dict] = []
        alerted: set = set()
        prev_levels: Dict[int, str] = {}
        track_summary: Dict[int, dict] = {}
        max_risk_seen: Dict[int, str] = {}

        frame_index = 0
        sample_count = 0
        total_detections = 0
        started = time.perf_counter()
        last_progress = 0.0

        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "frame_index", "timestamp_seconds", "track_id", "confirmed",
            "risk_level", "risk_score", "risk_confidence",
            "detection_confidence", "movement_status", "speed_px_s",
            "stationary_seconds", "in_water", "water_ratio",
            "body_angle_deg", "pose_visibility", "source",
            "geofence_breach", "geofence_zone", "on_platform",
            "x1", "y1", "x2", "y2", "reasons",
        ])

        # Optional feature dump for tools/train_risk_model.py.  The ``label``
        # column is deliberately left empty: label it by hand, or train with
        # ``--pseudo-labels`` to calibrate against the rule engine.
        feature_file = None
        feature_writer = None
        feature_columns = []
        if dump_features:
            from risk_engine import FEATURE_ORDER

            feature_columns = list(FEATURE_ORDER)
            feature_path = Path(dump_features)
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            feature_file = feature_path.open("w", newline="", encoding="utf-8")
            feature_writer = csv.writer(feature_file)
            feature_writer.writerow(
                [
                    "source", "frame_index", "timestamp_seconds",
                    "track_id", "label", "risk_level", "risk_score",
                    "confirmed", "in_water", "scene_type",
                ] + feature_columns
            )

        try:
            while True:
                ok, frame = video.read()
                if not ok:
                    break

                if frame_index % frame_step != 0:
                    frame_index += 1
                    continue

                timestamp = frame_index / source_fps

                results = self.pipeline.process_frame(
                    frame,
                    frame_index=frame_index,
                    source_fps=source_fps,
                    force_underwater=force_underwater,
                )

                annotated = self.pipeline.annotate_frame(frame, results)
                writer.write(annotated)

                if sample_count == 0:
                    cv2.imwrite(str(preview_path), annotated)

                for entry in results:
                    tid = int(entry["track_id"])
                    level = entry["risk_level"]

                    # Per-track summary
                    summary = track_summary.setdefault(tid, {
                        "track_id": tid,
                        "first_seen": entry["timestamp"],
                        "last_seen": entry["timestamp"],
                        "max_risk": "NORMAL",
                        "in_water": False,
                        "geofence": False,
                        "best_confidence": 0.0,
                        "reasons": [],
                    })
                    summary["last_seen"] = entry["timestamp"]
                    summary["in_water"] = summary["in_water"] or bool(entry["in_water"])
                    summary["geofence"] = summary["geofence"] or bool(
                        entry.get("geofence_breach")
                    )
                    if RISK_PRIORITY.get(level, 0) > RISK_PRIORITY.get(summary["max_risk"], 0):
                        summary["max_risk"] = level
                        summary["best_confidence"] = entry["confidence"]
                        summary["reasons"] = entry["reasons"][:3]
                    max_risk_seen[tid] = summary["max_risk"]

                    # Alert on upward transition only
                    previous = prev_levels.get(tid, "UNKNOWN")
                    if (
                        RISK_PRIORITY.get(level, 0) > RISK_PRIORITY.get(previous, 0)
                        and level in ("DROWNING", "DISTRESS")
                        and entry["confirmed"]
                    ):
                        key = (tid, level)
                        if key not in alerted:
                            alerted.add(key)
                            alerts.append({
                                "timestamp": entry["timestamp"],
                                "frame": entry["frame_index"],
                                "track_id": tid,
                                "risk_level": level,
                                "alert_type": "RISK",
                                "geofence_zone": None,
                                "confidence": entry["confidence"],
                                "score": entry["score"],
                                "in_water": entry["in_water"],
                                "reasons": "; ".join(entry["reasons"][:4]),
                            })
                    # Geofence (rip-current) alert: one per track crossing.
                    if (
                        entry.get("geofence_breach")
                        and entry.get("geofence_event")
                        and entry["confirmed"]
                    ):
                        key = (tid, "GEOFENCE")
                        if key not in alerted:
                            alerted.add(key)
                            alerts.append({
                                "timestamp": entry["timestamp"],
                                "frame": entry["frame_index"],
                                "track_id": tid,
                                "risk_level": "GEOFENCE",
                                "alert_type": "GEOFENCE",
                                "geofence_zone": entry.get("geofence_zone"),
                                "confidence": entry["confidence"],
                                "score": entry["score"],
                                "in_water": entry["in_water"],
                                "reasons": (
                                    entry.get("geofence_reason")
                                    or "Person crossed the safety geofence"
                                ),
                            })
                    prev_levels[tid] = level

                    csv_writer.writerow([
                        entry["frame_index"],
                        entry["timestamp"],
                        tid,
                        entry["confirmed"],
                        level,
                        entry["score"],
                        entry["confidence"],
                        entry["detection_confidence"],
                        entry["movement_status"],
                        entry["speed"],
                        entry["stationary_duration"],
                        entry["in_water"],
                        entry["water_ratio"],
                        entry["body_angle_deg"],
                        entry["pose_visibility"],
                        entry["source"],
                        entry.get("geofence_breach", False),
                        entry.get("geofence_zone") or "",
                        entry.get("on_platform", False),
                        round(entry["box"][0], 2),
                        round(entry["box"][1], 2),
                        round(entry["box"][2], 2),
                        round(entry["box"][3], 2),
                        " | ".join(entry["reasons"][:4]),
                    ])

                    if feature_writer is not None:
                        feats = entry.get("features") or {}
                        feature_writer.writerow(
                            [
                                Path(input_path).name,
                                entry["frame_index"],
                                entry["timestamp"],
                                tid,
                                "",
                                level,
                                entry["score"],
                                entry["confirmed"],
                                entry["in_water"],
                                entry.get("scene_type", ""),
                            ]
                            + [feats.get(name, 0.0) for name in feature_columns]
                        )

                total_detections += len(results)
                sample_count += 1

                if progress is not None:
                    if total_frames > 0:
                        fraction = min(0.97, frame_index / total_frames)
                    else:
                        fraction = 0.5
                    # Throttle callbacks to ~1 % steps.
                    if fraction - last_progress >= 0.01:
                        last_progress = fraction
                        progress(
                            fraction,
                            f"Analysing frame {frame_index}/{total_frames or '?'} "
                            f"({len(results)} detections)",
                        )

                if max_frames and sample_count >= max_frames:
                    break

                frame_index += 1

        except Exception as exc:
            video.release()
            writer.release()
            csv_file.close()
            if feature_file is not None:
                feature_file.close()
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            return ProcessingResult(success=False, error=f"Processing failed: {exc}")

        finally:
            video.release()
            if writer is not None:
                writer.release()
            if not csv_file.closed:
                csv_file.close()
            if feature_file is not None and not feature_file.closed:
                feature_file.close()

        if sample_count == 0:
            temp_path.unlink(missing_ok=True)
            return ProcessingResult(success=False, error="No video frames could be processed.")

        # --------------------------------------------------
        # ENCODE
        # --------------------------------------------------
        if progress is not None:
            progress(0.98, "Encoding output video…")

        encoded = self._encode_h264(temp_path, output_path)
        if encoded:
            temp_path.unlink(missing_ok=True)
        elif temp_path.exists():
            # Fall back to the mp4v file so the UI still has a video.
            shutil.move(str(temp_path), str(output_path))

        if not output_path.exists():
            return ProcessingResult(success=False, error="Could not write output video.")

        # --------------------------------------------------
        # ALERT CSV
        # --------------------------------------------------
        with alerts_csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer_csv = csv.writer(fh)
            writer_csv.writerow([
                "timestamp_seconds", "frame", "track_id", "risk_level",
                "alert_type", "geofence_zone", "confidence", "score",
                "in_water", "reasons",
            ])
            for alert in alerts:
                writer_csv.writerow([
                    alert["timestamp"], alert["frame"], alert["track_id"],
                    alert["risk_level"], alert.get("alert_type", "RISK"),
                    alert.get("geofence_zone") or "",
                    alert["confidence"], alert["score"],
                    alert["in_water"], alert["reasons"],
                ])

        # --------------------------------------------------
        # STATS
        # --------------------------------------------------
        drowning_tracks = {s["track_id"] for s in track_summary.values()
                           if s["max_risk"] == "DROWNING"}
        distress_tracks = {s["track_id"] for s in track_summary.values()
                           if s["max_risk"] == "DISTRESS"}
        normal_tracks = {s["track_id"] for s in track_summary.values()
                         if s["max_risk"] == "NORMAL"}
        unknown_tracks = {s["track_id"] for s in track_summary.values()
                          if s["max_risk"] == "UNKNOWN"}

        elapsed = time.perf_counter() - started
        stats = {
            "source_name": Path(input_path).name,
            "width": width,
            "height": height,
            "fps": round(source_fps, 2),
            "output_fps": round(output_fps, 2),
            "frame_count": total_frames,
            "processed_count": sample_count,
            "duration": round(duration, 2),
            "sample_fps": round(sample_fps, 2),
            "elapsed_seconds": round(elapsed, 1),
            "throughput_fps": round(sample_count / elapsed, 2) if elapsed > 0 else 0.0,
            "scene_type": self.pipeline.scene_type,
            "conf_threshold": self.pipeline.confidence_threshold,
            "total_detections": total_detections,
            "unique_tracks": len(track_summary),
            "confirmed_tracks": sum(1 for s in track_summary.values()
                                    if s["max_risk"] != "UNKNOWN"),
            "drowning_count": len(drowning_tracks),
            "distress_count": len(distress_tracks),
            "normal_count": len(normal_tracks),
            "unknown_count": len(unknown_tracks),
            "geofence_count": sum(
                1 for s in track_summary.values() if s.get("geofence")
            ),
            "geofence_alert_count": sum(
                1 for a in alerts if a.get("alert_type") == "GEOFENCE"
            ),
            "water_scene": self.pipeline.scene_type in
                           ("underwater", "sonar") or
                           any(s["in_water"] for s in track_summary.values()),
            "alert_count": len(alerts),
        }

        tracks = sorted(
            track_summary.values(),
            key=lambda s: (RISK_PRIORITY.get(s["max_risk"], 0), s["last_seen"]),
            reverse=True,
        )

        if progress is not None:
            progress(1.0, "Done")

        return ProcessingResult(
            success=True,
            output_name=output_name,
            preview_name=preview_name if preview_path.exists() else None,
            csv_name=csv_name,
            alerts_csv_name=alerts_csv_name,
            alerts=alerts,
            stats=stats,
            tracks=tracks,
        )

    # --------------------------------------------------
    # IMAGE
    # --------------------------------------------------

    def process_image(
        self,
        input_path: str,
        output_dir: str,
        force_underwater: Optional[bool] = None,
        progress: Optional[ProgressCallback] = None,
        job_id: Optional[str] = None,
    ) -> ProcessingResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(str(input_path))
        if image is None:
            return ProcessingResult(success=False, error="Could not read image.")

        run_id = job_id or uuid.uuid4().hex[:12]
        base = Path(input_path).stem or "image"
        output_name = f"{run_id}_{base}_processed.jpg"
        output_path = output_dir / output_name

        self.pipeline.reset()
        if progress is not None:
            progress(0.2, "Preparing image…")

        results = self.pipeline.process_frame(
            image, frame_index=0, source_fps=1.0,
            force_underwater=force_underwater,
        )

        # Still images get a pose-only assessment in addition to the
        # movement-based result (which is UNKNOWN without history).
        from water_mask import WaterMaskEstimator

        water = WaterMaskEstimator().analyse(image, self.pipeline.scene_type)
        for entry in results:
            if entry["risk_level"] == "UNKNOWN":
                static = self.pipeline._classifier.classify_static(
                    image,
                    entry["box"],
                    in_water_ratio=(
                        water.in_water_ratio(entry["box"])
                        if water.reliable else
                        (1.0 if self.pipeline.scene_type in ("underwater", "sonar") else None)
                    ),
                    scene_type=self.pipeline.scene_type,
                )
                entry["risk_level"] = static.risk_level
                entry["confidence"] = round(static.confidence, 3)
                entry["reasons"] = static.reasons
                entry["score"] = round(static.score, 3)
                entry["in_water"] = static.in_water
                entry["color"] = {
                    "NORMAL": (0, 200, 80),
                    "DISTRESS": (0, 200, 255),
                    "DROWNING": (0, 0, 255),
                    "UNKNOWN": (180, 180, 180),
                }[static.risk_level]

        annotated = self.pipeline.annotate_frame(image, results)
        cv2.imwrite(str(output_path), annotated)

        alerts = []
        for entry in results:
            if entry["risk_level"] in ("DROWNING", "DISTRESS"):
                alerts.append({
                    "timestamp": 0.0,
                    "frame": 0,
                    "track_id": entry["track_id"],
                    "risk_level": entry["risk_level"],
                    "alert_type": "RISK",
                    "geofence_zone": None,
                    "confidence": entry["confidence"],
                    "score": entry["score"],
                    "in_water": entry["in_water"],
                    "reasons": "; ".join(entry["reasons"][:4]),
                })

        if progress is not None:
            progress(1.0, "Done")

        return ProcessingResult(
            success=True,
            output_name=output_name,
            preview_name=output_name,
            alerts=alerts,
            stats={
                "source_name": Path(input_path).name,
                "width": image.shape[1],
                "height": image.shape[0],
                "scene_type": self.pipeline.scene_type,
                "conf_threshold": self.pipeline.confidence_threshold,
                "unique_tracks": len(results),
                "total_detections": len(results),
                "alert_count": len(alerts),
                "drowning_count": sum(1 for r in results if r["risk_level"] == "DROWNING"),
                "distress_count": sum(1 for r in results if r["risk_level"] == "DISTRESS"),
                "normal_count": sum(1 for r in results if r["risk_level"] == "NORMAL"),
            },
        )

    # --------------------------------------------------
    # ENCODING
    # --------------------------------------------------

    @staticmethod
    def _encode_h264(source: Path, destination: Path) -> bool:
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            return False

        command = [
            ffmpeg, "-y", "-loglevel", "error",
            "-i", str(source),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            str(destination),
        ]
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=1800
            )
            return result.returncode == 0 and destination.exists()
        except Exception:
            return False


# --------------------------------------------------
# CLI
# --------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the full drowning-detection pipeline on a file."
    )
    parser.add_argument("input")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--config", default=None,
                        help="YAML config (default: configs/pipeline.yaml)")
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--pose", default=None)
    parser.add_argument("--sample-fps", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--force-underwater", action="store_true")
    parser.add_argument("--tta", action="store_true", help="Enable flip TTA (slower)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--dump-features",
        default=None,
        help="Write per-track risk features to this CSV for "
             "tools/train_risk_model.py",
    )

    args = parser.parse_args()

    from config import load_config, pipeline_kwargs
    from detection_pipeline import PipelineConfig

    loaded = load_config(args.config)
    kwargs = pipeline_kwargs(loaded)
    if args.onnx:
        kwargs["onnx_path"] = args.onnx
    if args.pose:
        kwargs["pose_model_path"] = args.pose
    if args.tta:
        kwargs["tta"] = True
    if args.batch_size:
        kwargs["batch_size"] = args.batch_size
    if args.force_underwater:
        kwargs["force_underwater"] = True

    sample_fps = (
        args.sample_fps
        if args.sample_fps is not None
        else float(loaded["processing"]["sample_fps"])
    )

    config = PipelineConfig(**kwargs)
    pipeline = DetectionPipeline(config=config)
    processor = VideoProcessor(pipeline)

    def report(fraction: float, message: str):
        print(f"[{fraction*100:5.1f}%] {message}")

    path = Path(args.input)
    force = True if args.force_underwater else None
    if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
        result = processor.process_image(
            str(path), args.output_dir,
            force_underwater=force, progress=report,
        )
    else:
        result = processor.process(
            str(path), args.output_dir,
            sample_fps=sample_fps,
            max_frames=args.max_frames,
            force_underwater=force,
            progress=report,
            dump_features=args.dump_features,
        )

    pipeline.close()

    if not result.success:
        print("ERROR:", result.error)
        raise SystemExit(1)

    print("\nDone.")
    for key, value in result.stats.items():
        print(f"  {key}: {value}")
    print(f"  output: {result.output_name}")
    if result.alerts:
        print(f"  alerts: {len(result.alerts)}")
        for alert in result.alerts[:10]:
            print(f"    {alert['timestamp']:>7.2f}s  #{alert['track_id']}  "
                  f"{alert['risk_level']}  {alert['reasons']}")


if __name__ == "__main__":
    main()
