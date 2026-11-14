"""
detection_pipeline.py
=====================
Unified drowning-detection pipeline.

Stages
------
1. ``UnderwaterPreprocessor``  scene detection + colour/contrast repair
2. ``WaterMaskEstimator``      is this person actually in the water?
3. D-FINE-S ONNX               tiled, batched person detection (no YOLO)
4. ``SonarDetector``           classical blob detector fused on sonar frames
5. ``PersonTracker``           multi-object tracking + motion features
6. ``DrowningClassifier``      MediaPipe pose + temporal ``RiskEngine``

Usage
-----
    pipeline = DetectionPipeline(
        onnx_path="models/dfine_s_coco.onnx",
        pose_model_path="models/pose_landmarker_lite.task",
    )
    results = pipeline.process_frame(frame_bgr, frame_index, source_fps)
    annotated = pipeline.annotate_frame(frame_bgr, results)
    pipeline.close()

Every entry in ``results`` contains:
    track_id, risk_level, confidence, score, box, reasons, color,
    movement_status, speed, stationary_duration, in_water, water_ratio,
    cues, scene_type, source, event, ...
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ORT_AVAILABLE = False
    print("[DetectionPipeline] WARNING: onnxruntime not installed.")

from drowning_classifier import DrowningClassifier, PoseFeatures
from geofence import Geofence
from risk_engine import RiskEngine, _noisy_or
from sonar_detector import SonarDetector, fuse_detections, is_probably_sonar
from underwater_preprocessor import UnderwaterPreprocessor
from water_mask import WaterMaskEstimator


# --------------------------------------------------
# DETECTION / TRACKING CONFIGURATION
# --------------------------------------------------

INPUT_SIZE = 640
OVERLAP = 0.22
CONFIDENCE_DEFAULT = 0.40
NMS_THRESHOLD = 0.50

# Tracking
CONFIRM_HITS = 3
STATIC_WINDOW_SECONDS = 5.0
STATIC_THRESHOLD_PIXELS = 10.0
LOW_MOVEMENT_SPEED = 2.0
SUDDEN_SPEED_THRESHOLD = 25.0
SUDDEN_SPEED_CHANGE = 20.0
STATIONARY_ALERT_SECONDS = 5.0

# Alerts
ALERT_COOLDOWN_SECONDS = 8.0

RISK_COLORS = {
    "NORMAL": (0, 200, 80),
    "DISTRESS": (0, 200, 255),
    "DROWNING": (0, 0, 255),
    "UNKNOWN": (180, 180, 180),
}

RISK_PRIORITY = {"DROWNING": 3, "DISTRESS": 2, "NORMAL": 1, "UNKNOWN": 0}


# --------------------------------------------------
# GEOMETRY HELPERS
# --------------------------------------------------

def tile_positions(length: int, tile_size: int, overlap: float) -> List[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, int(tile_size * (1.0 - overlap)))
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def prepare_tile(rgb_tile: np.ndarray):
    image = Image.fromarray(rgb_tile)
    width, height = image.size

    ratio = min(INPUT_SIZE / width, INPUT_SIZE / height)
    new_w, new_h = max(1, int(width * ratio)), max(1, int(height * ratio))

    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    pad_w = (INPUT_SIZE - new_w) // 2
    pad_h = (INPUT_SIZE - new_h) // 2

    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE))
    canvas.paste(resized, (pad_w, pad_h))

    data = np.asarray(canvas, dtype=np.float32) / 255.0
    data = np.transpose(data, (2, 0, 1))

    return data, ratio, pad_w, pad_h


def calculate_iou(a, b) -> float:
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(detections: list, iou_threshold: float = NMS_THRESHOLD) -> list:
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept = []
    while detections:
        best = detections.pop(0)
        kept.append(best)
        detections = [
            item for item in detections
            if item["class_id"] != best["class_id"]
            or calculate_iou(best["box"], item["box"]) < iou_threshold
        ]
    return kept


# --------------------------------------------------
# PERSON TRACKER
# --------------------------------------------------

class PersonTracker:
    """IoU + distance tracker with constant-velocity prediction."""

    def __init__(self, max_age: int = 3, iou_threshold: float = 0.10):
        self.next_id = 1
        self.tracks: Dict[int, dict] = {}
        self.max_age = max_age
        self.iou_threshold = iou_threshold

    @staticmethod
    def center(box):
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    @staticmethod
    def diagonal(box):
        return math.hypot(max(1.0, box[2] - box[0]), max(1.0, box[3] - box[1]))

    def _predicted_box(self, track):
        vx, vy = track.get("velocity", (0.0, 0.0))
        box = track["box"]
        return [box[0] + vx, box[1] + vy, box[2] + vx, box[3] + vy]

    def match_score(self, old_box, new_box):
        iou = calculate_iou(old_box, new_box)
        distance = math.dist(self.center(old_box), self.center(new_box))
        max_dist = max(20.0, self.diagonal(old_box) * 1.6)
        if iou < self.iou_threshold and distance > max_dist:
            return None
        return 0.62 * iou + 0.38 * max(0.0, 1.0 - distance / max_dist)

    # --------------------------------------------------
    # MOVEMENT FEATURES
    # --------------------------------------------------

    def movement_status(self, track, timestamp):
        history = track["history"]
        while history and history[0][0] < timestamp - STATIC_WINDOW_SECONDS:
            history.popleft()

        if len(history) < 2:
            track["previous_speed"] = 0.0
            track["stationary_since"] = None
            return "INSUFFICIENT_HISTORY", 0.0, 0.0, False

        prev_time, prev_center = history[-2]
        curr_center = history[-1][1]
        elapsed = timestamp - prev_time
        speed = math.dist(prev_center, curr_center) / elapsed if elapsed > 0 else 0.0

        # Smooth the raw speed so single-frame jitter does not trigger
        # "sudden movement".
        smoothed = 0.6 * speed + 0.4 * track.get("previous_speed", 0.0)
        previous_smoothed = track.get("previous_smoothed_speed", 0.0)
        track["previous_smoothed_speed"] = smoothed

        sudden = (
            smoothed >= SUDDEN_SPEED_THRESHOLD
            or (len(history) >= 3
                and smoothed - previous_smoothed >= SUDDEN_SPEED_CHANGE)
        )
        track["previous_speed"] = smoothed

        if smoothed <= LOW_MOVEMENT_SPEED:
            if track.get("stationary_since") is None:
                track["stationary_since"] = prev_time
        else:
            track["stationary_since"] = None

        stationary_dur = (
            timestamp - track["stationary_since"]
            if track.get("stationary_since") is not None else 0.0
        )

        possibly_static = False
        if timestamp - history[0][0] >= STATIC_WINDOW_SECONDS:
            first_center = history[0][1]
            max_disp = max(math.dist(first_center, pt) for _, pt in history)
            possibly_static = max_disp <= STATIC_THRESHOLD_PIXELS

        if possibly_static or stationary_dur >= STATIONARY_ALERT_SECONDS:
            status = "POSSIBLY_STATIC"
        elif sudden:
            status = "SUDDEN_MOVEMENT"
        elif smoothed > LOW_MOVEMENT_SPEED:
            status = "MOVING"
        else:
            status = "LOW_MOVEMENT"

        return status, smoothed, stationary_dur, sudden

    # --------------------------------------------------
    # UPDATE
    # --------------------------------------------------

    def update(self, detections, timestamp):
        track_ids = list(self.tracks.keys())
        candidates = []

        for track_id in track_ids:
            old_box = self._predicted_box(self.tracks[track_id])
            for idx, det in enumerate(detections):
                score = self.match_score(old_box, det["box"])
                if score is not None:
                    candidates.append((score, track_id, idx))

        candidates.sort(reverse=True)
        matched_tracks, matched_dets = set(), set()
        assignments = {}

        for score, track_id, idx in candidates:
            if track_id in matched_tracks or idx in matched_dets:
                continue
            track = self.tracks[track_id]
            det = detections[idx]

            if track["missed"] == 0:
                track["consecutive_hits"] += 1
            else:
                track["consecutive_hits"] = 1

            old_center = self.center(track["box"])
            new_center = self.center(det["box"])
            dt = max(1e-3, timestamp - track.get("last_update", timestamp - 0.5))
            track["velocity"] = (
                0.5 * track.get("velocity", (0.0, 0.0))[0]
                + 0.5 * (new_center[0] - old_center[0]) / dt,
                0.5 * track.get("velocity", (0.0, 0.0))[1]
                + 0.5 * (new_center[1] - old_center[1]) / dt,
            )

            track["hits"] += 1
            track["missed"] = 0
            track["box"] = det["box"]
            track["last_update"] = timestamp
            track["history"].append((timestamp, new_center))

            status, speed, stat_dur, sudden = self.movement_status(track, timestamp)

            if track["consecutive_hits"] >= CONFIRM_HITS:
                track["confirmed"] = True

            assignments[idx] = {
                "track_id": track_id,
                "confirmed": track["confirmed"],
                "consecutive_hits": track["consecutive_hits"],
                "movement_status": status,
                "speed": speed,
                "stationary_duration": stat_dur,
                "sudden_movement": sudden,
                "track_age": timestamp - track["created"],
            }
            matched_tracks.add(track_id)
            matched_dets.add(idx)

        for track_id in track_ids:
            if track_id not in matched_tracks:
                self.tracks[track_id]["missed"] += 1
                self.tracks[track_id]["consecutive_hits"] = 0
                self.tracks[track_id]["velocity"] = (0.0, 0.0)

        expired = [tid for tid, t in self.tracks.items() if t["missed"] > self.max_age]
        for tid in expired:
            del self.tracks[tid]

        for idx, det in enumerate(detections):
            if idx in matched_dets:
                continue
            tid = self.next_id
            self.next_id += 1
            center = self.center(det["box"])
            self.tracks[tid] = {
                "box": det["box"],
                "hits": 1,
                "consecutive_hits": 1,
                "missed": 0,
                "confirmed": False,
                "history": deque([(timestamp, center)]),
                "previous_speed": 0.0,
                "previous_smoothed_speed": 0.0,
                "stationary_since": None,
                "velocity": (0.0, 0.0),
                "created": timestamp,
                "last_update": timestamp,
            }
            assignments[idx] = {
                "track_id": tid,
                "confirmed": False,
                "consecutive_hits": 1,
                "movement_status": "INSUFFICIENT_HISTORY",
                "speed": 0.0,
                "stationary_duration": 0.0,
                "sudden_movement": False,
                "track_age": 0.0,
            }

        return assignments

    # --------------------------------------------------
    # INTROSPECTION
    # --------------------------------------------------

    def confirmed_count(self) -> int:
        return sum(1 for t in self.tracks.values() if t["confirmed"])

    def active_count(self) -> int:
        return len(self.tracks)

    def reset(self):
        self.next_id = 1
        self.tracks.clear()


# --------------------------------------------------
# PIPELINE CONFIG
# --------------------------------------------------

@dataclass
class PipelineConfig:
    onnx_path: str = "models/dfine_s_coco.onnx"
    pose_model_path: str = "models/pose_landmarker_lite.task"
    max_age: int = 3
    tile_w: Optional[int] = None
    tile_h: Optional[int] = None
    overlap: float = OVERLAP
    batch_size: int = 8
    tta: bool = False
    enable_sonar: bool = True
    enable_water_mask: bool = True
    force_underwater: bool = False
    max_pose_tracks: int = 8
    use_learned_risk_model: bool = True
    alert_cooldown: float = ALERT_COOLDOWN_SECONDS
    geofence: Optional[dict] = None


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

class DetectionPipeline:
    """End-to-end drowning detection pipeline."""

    def __init__(
        self,
        onnx_path: str = "models/dfine_s_coco.onnx",
        pose_model_path: str = "models/pose_landmarker_lite.task",
        max_age: int = 3,
        config: Optional[PipelineConfig] = None,
    ):
        if config is None:
            config = PipelineConfig(
                onnx_path=onnx_path,
                pose_model_path=pose_model_path,
                max_age=max_age,
            )
        self.config = config

        self._preprocessor = UnderwaterPreprocessor()
        self._water = WaterMaskEstimator()
        self._sonar = SonarDetector()
        self._classifier = DrowningClassifier(
            model_path=config.pose_model_path,
            risk_engine=RiskEngine(
                use_learned_model=config.use_learned_risk_model
            ),
        )
        self._tracker = PersonTracker(max_age=config.max_age)
        self._geofence = Geofence.from_config(
            config.geofence, base_dir=Path(__file__).resolve().parent
        )
        self._session = None
        self._session_lock = threading.Lock()
        self._scene_type = "surface"
        self._conf_threshold = CONFIDENCE_DEFAULT
        self._tile_cache: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self._last_levels: Dict[int, str] = {}
        self._last_alert_time: Dict[Tuple[int, str], float] = {}
        self._alert_lock = threading.Lock()
        self._prev_motion_gray: Optional[np.ndarray] = None
        self._prev_motion_time: Optional[float] = None

        self.alerts: List[dict] = []

        if _ORT_AVAILABLE:
            self._load_model(config.onnx_path)

    # --------------------------------------------------
    # CAMERA MOTION ESTIMATION
    # --------------------------------------------------

    def _estimate_camera_motion(self, frame_bgr: np.ndarray, timestamp: float) -> Optional[Sequence[float]]:
        """Estimate global camera panning/zooming motion using phase correlation.

        Returns [dx, dy] in pixels, or None if estimation fails.
        These values are subtracted from track features in the risk engine
        so that a panning camera does not look like swimming/sinking.
        """
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            if self._prev_motion_gray is None:
                self._prev_motion_gray = gray
                self._prev_motion_time = timestamp
                return [0.0, 0.0]

            # Compute phase correlation between current and previous frame.
            try:
                dx, dy, _ = cv2.phaseCorrelate(self._prev_motion_gray, gray)
            except cv2.error:
                dx, dy = 0.0, 0.0

            # Store current as previous for next frame.
            self._prev_motion_gray = gray
            self._prev_motion_time = timestamp

            return [float(dx), float(dy)]
        except Exception:
            return [0.0, 0.0]

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------

    def _load_model(self, onnx_path: str):
        if not Path(onnx_path).exists():
            raise FileNotFoundError(
                f"D-FINE ONNX model not found: {onnx_path}"
            )

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 0

        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            print("[DetectionPipeline] Using CUDA for inference.")
        else:
            providers = ["CPUExecutionProvider"]
            print("[DetectionPipeline] Using CPU for inference.")

        self._session = ort.InferenceSession(
            onnx_path, sess_options=opts, providers=providers
        )
        print(f"[DetectionPipeline] D-FINE model loaded: {onnx_path}")

    # --------------------------------------------------
    # MAIN ENTRY POINT
    # --------------------------------------------------

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        source_fps: float,
        force_underwater: Optional[bool] = None,
    ) -> List[dict]:
        timestamp = frame_index / max(source_fps, 1.0)

        # Estimate global camera motion once per frame so the risk engine can
        # distinguish a panning camera from a swimmer / sinking person.
        camera_motion = self._estimate_camera_motion(frame_bgr, timestamp)

        force = (
            self.config.force_underwater
            if force_underwater is None else bool(force_underwater)
        )

        # 1. Scene analysis + enhancement
        pre = self._preprocessor.process_detailed(
            frame_bgr, force_underwater=force
        )
        self._scene_type = pre.scene_type
        self._conf_threshold = pre.confidence_threshold

        # 2. Water mask (uses the original frame, not the enhanced one)
        if self.config.enable_water_mask:
            water = self._water.analyse(frame_bgr, pre.scene_type)
        else:
            water = None

        # 3. Detection (persons plus boat/surfboard boxes used for
        #    geofence exclusion of passengers).
        detections, platform_boxes = self._detect(
            pre.frame, pre.confidence_threshold, pre.scene_type
        )

        # 4. Sonar fusion
        if (
            self.config.enable_sonar
            and pre.scene_type in ("sonar", "grayscale")
        ):
            sonar_dets = self._sonar.detect(
                pre.frame, confidence_threshold=pre.confidence_threshold
            )
            if sonar_dets:
                detections = fuse_detections(detections, sonar_dets)

        # 5. Tracking
        assignments = self._tracker.update(detections, timestamp)

        # 6. Pose scheduling: only the largest in-water tracks get MediaPipe
        pose_targets = self._select_pose_targets(
            detections, assignments, water, pre.scene_type
        )

        # 7. Classification
        results: List[dict] = []
        for idx, det in enumerate(detections):
            info = assignments[idx]
            track_id = int(info["track_id"])

            in_water_ratio = None
            if water is not None:
                in_water_ratio = water.in_water_ratio(det["box"])

            pose: Optional[PoseFeatures] = None
            analyze_pose = idx in pose_targets and det.get("source") != "sonar"

            if analyze_pose:
                pose = self._classifier.pose.analyze(
                    frame_bgr, det["box"], track_id, timestamp,
                    allow_cached=True,
                )

            clf = self._classifier.classify(
                frame_bgr=frame_bgr,
                track_info=info,
                box=det["box"],
                timestamp=timestamp,
                in_water_ratio=in_water_ratio,
                scene_type=pre.scene_type,
                analyze_pose=False,
                pose=pose,
                camera_motion=camera_motion,
            )

            # Upward level transition = event (used by the alert log).
            previous = self._last_levels.get(track_id, "UNKNOWN")
            event = RISK_PRIORITY.get(clf.risk_level, 0) > RISK_PRIORITY.get(previous, 0)
            self._last_levels[track_id] = clf.risk_level

            # Geofence (rip-current) check runs for every person track.
            # Platform boxes (boats / surfboards) are passed so passengers
            # are excluded from geofence alerts.
            geofence_state = self._geofence.update(
                track_id=track_id,
                timestamp=timestamp,
                box=det["box"],
                frame_width=frame_bgr.shape[1],
                frame_height=frame_bgr.shape[0],
                confirmed=bool(info["confirmed"]),
                track_age=float(info.get("track_age", 0.0)),
                in_water_ratio=in_water_ratio,
                scene_type=pre.scene_type,
                platform_boxes=platform_boxes,
                water_mask=water,
            )

            entry = {
                "track_id": track_id,
                "risk_level": clf.risk_level,
                "alert_level": (
                    "GEOFENCE" if geofence_state.breached else clf.risk_level
                ),
                "confidence": round(clf.confidence, 3),
                "score": round(clf.score, 3),
                "detection_confidence": round(det["confidence"], 3),
                "box": det["box"],
                "reasons": clf.reasons,
                "color": RISK_COLORS.get(clf.risk_level, RISK_COLORS["UNKNOWN"]),
                "movement_status": info["movement_status"],
                "speed": round(info["speed"], 2),
                "stationary_duration": round(info["stationary_duration"], 2),
                "sudden_movement": info["sudden_movement"],
                "confirmed": info["confirmed"],
                "track_age": round(info.get("track_age", 0.0), 2),
                "body_angle_deg": round(clf.body_angle_deg, 1),
                "pose_visibility": round(clf.pose_visibility_fraction, 2),
                "arm_asymmetry": round(clf.arm_asymmetry_ratio, 2),
                "head_below_hip": clf.head_below_hip,
                "in_water": clf.in_water,
                "water_ratio": (
                    round(in_water_ratio, 3)
                    if in_water_ratio is not None else None
                ),
                "cues": {k: round(v, 3) for k, v in clf.cues.items()},
                "features": dict(clf.features),
                "scene_type": pre.scene_type,
                "source": det.get("source", "dfine"),
                "event": bool(event),
                "geofence_breach": bool(geofence_state.breached),
                "geofence_event": bool(geofence_state.event),
                "geofence_zone": geofence_state.zone,
                "geofence_reason": geofence_state.reason,
                "on_platform": bool(geofence_state.excluded),
                "timestamp": round(timestamp, 3),
                "frame_index": frame_index,
            }

            self._maybe_record_alert(entry, timestamp)
            results.append(entry)

        # Forget state for tracks that the tracker dropped.
        self._prune_state()

        return results

    # --------------------------------------------------
    # POSE TARGET SELECTION
    # --------------------------------------------------

    def _select_pose_targets(
        self,
        detections: List[dict],
        assignments: Dict[int, dict],
        water,
        scene_type: str,
    ) -> set:
        if not self._classifier.pose.available:
            return set()

        candidates = []
        for idx, det in enumerate(detections):
            box = det["box"]
            h = box[3] - box[1]
            w = box[2] - box[0]
            if h < 44 or w < 18:
                continue
            if det.get("source") == "sonar":
                continue

            ratio = None
            if water is not None:
                ratio = water.in_water_ratio(box)

            if scene_type in ("underwater", "sonar"):
                in_water = True
            elif ratio is not None:
                in_water = ratio >= 0.30
            else:
                in_water = True  # unknown scene: do not exclude

            if not in_water:
                continue
            candidates.append((h * w, idx))

        candidates.sort(reverse=True)
        return {idx for _, idx in candidates[: self.config.max_pose_tracks]}

    # --------------------------------------------------
    # ALERTS
    # --------------------------------------------------

    def _maybe_record_alert(self, entry: dict, timestamp: float):
        """Record risk and geofence alerts independently.

        A swimmer can cross the geofence and later start drowning; both
        events matter, so neither alert type should suppress the other.
        """
        if not entry["confirmed"]:
            return

        track_id = entry["track_id"]
        pending = []  # (kind, reasons, zone)

        # Geofence is edge-triggered: one alert per crossing.  The
        # annotation keeps showing the breach until it clears.
        if entry.get("geofence_breach") and entry.get("geofence_event"):
            pending.append((
                "GEOFENCE",
                entry.get("geofence_reason") or "Geofence breach",
                entry.get("geofence_zone"),
            ))

        # Drowning / distress use the existing event + cooldown rule.
        if entry["risk_level"] in ("DROWNING", "DISTRESS"):
            kind = entry["risk_level"]
            key = (track_id, kind)
            with self._alert_lock:
                last = self._last_alert_time.get(key, -1e9)
                allow = bool(entry["event"]) or (
                    timestamp - last >= self.config.alert_cooldown
                )
                if allow:
                    self._last_alert_time[key] = timestamp
            if allow:
                pending.append((
                    kind,
                    "; ".join(entry["reasons"][:4]),
                    None,
                ))

        if not pending:
            return

        with self._alert_lock:
            for kind, reasons, zone in pending:
                self.alerts.append({
                    "timestamp": entry["timestamp"],
                    "frame": entry["frame_index"],
                    "track_id": track_id,
                    "risk_level": kind,
                    "alert_type": kind,
                    "geofence_zone": zone,
                    "confidence": entry["confidence"],
                    "score": entry["score"],
                    "in_water": entry["in_water"],
                    "reasons": reasons,
                })

    # --------------------------------------------------
    # DETECTION
    # --------------------------------------------------

    def _tile_size(self, width: int, height: int) -> Tuple[int, int]:
        key = (width, height)
        if key in self._tile_cache:
            return self._tile_cache[key]

        if self.config.tile_w and self.config.tile_h:
            size = (self.config.tile_w, self.config.tile_h)
        elif width <= 1920:
            size = (800, 640)
        elif width <= 3000:
            size = (1120, 800)
        else:
            size = (1280, 960)

        size = (min(size[0], width), min(size[1], height))
        self._tile_cache[key] = size
        return size

    def _detect(
        self,
        frame_bgr: np.ndarray,
        confidence_threshold: float,
        scene_type: str = "surface",
    ) -> Tuple[List[dict], List[dict]]:
        """Run tiled D-FINE inference.

        Returns ``(person_detections, platform_boxes)``.  Platform boxes are
        COCO boats / surfboards and are used only to exclude passengers from
        geofence alerts; they are never tracked or classified as people.
        """
        if self._session is None:
            return [], []

        platform_classes = set()
        if self._geofence.active and self._geofence.exclude_platforms:
            platform_classes = set(self._geofence.platform_class_ids)
        keep_classes = {0} | platform_classes

        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        tile_w, tile_h = self._tile_size(width, height)
        xs = tile_positions(width, tile_w, self.config.overlap)
        ys = tile_positions(height, tile_h, self.config.overlap)

        tiles = []
        for y in ys:
            for x in xs:
                tile = rgb[y: min(y + tile_h, height), x: min(x + tile_w, width)]
                if tile.size == 0:
                    continue
                data, ratio, pad_w, pad_h = prepare_tile(tile)
                tiles.append((data, ratio, pad_w, pad_h, x, y))

        if not tiles:
            return [], []

        detections: List[dict] = []
        batch_size = max(1, self.config.batch_size)

        for start in range(0, len(tiles), batch_size):
            batch = tiles[start:start + batch_size]
            images = np.stack([item[0] for item in batch], axis=0)
            sizes = np.tile(
                np.array([[INPUT_SIZE, INPUT_SIZE]], dtype=np.int64),
                (len(batch), 1),
            )

            with self._session_lock:
                labels, boxes, scores = self._session.run(
                    None,
                    {"images": images, "orig_target_sizes": sizes},
                )

            for b, (_, ratio, pad_w, pad_h, tile_x, tile_y) in enumerate(batch):
                for label, box, score in zip(labels[b], boxes[b], scores[b]):
                    conf = float(score)
                    class_id = int(label)
                    if class_id not in keep_classes or conf < confidence_threshold:
                        continue
                    self._append_box(
                        detections, box, conf, ratio, pad_w, pad_h,
                        tile_x, tile_y, width, height, source="dfine",
                        class_id=class_id,
                    )

            # Test-time augmentation (horizontal flip) in the same batch.
            if self.config.tta:
                flipped = np.ascontiguousarray(images[:, :, :, ::-1])
                with self._session_lock:
                    labels_f, boxes_f, scores_f = self._session.run(
                        None,
                        {"images": flipped, "orig_target_sizes": sizes},
                    )
                for b, (_, ratio, pad_w, pad_h, tile_x, tile_y) in enumerate(batch):
                    for label, box, score in zip(
                        labels_f[b], boxes_f[b], scores_f[b]
                    ):
                        conf = float(score) * 0.98
                        class_id = int(label)
                        if class_id not in keep_classes or conf < confidence_threshold:
                            continue
                        # Undo the horizontal flip in the letterboxed space.
                        x1 = INPUT_SIZE - float(box[2])
                        x2 = INPUT_SIZE - float(box[0])
                        box_unflipped = [x1, box[1], x2, box[3]]
                        self._append_box(
                            detections, box_unflipped, conf, ratio,
                            pad_w, pad_h, tile_x, tile_y, width, height,
                            source="dfine_tta", class_id=class_id,
                        )

        kept = nms(detections)
        persons = [d for d in kept if d["class_id"] == 0]
        platforms = [d for d in kept if d["class_id"] in platform_classes]
        return persons, [d["box"] for d in platforms]

    def _append_box(
        self,
        detections: List[dict],
        box,
        confidence: float,
        ratio: float,
        pad_w: int,
        pad_h: int,
        tile_x: int,
        tile_y: int,
        width: int,
        height: int,
        source: str,
        class_id: int = 0,
    ):
        x1 = max(0.0, min(width, (float(box[0]) - pad_w) / ratio + tile_x))
        y1 = max(0.0, min(height, (float(box[1]) - pad_h) / ratio + tile_y))
        x2 = max(0.0, min(width, (float(box[2]) - pad_w) / ratio + tile_x))
        y2 = max(0.0, min(height, (float(box[3]) - pad_h) / ratio + tile_y))

        if x2 - x1 < 2 or y2 - y1 < 2:
            return

        detections.append({
            "class_id": int(class_id),
            "confidence": confidence,
            "box": [x1, y1, x2, y2],
            "source": source,
        })

    def _prune_state(self):
        active = set(self._tracker.tracks.keys())
        for tid in list(self._last_levels.keys()):
            if tid not in active:
                self._last_levels.pop(tid, None)
                self._classifier.forget_track(tid)
                self._geofence.forget_track(tid)
        for key in list(self._last_alert_time.keys()):
            if key[0] not in active:
                self._last_alert_time.pop(key, None)

    # --------------------------------------------------
    # ANNOTATION
    # --------------------------------------------------

    def _draw_geofence_zones(self, annotated: np.ndarray):
        """Draw the configured safe / danger polygons or dynamic water boundary onto the frame."""
        gf = self._geofence
        if not gf.active:
            return

        h_img, w_img = annotated.shape[:2]

        # 1. Draw static polygon zones if configured
        for zone in gf.zones:
            pts = np.asarray(
                [
                    [
                        int(max(0.0, min(1.0, x)) * w_img),
                        int(max(0.0, min(1.0, y)) * h_img),
                    ]
                    for x, y in zone.points
                ],
                dtype=np.int32,
            )
            if len(pts) < 3:
                continue

            if zone.is_danger:
                color = (0, 0, 220)
                alpha = 0.16
                label = f"DANGER / RIP: {zone.name}"
            else:
                color = (0, 190, 90)
                alpha = 0.10
                label = f"SAFE ZONE: {zone.name}"

            overlay = annotated.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0, annotated)
            cv2.polylines(annotated, [pts], True, color, 2)

            lx, ly = pts[np.argmin(pts[:, 1])]
            cv2.putText(
                annotated, label[:52],
                (int(lx) + 4, max(16, int(ly) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1,
            )

        # 2. Draw Dynamic Water Geofence Boundary
        if gf.dynamic_water_mode:
            scene = (self._scene_type or "surface").lower()
            is_deep_pure_water = scene in ("underwater", "sonar")

            if self._water is not None and hasattr(self._water, "_smoothed_mask") and self._water._smoothed_mask is not None:
                mask = (self._water._smoothed_mask >= 128).astype(np.uint8) * 255
                row_bottom = mask[-max(1, mask.shape[0]//5):, :]
                if np.mean(row_bottom > 128) > 0.85:
                    is_deep_pure_water = True

            if is_deep_pure_water:
                # Direct bottom of the video frame
                line_y = int(0.95 * h_img)
                cv2.line(annotated, (0, line_y), (w_img, line_y), (0, 0, 255), 3)
                cv2.putText(
                    annotated, "DYNAMIC GEOFENCE: BOTTOM BOUNDARY (NO LAND)",
                    (20, line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                )
            else:
                # Land observed: compute shoreline and 4-5cm offset boundary
                if self._water is not None and hasattr(self._water, "_smoothed_mask") and self._water._smoothed_mask is not None:
                    mask = (self._water._smoothed_mask >= 128).astype(np.uint8) * 255
                    mh, mw = mask.shape
                    boundary_pts = []
                    shore_pts = []
                    step = max(1, mw // 64)
                    for c in range(0, mw, step):
                        col_water = np.where(mask[:, c] > 128)[0]
                        if len(col_water) > 0:
                            s_y = float(col_water.max()) / float(mh)
                        else:
                            s_y = 0.55
                        b_y = s_y - gf.water_offset_ratio
                        fx = int((c / mw) * w_img)
                        shore_pts.append([fx, int(s_y * h_img)])
                        boundary_pts.append([fx, int(max(0.0, b_y) * h_img)])

                    # Draw green dashed/smooth shoreline
                    if len(shore_pts) > 1:
                        cv2.polylines(annotated, [np.array(shore_pts, dtype=np.int32)], False, (0, 220, 100), 2)
                        cv2.putText(
                            annotated, "OBSERVED SHORELINE",
                            (shore_pts[0][0] + 10, shore_pts[0][1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 100), 1
                        )
                    # Draw red/orange 4-5cm offset geofence boundary line
                    if len(boundary_pts) > 1:
                        cv2.polylines(annotated, [np.array(boundary_pts, dtype=np.int32)], False, (0, 140, 255), 3)
                        cv2.putText(
                            annotated, "DYNAMIC GEOFENCE: 4-5CM WATER BOUNDARY",
                            (boundary_pts[0][0] + 10, max(25, boundary_pts[0][1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2
                        )

    def annotate_frame(
        self,
        frame_bgr: np.ndarray,
        results: List[dict],
        scene_type: Optional[str] = None,
        show_cues: bool = True,
    ) -> np.ndarray:
        annotated = frame_bgr.copy()
        scene = (scene_type or self._scene_type or "surface").upper()
        h_img, w_img = annotated.shape[:2]

        # Geofence zones go underneath the person boxes.
        self._draw_geofence_zones(annotated)

        # Scene badge
        badge_text = f"Scene: {scene}"
        badge_color = (170, 90, 0) if scene in ("UNDERWATER", "SONAR") else (60, 60, 60)
        cv2.rectangle(annotated, (w_img - 250, 10), (w_img - 10, 42), badge_color, -1)
        cv2.putText(
            annotated, badge_text, (w_img - 244, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2,
        )

        counts = {"DROWNING": 0, "DISTRESS": 0, "NORMAL": 0, "UNKNOWN": 0}
        geofence_count = 0
        geofence_color = (0, 140, 255)

        for entry in results:
            x1, y1, x2, y2 = [int(v) for v in entry["box"]]
            risk = entry["risk_level"]
            color = entry["color"]
            tid = entry["track_id"]
            conf = entry["confidence"]
            confirmed = entry["confirmed"]
            breach = bool(entry.get("geofence_breach"))
            if breach:
                geofence_count += 1

            counts[risk] = counts.get(risk, 0) + 1

            box_color = geofence_color if breach else color
            thickness = 4 if (breach or risk == "DROWNING") else (
                3 if risk == "DISTRESS" else 2
            )
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

            tag = {"DROWNING": "[!]", "DISTRESS": "[?]"}.get(risk, "")
            if breach:
                label = f"[GEO] {risk} #{tid}"
            else:
                label = f"{tag} {risk} #{tid}"
            if confirmed:
                label += f" {conf:.0%}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.53, 2)
            lx = x1
            ly = max(th + 8, y1 - 4)
            cv2.rectangle(annotated, (lx, ly - th - 5), (lx + tw + 5, ly), box_color, -1)
            cv2.putText(
                annotated, label, (lx + 2, ly - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.53, (0, 0, 0), 2,
            )

            # Geofence anchor marker (waterline / feet point).
            if breach:
                if self._geofence.anchor == "center":
                    ax = int((x1 + x2) * 0.5)
                    ay = int((y1 + y2) * 0.5)
                else:
                    ax = int((x1 + x2) * 0.5)
                    ay = int(y2)
                cv2.circle(annotated, (ax, ay), 7, geofence_color, -1)
                cv2.circle(annotated, (ax, ay), 10, (255, 255, 255), 2)

            # Water state chip
            water_ratio = entry.get("water_ratio")
            if water_ratio is not None:
                water_text = f"water {water_ratio:.0%}"
                cv2.putText(
                    annotated, water_text, (x1 + 2, min(h_img - 4, y2 + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.39,
                    (220, 220, 220) if entry.get("in_water") else (150, 150, 150), 1,
                )

            if breach and entry.get("geofence_reason"):
                cv2.putText(
                    annotated, entry["geofence_reason"][:70],
                    (x1, min(h_img - 18, y2 + 31)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, geofence_color, 1,
                )
            elif show_cues and risk in ("DROWNING", "DISTRESS") and entry["reasons"]:
                reason_text = entry["reasons"][0][:64]
                cv2.putText(
                    annotated, reason_text,
                    (x1, min(h_img - 18, y2 + 31)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1,
                )

        if counts["DROWNING"] > 0:
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (w_img, 48), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
            cv2.putText(
                annotated,
                f"  *** DROWNING ALERT — {counts['DROWNING']} person(s) at risk! ***",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2,
            )
        elif geofence_count > 0:
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (w_img, 48), (0, 120, 220), -1)
            cv2.addWeighted(overlay, 0.72, annotated, 0.28, 0, annotated)
            cv2.putText(
                annotated,
                f"  GEOFENCE BREACH — {geofence_count} person(s) outside safe area "
                "(possible rip current)",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2,
            )
        elif counts["DISTRESS"] > 0:
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (w_img, 40), (0, 130, 200), -1)
            cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)
            cv2.putText(
                annotated,
                f"  DISTRESS detected — {counts['DISTRESS']} person(s) need attention",
                (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2,
            )

        stats_text = (
            f"Drowning: {counts['DROWNING']}  "
            f"Distress: {counts['DISTRESS']}  "
            f"Geofence: {geofence_count}  "
            f"Normal: {counts['NORMAL']}  "
            f"Tracks: {self._tracker.active_count()}"
        )
        cv2.putText(
            annotated, stats_text, (10, h_img - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1,
        )

        return annotated

    # --------------------------------------------------
    # LIFECYCLE
    # --------------------------------------------------

    def reset(self):
        self._tracker.reset()
        self._classifier.reset()
        self._water.reset()
        self._sonar.reset()
        self._geofence.reset()
        self._last_levels.clear()
        self._last_alert_time.clear()
        self._prev_motion_gray = None
        self._prev_motion_time = None
        with self._alert_lock:
            self.alerts.clear()

    # --------------------------------------------------
    # GEOFENCE CONFIGURATION
    # --------------------------------------------------

    @property
    def geofence(self) -> Geofence:
        return self._geofence

    def set_geofence(self, payload: Optional[dict]) -> dict:
        """Replace the geofence at runtime (used by the web UI endpoint)."""
        self._geofence = Geofence.from_config(
            payload, base_dir=Path(__file__).resolve().parent
        )
        return self._geofence.as_dict()

    @property
    def scene_type(self) -> str:
        return self._scene_type

    @property
    def confidence_threshold(self) -> float:
        return self._conf_threshold

    def close(self):
        self._classifier.close()


# --------------------------------------------------
# OPTIONAL: IMAGE-ONLY HELPER
# --------------------------------------------------

def detect_image(
    pipeline: DetectionPipeline,
    image_bgr: np.ndarray,
    force_underwater: bool = False,
) -> Tuple[List[dict], np.ndarray]:
    """Run detection once on a still image and return annotated output."""
    results = pipeline.process_frame(
        image_bgr, frame_index=0, source_fps=1.0,
        force_underwater=force_underwater,
    )
    return results, pipeline.annotate_frame(image_bgr, results)
