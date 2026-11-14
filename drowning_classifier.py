"""
drowning_classifier.py
======================
Per-person pose analysis and risk classification.

This module contains two pieces:

``PoseAnalyzer``
    Runs MediaPipe Pose on a person crop and extracts the geometric
    features used by the risk engine: torso angle, head/hip relationship,
    landmark visibility and arm asymmetry.

``DrowningClassifier``
    Thin orchestration layer that feeds pose features + movement history
    into :class:`risk_engine.RiskEngine` and returns a
    :class:`ClassificationResult` compatible with the rest of the project.

Convention
----------
``body_angle_deg`` is measured from the horizontal:
    0  deg -> torso horizontal (swimming / floating)
    90 deg -> torso upright (standing / treading water)
The earlier implementation returned the complement, which classified
upright people as "nearly horizontal" and produced a flood of false
drowning alerts.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _MEDIAPIPE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MEDIAPIPE_AVAILABLE = False

try:
    from risk_engine import RiskEngine, RiskAssessment
    _RISK_ENGINE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RISK_ENGINE_AVAILABLE = False


# --------------------------------------------------
# LANDMARK INDICES (MediaPipe Pose, 33 landmarks)
# --------------------------------------------------

LM_NOSE = 0
LM_LEFT_EAR = 7
LM_RIGHT_EAR = 8
LM_LEFT_SHOULDER = 11
LM_RIGHT_SHOULDER = 12
LM_LEFT_ELBOW = 13
LM_RIGHT_ELBOW = 14
LM_LEFT_WRIST = 15
LM_RIGHT_WRIST = 16
LM_LEFT_HIP = 23
LM_RIGHT_HIP = 24
LM_LEFT_KNEE = 25
LM_RIGHT_KNEE = 26
LM_LEFT_ANKLE = 27
LM_RIGHT_ANKLE = 28

VISIBILITY_MIN = 0.35

# Do not waste time running pose on tiny distant detections.
MIN_POSE_BOX_HEIGHT = 44
MIN_POSE_BOX_WIDTH = 18

# Cache pose features for a track for this long when it is skipped.
POSE_CACHE_SECONDS = 0.8


# --------------------------------------------------
# RESULT TYPES
# --------------------------------------------------

@dataclass
class PoseFeatures:
    visibility_fraction: float = 0.0
    body_angle_deg: float = 90.0
    head_below_hip: bool = False
    head_visible: bool = True
    arm_asymmetry: float = 0.0
    shoulder_width: float = 0.0
    timestamp: float = 0.0
    landmarks: Optional[List[dict]] = None

    def as_dict(self) -> dict:
        return {
            "visibility_fraction": self.visibility_fraction,
            "body_angle_deg": self.body_angle_deg,
            "head_below_hip": self.head_below_hip,
            "head_visible": self.head_visible,
            "arm_asymmetry": self.arm_asymmetry,
            "shoulder_width": self.shoulder_width,
        }


@dataclass
class ClassificationResult:
    risk_level: str = "UNKNOWN"
    confidence: float = 0.0
    reasons: List[str] = field(default_factory=list)
    body_angle_deg: float = 0.0
    pose_visibility_fraction: float = 0.0
    arm_asymmetry_ratio: float = 0.0
    head_below_hip: bool = False
    score: float = 0.0
    in_water: bool = False
    water_ratio: Optional[float] = None
    cues: Dict[str, float] = field(default_factory=dict)
    event: bool = False
    features: Dict[str, float] = field(default_factory=dict)


# --------------------------------------------------
# POSE ANALYZER
# --------------------------------------------------

class PoseAnalyzer:
    """MediaPipe Pose wrapper that extracts compact geometric features."""

    def __init__(
        self,
        model_path: str = "models/pose_landmarker_lite.task",
        min_detection_confidence: float = 0.30,
    ):
        self._model_path = model_path
        self._lock = threading.Lock()
        self._landmarker = None
        self._wrist_history: Dict[int, Deque[Tuple[float, Tuple[float, float], Tuple[float, float]]]] = {}
        self._cache: Dict[int, PoseFeatures] = {}
        self._last_visibility: Dict[int, float] = {}

        if _MEDIAPIPE_AVAILABLE:
            self._init_mediapipe(min_detection_confidence)
        else:
            print(
                "[PoseAnalyzer] WARNING: mediapipe not installed. "
                "Pose cues disabled; movement cues still work."
            )

    def _init_mediapipe(self, min_detection_confidence: float):
        try:
            options = mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=self._model_path
                ),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=min_detection_confidence,
                min_pose_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_detection_confidence,
            )
            self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
            print(f"[PoseAnalyzer] MediaPipe pose loaded: {self._model_path}")
        except Exception as exc:
            print(f"[PoseAnalyzer] Could not load MediaPipe model: {exc}")
            self._landmarker = None

    @property
    def available(self) -> bool:
        return self._landmarker is not None

    def reset(self):
        self._wrist_history.clear()
        self._cache.clear()
        self._last_visibility.clear()

    def forget_track(self, track_id: int):
        self._wrist_history.pop(track_id, None)
        self._cache.pop(track_id, None)
        self._last_visibility.pop(track_id, None)

    def analyze(
        self,
        frame_bgr: np.ndarray,
        box: List[float],
        track_id: int,
        timestamp: float,
        allow_cached: bool = True,
    ) -> Optional[PoseFeatures]:
        """
        Run pose on a crop and return features.

        When ``allow_cached`` is True and the box is too small / pose fails,
        the last successful features are returned if they are recent.  That
        keeps the temporal risk engine from flickering between "pose lost"
        and "pose found" on every other sampled frame.
        """
        if self._landmarker is None:
            return None

        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = (
            max(0, int(box[0])), max(0, int(box[1])),
            min(w, int(box[2])), min(h, int(box[3])),
        )

        box_w, box_h = x2 - x1, y2 - y1
        if box_w < MIN_POSE_BOX_WIDTH or box_h < MIN_POSE_BOX_HEIGHT:
            return self._cached(track_id, timestamp, allow_cached)

        pad_x = int(box_w * 0.12)
        pad_y = int(box_h * 0.10)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)

        crop = frame_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return self._cached(track_id, timestamp, allow_cached)

        # Upscale small crops; MediaPipe accuracy degrades badly below
        # roughly 200 px.
        crop_h, crop_w = crop.shape[:2]
        scale = max(1.0, 220.0 / max(1, max(crop_h, crop_w)))
        if scale > 1.0:
            crop = cv2.resize(
                crop,
                (int(crop_w * scale), int(crop_h * scale)),
                interpolation=cv2.INTER_CUBIC,
            )

        # Mild contrast help for underwater crops.
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_ch = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_ch)
        crop = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            with self._lock:
                detection = self._landmarker.detect(mp_image)
        except Exception:
            return self._cached(track_id, timestamp, allow_cached)

        if not detection.pose_landmarks:
            features = self._cached(track_id, timestamp, allow_cached)
            if features is not None:
                # Keep the old geometry but flag low visibility so the risk
                # engine can recognise a "pose lost" event.
                features = PoseFeatures(
                    visibility_fraction=0.0,
                    body_angle_deg=features.body_angle_deg,
                    head_below_hip=False,
                    head_visible=False,
                    arm_asymmetry=features.arm_asymmetry,
                    shoulder_width=features.shoulder_width,
                    timestamp=timestamp,
                )
            return features

        landmarks_raw = detection.pose_landmarks[0]
        crop_w2, crop_h2 = crop.shape[1], crop.shape[0]
        inv_scale = 1.0 / scale

        landmarks = []
        for lm in landmarks_raw:
            lx = cx1 + (lm.x * crop_w2 * inv_scale)
            ly = cy1 + (lm.y * crop_h2 * inv_scale)
            landmarks.append({
                "x": lx,
                "y": ly,
                "visibility": float(getattr(lm, "visibility", 1.0)),
            })

        features = self._build_features(landmarks, track_id, timestamp)
        features.landmarks = landmarks

        # Cache successful features.
        self._cache[track_id] = features
        self._last_visibility[track_id] = features.visibility_fraction
        return features

    def _cached(self, track_id: int, timestamp: float, allow_cached: bool) -> Optional[PoseFeatures]:
        if not allow_cached:
            return None
        cached = self._cache.get(track_id)
        if cached is None:
            return None
        if timestamp - cached.timestamp > POSE_CACHE_SECONDS:
            return None
        # Return a copy with the new timestamp, preserving the geometry.
        return PoseFeatures(
            visibility_fraction=cached.visibility_fraction,
            body_angle_deg=cached.body_angle_deg,
            head_below_hip=cached.head_below_hip,
            head_visible=cached.head_visible,
            arm_asymmetry=cached.arm_asymmetry,
            shoulder_width=cached.shoulder_width,
            timestamp=timestamp,
            landmarks=cached.landmarks,
        )

    # --------------------------------------------------
    # FEATURE EXTRACTION
    # --------------------------------------------------

    def _build_features(
        self,
        landmarks: List[dict],
        track_id: int,
        timestamp: float,
    ) -> PoseFeatures:
        visible_count = sum(
            1 for lm in landmarks if lm["visibility"] >= VISIBILITY_MIN
        )
        vis_frac = visible_count / max(1, len(landmarks))

        angle = self._body_angle(landmarks)
        head_below, head_visible = self._head_relative_to_hips(landmarks)

        shoulder_width = 0.0
        if len(landmarks) > max(LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER):
            ls = landmarks[LM_LEFT_SHOULDER]
            rs = landmarks[LM_RIGHT_SHOULDER]
            shoulder_width = math.dist((ls["x"], ls["y"]), (rs["x"], rs["y"]))

        asymmetry = self._update_wrist_history(track_id, landmarks, timestamp)

        return PoseFeatures(
            visibility_fraction=vis_frac,
            body_angle_deg=angle if angle is not None else 90.0,
            head_below_hip=head_below,
            head_visible=head_visible,
            arm_asymmetry=asymmetry,
            shoulder_width=shoulder_width,
            timestamp=timestamp,
        )

    @staticmethod
    def _body_angle(landmarks: List[dict]) -> Optional[float]:
        """Torso angle from horizontal (0 = horizontal, 90 = upright)."""
        required = [LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER, LM_LEFT_HIP, LM_RIGHT_HIP]
        if len(landmarks) <= max(required):
            return None

        for idx in required:
            if landmarks[idx]["visibility"] < VISIBILITY_MIN:
                return None

        ls, rs = landmarks[LM_LEFT_SHOULDER], landmarks[LM_RIGHT_SHOULDER]
        lh, rh = landmarks[LM_LEFT_HIP], landmarks[LM_RIGHT_HIP]

        mid_shoulder = ((ls["x"] + rs["x"]) / 2, (ls["y"] + rs["y"]) / 2)
        mid_hip = ((lh["x"] + rh["x"]) / 2, (lh["y"] + rh["y"]) / 2)

        dx = mid_hip[0] - mid_shoulder[0]
        dy = mid_hip[1] - mid_shoulder[1]

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 90.0

        # atan2(|dy|, |dx|): horizontal torso -> 0 deg, vertical -> 90 deg.
        return math.degrees(math.atan2(abs(dy), abs(dx)))


    @staticmethod
    def _head_relative_to_hips(landmarks: List[dict]) -> Tuple[bool, bool]:
        required = [LM_NOSE, LM_LEFT_HIP, LM_RIGHT_HIP]
        if len(landmarks) <= max(required):
            return False, True

        nose = landmarks[LM_NOSE]
        lh, rh = landmarks[LM_LEFT_HIP], landmarks[LM_RIGHT_HIP]

        if lh["visibility"] < VISIBILITY_MIN or rh["visibility"] < VISIBILITY_MIN:
            return False, True

        head_visible = nose["visibility"] >= VISIBILITY_MIN
        mid_hip_y = (lh["y"] + rh["y"]) / 2

        # In image coordinates larger y is lower on screen.
        return (head_visible and nose["y"] > mid_hip_y), head_visible


    def _update_wrist_history(
        self,
        track_id: int,
        landmarks: List[dict],
        timestamp: float,
    ) -> float:
        if len(landmarks) <= max(LM_LEFT_WRIST, LM_RIGHT_WRIST):
            return 0.0

        lw = landmarks[LM_LEFT_WRIST]
        rw = landmarks[LM_RIGHT_WRIST]
        if lw["visibility"] < VISIBILITY_MIN or rw["visibility"] < VISIBILITY_MIN:
            return self._arm_asymmetry(track_id)

        history = self._wrist_history.setdefault(
            track_id, deque(maxlen=15)
        )
        history.append((timestamp, (lw["x"], lw["y"]), (rw["x"], rw["y"])))

        return self._arm_asymmetry(track_id)

    def _arm_asymmetry(self, track_id: int) -> float:
        history = list(self._wrist_history.get(track_id, ()))
        if len(history) < 2:
            return 0.0

        left_speeds, right_speeds = [], []
        for (t0, lw0, rw0), (t1, lw1, rw1) in zip(history[:-1], history[1:]):
            dt = t1 - t0
            if dt <= 0:
                continue
            left_speeds.append(math.dist(lw0, lw1) / dt)
            right_speeds.append(math.dist(rw0, rw1) / dt)

        if not left_speeds or not right_speeds:
            return 0.0

        avg_left = max(sum(left_speeds) / len(left_speeds), 0.5)
        avg_right = max(sum(right_speeds) / len(right_speeds), 0.5)
        return max(avg_left / avg_right, avg_right / avg_left)


# --------------------------------------------------
# CLASSIFIER
# --------------------------------------------------

class DrowningClassifier:
    """Combines pose features and the temporal risk engine."""

    def __init__(
        self,
        model_path: str = "models/pose_landmarker_lite.task",
        risk_engine: Optional["RiskEngine"] = None,
        use_learned_risk_model: bool = True,
    ):
        self.pose = PoseAnalyzer(model_path=model_path)

        if risk_engine is not None:
            self.risk = risk_engine
        elif _RISK_ENGINE_AVAILABLE:
            self.risk = RiskEngine(use_learned_model=use_learned_risk_model)
        else:  # pragma: no cover
            self.risk = None
            print("[DrowningClassifier] WARNING: risk_engine unavailable.")

    # --------------------------------------------------
    # MAIN API
    # --------------------------------------------------

    def classify(
        self,
        frame_bgr: np.ndarray,
        track_info: dict,
        box: List[float],
        timestamp: float,
        in_water_ratio: Optional[float] = None,
        scene_type: str = "surface",
        analyze_pose: bool = True,
        pose: Optional[PoseFeatures] = None,
        camera_motion: Optional[Sequence[float]] = None,
    ) -> ClassificationResult:
        track_id = int(track_info.get("track_id", -1))

        if pose is None and analyze_pose:
            pose = self.pose.analyze(
                frame_bgr, box, track_id, timestamp, allow_cached=True
            )

        features = pose.as_dict() if pose is not None else {}

        if self.risk is None:  # pragma: no cover - fallback
            return ClassificationResult(
                risk_level="UNKNOWN", confidence=0.0,
                reasons=["Risk engine unavailable"],
            )

        assessment: RiskAssessment = self.risk.update(
            track_id=track_id,
            timestamp=timestamp,
            box=box,
            speed_px_s=float(track_info.get("speed", 0.0)),
            stationary_duration=float(track_info.get("stationary_duration", 0.0)),
            sudden_movement=bool(track_info.get("sudden_movement", False)),
            confirmed=bool(track_info.get("confirmed", False)),
            in_water_ratio=in_water_ratio,
            scene_type=scene_type,
            pose=features,
            camera_motion=camera_motion,
        )

        return ClassificationResult(
            risk_level=assessment.level,
            confidence=assessment.confidence,
            reasons=assessment.reasons,
            body_angle_deg=float(features.get("body_angle_deg", 0.0)),
            pose_visibility_fraction=float(features.get("visibility_fraction", 0.0)),
            arm_asymmetry_ratio=float(features.get("arm_asymmetry", 0.0)),
            head_below_hip=bool(features.get("head_below_hip", False)),
            score=assessment.score,
            in_water=assessment.in_water,
            water_ratio=assessment.water_ratio,
            cues=assessment.cues,
            event=assessment.event,
            features=dict(assessment.features),
        )

    # --------------------------------------------------
    # STATIC IMAGE SUPPORT
    # --------------------------------------------------

    def classify_static(
        self,
        frame_bgr: np.ndarray,
        box: List[float],
        in_water_ratio: Optional[float] = None,
        scene_type: str = "surface",
    ) -> ClassificationResult:
        """
        Pose-only assessment for a single photo.

        Without motion history we cannot judge drowning reliably, so the
        output is capped at DISTRESS and marked as a static estimate.
        """
        pose = self.pose.analyze(frame_bgr, box, -1, 0.0, allow_cached=False)
        if pose is None:
            return ClassificationResult(
                risk_level="UNKNOWN",
                confidence=0.0,
                reasons=["No pose detected (too small or occluded)"],
                water_ratio=in_water_ratio,
                in_water=(in_water_ratio or 0.0) >= 0.35
                or scene_type in ("underwater", "sonar"),
            )

        in_water = (
            (in_water_ratio or 0.0) >= 0.35
            or scene_type in ("underwater", "sonar")
        )

        reasons = []
        score = 0.0

        if in_water and pose.head_below_hip and not pose.head_visible:
            score += 0.55
            reasons.append("Head below hip line — possible submersion")
        if in_water and pose.visibility_fraction < 0.25:
            score += 0.30
            reasons.append("Most landmarks invisible — possible submersion")
        if in_water and pose.body_angle_deg < 25.0:
            score += 0.15
            reasons.append("Horizontal posture in water")

        level = "DISTRESS" if score >= 0.45 else ("NORMAL" if in_water else "UNKNOWN")
        if not reasons:
            reasons = ["Static frame — no acute risk visible"]

        return ClassificationResult(
            risk_level=level,
            confidence=min(0.7, 0.35 + score * 0.5),
            reasons=reasons,
            body_angle_deg=pose.body_angle_deg,
            pose_visibility_fraction=pose.visibility_fraction,
            arm_asymmetry_ratio=pose.arm_asymmetry,
            head_below_hip=pose.head_below_hip,
            score=score,
            in_water=in_water,
            water_ratio=in_water_ratio,
        )

    # --------------------------------------------------
    # LIFECYCLE
    # --------------------------------------------------

    def forget_track(self, track_id: int):
        self.pose.forget_track(track_id)
        if self.risk is not None:
            self.risk.forget_track(track_id)

    def reset(self):
        self.pose.reset()
        if self.risk is not None:
            self.risk.reset()

    def close(self):
        if self.pose._landmarker is not None:
            try:
                self.pose._landmarker.close()
            except Exception:
                pass
            self.pose._landmarker = None
