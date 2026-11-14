"""
risk_engine.py
==============
Temporal drowning-risk scoring.

The original classifier made a hard decision on every frame using
thresholds such as "body angle < 30 deg -> drowning".  That produces a
large number of false alarms because swimmers are horizontal, because
tiny distant detections jitter, and because single-frame pose estimates
are noisy.

This module replaces those rules with:

1. **Scale-invariant motion features** measured in body lengths per
   second, so the same thresholds work for a swimmer 15 px tall and one
   400 px tall.
2. **Multiple independent cues** (inactivity, thrashing in place, vertical
   bobbing, sinking, head submersion, pose loss, arm asymmetry, horizontal
   posture without forward progress) combined with a noisy-OR model.
3. **Temporal smoothing + hysteresis** so a level must be sustained before
   it is reported and must clear before the alert is dropped.
4. **Water gating** so people on sand are not scored as drowning.
5. Optional **learned calibration**: if ``models/risk_model.pkl`` exists
   (produced by ``tools/train_risk_model.py``) its probability is blended
   with the rule score.

The public API is:

    engine = RiskEngine()
    assessment = engine.update(track_id=..., timestamp=..., box=..., ...)
    assessment.level  # NORMAL / DISTRESS / DROWNING / UNKNOWN
"""

from __future__ import annotations

import math
import pickle
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

HISTORY_SECONDS = 9.0
POSE_HISTORY_SECONDS = 9.0

# EMA time constant for the risk score (seconds).
SCORE_TIME_CONSTANT = 0.85

# Level thresholds on the smoothed score.
DROWNING_ENTER = 0.62
DROWNING_EXIT = 0.42
DISTRESS_ENTER = 0.36
DISTRESS_EXIT = 0.24

# How long a threshold must be held.
DROWNING_HOLD_SECONDS = 1.10
DISTRESS_HOLD_SECONDS = 0.55
DROWNING_EXIT_SECONDS = 2.20
DISTRESS_EXIT_SECONDS = 1.20

MIN_TRACK_AGE_SECONDS = 0.70

# Water gating.
IN_WATER_THRESHOLD = 0.35
LAND_SCORE_MULTIPLIER = 0.12

# Minimum number of critical cues to allow a fast escalation.
FAST_ESCALATION_CRITICAL_CUES = 2

# Feature order for the optional learned model.  Keep this stable.
FEATURE_ORDER = (
    "speed_norm",
    "stationary_seconds",
    "straightness",
    "bob_std_norm",
    "bob_cycles",
    "sink_slope_norm",
    "water_ratio",
    "head_below",
    "pose_lost",
    "arm_asymmetry",
    "body_horizontal",
    "ema_score",
)


# --------------------------------------------------
# RESULT
# --------------------------------------------------

@dataclass
class RiskAssessment:
    level: str = "UNKNOWN"
    score: float = 0.0
    confidence: float = 0.0
    reasons: List[str] = field(default_factory=list)
    cues: Dict[str, float] = field(default_factory=dict)
    in_water: bool = False
    water_ratio: Optional[float] = None
    critical_cues: int = 0
    event: bool = False          # True on the frame where DROWNING begins
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class _TrackState:
    track_id: int
    history: Deque[Tuple[float, float, float, float, float]] = field(
        default_factory=lambda: deque(maxlen=128)
    )  # (t, cx, cy, height, speed_px_s)
    created: float = 0.0
    last_t: Optional[float] = None
    ema: float = 0.0
    level: str = "UNKNOWN"
    level_since: float = 0.0
    high_since: Optional[float] = None
    distress_since: Optional[float] = None
    low_since: Optional[float] = None
    was_high: bool = False
    last_pose_vis: float = 0.0
    pose_lost_since: Optional[float] = None


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def _smoothstep(value: float, edge0: float, edge1: float) -> float:
    if edge1 <= edge0:
        return 1.0 if value >= edge1 else 0.0
    x = min(1.0, max(0.0, (value - edge0) / (edge1 - edge0)))
    return x * x * (3.0 - 2.0 * x)


def _noisy_or(weights: Sequence[float]) -> float:
    product = 1.0
    for w in weights:
        product *= (1.0 - min(1.0, max(0.0, w)))
    return 1.0 - product


def _linear_slope(values: Sequence[float], times: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    t = t - t.mean()
    denominator = float(np.sum(t * t))
    if denominator <= 1e-9:
        return 0.0
    return float(np.sum(t * (v - v.mean())) / denominator)


# --------------------------------------------------
# ENGINE
# --------------------------------------------------

class RiskEngine:
    """Accumulates per-track evidence and emits a stable risk level."""

    def __init__(
        self,
        model_path: Optional[str] = "models/risk_model.pkl",
        use_learned_model: bool = True,
        drowning_enter: float = DROWNING_ENTER,
        drowning_exit: float = DROWNING_EXIT,
        distress_enter: float = DISTRESS_ENTER,
        distress_exit: float = DISTRESS_EXIT,
    ):
        self._states: Dict[int, _TrackState] = {}
        self._model = None
        self._model_features = None

        self.drowning_enter = drowning_enter
        self.drowning_exit = drowning_exit
        self.distress_enter = distress_enter
        self.distress_exit = distress_exit

        if use_learned_model and model_path:
            self.load_model(model_path)

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------

    def load_model(self, model_path: str) -> bool:
        path = Path(model_path)
        if not path.exists():
            return False
        try:
            with path.open("rb") as fh:
                bundle = pickle.load(fh)
            if isinstance(bundle, dict) and "model" in bundle:
                self._model = bundle["model"]
                self._model_features = bundle.get("features", list(FEATURE_ORDER))
            else:
                self._model = bundle
                self._model_features = list(FEATURE_ORDER)
            print(f"[RiskEngine] Loaded learned risk model: {model_path}")
            return True
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[RiskEngine] Could not load risk model ({exc}); using rules.")
            self._model = None
            return False

    @property
    def has_learned_model(self) -> bool:
        return self._model is not None

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------

    def reset(self):
        self._states.clear()

    def forget_track(self, track_id: int):
        self._states.pop(track_id, None)

    def update(
        self,
        track_id: int,
        timestamp: float,
        box: Sequence[float],
        speed_px_s: float = 0.0,
        stationary_duration: float = 0.0,
        sudden_movement: bool = False,
        confirmed: bool = True,
        in_water_ratio: Optional[float] = None,
        scene_type: str = "surface",
        pose: Optional[dict] = None,
        body_horizontal_progress: Optional[float] = None,
        camera_motion: Optional[Sequence[float]] = None,
    ) -> RiskAssessment:
        """
        Update risk for one track.

        Parameters
        ----------
        box : [x1, y1, x2, y2]
        speed_px_s : centroid speed in pixels / second
        stationary_duration : seconds the track has been nearly still
        in_water_ratio : 0..1 from WaterMaskEstimator, or None if unknown
        pose : dict with visibility_fraction, body_angle_deg,
               head_below_hip, arm_asymmetry, head_visible
        body_horizontal_progress : optional forward speed (body lengths/s)
        """
        x1, y1, x2, y2 = [float(v) for v in box]
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        scale = max(1.0, 0.5 * (height + width))
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5

        state = self._states.get(track_id)
        if state is None:
            state = _TrackState(track_id=track_id, created=timestamp)
            self._states[track_id] = state

        if state.last_t is None:
            dt = 0.0
        else:
            dt = max(1e-3, timestamp - state.last_t)
        state.last_t = timestamp

        state.history.append((timestamp, cx, cy, height, float(speed_px_s)))
        while state.history and state.history[0][0] < timestamp - HISTORY_SECONDS:
            state.history.popleft()

        history = list(state.history)
        track_age = timestamp - state.created

        # ---------------- camera motion compensation ----------------
        # A panning / zooming camera moves every box, which otherwise looks
        # like swimming or sinking -- especially for small distant people.
        if camera_motion is not None and len(camera_motion) >= 2:
            camera_dx = float(camera_motion[0] or 0.0)
            camera_dy = float(camera_motion[1] or 0.0)
        else:
            camera_dx = 0.0
            camera_dy = 0.0
        camera_speed_norm = math.hypot(camera_dx, camera_dy) / scale

        # ---------------- scale-invariant motion features ----------------
        # Remove the global scene velocity from the track velocity.
        speed_norm = max(0.0, float(speed_px_s) / scale - camera_speed_norm)

        recent = [h for h in history if h[0] >= timestamp - 3.0]
        straightness = 1.0
        if len(recent) >= 2:
            path = 0.0
            for prev, curr in zip(recent[:-1], recent[1:]):
                path += math.dist(prev[1:3], curr[1:3])
            net = math.dist(recent[0][1:3], recent[-1][1:3])
            straightness = net / path if path > 1e-6 else 1.0

        horizontal_speed_norm = 0.0
        if len(recent) >= 2:
            dt_recent = max(1e-3, recent[-1][0] - recent[0][0])
            horizontal_speed_norm = max(
                0.0,
                abs(recent[-1][1] - recent[0][1]) / dt_recent / scale
                - abs(camera_dx) / scale,
            )
        if body_horizontal_progress is not None:
            horizontal_speed_norm = float(body_horizontal_progress)

        # Vertical bobbing (detrended centroid-y oscillation in body lengths)
        bob_std = 0.0
        bob_cycles = 0
        window = [h for h in history if h[0] >= timestamp - 4.0]
        if len(window) >= 5:
            ys = np.asarray([h[2] for h in window], dtype=np.float64)
            ts = np.asarray([h[0] for h in window], dtype=np.float64)
            slope = _linear_slope(ys, ts)
            detrended = ys - (ts - ts[0]) * slope
            bob_std = float(np.std(detrended)) / scale
            signs = np.sign(detrended - np.mean(detrended))
            signs = signs[signs != 0]
            if len(signs) >= 2:
                bob_cycles = int(np.sum(signs[1:] != signs[:-1]))

        # Sinking trend: positive means moving down the image.
        sink_slope = 0.0
        if len(window) >= 3:
            sink_slope = (
                _linear_slope(
                    [h[2] for h in window], [h[0] for h in window]
                ) - camera_dy
            ) / scale

        # ---------------- water gating ----------------
        scene = (scene_type or "surface").lower()
        if scene in ("underwater", "sonar"):
            water_ratio = 1.0
        else:
            water_ratio = in_water_ratio
        in_water = water_ratio is not None and water_ratio >= IN_WATER_THRESHOLD

        # ---------------- cue weights ----------------
        cues: Dict[str, float] = {}
        active_notes: List[Tuple[float, str]] = []

        if in_water:
            # 1. Prolonged inactivity in the water.
            inactivity = 0.88 * _smoothstep(stationary_duration, 3.0, 12.0)
            if inactivity > 0.05:
                cues["inactivity"] = inactivity
                active_notes.append((
                    inactivity,
                    f"No movement for {stationary_duration:.1f}s in water",
                ))

            # 2. Thrashing without progress (struggle in place).
            struggle = (
                0.78
                * _smoothstep(speed_norm, 0.55, 1.60)
                * (1.0 - min(1.0, straightness / 0.45))
            )
            if struggle > 0.08:
                cues["struggle"] = struggle
                active_notes.append((
                    struggle,
                    f"Thrashing in place (speed {speed_norm:.2f} body/s, "
                    f"straightness {straightness:.2f})",
                ))

            # 3. Vertical bobbing (repeated up/down struggle for air).
            # A swimmer strokes forward; bobbing without horizontal progress
            # is the suspicious pattern.
            if (
                bob_cycles >= 3
                and bob_std > 0.05
                and horizontal_speed_norm < 0.18
            ):
                bob = 0.52 * _smoothstep(bob_std, 0.06, 0.30) * min(
                    1.0, bob_cycles / 4.0
                )
                if bob > 0.06:
                    cues["bobbing"] = bob
                    active_notes.append((
                        bob,
                        f"Vertical bobbing {bob_cycles}x "
                        f"(range {bob_std:.2f} body lengths)",
                    ))

            # 4. Sinking.  Only meaningful for a person who is otherwise
            # staying put; a diver / swimmer moving down is not sinking.
            if (
                sink_slope > 0.03
                and speed_norm < 0.22
                and horizontal_speed_norm < 0.18
            ):
                sinking = 0.66 * _smoothstep(sink_slope, 0.03, 0.18)
                if sinking > 0.06:
                    cues["sinking"] = sinking
                    active_notes.append((
                        sinking,
                        f"Downward drift {sink_slope:.2f} body/s (sinking)",
                    ))

        # Pose-based cues work in any scene but need water context to be
        # decisive.  On land they only add a small amount.
        pose = pose or {}
        vis_frac = float(pose.get("visibility_fraction", 0.0) or 0.0)
        body_angle = pose.get("body_angle_deg")
        head_below = bool(pose.get("head_below_hip", False))
        head_visible = bool(pose.get("head_visible", True))
        asymmetry = float(pose.get("arm_asymmetry", 0.0) or 0.0)

        land_factor = 1.0 if in_water else LAND_SCORE_MULTIPLIER

        # Head below the hips is normal for freestyle swimming when the head
        # is still visible.  Only treat it as evidence when the pose is
        # consistent with a face-down body (horizontal torso) or when the
        # head landmarks are missing entirely.
        angle_from_horizontal = 90.0
        if body_angle is not None:
            try:
                angle_from_horizontal = abs(float(body_angle))
            except (TypeError, ValueError):
                angle_from_horizontal = 90.0

        face_down = angle_from_horizontal < 55.0 or vis_frac < 0.40
        if head_below and in_water and not head_visible:
            cues["head_submerged"] = 0.90
            active_notes.append((0.90, "Head/face below hip line while in water"))
        elif head_below and in_water and face_down:
            cues["head_below"] = 0.50
            active_notes.append((0.50, "Head below hip line"))

        # Pose loss after a previously reliable pose.
        if vis_frac > 0.45:
            state.last_pose_vis = vis_frac
            state.pose_lost_since = None
        elif state.last_pose_vis > 0.45 and vis_frac < 0.20:
            if state.pose_lost_since is None:
                state.pose_lost_since = timestamp
            lost_for = timestamp - state.pose_lost_since
            # Losing pose while still swimming is usually just MediaPipe
            # failing on a splash; only count it when the person is not
            # progressing through the water.
            if (
                lost_for >= 0.4
                and speed_norm < 0.30
                and horizontal_speed_norm < 0.22
            ):
                pose_lost = 0.62 * _smoothstep(lost_for, 0.4, 2.0) * land_factor
                cues["pose_lost"] = pose_lost
                active_notes.append((
                    pose_lost,
                    f"Pose lost {lost_for:.1f}s after being visible "
                    "(possible submersion)",
                ))
        else:
            state.pose_lost_since = None

        # Horizontal posture is only suspicious when there is no forward
        # progress.  A swimmer is horizontal AND moving forward.
        body_horizontal = False
        if body_angle is not None:
            # Convention used by PoseAnalyzer: 0 deg = torso horizontal
            # (swimming), 90 deg = torso upright (standing / treading).
            try:
                angle_from_horizontal = abs(float(body_angle))
            except (TypeError, ValueError):
                angle_from_horizontal = 90.0

            body_horizontal = angle_from_horizontal < 28.0
            # A calm float produces this cue; a thrashing swimmer is handled
            # by the "struggle" cue instead.  The weight stays below the
            # DISTRESS threshold so posture alone never raises an alert.
            if (
                body_horizontal
                and horizontal_speed_norm < 0.22
                and speed_norm < 0.30
            ):
                cue = 0.34 * land_factor
                cues["horizontal_no_progress"] = cue
                active_notes.append((
                    cue,
                    "Horizontal posture with no forward progress",
                ))

        # Arm asymmetry: weak on its own, useful when combined.
        if asymmetry >= 3.0:
            asym_cue = 0.26 * _smoothstep(asymmetry, 3.0, 8.0) * (
                1.0 if in_water else 0.3
            )
            cues["arm_asymmetry"] = asym_cue
            active_notes.append((
                asym_cue,
                f"Asymmetric arm thrashing ({asymmetry:.1f}x)",
            ))

        if sudden_movement and in_water:
            cues["sudden_movement"] = 0.18
            active_notes.append((0.18, "Sudden abnormal movement"))

        # ---------------- combine ----------------
        raw_score = _noisy_or(list(cues.values()))
        if not in_water:
            # Non-water cues are already scaled; apply one more damping so
            # a sprinter on the beach never becomes a drowning alert.
            raw_score *= 0.55

        # Learned calibration (optional).
        features = {
            "speed_norm": speed_norm,
            "stationary_seconds": float(stationary_duration),
            "straightness": float(straightness),
            "bob_std_norm": float(bob_std),
            "bob_cycles": float(bob_cycles),
            "sink_slope_norm": float(sink_slope),
            "water_ratio": float(water_ratio) if water_ratio is not None else -1.0,
            "head_below": 1.0 if head_below else 0.0,
            "pose_lost": 1.0 if "pose_lost" in cues else 0.0,
            "arm_asymmetry": float(asymmetry),
            "body_horizontal": 1.0 if body_horizontal else 0.0,
            "ema_score": float(state.ema),
        }
        if self._model is not None:
            try:
                names = self._model_features or list(FEATURE_ORDER)
                vector = np.asarray([[features.get(n, 0.0) for n in names]])
                proba = float(self._model.predict_proba(vector)[0][1])
                raw_score = 0.60 * raw_score + 0.40 * proba
                cues["learned_model"] = proba
            except Exception:
                pass

        # ---------------- temporal smoothing ----------------
        if dt <= 0:
            alpha = 1.0
        else:
            alpha = 1.0 - math.exp(-dt / SCORE_TIME_CONSTANT)
        state.ema = alpha * raw_score + (1.0 - alpha) * state.ema

        critical_names = ("inactivity", "head_submerged", "head_below",
                          "sinking", "pose_lost")
        critical_cues = sum(
            1 for name in critical_names if cues.get(name, 0.0) >= 0.45
        )

        # DROWNING is reserved for situations that include at least one
        # strong, sustained signal.  Bobbing / thrashing alone is common in
        # ordinary swimming and should not escalate past DISTRESS.
        high_evidence = (
            cues.get("inactivity", 0.0) >= 0.40
            or cues.get("head_submerged", 0.0) >= 0.45
            or (
                cues.get("sinking", 0.0) >= 0.45
                and stationary_duration >= 1.0
            )
            or (
                cues.get("pose_lost", 0.0) >= 0.45
                and stationary_duration >= 1.0
            )
        )

        # ---------------- hysteresis ----------------
        now = timestamp
        if state.ema >= self.drowning_enter:
            if state.high_since is None:
                state.high_since = now
        else:
            state.high_since = None

        if state.ema >= self.distress_enter:
            if state.distress_since is None:
                state.distress_since = now
        else:
            state.distress_since = None

        if state.ema <= self.distress_exit:
            if state.low_since is None:
                state.low_since = now
        else:
            state.low_since = None

        if track_age < MIN_TRACK_AGE_SECONDS:
            new_level = "UNKNOWN"
        elif not confirmed:
            new_level = "UNKNOWN"
        else:
            drowning_ready = (
                state.high_since is not None
                and (now - state.high_since) >= DROWNING_HOLD_SECONDS
                and high_evidence
            )
            fast_escalation = (
                critical_cues >= FAST_ESCALATION_CRITICAL_CUES
                and state.ema >= 0.55
                and high_evidence
            )
            distress_ready = (
                state.distress_since is not None
                and (now - state.distress_since) >= DISTRESS_HOLD_SECONDS
            )

            if drowning_ready or fast_escalation:
                new_level = "DROWNING"
            elif distress_ready:
                new_level = "DISTRESS"
            else:
                new_level = "NORMAL"

        # Keep the current level until it has clearly cleared.
        if state.level == "DROWNING" and new_level != "DROWNING":
            cleared_for = (
                now - state.low_since if state.low_since is not None else 0.0
            )
            if state.ema >= self.drowning_exit or cleared_for < DROWNING_EXIT_SECONDS:
                new_level = "DROWNING"
        elif state.level == "DISTRESS" and new_level == "NORMAL":
            cleared_for = (
                now - state.low_since if state.low_since is not None else 0.0
            )
            if state.ema >= self.distress_exit or cleared_for < DISTRESS_EXIT_SECONDS:
                new_level = "DISTRESS"

        event = new_level == "DROWNING" and state.level != "DROWNING"
        if new_level != state.level:
            state.level = new_level
            state.level_since = now

        # ---------------- reasons / confidence ----------------
        active_notes.sort(key=lambda item: item[0], reverse=True)
        reasons = [note for weight, note in active_notes if weight >= 0.05]

        if in_water and water_ratio is not None:
            reasons.insert(0, f"In water ({water_ratio:.0%} water in lower body)")
        elif not in_water and water_ratio is not None:
            reasons.insert(0, f"On land ({1.0 - water_ratio:.0%} non-water)")

        if new_level == "NORMAL" and not reasons:
            reasons = ["Normal movement pattern"]
        elif new_level == "UNKNOWN" and not reasons:
            reasons = ["Building tracking history"]

        if new_level == "DROWNING":
            confidence = min(0.99, 0.58 + 0.38 * state.ema + 0.02 * critical_cues)
        elif new_level == "DISTRESS":
            confidence = min(0.95, 0.42 + 0.45 * state.ema)
        elif new_level == "NORMAL":
            confidence = min(0.97, 0.55 + 0.40 * (1.0 - state.ema))
        else:
            confidence = 0.0

        return RiskAssessment(
            level=new_level,
            score=float(state.ema),
            confidence=float(confidence),
            reasons=reasons,
            cues={k: float(v) for k, v in cues.items()},
            in_water=bool(in_water),
            water_ratio=water_ratio,
            critical_cues=int(critical_cues),
            event=bool(event),
            features=features,
        )


# --------------------------------------------------
# SELF-TEST
# --------------------------------------------------

if __name__ == "__main__":
    engine = RiskEngine(use_learned_model=False)

    # Simulated swimmer: moving forward at a healthy pace in water.
    for i in range(40):
        t = i * 0.5
        result = engine.update(
            track_id=1, timestamp=t, box=[100, 100, 200, 160],
            speed_px_s=55.0, stationary_duration=0.0,
            in_water_ratio=0.9, scene_type="surface",
            pose={"body_angle_deg": 5.0, "visibility_fraction": 0.8,
                  "head_visible": True, "head_below_hip": False,
                  "arm_asymmetry": 1.1},
        )
    print("Swimmer ->", result.level, round(result.score, 3))

    # Simulated drowning: still in water for 15 s, head going under.
    for i in range(40):
        t = 30 + i * 0.5
        result = engine.update(
            track_id=2, timestamp=t, box=[100, 100, 130, 130],
            speed_px_s=1.0, stationary_duration=max(0.0, 15.0),
            in_water_ratio=0.85, scene_type="surface",
            pose={"body_angle_deg": 10.0, "visibility_fraction": 0.25,
                  "head_visible": False, "head_below_hip": True,
                  "arm_asymmetry": 1.0},
        )
    print("Drowning ->", result.level, round(result.score, 3),
          "| conf", round(result.confidence, 2))

    # Simulated person lying on sand (horizontal, no water).
    for i in range(40):
        t = 60 + i * 0.5
        result = engine.update(
            track_id=3, timestamp=t, box=[400, 400, 500, 470],
            speed_px_s=2.0, stationary_duration=12.0,
            in_water_ratio=0.02, scene_type="surface",
            pose={"body_angle_deg": 5.0, "visibility_fraction": 0.9,
                  "head_visible": True, "head_below_hip": False,
                  "arm_asymmetry": 1.0},
        )
    print("Sunbather ->", result.level, round(result.score, 3))
