"""
water_mask.py
=============
Water / land segmentation used to decide whether a detected person is
actually in the water.

Why this matters
----------------
The previous classifier flagged anyone whose body was horizontal as
"drowning".  On a beach camera most horizontal people are swimmers, and
half of the detections are people walking on sand.  Gating every aquatic
risk cue on an *in-water* measurement removes most of those false alarms.

The estimator is deliberately lightweight (runs on a 320 px wide copy of
the frame) and works on ordinary RGB CCTV / drone / underwater footage.
Sonar and underwater frames are handled by scene type instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np


# --------------------------------------------------
# TUNABLES
# --------------------------------------------------

# HSV ranges (OpenCV: H 0-179, S 0-255, V 0-255)
WATER_HUE_LOW = 74      # green-cyan
WATER_HUE_HIGH = 138    # deep blue
WATER_SAT_MIN = 38      # ignore pale foam / haze
WATER_VAL_MIN = 22

# Fraction of the frame that must look like water before the scene is
# considered aquatic at all.
SCENE_WATER_FRACTION = 0.10

# Portion of a person's box (measured from the bottom) used for the
# "is this person in the water" probe.  Legs are the best indicator.
LOWER_BODY_FRACTION = 0.60

# Temporal smoothing of the mask (0 = no smoothing, 0.9 = very smooth).
DEFAULT_MASK_SMOOTHING = 0.55


@dataclass
class WaterAnalysis:
    mask: Optional[np.ndarray]     # uint8 0/255 at reduced resolution
    fraction: float                # water fraction of the whole frame
    is_water_scene: bool
    reliable: bool                 # False for unknown / grayscale scenes
    scene_type: str = "surface"

    def in_water_ratio(self, box: Sequence[float]) -> Optional[float]:
        """
        Fraction of water pixels inside the lower part of a person box.

        Returns ``None`` when the scene gives us no colour information
        (plain grayscale footage that is not flagged as underwater/sonar).
        """
        # Underwater / sonar: the camera is inside the water, so every
        # person in view is in water by definition.
        if self.scene_type in ("underwater", "sonar"):
            return 1.0

        if not self.reliable or self.mask is None:
            return None

        mask_h, mask_w = self.mask.shape[:2]
        if mask_w == 0 or mask_h == 0:
            return None

        x1, y1, x2, y2 = [float(v) for v in box]
        # Convert frame coordinates to mask coordinates.
        fx = mask_w / max(1, self._frame_width)
        fy = mask_h / max(1, self._frame_height)

        x1 = max(0, int(x1 * fx))
        x2 = min(mask_w, int(np.ceil(x2 * fx)))
        y1 = max(0, int(y1 * fy))
        y2 = min(mask_h, int(np.ceil(y2 * fy)))

        if x2 <= x1 or y2 <= y1:
            return None

        body_h = y2 - y1
        lower_y1 = y2 - max(1, int(body_h * LOWER_BODY_FRACTION))
        lower = self.mask[lower_y1:y2, x1:x2]

        if lower.size == 0:
            return None

        return float(np.count_nonzero(lower)) / float(lower.size)

    # The mask is stored at reduced resolution; remember the original size
    # so boxes can be mapped back correctly.
    _frame_width: int = 1
    _frame_height: int = 1


class WaterMaskEstimator:
    """Segment water in RGB footage and answer per-person water ratios."""

    def __init__(
        self,
        mask_smoothing: float = DEFAULT_MASK_SMOOTHING,
        scene_water_fraction: float = SCENE_WATER_FRACTION,
        working_width: int = 320,
    ):
        self.mask_smoothing = float(np.clip(mask_smoothing, 0.0, 0.95))
        self.scene_water_fraction = scene_water_fraction
        self.working_width = working_width
        self._smoothed_mask: Optional[np.ndarray] = None
        self._last_scene: Optional[str] = None

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------

    def reset(self):
        self._smoothed_mask = None
        self._last_scene = None

    def analyse(
        self,
        frame_bgr: np.ndarray,
        scene_type: str = "surface",
    ) -> WaterAnalysis:
        scene_type = (scene_type or "surface").lower()
        frame_h, frame_w = frame_bgr.shape[:2]

        # Underwater / sonar cameras are *inside* the water: every person
        # in view is treated as being in water.
        if scene_type in ("underwater", "sonar"):
            return WaterAnalysis(
                mask=None,
                fraction=1.0,
                is_water_scene=True,
                reliable=True,
                scene_type=scene_type,
            )

        # A plain grayscale frame could be sonar, night vision or a grey
        # documentary clip.  Colour cues are unavailable -> unreliable.
        if scene_type == "grayscale":
            self._last_scene = scene_type
            blank = np.zeros((frame_h, frame_w), dtype=np.uint8)
            return WaterAnalysis(
                mask=blank,
                fraction=0.0,
                is_water_scene=False,
                reliable=False,
                scene_type=scene_type,
            )

        self._last_scene = scene_type

        mask, fraction = self._segment(frame_bgr)

        # Temporal smoothing to stop wavelets from flickering the decision.
        if self._smoothed_mask is not None and self._smoothed_mask.shape == mask.shape:
            s = self.mask_smoothing
            self._smoothed_mask = cv2.addWeighted(
                mask.astype(np.float32), 1.0 - s, self._smoothed_mask, s, 0,
                dtype=cv2.CV_32F,
            )
        else:
            self._smoothed_mask = mask.astype(np.float32)

        smoothed = (self._smoothed_mask >= 128).astype(np.uint8) * 255

        analysis = WaterAnalysis(
            mask=smoothed,
            fraction=float(fraction),
            is_water_scene=fraction >= self.scene_water_fraction,
            reliable=True,
            scene_type=scene_type,
        )
        analysis._frame_width = frame_w
        analysis._frame_height = frame_h
        return analysis

    def water_fraction(self, frame_bgr: np.ndarray) -> float:
        """Cheap helper used by the scene detector."""
        _, fraction = self._segment(frame_bgr)
        return fraction

    # --------------------------------------------------
    # SEGMENTATION
    # --------------------------------------------------

    def _segment(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]
        if w == 0 or h == 0:
            return np.zeros((1, 1), np.uint8), 0.0

        scale = self.working_width / float(w)
        small = cv2.resize(
            frame_bgr,
            (self.working_width, max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)

        hue_mask = cv2.inRange(hue, WATER_HUE_LOW, WATER_HUE_HIGH)
        sat_mask = cv2.inRange(sat, WATER_SAT_MIN, 255)
        val_mask = cv2.inRange(val, WATER_VAL_MIN, 255)

        water = cv2.bitwise_and(hue_mask, cv2.bitwise_and(sat_mask, val_mask))

        # Clean up: remove speckle, close wave gaps.
        kernel = np.ones((3, 3), np.uint8)
        water = cv2.morphologyEx(water, cv2.MORPH_OPEN, kernel, iterations=1)
        water = cv2.morphologyEx(water, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Keep only reasonably large regions (removes blue clothing etc.).
        water = self._keep_large_components(water, min_area_ratio=0.004)

        fraction = float(np.count_nonzero(water)) / float(water.size)
        return water, fraction

    @staticmethod
    def _keep_large_components(mask: np.ndarray, min_area_ratio: float) -> np.ndarray:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num <= 1:
            return mask

        h, w = mask.shape[:2]
        min_area = max(24, int(min_area_ratio * h * w))
        out = np.zeros_like(mask)

        for label in range(1, num):
            if stats[label, cv2.CC_STAT_AREA] >= min_area:
                out[labels == label] = 255

        return out


# --------------------------------------------------
# STANDALONE DEMO
# --------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python water_mask.py <image>")
        raise SystemExit(1)

    image = cv2.imread(sys.argv[1])
    if image is None:
        raise SystemExit(f"Could not read {sys.argv[1]}")

    estimator = WaterMaskEstimator()
    analysis = estimator.analyse(image, "surface")
    print(f"Water fraction: {analysis.fraction:.3f}")
    print(f"Is water scene: {analysis.is_water_scene}")

    vis = image.copy()
    if analysis.mask is not None:
        mask = cv2.resize(
            analysis.mask, (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        overlay = np.zeros_like(vis)
        overlay[:, :, 2] = mask
        vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)

    cv2.imshow("Water mask", vis)
    cv2.waitKey(0)
