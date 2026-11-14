"""
underwater_preprocessor.py
==========================
Pre-processing for beach CCTV, underwater RGB and sonar footage.

Responsibilities
----------------
* Detect the scene type: ``surface``, ``underwater``, ``grayscale`` or
  ``sonar``.
* Correct the blue-green cast of underwater footage.
* Apply colour-balanced CLAHE (and optional dark-channel dehazing).
* Prepare monochrome / sonar frames for the detector.
* Recommend a detection confidence threshold per scene.

The public API is unchanged::

    prep = UnderwaterPreprocessor()
    enhanced, scene_type, conf_threshold = prep.process(frame_bgr)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


# --------------------------------------------------
# SCENE THRESHOLDS
# --------------------------------------------------

UNDERWATER_HUE_FRACTION = 0.32
HUE_LOWER = 76      # ~cyan-green start
HUE_UPPER = 138     # ~deep blue end

CONF_SURFACE = 0.40
CONF_UNDERWATER = 0.26
CONF_SONAR = 0.18
CONF_GRAYSCALE = 0.30

# Edge density below which a monochrome frame may be sonar.
SONAR_EDGE_DENSITY = 0.09


@dataclass
class PreprocessResult:
    frame: np.ndarray
    scene_type: str
    confidence_threshold: float
    water_fraction: float = 0.0


class UnderwaterPreprocessor:
    """Detect scene type and enhance frames for D-FINE inference."""

    def __init__(
        self,
        clahe_clip_limit: float = 3.0,
        clahe_tile_grid: int = 8,
        underwater_hue_fraction: float = UNDERWATER_HUE_FRACTION,
        enable_dehaze: bool = True,
        enable_denoise: bool = False,
    ):
        self._clip = clahe_clip_limit
        self._tile = clahe_tile_grid
        self._hue_frac = underwater_hue_fraction
        self._enable_dehaze = enable_dehaze
        self._enable_denoise = enable_denoise

        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_tile_grid, clahe_tile_grid),
        )
        self._last_scene: Optional[str] = None

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------

    def process(
        self,
        frame_bgr: np.ndarray,
        force_underwater: bool = False,
    ) -> Tuple[np.ndarray, str, float]:
        """Backwards-compatible tuple API used by the pipeline."""
        result = self.process_detailed(frame_bgr, force_underwater=force_underwater)
        return result.frame, result.scene_type, result.confidence_threshold

    def process_detailed(
        self,
        frame_bgr: np.ndarray,
        force_underwater: bool = False,
    ) -> PreprocessResult:
        scene_type, water_fraction = self.detect_scene(
            frame_bgr, force_underwater=force_underwater
        )

        if scene_type == "sonar":
            enhanced = self._enhance_sonar(frame_bgr)
        elif scene_type == "grayscale":
            enhanced = self._enhance_grayscale(frame_bgr)
        elif scene_type == "underwater":
            enhanced = self._enhance_underwater(frame_bgr)
        else:
            enhanced = self._enhance_surface(frame_bgr)

        thresholds = {
            "surface": CONF_SURFACE,
            "underwater": CONF_UNDERWATER,
            "sonar": CONF_SONAR,
            "grayscale": CONF_GRAYSCALE,
        }

        return PreprocessResult(
            frame=enhanced,
            scene_type=scene_type,
            confidence_threshold=thresholds[scene_type],
            water_fraction=water_fraction,
        )

    # --------------------------------------------------
    # SCENE DETECTION
    # --------------------------------------------------

    def detect_scene(
        self,
        frame_bgr: np.ndarray,
        force_underwater: bool = False,
    ) -> Tuple[str, float]:
        water_fraction = self._water_fraction(frame_bgr)

        if force_underwater:
            self._last_scene = "underwater"
            return "underwater", max(water_fraction, 1.0)

        if self._is_grayscale(frame_bgr):
            if self._looks_like_sonar(frame_bgr):
                self._last_scene = "sonar"
                return "sonar", 1.0
            self._last_scene = "grayscale"
            return "grayscale", 1.0

        if self._is_underwater(frame_bgr):
            self._last_scene = "underwater"
            return "underwater", max(water_fraction, 0.5)

        self._last_scene = "surface"
        return "surface", water_fraction

    def _is_grayscale(self, frame_bgr: np.ndarray) -> bool:
        b, g, r = cv2.split(frame_bgr)
        bg = np.mean(np.abs(b.astype(np.int16) - g.astype(np.int16)))
        gr = np.mean(np.abs(g.astype(np.int16) - r.astype(np.int16)))
        return float(bg + gr) < 6.0

    def _is_underwater(self, frame_bgr: np.ndarray) -> bool:
        """
        Underwater RGB footage fills the frame with blue-green, including the
        bottom band.  A beach shot also has a lot of blue (the sky) but its
        bottom band is sand, which is the cue used here to avoid classifying
        every bright beach scene as underwater.
        """
        small = cv2.resize(frame_bgr, (160, 120), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)
        mask = (
            (hue >= HUE_LOWER) & (hue <= HUE_UPPER)
            & (sat >= 35) & (val >= 20)
        )
        full = float(mask.mean())
        bottom = float(mask[int(mask.shape[0] * 0.60):].mean())
        return full >= 0.55 and bottom >= 0.50

    def _water_fraction(self, frame_bgr: np.ndarray) -> float:
        small = cv2.resize(frame_bgr, (160, 120), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        in_range = (
            (hue >= HUE_LOWER) & (hue <= HUE_UPPER)
            & (sat >= 35) & (val >= 20)
        )
        return float(np.count_nonzero(in_range)) / float(in_range.size)

    def _looks_like_sonar(self, frame_bgr: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)
        edges = cv2.Canny(small, 40, 120)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)

        hist = cv2.calcHist([small], [0], None, [32], [0, 256]).ravel()
        hist /= max(1.0, float(hist.sum()))
        concentration = float(np.max(hist))

        return edge_density < SONAR_EDGE_DENSITY or concentration > 0.32

    # --------------------------------------------------
    # ENHANCEMENT
    # --------------------------------------------------

    def _enhance_surface(self, frame_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_ch = self._clahe.apply(l_ch)
        return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    def _enhance_underwater(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame_float = frame_bgr.astype(np.float32)
        b_ch, g_ch, r_ch = cv2.split(frame_float)

        mean_b = float(np.mean(b_ch)) + 1e-6
        mean_g = float(np.mean(g_ch)) + 1e-6
        mean_r = float(np.mean(r_ch)) + 1e-6

        # Gray-world style red restoration, but bounded so dark scenes are
        # not blown out.
        red_gain = float(np.clip(mean_b / mean_r, 1.0, 2.6))
        green_gain = float(np.clip(mean_b / mean_g, 1.0, 1.8))

        r_ch = np.clip(r_ch * red_gain, 0, 255)
        g_ch = np.clip(g_ch * green_gain, 0, 255)
        corrected = cv2.merge([b_ch, g_ch, r_ch]).astype(np.uint8)

        if self._enable_dehaze:
            corrected = self._dark_channel_dehaze(corrected)

        lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch2 = cv2.split(lab)
        l_ch = self._clahe.apply(l_ch)
        corrected = cv2.cvtColor(
            cv2.merge([l_ch, a_ch, b_ch2]), cv2.COLOR_LAB2BGR
        )

        blurred = cv2.GaussianBlur(corrected, (0, 0), 2.0)
        return cv2.addWeighted(corrected, 1.25, blurred, -0.25, 0)

    def _enhance_grayscale(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        equalized = self._clahe.apply(gray)

        if self._enable_denoise:
            # Median is ~20x faster than non-local means and is enough for
            # compression noise; heavy NLM made long videos unusable.
            equalized = cv2.medianBlur(equalized, 3)

        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    def _enhance_sonar(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Sonar is often very dark with a few bright returns; normalise the
        # robust intensity range before CLAHE.
        low = float(np.percentile(gray, 2))
        high = float(np.percentile(gray, 98))
        if high - low > 8:
            gray = np.clip((gray.astype(np.float32) - low) * (255.0 / (high - low)),
                           0, 255).astype(np.uint8)

        equalized = self._clahe.apply(gray)
        bright = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

        # A subtle false-colour version helps the learned detector, which
        # expects three-channel input.  Detection is still driven by the
        # sonar blob detector.
        coloured = cv2.applyColorMap(equalized, cv2.COLORMAP_TURBO)
        return cv2.addWeighted(bright, 0.55, coloured, 0.45, 0)

    # --------------------------------------------------
    # DEHAZING
    # --------------------------------------------------

    @staticmethod
    def _dark_channel_dehaze(
        frame_bgr: np.ndarray,
        patch_size: int = 7,
        omega: float = 0.75,
        t_min: float = 0.28,
    ) -> np.ndarray:
        """
        Fast single-scale dark-channel-prior dehazing.

        A full guided-filter implementation is overkill for detection
        pre-processing; a box filter gives most of the contrast recovery.
        """
        img = frame_bgr.astype(np.float32) / 255.0
        dark = np.min(img, axis=2)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (patch_size, patch_size)
        )
        dark = cv2.erode(dark, kernel)

        # Atmospheric light from the brightest 0.1 % of the dark channel.
        flat = dark.ravel()
        count = max(1, int(flat.size * 0.001))
        idx = np.argpartition(flat, -count)[-count:]
        atmospheric = np.max(img.reshape(-1, 3)[idx], axis=0) + 1e-6

        transmission = 1.0 - omega * cv2.erode(
            np.min(img / atmospheric.reshape(1, 1, 3), axis=2), kernel
        )
        transmission = np.clip(transmission, t_min, 1.0)
        transmission = cv2.blur(transmission, (patch_size, patch_size))
        transmission = np.clip(transmission, t_min, 1.0)[:, :, None]

        recovered = (img - atmospheric.reshape(1, 1, 3)) / transmission \
            + atmospheric.reshape(1, 1, 3)
        return np.clip(recovered * 255.0, 0, 255).astype(np.uint8)


# --------------------------------------------------
# STANDALONE TEST
# --------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python underwater_preprocessor.py <image_or_video>")
        raise SystemExit(1)

    path = sys.argv[1]
    prep = UnderwaterPreprocessor()

    if path.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
    else:
        frame = cv2.imread(path)
        ok = frame is not None

    if not ok:
        raise SystemExit(f"Could not read {path}")

    result = prep.process_detailed(frame)
    print(f"Scene type            : {result.scene_type}")
    print(f"Water fraction        : {result.water_fraction:.3f}")
    print(f"Recommended threshold : {result.confidence_threshold}")

    cv2.imshow("Original", frame)
    cv2.imshow("Enhanced", result.frame)
    cv2.waitKey(0)
