"""
sonar_detector.py
=================
Detection of diver / swimmer signatures in sonar and monochrome
underwater imagery.

D-FINE is trained on COCO photographs, so its person recall drops sharply
on side-scan / forward-looking sonar returns and on heavily degraded
underwater grayscale video.  This module adds a classical blob detector
that works without training data and fuses its detections with the
learned detector.

Signal model
------------
Sonar returns are bright (or dark) compact blobs against a slowly varying
background.  A large-kernel background estimate is subtracted, the
residual is thresholded adaptively, and connected components are filtered
by size, aspect ratio, fill ratio and edge contrast.  For video, frame
differencing picks up moving targets as an additional cue.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MIN_AREA_RATIO = 0.00035      # of frame area
MAX_AREA_RATIO = 0.12
MIN_WIDTH = 8
MIN_HEIGHT = 12
MIN_ASPECT = 0.12             # w / h
MAX_ASPECT = 7.0
MIN_FILL = 0.22               # contour area / box area
MIN_EXTENT = 0.18

# Limit the number of classical detections to keep the UI readable.
MAX_DETECTIONS = 40


@dataclass
class SonarDetection:
    box: List[float]
    confidence: float
    source: str = "sonar"

    def as_detection(self) -> dict:
        return {
            "class_id": 0,
            "confidence": float(self.confidence),
            "box": [float(v) for v in self.box],
            "source": self.source,
        }


class SonarDetector:
    """Classical blob detector for sonar / monochrome underwater frames."""

    def __init__(
        self,
        min_area_ratio: float = MIN_AREA_RATIO,
        max_area_ratio: float = MAX_AREA_RATIO,
        working_width: int = 960,
        use_motion: bool = True,
    ):
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.working_width = working_width
        self.use_motion = use_motion
        self._prev_gray: Optional[np.ndarray] = None

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------

    def reset(self):
        self._prev_gray = None

    def detect(
        self,
        frame_bgr: np.ndarray,
        confidence_threshold: float = 0.22,
    ) -> List[dict]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        h, w = frame_bgr.shape[:2]
        if w < 16 or h < 16:
            return []

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        scale = min(1.0, self.working_width / float(w))
        if scale < 1.0:
            small = cv2.resize(
                gray,
                (max(16, int(round(w * scale))), max(16, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = gray

        sh, sw = small.shape[:2]

        # Blur only mildly: sonar speckle is high frequency, targets are
        # larger than the speckle.
        base = cv2.GaussianBlur(small, (5, 5), 0)
        background = cv2.GaussianBlur(base, (0, 0), max(7.0, min(sh, sw) / 26.0))
        residual = cv2.absdiff(base, background)

        candidates: List[dict] = []
        candidates += self._components_from(
            base, residual, sw, sh, bright=True
        )
        candidates += self._components_from(
            base, residual, sw, sh, bright=False
        )

        if self.use_motion and self._prev_gray is not None:
            if self._prev_gray.shape == small.shape:
                motion = cv2.absdiff(base, self._prev_gray)
                motion = cv2.GaussianBlur(motion, (5, 5), 0)
                candidates += self._components_from(
                    base, motion, sw, sh, bright=True, motion_only=True
                )

        self._prev_gray = base

        candidates = self._nms(candidates, iou_threshold=0.45)
        candidates.sort(key=lambda d: d["confidence"], reverse=True)
        candidates = candidates[:MAX_DETECTIONS]

        inv = 1.0 / scale if scale > 0 else 1.0
        results = []
        for det in candidates:
            if det["confidence"] < confidence_threshold:
                continue
            x1, y1, x2, y2 = det["box"]
            results.append({
                "class_id": 0,
                "confidence": float(det["confidence"]),
                "box": [
                    float(x1 * inv), float(y1 * inv),
                    float(x2 * inv), float(y2 * inv),
                ],
                "source": "sonar",
            })

        return results

    # --------------------------------------------------
    # COMPONENT EXTRACTION
    # --------------------------------------------------

    def _components_from(
        self,
        gray: np.ndarray,
        response: np.ndarray,
        width: int,
        height: int,
        bright: bool,
        motion_only: bool = False,
    ) -> List[dict]:
        if response.size == 0:
            return []

        resp = response.astype(np.float64)
        resp = cv2.normalize(resp, None, 0, 255, cv2.NORM_MINMAX)
        resp = resp.astype(np.uint8)

        # Otsu on the residual isolates the strongest returns.
        threshold, _ = cv2.threshold(
            resp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        threshold = max(threshold, 18)
        binary = cv2.inRange(resp, int(threshold), 255)

        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        frame_area = float(width * height)
        detections: List[dict] = []

        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw < MIN_WIDTH or bh < MIN_HEIGHT:
                continue

            area = float(cv2.contourArea(contour))
            area_ratio = area / frame_area
            if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                continue

            aspect = bw / max(1.0, float(bh))
            if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
                continue

            fill = area / float(bw * bh)
            if fill < MIN_FILL:
                continue

            # Mean contrast of the blob against its immediate surroundings.
            pad = 3
            rx1, ry1 = max(0, x - pad), max(0, y - pad)
            rx2, ry2 = min(width, x + bw + pad), min(height, y + bh + pad)
            patch = gray[ry1:ry2, rx1:rx2]

            if patch.size == 0:
                continue

            mask = np.zeros_like(patch, dtype=bool)
            mx1, my1 = x - rx1, y - ry1
            mask[my1:my1 + bh, mx1:mx1 + bw] = True

            inside = patch[mask]
            outside = patch[~mask]
            if inside.size == 0 or outside.size == 0:
                continue

            inner_mean = float(np.mean(inside))
            outer_mean = float(np.mean(outside))
            contrast = abs(inner_mean - outer_mean) / 255.0

            if bright and inner_mean < outer_mean and not motion_only:
                # Dark blob handled by the dark pass.
                continue
            if not bright and inner_mean > outer_mean:
                continue

            extent = max(0.0, min(1.0, (area / float(bw * bh) - MIN_EXTENT) /
                                  max(1e-6, 1.0 - MIN_EXTENT)))

            confidence = (
                0.30
                + 0.42 * min(1.0, contrast / 0.22)
                + 0.18 * extent
                + 0.10 * min(1.0, area_ratio / 0.01)
            )
            if motion_only:
                confidence = min(confidence, 0.58)

            detections.append({
                "box": [float(x), float(y), float(x + bw), float(y + bh)],
                "confidence": float(min(0.95, max(0.0, confidence))),
            })

        return detections

    # --------------------------------------------------
    # NMS
    # --------------------------------------------------

    @staticmethod
    def _nms(detections: List[dict], iou_threshold: float = 0.45) -> List[dict]:
        if not detections:
            return []

        detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
        kept: List[dict] = []

        while detections:
            best = detections.pop(0)
            kept.append(best)
            detections = [
                other for other in detections
                if _iou(best["box"], other["box"]) < iou_threshold
            ]

        return kept


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    intersection = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def fuse_detections(
    primary: List[dict],
    secondary: List[dict],
    iou_threshold: float = 0.45,
) -> List[dict]:
    """
    Merge two detection lists, keeping the higher-confidence box when they
    overlap.  Classical sonar hits that already have a learned detection are
    dropped (the learned one is usually better localised).
    """
    merged = list(primary)

    for candidate in sorted(secondary, key=lambda d: d["confidence"], reverse=True):
        if any(_iou(candidate["box"], existing["box"]) >= iou_threshold
               for existing in merged):
            continue
        merged.append(candidate)

    return merged


def is_probably_sonar(frame_bgr: np.ndarray) -> bool:
    """
    Heuristic sonar / scan-converter frame test.

    Sonar frames are monochrome, low-texture and have a strongly
    concentrated intensity histogram.  That distinguishes them from a
    grayscale photograph of a forest, for example.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return False

    b, g, r = cv2.split(frame_bgr)
    bg = float(np.mean(np.abs(b.astype(np.int16) - g.astype(np.int16))))
    gr = float(np.mean(np.abs(g.astype(np.int16) - r.astype(np.int16))))
    if bg + gr > 8.0:
        return False

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)

    # Edge density: photographs have far more edges than sonar returns.
    edges = cv2.Canny(small, 40, 120)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)
    if edge_density > 0.11:
        return False

    # Histogram concentration over 32 bins.
    hist = cv2.calcHist([small], [0], None, [32], [0, 256]).ravel()
    hist /= max(1.0, hist.sum())
    concentration = float(np.max(hist))

    return edge_density < 0.075 or concentration > 0.30


# --------------------------------------------------
# STANDALONE TEST
# --------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python sonar_detector.py <image_or_video>")
        raise SystemExit(1)

    source = sys.argv[1]
    detector = SonarDetector()

    if source.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
        cap = cv2.VideoCapture(source)
        ok, frame = cap.read()
        cap.release()
    else:
        frame = cv2.imread(source)
        ok = frame is not None

    if not ok:
        raise SystemExit(f"Could not read {source}")

    print("Sonar-like:", is_probably_sonar(frame))
    detections = detector.detect(frame)
    print("Detections:", len(detections))

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            frame, f"{det['confidence']:.2f}", (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
        )

    cv2.imshow("Sonar detections", frame)
    cv2.waitKey(0)
