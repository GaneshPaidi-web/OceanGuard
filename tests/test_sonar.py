"""Tests for the classical sonar blob detector and fusion."""

import unittest

import cv2
import numpy as np

from sonar_detector import SonarDetector, fuse_detections, is_probably_sonar


def sonar_frame():
    frame = np.full((480, 640, 3), 28, dtype=np.uint8)
    # Two bright returns against the dark background.
    cv2.ellipse(frame, (180, 220), (34, 18), 0, 0, 360, (225, 225, 225), -1)
    cv2.ellipse(frame, (430, 300), (26, 14), 15, 0, 360, (200, 200, 200), -1)
    return frame


class SonarTests(unittest.TestCase):
    def setUp(self):
        self.detector = SonarDetector(use_motion=False)

    def test_detects_bright_blobs(self):
        detections = self.detector.detect(sonar_frame(), confidence_threshold=0.2)
        self.assertGreaterEqual(len(detections), 1)
        boxes = [d["box"] for d in detections]
        matched = any(
            abs((box[0] + box[2]) / 2 - 180) < 60
            and abs((box[1] + box[3]) / 2 - 220) < 60
            for box in boxes
        )
        self.assertTrue(matched, boxes)

    def test_blank_frame_has_no_detections(self):
        blank = np.full((240, 320, 3), 20, dtype=np.uint8)
        detections = self.detector.detect(blank, confidence_threshold=0.2)
        self.assertEqual(detections, [])

    def test_fusion_keeps_non_overlapping(self):
        primary = [{"box": [10, 10, 100, 120], "confidence": 0.9, "source": "dfine"}]
        secondary = [
            {"box": [12, 12, 102, 122], "confidence": 0.6, "source": "sonar"},
            {"box": [300, 200, 360, 280], "confidence": 0.5, "source": "sonar"},
        ]
        merged = fuse_detections(primary, secondary)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["source"], "dfine")
        self.assertEqual(merged[1]["source"], "sonar")

    def test_sonar_frame_heuristic(self):
        self.assertTrue(is_probably_sonar(sonar_frame()))

    def test_colour_photo_is_not_sonar(self):
        photo = np.zeros((240, 320, 3), dtype=np.uint8)
        photo[:, :] = (60, 120, 200)
        self.assertFalse(is_probably_sonar(photo))


if __name__ == "__main__":
    unittest.main()
