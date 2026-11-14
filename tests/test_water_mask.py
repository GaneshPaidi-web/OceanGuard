"""Tests for water / land segmentation."""

import unittest

import numpy as np

from water_mask import WaterMaskEstimator


def beach_frame(width=640, height=360):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    split = int(height * 0.55)
    frame[:split] = (180, 110, 40)     # blue water (BGR)
    frame[split:] = (150, 190, 215)    # sand
    return frame


class WaterMaskTests(unittest.TestCase):
    def setUp(self):
        self.estimator = WaterMaskEstimator()
        self.frame = beach_frame()

    def test_water_fraction_on_beach(self):
        analysis = self.estimator.analyse(self.frame, "surface")
        self.assertTrue(analysis.reliable)
        self.assertTrue(analysis.is_water_scene)
        self.assertGreater(analysis.fraction, 0.4)

    def test_person_in_water_ratio(self):
        analysis = self.estimator.analyse(self.frame, "surface")
        in_water = analysis.in_water_ratio([200, 40, 260, 140])
        on_sand = analysis.in_water_ratio([200, 250, 260, 340])
        self.assertIsNotNone(in_water)
        self.assertGreater(in_water, 0.6)
        self.assertLess(on_sand, 0.35)

    def test_underwater_scene_is_all_water(self):
        analysis = self.estimator.analyse(self.frame, "underwater")
        self.assertTrue(analysis.reliable)
        self.assertEqual(analysis.fraction, 1.0)
        self.assertEqual(analysis.in_water_ratio([10, 10, 100, 100]), 1.0)

    def test_grayscale_is_unreliable(self):
        analysis = self.estimator.analyse(self.frame, "grayscale")
        self.assertFalse(analysis.reliable)
        self.assertIsNone(analysis.in_water_ratio([10, 10, 100, 100]))

    def test_land_only_frame(self):
        sand = np.zeros((240, 320, 3), dtype=np.uint8)
        sand[:] = (150, 190, 215)
        analysis = self.estimator.analyse(sand, "surface")
        self.assertFalse(analysis.is_water_scene)
        self.assertLess(analysis.fraction, 0.1)


if __name__ == "__main__":
    unittest.main()
