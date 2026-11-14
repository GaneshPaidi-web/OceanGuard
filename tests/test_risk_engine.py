"""Behavioural tests for the temporal risk engine."""

import unittest

from risk_engine import RiskEngine


def simulate(
    engine: RiskEngine,
    track_id: int,
    *,
    start: float = 0.0,
    seconds: float = 20.0,
    step: float = 0.5,
    speed: float = 0.0,
    stationary: float = 0.0,
    water_ratio=0.9,
    scene="surface",
    body_angle=90.0,
    visibility=0.8,
    head_visible=True,
    head_below=False,
    asymmetry=1.0,
):
    result = None
    timestamp = start
    while timestamp < start + seconds:
        # Move the box so centroid-derived speed and straightness are
        # consistent with ``speed`` (a swimmer actually progresses).
        offset = speed * (timestamp - start)
        result = engine.update(
            track_id=track_id,
            timestamp=timestamp,
            box=[100 + offset, 100, 180 + offset, 160],
            speed_px_s=speed,
            stationary_duration=stationary,
            in_water_ratio=water_ratio,
            scene_type=scene,
            pose={
                "body_angle_deg": body_angle,
                "visibility_fraction": visibility,
                "head_visible": head_visible,
                "head_below_hip": head_below,
                "arm_asymmetry": asymmetry,
            },
        )
        timestamp += step
    return result


class RiskEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = RiskEngine(use_learned_model=False)

    def test_swimmer_stays_normal(self):
        result = simulate(
            self.engine, 1, speed=55.0, body_angle=5.0, water_ratio=0.9
        )
        self.assertEqual(result.level, "NORMAL")
        self.assertTrue(result.in_water)

    def test_drowning_escalates(self):
        result = simulate(
            self.engine, 2, speed=1.0, stationary=15.0, body_angle=10.0,
            visibility=0.25, head_visible=False, head_below=True,
            water_ratio=0.85,
        )
        self.assertEqual(result.level, "DROWNING")
        self.assertGreater(result.confidence, 0.5)

    def test_person_on_sand_is_not_drowning(self):
        result = simulate(
            self.engine, 3, speed=2.0, stationary=15.0, body_angle=5.0,
            water_ratio=0.02,
        )
        self.assertEqual(result.level, "NORMAL")
        self.assertFalse(result.in_water)
        self.assertLess(result.score, 0.3)

    def test_underwater_scene_forces_water(self):
        result = simulate(
            self.engine, 4, speed=1.0, stationary=15.0, body_angle=10.0,
            visibility=0.2, head_visible=False, head_below=True,
            water_ratio=None, scene="underwater",
        )
        self.assertTrue(result.in_water)
        self.assertEqual(result.level, "DROWNING")

    def test_new_track_is_unknown(self):
        result = self.engine.update(
            track_id=5, timestamp=0.0, box=[0, 0, 50, 50],
            speed_px_s=0.0, stationary_duration=0.0,
            in_water_ratio=0.9, confirmed=False,
        )
        self.assertEqual(result.level, "UNKNOWN")

    def test_drowning_event_fires_once(self):
        events = 0
        for i in range(40):
            result = self.engine.update(
                track_id=6, timestamp=i * 0.5, box=[100, 100, 130, 130],
                speed_px_s=1.0, stationary_duration=15.0,
                in_water_ratio=0.9, scene_type="surface",
                pose={"body_angle_deg": 10.0, "visibility_fraction": 0.2,
                      "head_visible": False, "head_below_hip": True},
            )
            events += int(result.event)
        self.assertEqual(events, 1)

    def test_reset_clears_history(self):
        simulate(self.engine, 7, speed=55.0)
        self.engine.reset()
        result = self.engine.update(
            track_id=7, timestamp=100.0, box=[0, 0, 50, 50],
            speed_px_s=0.0, in_water_ratio=0.9, confirmed=True,
        )
        self.assertEqual(result.level, "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
