"""Behavioural tests for camera-view geofencing."""

import tempfile
import unittest
from pathlib import Path

from geofence import Geofence, point_in_polygon


W, H = 1920, 1080

SAFE_ZONE = {
    "name": "guarded-beach",
    "mode": "safe",
    "points": [
        [0.05, 0.55],
        [0.95, 0.55],
        [0.95, 0.99],
        [0.05, 0.99],
    ],
}

INSIDE_BOX = [400, 700, 480, 800]     # anchor (440, 800) -> y=0.741
OUTSIDE_BOX = [900, 300, 960, 380]    # anchor (930, 380) -> y=0.352


def make_fence(**overrides):
    payload = {"enabled": True, "zones": [SAFE_ZONE]}
    payload.update(overrides)
    return Geofence.from_config(payload)


class GeometryTests(unittest.TestCase):
    def test_point_in_polygon(self):
        square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        self.assertTrue(point_in_polygon((0.5, 0.5), square))
        self.assertFalse(point_in_polygon((1.5, 0.5), square))

    def test_concave_polygon(self):
        # U-shape: the notch is outside.
        poly = [
            (0.0, 0.0), (0.4, 0.0), (0.4, 0.4), (0.6, 0.4),
            (0.6, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0),
        ]
        self.assertFalse(point_in_polygon((0.5, 0.2), poly))
        self.assertTrue(point_in_polygon((0.2, 0.7), poly))


class SafeZoneTests(unittest.TestCase):
    def setUp(self):
        self.fence = make_fence()

    def test_inside_safe_zone_is_clear(self):
        for i in range(10):
            result = self.fence.update(
                1, i * 0.5, INSIDE_BOX, W, H,
                track_age=5.0, in_water_ratio=0.9,
            )
        self.assertFalse(result.breached)
        self.assertIsNone(result.zone)

    def test_crossing_alerts_after_hold(self):
        states = []
        for i in range(8):
            states.append(self.fence.update(
                2, i * 0.5, OUTSIDE_BOX, W, H,
                track_age=5.0, in_water_ratio=0.9,
            ))
        # 0.0s and 0.5s are pending; 1.5s is the first confirmed breach.
        self.assertFalse(states[1].breached)
        self.assertTrue(states[3].breached)
        self.assertTrue(states[3].event)
        self.assertIn("outside the safe swimming area", states[3].reason)

    def test_event_fires_once(self):
        events = 0
        for i in range(10):
            result = self.fence.update(
                3, i * 0.5, OUTSIDE_BOX, W, H,
                track_age=5.0, in_water_ratio=0.9,
            )
            events += int(result.event)
        self.assertEqual(events, 1)

    def test_clears_with_hysteresis(self):
        for i in range(8):
            self.fence.update(
                4, i * 0.5, OUTSIDE_BOX, W, H,
                track_age=5.0, in_water_ratio=0.9,
            )
        # Return inside: alert stays up for clear_seconds (2.5 s).
        early = self.fence.update(
            4, 4.0, INSIDE_BOX, W, H, track_age=9.0, in_water_ratio=0.9
        )
        later = self.fence.update(
            4, 7.0, INSIDE_BOX, W, H, track_age=12.0, in_water_ratio=0.9
        )
        self.assertTrue(early.breached)   # still clearing
        self.assertFalse(later.breached)

    def test_new_track_is_ignored(self):
        result = self.fence.update(
            5, 0.0, OUTSIDE_BOX, W, H,
            track_age=0.1, in_water_ratio=0.9,
        )
        self.assertFalse(result.breached)

    def test_unconfirmed_track_is_ignored(self):
        result = self.fence.update(
            6, 0.0, OUTSIDE_BOX, W, H,
            confirmed=False, track_age=5.0, in_water_ratio=0.9,
        )
        self.assertFalse(result.breached)

    def test_dry_sand_is_ignored(self):
        for i in range(10):
            result = self.fence.update(
                7, i * 0.5, OUTSIDE_BOX, W, H,
                track_age=5.0, in_water_ratio=0.10,
            )
        self.assertFalse(result.breached)

    def test_underwater_scene_ignores_water_ratio(self):
        for i in range(6):
            result = self.fence.update(
                8, i * 0.5, OUTSIDE_BOX, W, H,
                track_age=5.0, in_water_ratio=None, scene_type="underwater",
            )
        self.assertTrue(result.breached)


class DangerZoneTests(unittest.TestCase):
    def test_inside_danger_zone_alerts(self):
        danger = {
            "name": "rip-channel",
            "mode": "danger",
            "points": [
                [0.30, 0.20],
                [0.70, 0.20],
                [0.80, 0.60],
                [0.20, 0.60],
            ],
        }
        fence = make_fence(zones=[danger])
        for i in range(6):
            result = fence.update(
                1, i * 0.5, OUTSIDE_BOX, W, H,
                track_age=5.0, in_water_ratio=0.9,
            )
        self.assertTrue(result.breached)
        self.assertEqual(result.zone, "rip-channel")
        self.assertIn("danger zone", result.reason)


class PlatformExclusionTests(unittest.TestCase):
    def setUp(self):
        self.fence = make_fence()

    def test_person_on_boat_is_excluded(self):
        boat = [860, 280, 1010, 420]
        for i in range(12):
            result = self.fence.update(
                1, i * 0.5, OUTSIDE_BOX, W, H,
                track_age=5.0, in_water_ratio=0.9,
                platform_boxes=[boat],
            )
        self.assertTrue(result.excluded)
        self.assertFalse(result.breached)

    def test_grace_period_after_boat_disappears(self):
        boat = [860, 280, 1010, 420]
        # Seen on a boat at t=0.
        self.fence.update(
            2, 0.0, OUTSIDE_BOX, W, H,
            track_age=5.0, in_water_ratio=0.9, platform_boxes=[boat],
        )
        # Boat box gone; still inside the 2 s grace period.
        grace = self.fence.update(
            2, 1.0, OUTSIDE_BOX, W, H,
            track_age=6.0, in_water_ratio=0.9, platform_boxes=[],
        )
        self.assertTrue(grace.excluded)
        self.assertFalse(grace.breached)

    def test_platform_exclusion_can_be_disabled(self):
        fence = make_fence(exclude_platforms=False)
        boat = [860, 280, 1010, 420]
        for i in range(6):
            result = fence.update(
                3, i * 0.5, OUTSIDE_BOX, W, H,
                track_age=5.0, in_water_ratio=0.9,
                platform_boxes=[boat],
            )
        self.assertFalse(result.excluded)
        self.assertTrue(result.breached)


class ConfigTests(unittest.TestCase):
    def test_no_zones_means_inactive(self):
        fence = Geofence.from_config({"enabled": True, "zones": []})
        self.assertFalse(fence.active)

    def test_disabled_fence_never_alerts(self):
        fence = make_fence(enabled=False)
        result = fence.update(
            1, 0.0, OUTSIDE_BOX, W, H,
            track_age=5.0, in_water_ratio=0.9,
        )
        self.assertFalse(result.breached)

    def test_invalid_zone_is_dropped(self):
        fence = Geofence.from_config({
            "enabled": True,
            "zones": [{"name": "bad", "points": [[0.1, 0.1], [0.2, 0.2]]}],
        })
        self.assertFalse(fence.active)

    def test_loads_from_json_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "geofence.json"
            path.write_text(
                '{"enabled": true, "zones": ['
                '{"name": "z", "mode": "safe", '
                '"points": [[0,0],[1,0],[1,1],[0,1]]}]}',
                encoding="utf-8",
            )
            fence = Geofence.from_config(
                {"config_file": str(path)}, base_dir=Path(tmp)
            )
        self.assertTrue(fence.active)
        self.assertEqual(len(fence.zones), 1)

    def test_reset_clears_track_state(self):
        fence = make_fence()
        for i in range(6):
            fence.update(
                9, i * 0.5, OUTSIDE_BOX, W, H,
                track_age=5.0, in_water_ratio=0.9,
            )
        fence.reset()
        result = fence.update(
            9, 100.0, OUTSIDE_BOX, W, H,
            track_age=5.0, in_water_ratio=0.9,
        )
        self.assertFalse(result.breached)   # hold timer restarted


if __name__ == "__main__":
    unittest.main()
