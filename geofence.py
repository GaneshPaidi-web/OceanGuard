"""
geofence.py
===========
Camera-view geofencing for rip-current / open-water risk.

A lifeguard marks the **safe swimming area** (or an explicit **danger /
rip-current channel**) once per camera.  The pipeline then watches every
tracked *person* and raises a geofence alert when the person's waterline
point leaves the safe area or enters the danger area.

Why a geofence?
---------------
Rip currents pull swimmers away from the beach and out to sea.  A person
crossing the outer boundary of the guarded area is an early warning even
before the drowning risk engine sees thrashing or submersion.

People on boats / surfboards are **not** alerting candidates
------------------------------------------------------------
The request was explicit: only a person who crosses the geofence should
alert, not people riding a boat or another floating object.  COCO class 8
(boat) and class 37 (surfboard) detections from the same D-FINE model are
used for exclusion, with a short grace period so a flickering boat box does
not turn a passenger into a swimmer for one frame.

Geofence output is deliberately separate from the drowning risk levels:
``risk_level`` stays NORMAL / DISTRESS / DROWNING, while ``geofence_breach``
is an additional flag.  The UI shows it as a distinct "GEOFENCE" alert.

Coordinates
-----------
Zone polygons are stored in **normalised frame coordinates** (0..1), so the
same geofence works on a 720p preview and a 4K analysed frame.  Convert to
pixels only when drawing.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]
Box = Sequence[float]


# --------------------------------------------------
# DEFAULTS
# --------------------------------------------------

GEOFENCE_HOLD_SECONDS = 1.5
GEOFENCE_CLEAR_SECONDS = 2.5
GEOFENCE_MIN_TRACK_AGE = 0.7
GEOFENCE_MIN_WATER_RATIO = 0.35

# COCO: 8 = boat, 37 = surfboard.  Used only to *exclude* people riding
# these objects from geofence alerts; they are never tracked as people.
DEFAULT_PLATFORM_CLASS_IDS = (8, 37)

# Once a track is seen on a platform, ignore it for this long even if the
# platform box flickers out for a frame or two.
DEFAULT_PLATFORM_GRACE_SECONDS = 2.0


# --------------------------------------------------
# GEOMETRY
# --------------------------------------------------

def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Ray-casting point-in-polygon test (works for concave polygons)."""
    if len(polygon) < 3:
        return False

    x, y = float(point[0]), float(point[1])
    inside = False
    j = len(polygon) - 1

    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        # Check if the horizontal ray at y crosses edge (i, j).
        intersects = (yi > y) != (yj > y)
        if intersects:
            denominator = (yj - yi) if abs(yj - yi) > 1e-12 else 1e-12
            x_cross = (xj - xi) * (y - yi) / denominator + xi
            if x < x_cross:
                inside = not inside
        j = i

    return inside


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    xx1 = max(float(a[0]), float(b[0]))
    yy1 = max(float(a[1]), float(b[1]))
    xx2 = min(float(a[2]), float(b[2]))
    yy2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(
        0.0, float(a[3]) - float(a[1])
    )
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(
        0.0, float(b[3]) - float(b[1])
    )
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _expand_box(box: Sequence[float], fx: float, fy_top: float, fy_bottom: float) -> List[float]:
    """Expand a platform box, mostly upwards, to include its occupants."""
    x1, y1, x2, y2 = [float(v) for v in box]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    return [
        x1 - fx * w,
        y1 - fy_top * h,
        x2 + fx * w,
        y2 + fy_bottom * h,
    ]


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def box_anchor(box: Box, anchor: str = "feet") -> Point:
    """Return the pixel anchor point of a box.

    ``feet`` (default) is the bottom-centre of the box.  For a person in
    the water this is the waterline where they meet the surface, which is
    much more stable than the bounding-box centre when only the head and
    shoulders are visible.
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    cx = (x1 + x2) * 0.5
    if (anchor or "feet").lower() == "center":
        cy = (y1 + y2) * 0.5
    else:
        cy = y2
    return (cx, cy)


# --------------------------------------------------
# CONFIG TYPES
# --------------------------------------------------

@dataclass
class GeofenceZone:
    name: str
    points: List[Point] = field(default_factory=list)
    mode: str = "safe"  # "safe" = allowed area, "danger" = forbidden area

    @property
    def is_danger(self) -> bool:
        return (self.mode or "safe").lower() in ("danger", "rip", "rip_current", "forbidden")

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "mode": self.mode,
            "points": [[round(x, 6), round(y, 6)] for x, y in self.points],
        }


@dataclass
class GeofenceState:
    breached: bool = False
    event: bool = False
    zone: Optional[str] = None
    reason: Optional[str] = None
    anchor: Optional[Point] = None      # normalised
    excluded: bool = False              # on boat / surfboard / on land
    exclusion_reason: Optional[str] = None
    pending_seconds: float = 0.0


# --------------------------------------------------
# TRACK STATE
# --------------------------------------------------

@dataclass
class _GeofenceTrack:
    candidate_since: Optional[float] = None
    breached: bool = False
    clear_since: Optional[float] = None
    zone: Optional[str] = None
    platform_until: float = -1e9


# --------------------------------------------------
# GEOFENCE
# --------------------------------------------------

class Geofence:
    """Per-track safe-area / danger-zone monitor."""

    def __init__(
        self,
        zones: Optional[Sequence[GeofenceZone]] = None,
        enabled: bool = True,
        anchor: str = "feet",
        hold_seconds: float = GEOFENCE_HOLD_SECONDS,
        clear_seconds: float = GEOFENCE_CLEAR_SECONDS,
        min_track_age: float = GEOFENCE_MIN_TRACK_AGE,
        require_in_water: bool = True,
        min_water_ratio: float = GEOFENCE_MIN_WATER_RATIO,
        exclude_platforms: bool = True,
        platform_class_ids: Sequence[int] = DEFAULT_PLATFORM_CLASS_IDS,
        platform_iou: float = 0.30,
        platform_grace_seconds: float = DEFAULT_PLATFORM_GRACE_SECONDS,
        dynamic_water_mode: bool = False,
        dynamic_water_threshold: float = 0.30,
        water_offset_ratio: float = 0.08,
    ):
        self.enabled = bool(enabled)
        self.zones: List[GeofenceZone] = list(zones or [])
        self.anchor = (anchor or "feet").lower()
        self.hold_seconds = float(hold_seconds)
        self.clear_seconds = float(clear_seconds)
        self.min_track_age = float(min_track_age)
        self.require_in_water = bool(require_in_water)
        self.min_water_ratio = float(min_water_ratio)
        self.exclude_platforms = bool(exclude_platforms)
        self.platform_class_ids = tuple(int(v) for v in platform_class_ids)
        self.platform_iou = float(platform_iou)
        self.platform_grace_seconds = float(platform_grace_seconds)
        self.dynamic_water_mode = bool(dynamic_water_mode)
        self.dynamic_water_threshold = float(dynamic_water_threshold)
        self.water_offset_ratio = float(water_offset_ratio)

        self._tracks: Dict[int, _GeofenceTrack] = {}

    # --------------------------------------------------
    # CONSTRUCTION
    # --------------------------------------------------

    @classmethod
    def from_config(cls, payload: Optional[dict], base_dir: Optional[Path] = None) -> "Geofence":
        """Build from a pipeline-config section, optionally loading a JSON file.

        ``payload`` may contain ``config_file``; that JSON file is loaded
        first and the inline payload keys override it.  This lets the web UI
        write ``configs/geofence.json`` while the YAML file keeps the
        enable/disable switch.
        """
        if not payload:
            return cls(enabled=False)

        data: Dict[str, Any] = {}
        config_file = payload.get("config_file")
        if config_file:
            path = Path(config_file)
            if base_dir is not None and not path.is_absolute():
                path = base_dir / path
            if path.is_file():
                try:
                    with path.open("r", encoding="utf-8") as fh:
                        loaded = json.load(fh) or {}
                    if isinstance(loaded, dict):
                        data.update(loaded)
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"[Geofence] Could not read {path}: {exc}")
            else:
                print(f"[Geofence] Config file not found: {path}")

        data.update({
            k: v for k, v in payload.items()
            if k != "config_file"
            and not (k == "enabled" and v is None)
            and not (k == "zones" and not v)
        })

        zones = []
        for raw in data.get("zones", []) or []:
            if not isinstance(raw, dict):
                continue
            points = []
            for point in raw.get("points", []) or []:
                try:
                    if isinstance(point, dict):
                        px, py = float(point.get("x", 0.0)), float(point.get("y", 0.0))
                    else:
                        px, py = float(point[0]), float(point[1])
                except (TypeError, ValueError, IndexError):
                    continue
                points.append((clamp01(px), clamp01(py)))
            if len(points) < 3:
                continue
            zones.append(
                GeofenceZone(
                    name=str(raw.get("name") or f"zone-{len(zones) + 1}"),
                    points=points,
                    mode=str(raw.get("mode") or "safe"),
                )
            )

        enabled = data.get("enabled", True)
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() in ("1", "true", "yes", "on")
        dynamic_water_mode = bool(data.get("dynamic_water_mode", False))
        if not zones and not dynamic_water_mode:
            enabled = False

        return cls(
            zones=zones,
            enabled=bool(enabled),
            anchor=str(data.get("anchor", "feet")),
            hold_seconds=float(data.get("hold_seconds", GEOFENCE_HOLD_SECONDS)),
            clear_seconds=float(data.get("clear_seconds", GEOFENCE_CLEAR_SECONDS)),
            min_track_age=float(data.get("min_track_age", GEOFENCE_MIN_TRACK_AGE)),
            require_in_water=bool(data.get("require_in_water", True)),
            min_water_ratio=float(data.get("min_water_ratio", GEOFENCE_MIN_WATER_RATIO)),
            exclude_platforms=bool(data.get("exclude_platforms", True)),
            platform_class_ids=tuple(
                data.get("platform_class_ids", DEFAULT_PLATFORM_CLASS_IDS)
            ),
            platform_iou=float(data.get("platform_iou", 0.30)),
            platform_grace_seconds=float(
                data.get("platform_grace_seconds", DEFAULT_PLATFORM_GRACE_SECONDS)
            ),
            dynamic_water_mode=bool(data.get("dynamic_water_mode", False)),
            dynamic_water_threshold=float(data.get("dynamic_water_threshold", 0.30)),
            water_offset_ratio=float(data.get("water_offset_ratio", 0.08)),
        )

    # --------------------------------------------------
    # INTROSPECTION / LIFECYCLE
    # --------------------------------------------------

    @property
    def active(self) -> bool:
        return self.enabled and (bool(self.zones) or self.dynamic_water_mode)

    @property
    def has_danger_zone(self) -> bool:
        return any(z.is_danger for z in self.zones)

    def safe_zones(self) -> List[GeofenceZone]:
        return [z for z in self.zones if not z.is_danger]

    def danger_zones(self) -> List[GeofenceZone]:
        return [z for z in self.zones if z.is_danger]

    def reset(self):
        self._tracks.clear()

    def forget_track(self, track_id: int):
        self._tracks.pop(int(track_id), None)

    def as_dict(self) -> dict:
        return {
            "enabled": bool(self.enabled),
            "anchor": self.anchor,
            "hold_seconds": self.hold_seconds,
            "clear_seconds": self.clear_seconds,
            "require_in_water": self.require_in_water,
            "min_water_ratio": self.min_water_ratio,
            "exclude_platforms": self.exclude_platforms,
            "platform_class_ids": list(self.platform_class_ids),
            "zones": [z.as_dict() for z in self.zones],
        }

    # --------------------------------------------------
    # PLATFORM (BOAT / SURFBOARD) EXCLUSION
    # --------------------------------------------------

    def is_on_platform(
        self,
        box: Box,
        platform_boxes: Sequence[Box],
        frame_width: int,
        frame_height: int,
    ) -> bool:
        """True when a person box sits on a detected boat / surfboard."""
        if not self.exclude_platforms or not platform_boxes:
            return False

        anchor_px = box_anchor(box, self.anchor)

        for platform in platform_boxes:
            if not platform or len(platform) < 4:
                continue

            if _iou(box, platform) >= self.platform_iou:
                return True

            expanded = _expand_box(platform, fx=0.15, fy_top=0.45, fy_bottom=0.12)
            ex1, ey1, ex2, ey2 = expanded
            if ex1 <= anchor_px[0] <= ex2 and ey1 <= anchor_px[1] <= ey2:
                return True

            # Also catch a passenger whose body centre is inside the hull.
            center = (
                (float(box[0]) + float(box[2])) * 0.5,
                (float(box[1]) + float(box[3])) * 0.5,
            )
            if ex1 <= center[0] <= ex2 and ey1 <= center[1] <= ey2:
                return True

        return False

    # --------------------------------------------------
    # BREACH GEOMETRY
    # --------------------------------------------------

    def locate(self, anchor_norm: Point) -> Tuple[bool, Optional[str], str]:
        """Return ``(breached, zone_name, description)`` for a normalised point."""
        # Explicit danger zones always win.
        for zone in self.danger_zones():
            if point_in_polygon(anchor_norm, zone.points):
                return True, zone.name, f"inside danger zone '{zone.name}'"

        safe = self.safe_zones()
        if safe:
            if any(point_in_polygon(anchor_norm, zone.points) for zone in safe):
                return False, None, "inside safe zone"
            return True, safe[0].name, "outside the safe swimming area"

        return False, None, "outside all zones"

    # --------------------------------------------------
    # PUBLIC UPDATE
    # --------------------------------------------------

    def update(
        self,
        track_id: int,
        timestamp: float,
        box: Box,
        frame_width: int,
        frame_height: int,
        *,
        confirmed: bool = True,
        track_age: float = 0.0,
        in_water_ratio: Optional[float] = None,
        scene_type: str = "surface",
        platform_boxes: Sequence[Box] = (),
        water_mask: Optional[Any] = None,
    ) -> GeofenceState:
        """Update one person track and return its geofence state."""
        if not self.active:
            return GeofenceState()

        track_id = int(track_id)
        state = self._tracks.get(track_id)
        if state is None:
            state = _GeofenceTrack()
            self._tracks[track_id] = state

        anchor_px = box_anchor(box, self.anchor)
        anchor_norm = (
            clamp01(anchor_px[0] / max(1.0, float(frame_width))),
            clamp01(anchor_px[1] / max(1.0, float(frame_height))),
        )

        # ---- platform / boat exclusion with a grace period ----
        on_platform = self.is_on_platform(
            box, platform_boxes, frame_width, frame_height
        )
        if on_platform:
            state.platform_until = timestamp + self.platform_grace_seconds
        excluded = on_platform or timestamp < state.platform_until

        # ---- water gating: people on sand are not geofence breaches ----
        scene = (scene_type or "surface").lower()
        if scene in ("underwater", "sonar"):
            in_water = True
        elif in_water_ratio is None:
            in_water = True
        else:
            in_water = float(in_water_ratio) >= self.min_water_ratio

        eligible = (
            confirmed
            and track_age >= self.min_track_age
            and (in_water or not self.require_in_water)
            and not excluded
        )

        breached_raw, zone_name, description = self.locate(anchor_norm)

        # Dynamic Water Geofence Mode:
        # 1. When all/mostly water is present and no land is observable (underwater, pool, full water):
        #    apply geofence directly to the bottom of the video frame.
        # 2. When land is observed:
        #    locate where water begins (shoreline) and apply geofence offset ~4-5cm (water_offset_ratio) into the water.
        if self.dynamic_water_mode:
            scene = (scene_type or "surface").lower()
            has_no_land = scene in ("underwater", "sonar")
            
            # Check if water_mask indicates no land
            if water_mask is not None and hasattr(water_mask, "mask") and water_mask.mask is not None:
                mask = water_mask.mask
                row_bottom = mask[-max(1, mask.shape[0]//5):, :]
                if np.mean(row_bottom > 128) > 0.85:
                    has_no_land = True

            if has_no_land:
                # Direct bottom boundary (bottom 5% of video)
                if anchor_norm[1] >= 0.95:
                    breached_raw = True
                    zone_name = "bottom-boundary"
                    description = "reached bottom boundary in open/deep water"
            else:
                # Land is observed: check if water is observed
                if in_water and in_water_ratio is not None and float(in_water_ratio) >= self.min_water_ratio:
                    # Find shoreline column Y
                    shoreline_y = 0.55 # default fallback
                    if water_mask is not None and hasattr(water_mask, "mask") and water_mask.mask is not None:
                        mask = water_mask.mask
                        col_idx = int(np.clip(anchor_norm[0] * mask.shape[1], 0, mask.shape[1] - 1))
                        water_rows = np.where(mask[:, col_idx] > 128)[0]
                        if len(water_rows) > 0:
                            shoreline_y = float(water_rows.max()) / float(mask.shape[0])
                    
                    # Apply geofence 4-5cm (~water_offset_ratio) past shoreline into water
                    # In perspective CCTV, water extends towards smaller Y (further away) or deeper Y
                    water_geofence_limit = shoreline_y - self.water_offset_ratio
                    if anchor_norm[1] <= water_geofence_limit or anchor_norm[1] >= 0.96:
                        breached_raw = True
                        zone_name = "water-shoreline-boundary"
                        description = f"crossed 4-5cm beyond safe shoreline into deep water"

        result = GeofenceState(
            anchor=anchor_norm,
            excluded=excluded,
            exclusion_reason=(
                "person is on a boat/floating object" if excluded else None
            ),
        )

        now = float(timestamp)
        if breached_raw and eligible:
            if state.candidate_since is None:
                state.candidate_since = now
            state.clear_since = None
            pending = now - state.candidate_since

            if pending >= self.hold_seconds:
                if not state.breached:
                    state.breached = True
                    result.event = True
                state.zone = zone_name
                result.breached = True
                result.zone = zone_name
                result.reason = (
                    f"Crossed geofence: {description} "
                    "(possible rip current)"
                )
                result.pending_seconds = pending
            else:
                # Still crossing, but not long enough to alert yet.
                result.zone = zone_name
                result.pending_seconds = pending
                result.reason = (
                    f"Approaching geofence boundary ({pending:.1f}s)"
                )
        else:
            state.candidate_since = None
            if state.breached:
                if state.clear_since is None:
                    state.clear_since = now
                if now - state.clear_since < self.clear_seconds:
                    # Hysteresis: keep the alert up briefly.
                    result.breached = True
                    result.zone = state.zone
                    result.reason = "Geofence breach clearing"
                else:
                    state.breached = False
                    state.zone = None
                    state.clear_since = None
            else:
                state.clear_since = None

        return result


# --------------------------------------------------
# SELF-TEST
# --------------------------------------------------

if __name__ == "__main__":
    # Safe area covers the bottom 45% of the frame; the rest is open sea.
    safe = GeofenceZone(
        name="guarded-beach",
        mode="safe",
        points=[(0.05, 0.55), (0.95, 0.55), (0.95, 0.99), (0.05, 0.99)],
    )
    fence = Geofence(zones=[safe], enabled=True)

    # Swimmer inside the safe area: no alert.
    for i in range(6):
        result = fence.update(
            1, i * 0.5, box=[400, 600, 480, 700],
            frame_width=1920, frame_height=1080,
            in_water_ratio=0.9, track_age=5.0,
        )
    print("Inside safe zone ->", result.breached)

    # Swimmer drifting outside: alert after the hold time.
    for i in range(10):
        result = fence.update(
            2, i * 0.5, box=[900, 300, 960, 380],
            frame_width=1920, frame_height=1080,
            in_water_ratio=0.9, track_age=5.0,
        )
    print("Crossed boundary ->", result.breached, "|", result.reason)

    # Passenger on a boat in the danger area: excluded.
    for i in range(10):
        result = fence.update(
            3, i * 0.5, box=[900, 300, 960, 380],
            frame_width=1920, frame_height=1080,
            in_water_ratio=0.9, track_age=5.0,
            platform_boxes=[[860, 280, 1010, 420]],
        )
    print("On a boat ->", result.breached, "| excluded:", result.excluded)
