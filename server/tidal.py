"""
tidal.py — Three nested environmental cycles: day/night, tidal, seasonal.
Pure math — no ocean logic, just phase computation. O(1) per tick.
"""
from __future__ import annotations

import math


class TidalEngine:
    """Three cosine cycles that modulate ocean parameters."""

    def __init__(self, config: dict):
        self.enabled = config.get("enabled", True)
        # Day/Night: period 50 ticks
        self.dn_period = config.get("day_night_period", 50)
        self.dn_predator_max = config.get("day_night_predator_max", 2.0)
        self.dn_sound_min = config.get("day_night_sound_min", 0.4)
        # Tidal: period 200 ticks
        self.tidal_period = config.get("tidal_period", 200)
        self.tidal_offset_max = config.get("tidal_offset_max", 2)
        # Seasonal: period 1000 ticks
        self.seasonal_period = config.get("seasonal_period", 1000)
        self.seasonal_min = config.get("seasonal_min_multiplier", 0.5)

        # Current phase values (updated each tick)
        self.day_night_phase: float = 0.0  # 0.0 = day, 1.0 = night
        self._tidal_angle: float = 0.0
        self.seasonal_multiplier: float = 1.0

    def tick(self, global_tick: int):
        """Recompute all three phases. Called once per tick."""
        if not self.enabled:
            self.day_night_phase = 0.0
            self._tidal_angle = 0.0
            self.seasonal_multiplier = 1.0
            return

        # Day/Night: phase = 0.5 + 0.5 * cos(2pi * t / period)
        # 0.0 = day, 1.0 = night
        self.day_night_phase = 0.5 + 0.5 * math.cos(
            2.0 * math.pi * global_tick / self.dn_period
        )

        # Tidal: angle for sin/cos offset
        self._tidal_angle = 2.0 * math.pi * global_tick / self.tidal_period

        # Seasonal: 0.5 + 0.5 * cos(2pi * t / period)
        # 1.0 = summer (peak), seasonal_min = winter (trough)
        raw = 0.5 + 0.5 * math.cos(
            2.0 * math.pi * global_tick / self.seasonal_period
        )
        self.seasonal_multiplier = self.seasonal_min + (1.0 - self.seasonal_min) * raw

    def get_predator_spawn_multiplier(self) -> float:
        """1.0 (day) → dn_predator_max (night)."""
        if not self.enabled:
            return 1.0
        return 1.0 + (self.dn_predator_max - 1.0) * self.day_night_phase

    def get_sound_range_multiplier(self) -> float:
        """1.0 (day) → dn_sound_min (night)."""
        if not self.enabled:
            return 1.0
        return 1.0 - (1.0 - self.dn_sound_min) * self.day_night_phase

    def get_rift_offset(self) -> tuple[int, int]:
        """Rift position offset at respawn: ±tidal_offset_max cells."""
        if not self.enabled:
            return (0, 0)
        dx = round(self.tidal_offset_max * math.sin(self._tidal_angle))
        dy = round(self.tidal_offset_max * math.cos(self._tidal_angle))
        return (dx, dy)

    def get_seasonal_multiplier(self) -> float:
        """Richness harvest multiplier: seasonal_min (winter) → 1.0 (summer)."""
        return self.seasonal_multiplier

    def get_viewer_data(self) -> dict:
        """Data for viewer UI indicators."""
        dx, dy = self.get_rift_offset()
        return {
            "day_night_phase": round(self.day_night_phase, 3),
            "tidal_dx": dx,
            "tidal_dy": dy,
            "seasonal_multiplier": round(self.seasonal_multiplier, 3),
        }

    def get_state_for_agent(self) -> dict:
        """Minimal state exposed to agents."""
        return {
            "is_night": self.day_night_phase > 0.5,
            "seasonal_multiplier": round(self.seasonal_multiplier, 3),
            "sound_range_multiplier": round(self.get_sound_range_multiplier(), 3),
        }
