"""
rift.py — Rift spawning, depletion, group bonus rewards, eruptions.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass


@dataclass
class Rift:
    id: str
    x: int
    y: int
    richness: float
    rift_type: str = "silver"
    spawn_tick: int = 0
    pressure: float = 0.0
    erupting: bool = False
    eruption_ticks_left: int = 0
    eruption_multiplier: float = 1.0
    eruption_predator_spawned: bool = False


class RiftManager:
    def __init__(self, config: dict):
        self.config = config
        self.active_rifts: list[Rift] = []
        self.next_rift_id: int = 1
        self.last_respawn_tick: int = 0

    def _pick_rift_type(self) -> str:
        """Weighted random choice of rift type from config."""
        types_cfg = self.config["rifts"]["types"]
        names = list(types_cfg.keys())
        weights = [types_cfg[n]["spawn_weight"] for n in names]
        return random.choices(names, weights=weights, k=1)[0]

    def _richness_for_type(self, rift_type: str) -> float:
        return self.config["rifts"]["types"][rift_type]["richness"]

    def depletion_rate_for_type(self, rift_type: str) -> float:
        return self.config["rifts"]["types"][rift_type]["depletion_rate"]

    def spawn_epoch_rifts(self, epoch: int, zone_min: int, zone_size: int):
        """Spawn rifts for a new epoch within the active zone. Clears old rifts."""
        self.active_rifts.clear()
        self._zone_min = zone_min
        self._zone_size = zone_size
        count = self.config["rifts"]["count_per_epoch"]

        for i in range(count):
            x, y = self._rift_position(epoch, i, zone_min, zone_size)
            rid = f"rift_{self.next_rift_id}"
            self.next_rift_id += 1
            rtype = self._pick_rift_type()
            # Stagger initial pressure so eruptions don't all fire at once
            initial_pressure = random.uniform(0, 0.3)
            self.active_rifts.append(
                Rift(id=rid, x=x, y=y, richness=self._richness_for_type(rtype),
                     rift_type=rtype, pressure=initial_pressure)
            )

    def _rift_position(self, epoch_seed: int, rift_index: int, zone_min: int, zone_size: int) -> tuple[int, int]:
        """Pseudo-random position from epoch seed, within active zone."""
        h = int(hashlib.md5(f"{epoch_seed}:{rift_index}".encode()).hexdigest(), 16)
        return (zone_min + h % zone_size, zone_min + (h >> 16) % zone_size)

    def calc_group_reward(self, n: int) -> float:
        """Group bonus: how much credit each lobster earns per tick.
        Sweet spot at 5; diminishing returns past that to motivate splitting."""
        if n == 0:
            return 0.0
        if n == 1:
            return 0.5  # raised from 0.1 — newcomer viability (Game Designer rec)
        if n == 2:
            return 1.0
        if n == 3:
            return 1.8
        if n == 4:
            return 2.8
        if n == 5:
            return 4.0
        # Negative marginal returns past sweet spot of 5
        if n == 6:
            return 3.8
        if n == 7:
            return 3.5
        if n == 8:
            return 3.2
        if n == 9:
            return 3.0
        return 2.8  # 10+

    def update_zone(self, zone_min: int, zone_size: int):
        """Update cached active zone bounds (called when agents join/leave)."""
        self._zone_min = zone_min
        self._zone_size = zone_size

    def tick_eruptions(self, current_tick: int, lobster_positions: list[tuple[int, int]]) -> list[dict]:
        """Accumulate pressure on all rifts, trigger eruptions when threshold reached.
        Returns list of eruption events [{rift_id, x, y, multiplier, phase}]."""
        ecfg = self.config.get("eruptions", {})
        if not ecfg.get("enabled", False):
            return []

        base_rate = ecfg.get("pressure_per_tick", 0.01)
        agent_pressure = ecfg.get("pressure_per_agent", 0.2)
        thresholds = {
            "gold": ecfg.get("threshold_gold", 1.5),
            "silver": ecfg.get("threshold_silver", 1.0),
            "copper": ecfg.get("threshold_copper", 0.7),
        }
        multipliers = {
            "gold": ecfg.get("multiplier_gold", 5.0),
            "silver": ecfg.get("multiplier_silver", 3.0),
            "copper": ecfg.get("multiplier_copper", 2.0),
        }
        duration = ecfg.get("duration_ticks", 20)
        cascade_radius = ecfg.get("cascade_radius", 10)
        cascade_chance = ecfg.get("cascade_chance", 0.30)
        cascade_boost = ecfg.get("cascade_pressure_boost", 0.5)
        restore_frac = ecfg.get("richness_restore_frac", 0.5)

        events: list[dict] = []

        # Process erupting rifts: decrement timer, emit events
        for rift in self.active_rifts:
            if rift.erupting:
                rift.eruption_ticks_left -= 1
                if rift.eruption_ticks_left <= 0:
                    rift.erupting = False
                    rift.eruption_multiplier = 1.0
                    events.append({
                        "rift_id": rift.id, "x": rift.x, "y": rift.y,
                        "multiplier": 1.0, "phase": "end",
                    })
                    print(f"[ERUPTION] {rift.id} ({rift.rift_type}) eruption ended at tick {current_tick}")
                else:
                    events.append({
                        "rift_id": rift.id, "x": rift.x, "y": rift.y,
                        "multiplier": rift.eruption_multiplier, "phase": "ongoing",
                    })

        # Accumulate pressure on non-erupting rifts
        cascade_targets: list[tuple[Rift, float]] = []
        for rift in self.active_rifts:
            if rift.erupting:
                continue

            # Count nearby agents (within rift radius)
            rift_radius = self.config["rifts"]["radius"]
            nearby_agents = sum(
                1 for lx, ly in lobster_positions
                if abs(lx - rift.x) <= rift_radius and abs(ly - rift.y) <= rift_radius
            )

            # Accumulate pressure with agent-density scaling + noise
            noise = random.gauss(0, 0.002)
            rift.pressure += base_rate * (1 + nearby_agents * agent_pressure) + noise
            rift.pressure = max(0.0, rift.pressure)  # clamp

            # Check threshold
            threshold = thresholds.get(rift.rift_type, 1.0)
            if rift.pressure >= threshold:
                # ERUPT
                rift.erupting = True
                rift.eruption_ticks_left = duration
                rift.eruption_multiplier = multipliers.get(rift.rift_type, 2.0)
                rift.eruption_predator_spawned = False
                rift.pressure = 0.0

                # Restore richness (eruption revives depleted rifts)
                max_richness = self._richness_for_type(rift.rift_type)
                rift.richness = max(rift.richness, max_richness * restore_frac)

                events.append({
                    "rift_id": rift.id, "x": rift.x, "y": rift.y,
                    "multiplier": rift.eruption_multiplier, "phase": "start",
                })
                print(f"[ERUPTION] {rift.id} ({rift.rift_type}) ERUPTED at ({rift.x},{rift.y}) "
                      f"x{rift.eruption_multiplier} for {duration} ticks")

                # Queue cascade to nearby rifts
                for other in self.active_rifts:
                    if other.id == rift.id or other.erupting:
                        continue
                    dist = math.sqrt((other.x - rift.x) ** 2 + (other.y - rift.y) ** 2)
                    if dist <= cascade_radius and random.random() < cascade_chance:
                        # Scale boost by distance (closer = stronger)
                        scale = 1.0 - (dist / cascade_radius)
                        cascade_targets.append((other, cascade_boost * scale))

        # Apply cascades (deferred to avoid modifying during iteration)
        for target_rift, boost in cascade_targets:
            target_rift.pressure += boost

        return events

    def get_erupting_needing_predator(self, delay: int) -> list[Rift]:
        """Return erupting rifts that have been erupting for `delay` ticks and haven't spawned a predator yet."""
        ecfg = self.config.get("eruptions", {})
        duration = ecfg.get("duration_ticks", 20)
        result = []
        for rift in self.active_rifts:
            if not rift.erupting or rift.eruption_predator_spawned:
                continue
            ticks_into_eruption = duration - rift.eruption_ticks_left
            if ticks_into_eruption >= delay:
                result.append(rift)
        return result

    def tick_rifts(self, current_tick: int, tidal_offset: tuple[int, int] = (0, 0)):
        """Remove depleted rifts (keep pressurized ones alive) and respawn if interval elapsed."""
        # Keep depleted rifts alive if they have significant pressure (can still erupt)
        self.active_rifts = [
            r for r in self.active_rifts
            if r.richness > 0 or r.erupting or r.pressure > 0.5
        ]

        respawn_interval = self.config["rifts"]["respawn_interval"]
        if current_tick - self.last_respawn_tick >= respawn_interval:
            # Respawn one rift if below count (within active zone)
            if len(self.active_rifts) < self.config["rifts"]["count_per_epoch"]:
                rid = f"rift_{self.next_rift_id}"
                self.next_rift_id += 1
                zm = getattr(self, '_zone_min', 0)
                zs = getattr(self, '_zone_size', 100)
                x = random.randint(zm, zm + zs - 1) + tidal_offset[0]
                y = random.randint(zm, zm + zs - 1) + tidal_offset[1]
                # Clamp to active zone
                x = max(zm, min(zm + zs - 1, x))
                y = max(zm, min(zm + zs - 1, y))
                rtype = self._pick_rift_type()
                initial_pressure = random.uniform(0, 0.3)
                self.active_rifts.append(
                    Rift(id=rid, x=x, y=y, richness=self._richness_for_type(rtype),
                         rift_type=rtype, spawn_tick=current_tick, pressure=initial_pressure)
                )
            self.last_respawn_tick = current_tick

    def reset(self):
        self.active_rifts.clear()
        self.next_rift_id = 1
        self.last_respawn_tick = 0
