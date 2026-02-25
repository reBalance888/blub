"""
rift.py — Rift spawning, depletion, group bonus rewards.
"""
from __future__ import annotations

import hashlib
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
            self.active_rifts.append(
                Rift(id=rid, x=x, y=y, richness=self._richness_for_type(rtype), rift_type=rtype)
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
            return 0.1
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

    def tick_rifts(self, current_tick: int):
        """Remove depleted rifts and respawn if interval elapsed."""
        self.active_rifts = [r for r in self.active_rifts if r.richness > 0]

        respawn_interval = self.config["rifts"]["respawn_interval"]
        if current_tick - self.last_respawn_tick >= respawn_interval:
            # Respawn one rift if below count (within active zone)
            if len(self.active_rifts) < self.config["rifts"]["count_per_epoch"]:
                rid = f"rift_{self.next_rift_id}"
                self.next_rift_id += 1
                zm = getattr(self, '_zone_min', 0)
                zs = getattr(self, '_zone_size', 100)
                x = random.randint(zm, zm + zs - 1)
                y = random.randint(zm, zm + zs - 1)
                rtype = self._pick_rift_type()
                self.active_rifts.append(
                    Rift(id=rid, x=x, y=y, richness=self._richness_for_type(rtype),
                         rift_type=rtype, spawn_tick=current_tick)
                )
            self.last_respawn_tick = current_tick

    def reset(self):
        self.active_rifts.clear()
        self.next_rift_id = 1
        self.last_respawn_tick = 0
