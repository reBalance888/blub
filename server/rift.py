"""
rift.py — Rift spawning, depletion, group bonus rewards.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass
class Rift:
    id: str
    x: int
    y: int
    richness: float
    spawn_tick: int = 0


class RiftManager:
    def __init__(self, config: dict):
        self.config = config
        self.active_rifts: list[Rift] = []
        self.next_rift_id: int = 1
        self.last_respawn_tick: int = 0

    def spawn_epoch_rifts(self, epoch: int, ocean_size: int):
        """Spawn rifts for a new epoch using deterministic positions."""
        count = self.config["rifts"]["count_per_epoch"]
        base_richness = self.config["rifts"]["base_richness"]

        for i in range(count):
            x, y = self._rift_position(epoch, i, ocean_size)
            rid = f"rift_{self.next_rift_id}"
            self.next_rift_id += 1
            self.active_rifts.append(
                Rift(id=rid, x=x, y=y, richness=base_richness)
            )

    def _rift_position(self, epoch_seed: int, rift_index: int, ocean_size: int) -> tuple[int, int]:
        """Pseudo-random position from epoch seed."""
        h = int(hashlib.md5(f"{epoch_seed}:{rift_index}".encode()).hexdigest(), 16)
        return (h % ocean_size, (h >> 16) % ocean_size)

    def calc_group_reward(self, n: int) -> float:
        """Group bonus: how much credit each lobster earns per tick."""
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
        return 4.0 + 0.2 * (n - 5)

    def tick_rifts(self, current_tick: int):
        """Remove depleted rifts and respawn if interval elapsed."""
        self.active_rifts = [r for r in self.active_rifts if r.richness > 0]

        respawn_interval = self.config["rifts"]["respawn_interval"]
        if current_tick - self.last_respawn_tick >= respawn_interval:
            # Respawn one rift if below count
            if len(self.active_rifts) < self.config["rifts"]["count_per_epoch"]:
                rid = f"rift_{self.next_rift_id}"
                self.next_rift_id += 1
                import random
                x = random.randint(0, 99)
                y = random.randint(0, 99)
                self.active_rifts.append(
                    Rift(id=rid, x=x, y=y, richness=self.config["rifts"]["base_richness"], spawn_tick=current_tick)
                )
            self.last_respawn_tick = current_tick

    def reset(self):
        self.active_rifts.clear()
        self.next_rift_id = 1
        self.last_respawn_tick = 0
