"""
predator.py — Predator spawning, movement, and killing.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class Predator:
    id: str
    x: float
    y: float
    lifetime_remaining: int
    target_x: int | None = None
    target_y: int | None = None


class PredatorManager:
    def __init__(self, config: dict):
        self.config = config
        self.active_predators: list[Predator] = []
        self.next_pred_id: int = 1

    def process_tick(self, lobster_positions: list[tuple[int, int]], ocean_size: int,
                     rift_positions: list[tuple[int, int]] | None = None):
        """Spawn new predators based on density (sigmoid), move existing ones, remove expired."""
        zone_size = self.config["predators"]["zone_size"]
        max_rate = self.config["predators"]["base_spawn_rate"]
        K = self.config["predators"].get("sigmoid_K", 12)
        speed = self.config["predators"]["speed"]
        lifetime = self.config["predators"]["lifetime"]
        rift_reduction = self.config["predators"].get("rift_zone_reduction", 0.6)

        # Count lobsters per zone
        zones: dict[tuple[int, int], int] = {}
        for lx, ly in lobster_positions:
            zx = lx // zone_size
            zy = ly // zone_size
            zones[(zx, zy)] = zones.get((zx, zy), 0) + 1

        # Spawn predators by density (Hill/sigmoid function)
        for (zx, zy), count in zones.items():
            chance = max_rate * (count ** 2) / (K ** 2 + count ** 2)

            # Reduce spawn rate near rifts
            if rift_positions:
                zone_cx = zx * zone_size + zone_size // 2
                zone_cy = zy * zone_size + zone_size // 2
                for rx, ry in rift_positions:
                    if abs(zone_cx - rx) <= zone_size and abs(zone_cy - ry) <= zone_size:
                        chance *= rift_reduction
                        break

            if random.random() < chance:
                # Spawn at zone edge
                edge = random.choice(["top", "bottom", "left", "right"])
                px = zx * zone_size
                py = zy * zone_size
                if edge == "top":
                    px += random.randint(0, zone_size - 1)
                elif edge == "bottom":
                    px += random.randint(0, zone_size - 1)
                    py += zone_size - 1
                elif edge == "left":
                    py += random.randint(0, zone_size - 1)
                elif edge == "right":
                    px += zone_size - 1
                    py += random.randint(0, zone_size - 1)

                px = max(0, min(ocean_size - 1, px))
                py = max(0, min(ocean_size - 1, py))

                pid = f"pred_{self.next_pred_id}"
                self.next_pred_id += 1
                self.active_predators.append(
                    Predator(id=pid, x=float(px), y=float(py), lifetime_remaining=lifetime)
                )

        # Move predators toward nearest lobster
        for pred in self.active_predators:
            pred.lifetime_remaining -= 1

            if not lobster_positions:
                continue

            # Find nearest lobster
            nearest = min(
                lobster_positions,
                key=lambda p: math.sqrt((p[0] - pred.x) ** 2 + (p[1] - pred.y) ** 2),
            )
            dx = nearest[0] - pred.x
            dy = nearest[1] - pred.y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist > 0:
                pred.x += (dx / dist) * speed
                pred.y += (dy / dist) * speed
                pred.x = max(0, min(ocean_size - 1, pred.x))
                pred.y = max(0, min(ocean_size - 1, pred.y))

        # Remove expired
        self.active_predators = [p for p in self.active_predators if p.lifetime_remaining > 0]

    def reset(self):
        self.active_predators.clear()
        self.next_pred_id = 1
