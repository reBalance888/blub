"""
predator.py — Predator spawning, movement, and killing.
Three predator types: shark (chase), eel (ambush near rifts), octopus (pheromone follow).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class Predator:
    id: str
    x: float
    y: float
    lifetime_remaining: int
    predator_type: str = "shark"
    speed: float = 1.0
    lethality: float = 0.80
    target_x: int | None = None
    target_y: int | None = None
    # Eel state
    has_attacked: bool = False
    attack_cooldown: int = 0
    respawn_delay: int = 10
    # Octopus fractional movement
    move_accumulator: float = 0.0


class PredatorManager:
    def __init__(self, config: dict):
        self.config = config
        self.active_predators: list[Predator] = []
        self.next_pred_id: int = 1

        # Parse type configs
        pred_cfg = config.get("predators", {})
        self.type_configs: dict[str, dict] = pred_cfg.get("types", {})
        self.min_population = pred_cfg.get("min_population_for_predators", 20)
        self.extinction_protection = pred_cfg.get("extinction_protection", True)

        # Build spawn weight list for weighted random selection
        self._type_names: list[str] = []
        self._spawn_weights: list[float] = []
        for tname, tcfg in self.type_configs.items():
            self._type_names.append(tname)
            self._spawn_weights.append(tcfg.get("spawn_weight", 0.33))

        # Fallback if no types configured
        if not self._type_names:
            self._type_names = ["shark"]
            self._spawn_weights = [1.0]
            self.type_configs = {"shark": {
                "speed": pred_cfg.get("speed", 1.0),
                "lethality": pred_cfg.get("kill_prob_base", 0.8),
                "lifetime": pred_cfg.get("lifetime", 30),
                "behavior": "chase",
            }}

    def _select_type(self) -> str:
        """Weighted random selection of predator type."""
        return random.choices(self._type_names, weights=self._spawn_weights, k=1)[0]

    def _spawn_positioned(self, ptype: str, zx: int, zy: int, zone_size: int,
                          ocean_size: int, rift_positions: list[tuple[int, int]]) -> Predator | None:
        """Spawn a predator of the given type with type-specific positioning."""
        tcfg = self.type_configs.get(ptype, {})
        lifetime = tcfg.get("lifetime", self.config["predators"]["lifetime"])
        speed = tcfg.get("speed", self.config["predators"]["speed"])
        lethality = tcfg.get("lethality", self.config["predators"].get("kill_prob_base", 0.8))
        behavior = tcfg.get("behavior", "chase")
        respawn_delay = tcfg.get("respawn_delay", 10)

        if behavior == "ambush":
            # Eel: spawn near a random active rift (within max_rift_distance)
            if not rift_positions:
                return None
            max_dist = tcfg.get("max_rift_distance", 2)
            rift = random.choice(rift_positions)
            px = rift[0] + random.randint(-max_dist, max_dist)
            py = rift[1] + random.randint(-max_dist, max_dist)
            px = max(0, min(ocean_size - 1, px))
            py = max(0, min(ocean_size - 1, py))

        elif behavior == "chase":
            # Shark: zone edge, far from rifts
            min_rift_dist = tcfg.get("min_rift_distance", 8)
            px, py = self._zone_edge_pos(zx, zy, zone_size, ocean_size)
            # Try up to 5 times to find a position far from rifts
            if rift_positions:
                for _ in range(5):
                    too_close = False
                    for rx, ry in rift_positions:
                        if abs(px - rx) + abs(py - ry) < min_rift_dist:
                            too_close = True
                            break
                    if not too_close:
                        break
                    px, py = self._zone_edge_pos(zx, zy, zone_size, ocean_size)

        else:
            # Octopus (pheromone_follow): zone edge (same as old default)
            px, py = self._zone_edge_pos(zx, zy, zone_size, ocean_size)

        pid = f"pred_{self.next_pred_id}"
        self.next_pred_id += 1
        return Predator(
            id=pid, x=float(px), y=float(py),
            lifetime_remaining=lifetime,
            predator_type=ptype, speed=speed, lethality=lethality,
            respawn_delay=respawn_delay,
        )

    @staticmethod
    def _zone_edge_pos(zx: int, zy: int, zone_size: int, ocean_size: int) -> tuple[int, int]:
        """Random position on edge of a zone."""
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
        return px, py

    def process_tick(self, lobster_positions: list[tuple[int, int]], ocean_size: int,
                     rift_positions: list[tuple[int, int]] | None = None,
                     pheromone_map=None,
                     spawn_rate_multiplier: float = 1.0,
                     alive_count: int = 0) -> int:
        """Spawn new predators based on density (sigmoid), move existing ones, remove expired.
        Returns number of predators spawned this tick."""
        spawned = 0

        # Extinction protection: skip spawns if population too low
        if alive_count < self.min_population and self.extinction_protection:
            # Still move and age existing predators
            self._move_all(lobster_positions, ocean_size, rift_positions, pheromone_map)
            self.active_predators = [p for p in self.active_predators if p.lifetime_remaining > 0]
            return 0

        zone_size = self.config["predators"]["zone_size"]
        max_rate = self.config["predators"]["base_spawn_rate"] * spawn_rate_multiplier
        K = self.config["predators"].get("sigmoid_K", 12)
        rift_reduction = self.config["predators"].get("rift_zone_reduction", 0.6)
        rp = rift_positions or []

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
            if rp:
                zone_cx = zx * zone_size + zone_size // 2
                zone_cy = zy * zone_size + zone_size // 2
                for rx, ry in rp:
                    if abs(zone_cx - rx) <= zone_size and abs(zone_cy - ry) <= zone_size:
                        chance *= rift_reduction
                        break

            if random.random() < chance:
                ptype = self._select_type()
                pred = self._spawn_positioned(ptype, zx, zy, zone_size, ocean_size, rp)
                if pred:
                    self.active_predators.append(pred)
                    spawned += 1

        # Move all predators (type-dispatched)
        self._move_all(lobster_positions, ocean_size, rift_positions, pheromone_map)

        # Remove expired
        self.active_predators = [p for p in self.active_predators if p.lifetime_remaining > 0]
        return spawned

    def _move_all(self, lobster_positions: list[tuple[int, int]], ocean_size: int,
                  rift_positions: list[tuple[int, int]] | None, pheromone_map):
        """Move all predators, dispatching by type."""
        rp = rift_positions or []
        for pred in self.active_predators:
            pred.lifetime_remaining -= 1

            if pred.predator_type == "shark":
                self._move_shark(pred, lobster_positions, ocean_size, pheromone_map)
            elif pred.predator_type == "eel":
                self._move_eel(pred, rp, ocean_size)
            elif pred.predator_type == "octopus":
                self._move_octopus(pred, lobster_positions, ocean_size, pheromone_map)
            else:
                # Fallback: chase (legacy behavior)
                self._move_shark(pred, lobster_positions, ocean_size, pheromone_map)

    def _move_shark(self, pred: Predator, lobster_positions: list[tuple[int, int]],
                    ocean_size: int, pheromone_map):
        """Shark: chase nearest target, speed 2.0, 70/30 lobster/danger blend."""
        dx, dy = 0.0, 0.0
        has_target = False

        if lobster_positions:
            nearest = min(
                lobster_positions,
                key=lambda p: math.sqrt((p[0] - pred.x) ** 2 + (p[1] - pred.y) ** 2),
            )
            dx = nearest[0] - pred.x
            dy = nearest[1] - pred.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 0:
                dx /= dist
                dy /= dist
                has_target = True

        # Blend with danger pheromone attraction (30% weight)
        if pheromone_map and has_target:
            danger = pheromone_map.read(int(pred.x), int(pred.y), "danger", radius=5)
            if danger:
                best = max(danger, key=lambda t: t["intensity"])
                if best["intensity"] > 1.0:
                    pdx, pdy = best["dx"], best["dy"]
                    pdist = math.sqrt(pdx * pdx + pdy * pdy)
                    if pdist > 0:
                        dx = dx * 0.7 + (pdx / pdist) * 0.3
                        dy = dy * 0.7 + (pdy / pdist) * 0.3

        if has_target or (dx != 0 or dy != 0):
            norm = math.sqrt(dx * dx + dy * dy)
            if norm > 0:
                pred.x += (dx / norm) * pred.speed
                pred.y += (dy / norm) * pred.speed
            pred.x = max(0, min(ocean_size - 1, pred.x))
            pred.y = max(0, min(ocean_size - 1, pred.y))

    def _move_eel(self, pred: Predator, rift_positions: list[tuple[int, int]], ocean_size: int):
        """Eel: stationary ambush. Tick cooldown. Respawn at new rift when cooldown expires."""
        if pred.attack_cooldown > 0:
            pred.attack_cooldown -= 1
            if pred.attack_cooldown <= 0 and rift_positions:
                # Respawn at new rift position
                tcfg = self.type_configs.get("eel", {})
                max_dist = tcfg.get("max_rift_distance", 2)
                rift = random.choice(rift_positions)
                pred.x = float(max(0, min(ocean_size - 1, rift[0] + random.randint(-max_dist, max_dist))))
                pred.y = float(max(0, min(ocean_size - 1, rift[1] + random.randint(-max_dist, max_dist))))
                pred.has_attacked = False
                pred.attack_cooldown = 0
        # Eels don't move (speed 0)

    def _move_octopus(self, pred: Predator, lobster_positions: list[tuple[int, int]],
                      ocean_size: int, pheromone_map):
        """Octopus: follow food pheromone gradient (60%), blend with chase (40%). Speed 0.5 via accumulator."""
        dx, dy = 0.0, 0.0
        has_food = False
        has_target = False

        tcfg = self.type_configs.get("octopus", {})
        pheromone_weight = tcfg.get("pheromone_weight", 0.6)
        chase_weight = 1.0 - pheromone_weight

        # Read food pheromone gradient
        if pheromone_map:
            food = pheromone_map.read(int(pred.x), int(pred.y), "food", radius=5)
            if food:
                best = max(food, key=lambda t: t["intensity"])
                if best["intensity"] > 0.1:
                    fdx, fdy = best["dx"], best["dy"]
                    fdist = math.sqrt(fdx * fdx + fdy * fdy)
                    if fdist > 0:
                        dx = (fdx / fdist) * pheromone_weight
                        dy = (fdy / fdist) * pheromone_weight
                        has_food = True

        # Chase component
        if lobster_positions:
            nearest = min(
                lobster_positions,
                key=lambda p: math.sqrt((p[0] - pred.x) ** 2 + (p[1] - pred.y) ** 2),
            )
            cdx = nearest[0] - pred.x
            cdy = nearest[1] - pred.y
            cdist = math.sqrt(cdx * cdx + cdy * cdy)
            if cdist > 0:
                if has_food:
                    dx += (cdx / cdist) * chase_weight
                    dy += (cdy / cdist) * chase_weight
                else:
                    # No food signal: random walk
                    dx = random.uniform(-1, 1)
                    dy = random.uniform(-1, 1)
                has_target = True

        if not has_food and not has_target:
            # Pure random walk
            dx = random.uniform(-1, 1)
            dy = random.uniform(-1, 1)

        # Fractional movement via accumulator (speed 0.5)
        norm = math.sqrt(dx * dx + dy * dy)
        if norm > 0:
            dx /= norm
            dy /= norm
            pred.move_accumulator += pred.speed
            if pred.move_accumulator >= 1.0:
                pred.move_accumulator -= 1.0
                pred.x += dx
                pred.y += dy
                pred.x = max(0, min(ocean_size - 1, pred.x))
                pred.y = max(0, min(ocean_size - 1, pred.y))

    def get_counts_by_type(self) -> dict[str, int]:
        """Return count of active predators by type."""
        counts: dict[str, int] = {}
        for pred in self.active_predators:
            counts[pred.predator_type] = counts.get(pred.predator_type, 0) + 1
        return counts

    def reset(self):
        self.active_predators.clear()
        self.next_pred_id = 1
