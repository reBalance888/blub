"""
pheromone.py — Four-layer chemical trail map:
  FOOD trails, DANGER trails, NO-ENTRY trails, COLONY SCENT.
Sparse dict architecture — only cells with non-zero values are stored.
"""
from __future__ import annotations


class PheromoneMap:
    """Four-layer chemical trail map."""

    def __init__(self, config: dict):
        self.cfg = config.get("pheromones", {})
        # Differential decay: food trails evaporate faster, danger persists longer
        self.food_decay = self.cfg.get("food_decay_rate", self.cfg.get("decay_rate", 0.93))
        self.danger_decay = self.cfg.get("danger_decay_rate", self.cfg.get("decay_rate", 0.97))
        self.noentry_decay = self.cfg.get("noentry_decay_rate", 0.90)
        self.colony_scent_decay = self.cfg.get("colony_scent_decay_rate", 0.95)
        self.diffusion_rate = self.cfg.get("diffusion_rate", 0.1)
        self.max_intensity = self.cfg.get("max_intensity", 10.0)
        self.evaporation_threshold = 0.01

        # Sparse: {(x,y): intensity}
        self.food_trails: dict[tuple[int, int], float] = {}
        self.danger_trails: dict[tuple[int, int], float] = {}
        self.noentry_trails: dict[tuple[int, int], float] = {}
        # Colony scent: {(x,y): {colony_id: intensity}}
        self.colony_scent: dict[tuple[int, int], dict[str, float]] = {}

    def deposit(self, x: int, y: int, layer: str, amount: float):
        """Agent deposits pheromone at position."""
        if layer == "food":
            trails = self.food_trails
        elif layer == "danger":
            trails = self.danger_trails
        elif layer == "noentry":
            trails = self.noentry_trails
        else:
            return
        current = trails.get((x, y), 0.0)
        trails[(x, y)] = min(current + amount, self.max_intensity)

    def deposit_colony_scent(self, x: int, y: int, colony_id: str, amount: float):
        """Deposit colony scent at position (nested: colony_id -> intensity)."""
        if (x, y) not in self.colony_scent:
            self.colony_scent[(x, y)] = {}
        cell = self.colony_scent[(x, y)]
        current = cell.get(colony_id, 0.0)
        cell[colony_id] = min(current + amount, self.max_intensity)

    def read(self, x: int, y: int, layer: str, radius: int = 3) -> list[dict]:
        """Read pheromone intensities within radius. Returns [{dx, dy, intensity}]."""
        if layer == "food":
            trails = self.food_trails
        elif layer == "danger":
            trails = self.danger_trails
        elif layer == "noentry":
            trails = self.noentry_trails
        else:
            return []
        result = []
        for (px, py), intensity in trails.items():
            dx, dy = px - x, py - y
            if abs(dx) <= radius and abs(dy) <= radius:
                result.append({"dx": dx, "dy": dy, "intensity": round(intensity, 3)})
        return result

    def read_colony_scent(self, x: int, y: int, radius: int,
                          my_colony_id: str | None,
                          same_weight: float, other_weight: float) -> list[dict]:
        """Read colony scent within radius, applying trust weights.
        Returns [{dx, dy, intensity, colony_id, trust}]."""
        result = []
        for (px, py), colonies in self.colony_scent.items():
            dx, dy = px - x, py - y
            if abs(dx) <= radius and abs(dy) <= radius:
                for cid, intensity in colonies.items():
                    is_same = (my_colony_id is not None and cid == my_colony_id)
                    trust = same_weight if is_same else other_weight
                    result.append({
                        "dx": dx, "dy": dy,
                        "intensity": round(intensity, 3),
                        "colony_id": cid,
                        "trust": trust,
                    })
        return result

    def tick(self):
        """Decay + diffusion each tick."""
        # Food, danger, noentry: decay + diffusion
        for trails, decay_rate in ((self.food_trails, self.food_decay),
                                   (self.danger_trails, self.danger_decay),
                                   (self.noentry_trails, self.noentry_decay)):
            # Decay
            to_remove = []
            new_vals = {}
            for pos, intensity in trails.items():
                new_val = intensity * decay_rate
                if new_val < self.evaporation_threshold:
                    to_remove.append(pos)
                else:
                    new_vals[pos] = new_val
            for pos in to_remove:
                del trails[pos]
            trails.update(new_vals)

            # Diffusion: spread fraction to 4 neighbors
            if self.diffusion_rate > 0:
                spread: dict[tuple[int, int], float] = {}
                for (x, y), intensity in list(trails.items()):
                    amt = intensity * self.diffusion_rate * 0.25
                    for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                        spread[(nx, ny)] = spread.get((nx, ny), 0) + amt
                    trails[(x, y)] = intensity * (1 - self.diffusion_rate)
                for pos, amt in spread.items():
                    if pos in trails:
                        trails[pos] = min(trails[pos] + amt, self.max_intensity)
                    elif amt >= self.evaporation_threshold:
                        trails[pos] = amt

        # Colony scent: decay only (NO diffusion — sharp boundaries create language pressure)
        to_remove_cells = []
        for pos, colonies in self.colony_scent.items():
            to_remove_ids = []
            for cid, intensity in colonies.items():
                new_val = intensity * self.colony_scent_decay
                if new_val < self.evaporation_threshold:
                    to_remove_ids.append(cid)
                else:
                    colonies[cid] = new_val
            for cid in to_remove_ids:
                del colonies[cid]
            if not colonies:
                to_remove_cells.append(pos)
        for pos in to_remove_cells:
            del self.colony_scent[pos]

    def get_viewer_data(self, zone_min: int, zone_max: int) -> list[dict]:
        """Pheromone data for viewer (only within active zone, above threshold)."""
        data = []
        min_display = 0.1
        for (x, y), intensity in self.food_trails.items():
            if zone_min <= x <= zone_max and zone_min <= y <= zone_max and intensity >= min_display:
                data.append({
                    "pos": [x, y], "type": "food",
                    "intensity": round(intensity / self.max_intensity, 2),
                })
        for (x, y), intensity in self.danger_trails.items():
            if zone_min <= x <= zone_max and zone_min <= y <= zone_max and intensity >= min_display:
                data.append({
                    "pos": [x, y], "type": "danger",
                    "intensity": round(intensity / self.max_intensity, 2),
                })
        for (x, y), intensity in self.noentry_trails.items():
            if zone_min <= x <= zone_max and zone_min <= y <= zone_max and intensity >= min_display:
                data.append({
                    "pos": [x, y], "type": "noentry",
                    "intensity": round(intensity / self.max_intensity, 2),
                })
        # Colony scent: aggregate per-cell, pick dominant colony
        for (x, y), colonies in self.colony_scent.items():
            if zone_min <= x <= zone_max and zone_min <= y <= zone_max and colonies:
                dominant_id = max(colonies, key=colonies.get)
                total_intensity = sum(colonies.values())
                if total_intensity >= min_display:
                    data.append({
                        "pos": [x, y], "type": "colony_scent",
                        "intensity": round(min(total_intensity / self.max_intensity, 1.0), 2),
                        "colony_id": dominant_id,
                    })
        return data

    def clear_zone(self, zone_min: int, zone_max: int):
        """Clear trails outside active zone (cleanup)."""
        for trails in (self.food_trails, self.danger_trails, self.noentry_trails):
            to_remove = [
                p for p in trails
                if not (zone_min <= p[0] <= zone_max and zone_min <= p[1] <= zone_max)
            ]
            for p in to_remove:
                del trails[p]
        # Colony scent cleanup
        to_remove = [
            p for p in self.colony_scent
            if not (zone_min <= p[0] <= zone_max and zone_min <= p[1] <= zone_max)
        ]
        for p in to_remove:
            del self.colony_scent[p]
