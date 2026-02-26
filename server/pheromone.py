"""
pheromone.py — Two-layer chemical trail map: FOOD trails + DANGER trails.
Sparse dict architecture — only cells with non-zero values are stored.
"""
from __future__ import annotations


class PheromoneMap:
    """Two-layer chemical trail map: FOOD trails + DANGER trails."""

    def __init__(self, config: dict):
        self.cfg = config.get("pheromones", {})
        self.decay_rate = self.cfg.get("decay_rate", 0.95)
        self.diffusion_rate = self.cfg.get("diffusion_rate", 0.1)
        self.max_intensity = self.cfg.get("max_intensity", 10.0)
        self.evaporation_threshold = 0.01

        # Sparse: {(x,y): intensity}
        self.food_trails: dict[tuple[int, int], float] = {}
        self.danger_trails: dict[tuple[int, int], float] = {}

    def deposit(self, x: int, y: int, layer: str, amount: float):
        """Agent deposits pheromone at position."""
        trails = self.food_trails if layer == "food" else self.danger_trails
        current = trails.get((x, y), 0.0)
        trails[(x, y)] = min(current + amount, self.max_intensity)

    def read(self, x: int, y: int, layer: str, radius: int = 3) -> list[dict]:
        """Read pheromone intensities within radius. Returns [{dx, dy, intensity}]."""
        trails = self.food_trails if layer == "food" else self.danger_trails
        result = []
        for (px, py), intensity in trails.items():
            dx, dy = px - x, py - y
            if abs(dx) <= radius and abs(dy) <= radius:
                result.append({"dx": dx, "dy": dy, "intensity": round(intensity, 3)})
        return result

    def tick(self):
        """Decay + diffusion each tick."""
        for trails in (self.food_trails, self.danger_trails):
            # Decay
            to_remove = []
            new_vals = {}
            for pos, intensity in trails.items():
                new_val = intensity * self.decay_rate
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
        return data

    def clear_zone(self, zone_min: int, zone_max: int):
        """Clear trails outside active zone (cleanup)."""
        for trails in (self.food_trails, self.danger_trails):
            to_remove = [
                p for p in trails
                if not (zone_min <= p[0] <= zone_max and zone_min <= p[1] <= zone_max)
            ]
            for p in to_remove:
                del trails[p]
