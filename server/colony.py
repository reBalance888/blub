"""
colony.py — ColonyManager: detects and manages persistent lobster clusters.
ColonyCacheManager: per-colony cultural caches with dormancy.
Lifecycle: detection → candidacy → formation → dissolution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Colony:
    id: str
    center_x: float
    center_y: float
    members: set[str]
    formed_tick: int
    rift_id: str | None = None
    total_reward: float = 0.0


class ColonyManager:
    """Detects and manages persistent lobster clusters."""

    def __init__(self, config: dict):
        self.cfg = config.get("colonies", {})
        self.formation_threshold = self.cfg.get("formation_threshold", 4)
        self.formation_radius = self.cfg.get("formation_radius", 3)
        self.persistence_ticks = self.cfg.get("persistence_ticks", 30)
        self.reward_bonus = self.cfg.get("reward_bonus", 1.2)
        self.max_colonies = self.cfg.get("max_colonies", 5)

        self.colonies: dict[str, Colony] = {}
        self._next_id = 1
        # Proto-colonies being tracked: {key: {members, center_x, center_y, first_tick}}
        self._candidates: dict[str, dict] = {}
        # Dissolution events from last tick: [(colony_id, center_x, center_y)]
        self._last_dissolved: list[tuple[str, float, float]] = []

    def tick(self, lobsters: dict, rifts: list, current_tick: int):
        """Update colonies: detect clusters, promote candidates, dissolve scattered."""
        self._last_dissolved = []
        # 1. Find current clusters
        clusters = self._detect_clusters(lobsters)

        # 2. Match clusters to existing colonies — update members/centers
        matched_colony_ids: set[str] = set()
        for cluster in clusters:
            best_colony = self._match_to_colony(cluster)
            if best_colony:
                matched_colony_ids.add(best_colony.id)
                best_colony.members = cluster["members"]
                best_colony.center_x = cluster["cx"]
                best_colony.center_y = cluster["cy"]
                # Associate with nearest rift
                best_colony.rift_id = self._nearest_rift_id(
                    cluster["cx"], cluster["cy"], rifts
                )
            else:
                # Match to candidate or create new candidate
                self._update_candidates(cluster, current_tick, rifts)

        # 3. Promote candidates that persisted
        self._promote_candidates(current_tick, rifts)

        # 4. Dissolve colonies whose members scattered
        to_dissolve = []
        for cid, colony in self.colonies.items():
            if cid not in matched_colony_ids:
                # No matching cluster found — check if still viable
                alive_members = sum(
                    1 for mid in colony.members
                    if mid in lobsters and lobsters[mid].alive
                )
                if alive_members < self.formation_threshold // 2:
                    to_dissolve.append(cid)
                else:
                    # Recalculate center from surviving members
                    xs, ys = [], []
                    for mid in colony.members:
                        lob = lobsters.get(mid)
                        if lob and lob.alive:
                            xs.append(lob.x)
                            ys.append(lob.y)
                    if xs:
                        colony.center_x = sum(xs) / len(xs)
                        colony.center_y = sum(ys) / len(ys)
        for cid in to_dissolve:
            colony = self.colonies[cid]
            self._last_dissolved.append((cid, colony.center_x, colony.center_y))
            print(f"[COLONY] Dissolved colony {cid}")
            del self.colonies[cid]

    def _detect_clusters(self, lobsters: dict) -> list[dict]:
        """Find groups of N+ alive agents within formation_radius (DBSCAN-like)."""
        alive = [
            lob for lob in lobsters.values()
            if lob.alive
        ]
        if len(alive) < self.formation_threshold:
            return []

        visited: set[str] = set()
        clusters = []
        radius = self.formation_radius

        for seed in alive:
            if seed.id in visited:
                continue
            # Expand cluster from seed
            members: set[str] = set()
            queue = [seed]
            while queue:
                lob = queue.pop()
                if lob.id in members:
                    continue
                members.add(lob.id)
                visited.add(lob.id)
                for other in alive:
                    if other.id not in members:
                        dist = abs(lob.x - other.x) + abs(lob.y - other.y)
                        if dist <= radius:
                            queue.append(other)

            if len(members) >= self.formation_threshold:
                xs = [lobsters[m].x for m in members]
                ys = [lobsters[m].y for m in members]
                clusters.append({
                    "members": members,
                    "cx": sum(xs) / len(xs),
                    "cy": sum(ys) / len(ys),
                })

        return clusters

    def _match_to_colony(self, cluster: dict) -> Colony | None:
        """Match a cluster to an existing colony by member overlap."""
        best = None
        best_overlap = 0
        for colony in self.colonies.values():
            overlap = len(colony.members & cluster["members"])
            if overlap > best_overlap:
                best_overlap = overlap
                best = colony
        # Require at least 50% overlap
        if best and best_overlap >= len(best.members) * 0.5:
            return best
        return None

    def _update_candidates(self, cluster: dict, current_tick: int, rifts: list):
        """Track proto-colonies."""
        # Try to match to existing candidate by center proximity
        for key, cand in self._candidates.items():
            dist = abs(cand["cx"] - cluster["cx"]) + abs(cand["cy"] - cluster["cy"])
            if dist <= self.formation_radius * 2:
                cand["members"] = cluster["members"]
                cand["cx"] = cluster["cx"]
                cand["cy"] = cluster["cy"]
                return

        # New candidate
        key = f"cand_{current_tick}_{len(self._candidates)}"
        self._candidates[key] = {
            "members": cluster["members"],
            "cx": cluster["cx"],
            "cy": cluster["cy"],
            "first_tick": current_tick,
        }

    def _promote_candidates(self, current_tick: int, rifts: list):
        """Promote candidates that have persisted long enough."""
        to_remove = []
        for key, cand in self._candidates.items():
            age = current_tick - cand["first_tick"]
            if age >= self.persistence_ticks:
                if len(self.colonies) < self.max_colonies:
                    cid = f"colony_{self._next_id}"
                    self._next_id += 1
                    rift_id = self._nearest_rift_id(cand["cx"], cand["cy"], rifts)
                    colony = Colony(
                        id=cid,
                        center_x=cand["cx"],
                        center_y=cand["cy"],
                        members=cand["members"],
                        formed_tick=current_tick,
                        rift_id=rift_id,
                    )
                    self.colonies[cid] = colony
                    print(
                        f"[COLONY] Formed {cid} at ({cand['cx']:.0f},{cand['cy']:.0f}) "
                        f"with {len(cand['members'])} members"
                    )
                to_remove.append(key)
        for key in to_remove:
            self._candidates.pop(key, None)

    @staticmethod
    def _nearest_rift_id(cx: float, cy: float, rifts: list) -> str | None:
        """Find nearest rift to a center position. rifts = [(x, y, id), ...]."""
        if not rifts:
            return None
        best = min(rifts, key=lambda r: abs(r[0] - cx) + abs(r[1] - cy))
        dist = abs(best[0] - cx) + abs(best[1] - cy)
        return best[2] if dist <= 5 else None

    def get_colony_for(self, agent_id: str) -> Colony | None:
        """Which colony does this agent belong to?"""
        for colony in self.colonies.values():
            if agent_id in colony.members:
                return colony
        return None

    def get_viewer_data(self) -> list[dict]:
        """Colony data for viewer rendering."""
        return [
            {
                "id": c.id,
                "center": [round(c.center_x), round(c.center_y)],
                "size": len(c.members),
                "total_reward": round(c.total_reward, 1),
            }
            for c in self.colonies.values()
        ]


class ColonyCacheManager:
    """Manages per-colony cultural caches, independent of Colony lifecycle.
    Colonies are recreated by DBSCAN each tick, but caches persist.
    Dissolved colony caches go dormant and can be revived."""

    def __init__(self, config: dict, cache_class: type):
        cc = config.get("colony_cache", {})
        self.enabled = cc.get("enabled", False)
        self.global_deposit_weight = cc.get("global_deposit_weight", 0.30)
        self.colony_inherit_frac = cc.get("colony_inherit_frac", 0.60)
        self.global_inherit_frac = cc.get("global_inherit_frac", 0.40)
        self.dormant_ttl = cc.get("dormant_ttl", 500)
        self.decay_rate = cc.get("cache_decay_rate", 0.98)
        self._config = config
        self._cache_class = cache_class

        # Active colony caches: {colony_id: CulturalCache}
        self.caches: dict[str, Any] = {}
        # Dormant caches: {colony_id: (CulturalCache, death_tick, cx, cy)}
        self.dormant: dict[str, tuple[Any, int, float, float]] = {}

    def get_or_create_cache(self, colony_id: str) -> Any:
        """Get existing cache for colony_id, revive from dormant, or create new."""
        if colony_id in self.caches:
            return self.caches[colony_id]
        # Check dormant
        if colony_id in self.dormant:
            cache, _, _, _ = self.dormant.pop(colony_id)
            self.caches[colony_id] = cache
            print(f"[COLONY_CACHE] Revived dormant cache for {colony_id}")
            return cache
        # Create new
        cache = self._cache_class(self._config)
        self.caches[colony_id] = cache
        print(f"[COLONY_CACHE] Created new cache for {colony_id}")
        return cache

    def on_colony_dissolved(self, colony_id: str, cx: float, cy: float, tick: int):
        """Move active cache to dormant when colony dissolves."""
        if colony_id in self.caches:
            cache = self.caches.pop(colony_id)
            self.dormant[colony_id] = (cache, tick, cx, cy)
            print(f"[COLONY_CACHE] {colony_id} cache moved to dormant (TTL={self.dormant_ttl})")

    def tick(self, current_tick: int, active_colony_ids: set[str]):
        """Expire dormant caches past TTL. Remove caches for colonies that no longer exist."""
        expired = []
        for cid, (cache, death_tick, cx, cy) in self.dormant.items():
            if current_tick - death_tick > self.dormant_ttl:
                expired.append(cid)
        for cid in expired:
            self.dormant.pop(cid)
            print(f"[COLONY_CACHE] Dormant cache {cid} expired")

        # Clean up active caches for colonies that no longer exist and aren't dormant
        orphaned = [cid for cid in self.caches if cid not in active_colony_ids]
        for cid in orphaned:
            # Move to dormant instead of deleting
            cache = self.caches.pop(cid)
            self.dormant[cid] = (cache, current_tick, 0.0, 0.0)

    def epoch_decay(self):
        """Decay all active + dormant caches."""
        for cache in self.caches.values():
            cache.epoch_decay()
        for cache, _, _, _ in self.dormant.values():
            cache.epoch_decay()
