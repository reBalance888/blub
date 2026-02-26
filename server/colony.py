"""
colony.py — ColonyManager: detects and manages persistent lobster clusters.
Lifecycle: detection → candidacy → formation → dissolution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


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

    def tick(self, lobsters: dict, rifts: list, current_tick: int):
        """Update colonies: detect clusters, promote candidates, dissolve scattered."""
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
            elif age > self.persistence_ticks * 3:
                # Stale candidate — remove
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
