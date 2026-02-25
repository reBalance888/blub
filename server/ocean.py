"""
ocean.py — Core world simulation: grid, lobsters, movement, tick processing.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from .rift import RiftManager, Rift
from .predator import PredatorManager, Predator
from .economy import EconomyManager
from .epoch import EpochManager


@dataclass
class Lobster:
    id: str
    name: str
    x: int
    y: int
    balance: float  # internal balance for MVP tier calc
    credits_earned: float = 0.0  # earned this epoch
    credits_spent: float = 0.0  # spent on sounds this epoch
    alive: bool = True
    death_timer: int = 0
    last_epoch_reward: float = 0.0
    agent_type: str = ""  # "social", "greedy", "random", or ""
    group_hits: int = 0  # times in group >=2 at rift this epoch
    speaking: list = field(default_factory=list)  # sounds this tick


class Ocean:
    """The 2D ocean world that runs the simulation."""

    def __init__(self, config: dict):
        self.config = config
        self.size: int = config["ocean"]["size"]
        self.tick_interval: float = config["ocean"]["tick_interval"]
        self.sounds: list[str] = config["sounds"]

        # Active zone parameters
        self.density_constant: float = config["ocean"].get("density_constant", 7)
        self.active_zone_min: int = config["ocean"].get("active_zone_min", 15)
        self.active_zone_size: int = self.active_zone_min

        self.tick: int = 0
        self.lobsters: dict[str, Lobster] = {}
        self.next_lobster_id: int = 1

        # Pending actions: {agent_id: action_dict}
        self.pending_actions: dict[str, dict] = {}

        # Sounds emitted this tick: [{from, sounds, pos}]
        self.current_sounds: list[dict] = []

        # Sub-managers
        self.rift_mgr = RiftManager(config)
        self.predator_mgr = PredatorManager(config)
        self.economy = EconomyManager(config)
        self.epoch_mgr = EpochManager(config)

        # Init first epoch rifts (within active zone)
        zone_min, zone_max = self._active_zone_bounds()
        self.rift_mgr.spawn_epoch_rifts(self.epoch_mgr.epoch, zone_min, self.active_zone_size)

    # ----- Active zone -----

    def _recalculate_active_zone(self):
        """Recalculate active zone size based on connected agent count."""
        n = len(self.lobsters)
        if n == 0:
            self.active_zone_size = self.active_zone_min
        else:
            raw = int(math.sqrt(n) * self.density_constant)
            self.active_zone_size = max(self.active_zone_min, min(self.size, raw))
        # Keep rift manager in sync
        zone_min, _ = self._active_zone_bounds()
        self.rift_mgr.update_zone(zone_min, self.active_zone_size)

    def _active_zone_bounds(self) -> tuple[int, int]:
        """Return (zone_min, zone_max) — active zone centered on the map."""
        offset = (self.size - self.active_zone_size) // 2
        return offset, offset + self.active_zone_size - 1

    def _random_pos_in_zone(self) -> tuple[int, int]:
        """Random position within the active zone."""
        zone_min, zone_max = self._active_zone_bounds()
        x = random.randint(zone_min, zone_max)
        y = random.randint(zone_min, zone_max)
        return x, y

    # ----- Agent management -----

    @staticmethod
    def _extract_agent_type(name: str) -> str:
        for prefix in ("social", "greedy", "random"):
            if name.startswith(prefix):
                return prefix
        return ""

    def add_lobster(self, name: str) -> Lobster:
        lid = f"lobster_{self.next_lobster_id}"
        self.next_lobster_id += 1
        balance = self.config["economy"]["starting_balance"]
        agent_type = self._extract_agent_type(name)
        # Temporarily add to recalculate zone, then spawn inside it
        lob = Lobster(id=lid, name=name, x=0, y=0, balance=balance, agent_type=agent_type)
        self.lobsters[lid] = lob
        self._recalculate_active_zone()
        x, y = self._random_pos_in_zone()
        lob.x = x
        lob.y = y
        zone_min, zone_max = self._active_zone_bounds()
        print(f"[+] {lid} ({name}) joined at ({x},{y}) | agents={len(self.lobsters)} active_zone={self.active_zone_size} [{zone_min}..{zone_max}]")
        return lob

    def remove_lobster(self, agent_id: str):
        self.lobsters.pop(agent_id, None)
        self._recalculate_active_zone()
        print(f"[-] {agent_id} left | agents={len(self.lobsters)} active_zone={self.active_zone_size}")

    # ----- Actions -----

    def submit_action(self, agent_id: str, actions: dict):
        if agent_id in self.lobsters:
            self.pending_actions[agent_id] = actions

    # ----- Tick -----

    def process_tick(self):
        self.tick += 1
        self.current_sounds = []

        # 1. Process movements
        self._process_movements()

        # 2. Process sounds (before rift rewards so we know who spoke)
        self._process_sounds()

        # 3. Rift rewards (group bonus)
        self._process_rift_rewards()

        # 4. Predator spawns & movement
        self.predator_mgr.process_tick(self._get_lobster_positions(), self.size)

        # 5. Predator kills
        self._process_predator_kills()

        # 6. Rift depletion & respawn
        self.rift_mgr.tick_rifts(self.tick)

        # 7. Death timers
        self._process_death_timers()

        # 8. Check epoch end
        if self.epoch_mgr.tick():
            self._end_epoch()

        # Clear pending actions
        self.pending_actions.clear()

        # Log every 100 ticks
        if self.tick % 100 == 0:
            self._log_stats()

    def _process_movements(self):
        directions = {
            "north": (0, -1),
            "south": (0, 1),
            "east": (1, 0),
            "west": (-1, 0),
            "stay": (0, 0),
        }
        zone_min, zone_max = self._active_zone_bounds()
        for agent_id, actions in self.pending_actions.items():
            lob = self.lobsters.get(agent_id)
            if not lob or not lob.alive:
                continue
            move = actions.get("move", "stay")
            dx, dy = directions.get(move, (0, 0))
            lob.x = max(zone_min, min(zone_max, lob.x + dx))
            lob.y = max(zone_min, min(zone_max, lob.y + dy))

    def _process_sounds(self):
        sound_cost = self.config["economy"]["sound_credit_cost"]
        valid_sounds = set(self.sounds)

        for agent_id, actions in self.pending_actions.items():
            lob = self.lobsters.get(agent_id)
            if not lob or not lob.alive:
                continue
            raw_sounds = actions.get("speak", [])
            if not raw_sounds:
                lob.speaking = []
                continue

            # Validate and limit to 5
            sounds = [s for s in raw_sounds if s in valid_sounds][:5]
            lob.speaking = sounds

            if sounds:
                cost = len(sounds) * sound_cost
                lob.credits_spent += cost
                self.current_sounds.append({
                    "from": agent_id,
                    "sounds": sounds,
                    "pos": [lob.x, lob.y],
                })

    def _process_rift_rewards(self):
        for rift in self.rift_mgr.active_rifts:
            nearby = self._lobsters_near(rift.x, rift.y, self.config["rifts"]["radius"])
            if not nearby:
                continue
            n = len(nearby)
            reward = self.rift_mgr.calc_group_reward(n)
            total_drain = n * reward
            if total_drain > rift.richness:
                reward = rift.richness / n if n > 0 else 0
                total_drain = rift.richness
            for lob in nearby:
                lob.credits_earned += reward
                if n >= 2:
                    lob.group_hits += 1
            rift.richness -= total_drain

    def _process_predator_kills(self):
        kill_radius = self.config["predators"]["kill_radius"]
        death_timeout = self.config["economy"]["death_timeout"]
        death_penalty = self.config["economy"]["death_credit_penalty"]

        for pred in self.predator_mgr.active_predators:
            for lob in self.lobsters.values():
                if not lob.alive:
                    continue
                dist = math.sqrt((lob.x - pred.x) ** 2 + (lob.y - pred.y) ** 2)
                if dist <= kill_radius:
                    lob.alive = False
                    lob.death_timer = death_timeout
                    penalty = lob.credits_earned * death_penalty
                    lob.credits_earned = max(0, lob.credits_earned - penalty)

    def _process_death_timers(self):
        for lob in self.lobsters.values():
            if not lob.alive:
                lob.death_timer -= 1
                if lob.death_timer <= 0:
                    lob.alive = True
                    # Respawn within active zone
                    lob.x, lob.y = self._random_pos_in_zone()

    def _end_epoch(self):
        """End of epoch: distribute rewards, reset credits, new rifts."""
        epoch_num = self.epoch_mgr.epoch - 1
        pool = self.config["economy"]["simulated_epoch_pool"]
        total_net = sum(
            max(0, lob.credits_earned - lob.credits_spent)
            for lob in self.lobsters.values()
        )

        # Collect per-agent stats BEFORE resetting
        # {type: {credits: [], rewards: [], group_hits: []}}
        type_stats: dict[str, dict[str, list]] = {}
        for lob in self.lobsters.values():
            net = max(0, lob.credits_earned - lob.credits_spent)
            if total_net > 0:
                reward = pool * (net / total_net)
            else:
                reward = 0
            lob.last_epoch_reward = reward
            lob.balance += reward

            agent_type = lob.agent_type or "unknown"

            if agent_type not in type_stats:
                type_stats[agent_type] = {"credits": [], "rewards": [], "group_hits": []}
            type_stats[agent_type]["credits"].append(lob.credits_earned - lob.credits_spent)
            type_stats[agent_type]["rewards"].append(reward)
            type_stats[agent_type]["group_hits"].append(lob.group_hits)

            # Reset credits and group_hits for new epoch
            lob.credits_earned = 0.0
            lob.credits_spent = 0.0
            lob.group_hits = 0

        # New rifts (within active zone)
        zone_min, zone_max = self._active_zone_bounds()
        self.rift_mgr.spawn_epoch_rifts(self.epoch_mgr.epoch, zone_min, self.active_zone_size)

        # Log epoch results
        print(f"\n=== EPOCH {epoch_num} COMPLETE ===")
        print(f"  Pool: {pool:,.0f} | Total net credits: {total_net:,.1f}")
        top = sorted(
            self.lobsters.values(),
            key=lambda l: l.last_epoch_reward,
            reverse=True,
        )[:5]
        for i, lob in enumerate(top):
            print(f"  #{i+1} {lob.id} ({lob.name}): {lob.last_epoch_reward:,.0f} BLUB")

        # Summary by agent type
        print("=== EPOCH SUMMARY ===")
        for atype in ("social", "greedy", "random"):
            stats = type_stats.get(atype)
            if not stats or not stats["credits"]:
                continue
            avg_credits = sum(stats["credits"]) / len(stats["credits"])
            total_reward = sum(stats["rewards"])
            total_group_hits = sum(stats["group_hits"])
            print(
                f"[EPOCH {epoch_num}] {atype}: "
                f"avg_credits={avg_credits:.1f} "
                f"total_reward={total_reward:,.0f} "
                f"group_hits={total_group_hits}"
            )
        print()

    # ----- Helpers -----

    def _lobsters_near(self, x: int, y: int, radius: int) -> list[Lobster]:
        result = []
        for lob in self.lobsters.values():
            if not lob.alive:
                continue
            if abs(lob.x - x) <= radius and abs(lob.y - y) <= radius:
                result.append(lob)
        return result

    def _get_lobster_positions(self) -> list[tuple[int, int]]:
        return [(lob.x, lob.y) for lob in self.lobsters.values() if lob.alive]

    def get_tier(self, lob: Lobster) -> str:
        tiers = self.config["tiers"]
        current = "shrimp"
        for name, info in tiers.items():
            if lob.balance >= info["min_hold"]:
                current = name
        return current

    def get_tier_info(self, tier: str) -> dict:
        return self.config["tiers"].get(tier, self.config["tiers"]["shrimp"])

    # ----- State for agents -----

    def get_agent_state(self, agent_id: str) -> Optional[dict]:
        lob = self.lobsters.get(agent_id)
        if not lob:
            return None

        tier = self.get_tier(lob)
        tier_info = self.get_tier_info(tier)
        vision = tier_info["vision"]
        sound_range = tier_info["sound_range"]

        # Nearby lobsters
        nearby_lobs = []
        for other in self.lobsters.values():
            if other.id == agent_id or not other.alive:
                continue
            dx = other.x - lob.x
            dy = other.y - lob.y
            if abs(dx) <= vision and abs(dy) <= vision:
                nearby_lobs.append({
                    "id": other.id,
                    "position": [other.x, other.y],
                    "relative": [dx, dy],
                })

        # Nearby rifts
        nearby_rifts = []
        for rift in self.rift_mgr.active_rifts:
            dx = rift.x - lob.x
            dy = rift.y - lob.y
            if abs(dx) <= vision and abs(dy) <= vision:
                pct = rift.richness / self.config["rifts"]["base_richness"] if self.config["rifts"]["base_richness"] > 0 else 0
                nearby_rifts.append({
                    "id": rift.id,
                    "position": [rift.x, rift.y],
                    "richness_pct": round(pct, 2),
                    "relative": [dx, dy],
                })

        # Nearby predators
        nearby_preds = []
        for pred in self.predator_mgr.active_predators:
            dx = pred.x - lob.x
            dy = pred.y - lob.y
            if abs(dx) <= vision and abs(dy) <= vision:
                nearby_preds.append({
                    "id": pred.id,
                    "position": [pred.x, pred.y],
                    "relative": [dx, dy],
                })

        # Sounds heard (within sound_range)
        sounds_heard = []
        for snd in self.current_sounds:
            if snd["from"] == agent_id:
                continue
            sx, sy = snd["pos"]
            dist = max(abs(sx - lob.x), abs(sy - lob.y))
            if dist <= sound_range:
                sounds_heard.append({
                    "from": snd["from"],
                    "sounds": snd["sounds"],
                    "distance": dist,
                    "tick": self.tick,
                })

        net_credits = lob.credits_earned - lob.credits_spent

        return {
            "tick": self.tick,
            "epoch": self.epoch_mgr.epoch,
            "epoch_ticks_remaining": self.epoch_mgr.ticks_remaining(),
            "my_position": [lob.x, lob.y],
            "my_credits": round(lob.credits_earned, 2),
            "my_credits_spent": round(lob.credits_spent, 2),
            "my_net_credits": round(net_credits, 2),
            "my_tier": tier,
            "nearby_lobsters": nearby_lobs,
            "nearby_rifts": nearby_rifts,
            "nearby_predators": nearby_preds,
            "sounds_heard": sounds_heard,
            "alive": lob.alive,
            "last_epoch_reward": round(lob.last_epoch_reward, 2),
        }

    # ----- Viewer state -----

    def get_viewer_state(self) -> dict:
        lobsters = []
        for lob in self.lobsters.values():
            lobsters.append({
                "id": lob.id,
                "name": lob.name,
                "agent_type": lob.agent_type,
                "pos": [lob.x, lob.y],
                "tier": self.get_tier(lob),
                "alive": lob.alive,
                "speaking": lob.speaking,
                "net_credits": round(lob.credits_earned - lob.credits_spent, 2),
            })

        rifts = []
        for rift in self.rift_mgr.active_rifts:
            base = self.config["rifts"]["base_richness"]
            pct = rift.richness / base if base > 0 else 0
            rifts.append({
                "id": rift.id,
                "pos": [rift.x, rift.y],
                "richness_pct": round(pct, 2),
            })

        predators = []
        for pred in self.predator_mgr.active_predators:
            predators.append({
                "id": pred.id,
                "pos": [pred.x, pred.y],
            })

        total_earned = sum(l.credits_earned for l in self.lobsters.values())
        total_spent = sum(l.credits_spent for l in self.lobsters.values())

        top_earner = max(
            self.lobsters.values(),
            key=lambda l: l.last_epoch_reward,
            default=None,
        )

        zone_min, zone_max = self._active_zone_bounds()

        return {
            "tick": self.tick,
            "epoch": self.epoch_mgr.epoch,
            "epoch_ticks_remaining": self.epoch_mgr.ticks_remaining(),
            "ocean_size": self.size,
            "active_zone": {
                "size": self.active_zone_size,
                "min": zone_min,
                "max": zone_max,
            },
            "lobsters": lobsters,
            "rifts": rifts,
            "predators": predators,
            "sounds": self.current_sounds,
            "stats": {
                "total_lobsters": len(self.lobsters),
                "alive_lobsters": sum(1 for l in self.lobsters.values() if l.alive),
                "active_rifts": len(self.rift_mgr.active_rifts),
                "total_credits_earned": round(total_earned, 2),
                "total_credits_spent_on_sounds": round(total_spent, 2),
                "epoch_pool": self.config["economy"]["simulated_epoch_pool"],
                "last_epoch_top_earner": {
                    "id": top_earner.id if top_earner else "",
                    "reward": round(top_earner.last_epoch_reward, 2) if top_earner else 0,
                },
            },
        }

    def _log_stats(self):
        alive = sum(1 for l in self.lobsters.values() if l.alive)
        total = len(self.lobsters)
        rifts = len(self.rift_mgr.active_rifts)
        preds = len(self.predator_mgr.active_predators)
        epoch = self.epoch_mgr.epoch
        remaining = self.epoch_mgr.ticks_remaining()
        print(
            f"[Tick {self.tick}] Epoch {epoch} ({remaining} left) | "
            f"Lobsters: {alive}/{total} | Rifts: {rifts} | Predators: {preds} | "
            f"Zone: {self.active_zone_size}"
        )

    def reset(self):
        """Reset the world for testing."""
        self.tick = 0
        self.lobsters.clear()
        self.next_lobster_id = 1
        self.pending_actions.clear()
        self.current_sounds.clear()
        self.rift_mgr.reset()
        self.predator_mgr.reset()
        self.epoch_mgr.reset()
        self._recalculate_active_zone()
        zone_min, zone_max = self._active_zone_bounds()
        self.rift_mgr.spawn_epoch_rifts(self.epoch_mgr.epoch, zone_min, self.active_zone_size)
