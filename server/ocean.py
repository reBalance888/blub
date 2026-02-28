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
from .metrics import MetricsLogger
from .cultural_cache import CulturalCache
from .pheromone import PheromoneMap
from .colony import ColonyManager, ColonyCacheManager
from .tidal import TidalEngine


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
    grace_ticks: int = 0  # immunity after spawn/respawn
    deaths_this_epoch: int = 0  # killed by predators count
    speaking: list = field(default_factory=list)  # sounds this tick
    age: int = 0  # ticks since spawn (for turnover)
    retired: bool = False  # flagged for replacement
    reward_this_tick: float = 0.0  # rift reward earned THIS tick (for pheromone deposit)
    path_history: list = field(default_factory=list)  # recent positions for trail deposit
    current_role: str = ""  # Phase 2: agent's dominant role (forage/scout/guard/teach)


class Ocean:
    """The 2D ocean world that runs the simulation."""

    def __init__(self, config: dict):
        self.config = config
        self.tick_interval: float = config["ocean"]["tick_interval"]
        self.sounds: list[str] = config["sounds"]

        # Active zone parameters — ocean expands dynamically with agent count
        self.density_constant: float = config["ocean"].get("density_constant", 7)
        self.active_zone_min: int = config["ocean"].get("active_zone_min", 15)
        self.active_zone_size: int = self.active_zone_min
        # size is dynamic: active_zone + margins, starts at minimum
        margin = max(10, self.active_zone_min // 5)
        self.size: int = self.active_zone_min + margin * 2

        self.tick: int = 0
        self.lobsters: dict[str, Lobster] = {}
        self.next_lobster_id: int = 1
        self.epoch_history: list[dict] = []

        # Pending actions: {agent_id: action_dict}
        self.pending_actions: dict[str, dict] = {}

        # Sounds emitted this tick: [{from, sounds, pos}]
        self.current_sounds: list[dict] = []

        # Server-side sound-context correlations for emergent dictionary
        # {sound: {context: count}}
        self.sound_correlations: dict[str, dict[str, int]] = {}

        # Sub-managers
        self.rift_mgr = RiftManager(config)
        self.predator_mgr = PredatorManager(config)
        self.economy = EconomyManager(config)
        self.epoch_mgr = EpochManager(config)
        self.metrics_logger = MetricsLogger()
        self.latest_metrics: dict = {}
        self.cultural_cache = CulturalCache(config)
        self.pheromone_map = PheromoneMap(config)
        self.colony_mgr = ColonyManager(config)
        self.colony_cache_mgr = ColonyCacheManager(config, CulturalCache)
        self.tidal = TidalEngine(config.get("tidal", {}))
        self.global_tick: int = 0  # never resets between epochs
        self._eruption_count: int = 0  # eruptions this epoch (for metrics)
        self._eruption_events: list[dict] = []  # current tick events (for viewer)

        # Tidal epoch accumulators (reset each epoch)
        self._tidal_predators_spawned_day: int = 0
        self._tidal_predators_spawned_night: int = 0
        self._tidal_credits_earned: float = 0.0
        self._tidal_deaths_day: int = 0
        self._tidal_deaths_night: int = 0

        # Predator mosaic accumulators (reset every 100 ticks with metrics)
        self._predators_by_type: dict[str, int] = {}
        self._deaths_by_predator_type: dict[str, int] = {}
        self._pheromone_clear_events: int = 0

        # CSR tracking: {agent_id: (heard_tick, agent_x, agent_y, speaker_type, listener_type)}
        self._csr_pending: dict[str, tuple[int, int, int, str, str]] = {}
        self._csr_heard_events: int = 0
        self._csr_successes: int = 0
        self._csr_social_heard: int = 0
        self._csr_social_successes: int = 0

        # PCA tracking: social→social with rift type matching
        # {listener_id: (heard_tick, speaker_rift_type)}
        self._pca_pending: dict[str, tuple[int, str]] = {}
        self._pca_events: int = 0
        self._pca_successes: int = 0

        # CIC (Causal Influence of Communication) tracking
        # Behavioral delta: heard-sound social agents vs silent social agents moving toward rifts
        # {agent_id: (tick, agent_x, agent_y)}
        self._cic_heard_pending: dict[str, tuple[int, int, int]] = {}
        self._cic_silent_pending: dict[str, tuple[int, int, int]] = {}
        self._cic_heard_events: int = 0
        self._cic_heard_moved: int = 0  # moved toward rift after hearing
        self._cic_silent_events: int = 0
        self._cic_silent_moved: int = 0  # moved toward rift without hearing

        # Listener feedback: {listener_id: (speaker_id, tick_heard)}
        self._heard_from: dict[str, tuple[str, int]] = {}

        # Speaker influence tracking (for causal influence intrinsic reward)
        # {speaker_id: [heard_count, moved_count, last_decay_tick]}
        self._speaker_influence: dict[str, list] = {}
        # {listener_id: (speaker_id, tick, orig_x, orig_y)}
        self._cic_speaker_pending: dict[str, tuple[str, int, int, int]] = {}

        # Phase 2: Factored Meaning Space — novelty holdout set
        self._novelty_holdout_set: set[tuple[int, int, int]] = set()
        lang_cfg = config.get("language", {})
        if lang_cfg.get("context_mode") == "factored":
            self._init_novelty_holdout(lang_cfg)

        # Phase 2: Bottleneck tracking
        self._bottleneck_retirements: int = 0

        # Phase 2: Specialization stimuli tracking
        self._ticks_since_new_rift: int = 0
        self._recent_deaths_window: list[int] = []  # tick numbers of recent deaths
        self._current_stimuli: list[float] = [1.0, 1.0, 1.0, 1.0]

        # Init first epoch rifts (within active zone)
        zone_min, zone_max = self._active_zone_bounds()
        self.rift_mgr.spawn_epoch_rifts(self.epoch_mgr.epoch, zone_min, self.active_zone_size)

    # ----- Active zone -----

    def _recalculate_active_zone(self):
        """Recalculate active zone size based on connected agent count.
        Ocean expands dynamically — no hard cap. More agents = bigger world."""
        n = len(self.lobsters)
        if n == 0:
            self.active_zone_size = self.active_zone_min
        else:
            self.active_zone_size = max(self.active_zone_min,
                                        int(math.sqrt(n) * self.density_constant))
        # Ocean size grows with zone (zone + margin for predator spawns)
        margin = max(10, self.active_zone_size // 5)
        self.size = self.active_zone_size + margin * 2
        # Keep rift manager in sync
        zone_min, _ = self._active_zone_bounds()
        self.rift_mgr.update_zone(zone_min, self.active_zone_size)

    def _active_zone_bounds(self) -> tuple[int, int]:
        """Return (zone_min, zone_max) — active zone centered on the map."""
        margin = max(10, self.active_zone_size // 5)
        return margin, margin + self.active_zone_size - 1

    def _random_pos_in_zone(self) -> tuple[int, int]:
        """Random position within the active zone."""
        zone_min, zone_max = self._active_zone_bounds()
        x = random.randint(zone_min, zone_max)
        y = random.randint(zone_min, zone_max)
        return x, y

    # ----- Factored Meaning Space (Phase 2) -----

    def _init_novelty_holdout(self, lang_cfg: dict):
        """Build deterministic set of held-out (situation, target, urgency) combos."""
        seed = lang_cfg.get("novelty_holdout_seed", 42)
        frac = lang_cfg.get("novelty_holdout_fraction", 0.20)
        n_sit = lang_cfg.get("features", {}).get("situation_types", 4)
        n_tgt = lang_cfg.get("features", {}).get("target_details", 5)
        n_urg = lang_cfg.get("features", {}).get("urgency_levels", 3)

        rng = random.Random(seed)
        # For each situation type, hold out ~frac of target x urgency pairs
        for sit in range(n_sit):
            combos = [(sit, tgt, urg) for tgt in range(n_tgt) for urg in range(n_urg)]
            n_holdout = max(1, int(len(combos) * frac))
            held = rng.sample(combos, n_holdout)
            self._novelty_holdout_set.update(held)
        print(f"[FACTORED] Novelty holdout: {len(self._novelty_holdout_set)} combos out of {n_sit * n_tgt * n_urg}")

    def _build_factored_context(self, lob: 'Lobster', state_rifts: list, state_preds: list) -> tuple[int, int, int]:
        """Convert raw agent surroundings into factored (situation, target, urgency) tuple."""
        # Get vision range for this lobster's tier
        tier = self.get_tier(lob)
        tier_info = self.get_tier_info(tier)
        vision = tier_info["vision"]

        # Nearby rifts within vision
        nearby_rifts = []
        for rift in self.rift_mgr.active_rifts:
            dx = abs(rift.x - lob.x)
            dy = abs(rift.y - lob.y)
            if dx <= vision and dy <= vision:
                max_r = self.rift_mgr._richness_for_type(rift.rift_type)
                pct = rift.richness / max_r if max_r > 0 else 0
                nearby_rifts.append({
                    "distance": dx + dy,
                    "richness_pct": pct,
                    "rift_type": rift.rift_type,
                })

        # Nearby predators within vision
        nearby_preds = []
        for pred in self.predator_mgr.active_predators:
            dx = abs(pred.x - lob.x)
            dy = abs(pred.y - lob.y)
            if dx <= vision and dy <= vision:
                nearby_preds.append({
                    "distance": dx + dy,
                    "type": pred.predator_type,
                })

        # Feature 1: SITUATION_TYPE
        nearest_rift = min(nearby_rifts, key=lambda r: r["distance"]) if nearby_rifts else None
        if nearest_rift:
            if nearest_rift["richness_pct"] > 0.20:
                situation = 0  # rift_nearby
            else:
                situation = 1  # rift_depleted
        elif nearby_preds:
            situation = 2  # predator_nearby
        else:
            situation = 3  # exploration

        # Feature 2: TARGET_DETAIL
        if situation in (0, 1):  # rift
            target = {"gold": 0, "silver": 1, "copper": 2}.get(nearest_rift["rift_type"], 1)
        elif situation == 2:  # predator
            pred_type = min(nearby_preds, key=lambda p: p["distance"])["type"]
            target = 3 if pred_type == "shark" else 4  # eel/octopus = 4
        else:  # exploration
            food_trails = self.pheromone_map.read(lob.x, lob.y, "food", vision)
            has_trail = any(t["intensity"] > 0.3 for t in food_trails)
            target = 0 if has_trail else 1

        # Feature 3: URGENCY
        # Eruption override: always max urgency when erupting rift is visible
        rift_is_erupting = any(
            r.erupting for r in self.rift_mgr.active_rifts
            if abs(r.x - lob.x) <= vision and abs(r.y - lob.y) <= vision
        )
        if rift_is_erupting:
            urgency = 2  # always maximum urgency for eruptions
        else:
            has_close_predator = any(p["distance"] <= 3 for p in nearby_preds)
            tidal_state = self.tidal.get_state_for_agent()
            is_night = tidal_state.get("is_night", False)
            is_winter = tidal_state.get("seasonal_multiplier", 1.0) < 0.7

            stress_count = sum([has_close_predator, is_night, is_winter])
            if stress_count >= 2:
                urgency = 2  # high
            elif stress_count == 1:
                urgency = 1  # medium
            else:
                urgency = 0  # low

        return (situation, target, urgency)

    def _get_factored_context_for_agent(self, lob: 'Lobster') -> tuple[int, int, int] | None:
        """Return factored context if enabled, else None."""
        if self.config.get("language", {}).get("context_mode") != "factored":
            return None
        return self._build_factored_context(lob, [], [])

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
        grace = self.config["predators"].get("grace_period", 60)
        # Stagger retirement: random initial age so agents don't all retire simultaneously
        agent_lifetime = self.config.get("agents", {}).get("agent_lifetime", 3000)
        initial_age = random.randint(0, agent_lifetime - 1)
        # Temporarily add to recalculate zone, then spawn inside it
        lob = Lobster(id=lid, name=name, x=0, y=0, balance=balance, agent_type=agent_type, grace_ticks=grace, age=initial_age)
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
            # Phase 2: capture role from agent action
            role = actions.get("role")
            if role is not None:
                self.lobsters[agent_id].current_role = role

    # ----- Tick -----

    def process_tick(self):
        self.tick += 1
        self.global_tick += 1
        self.current_sounds = []

        # Advance tidal cycles
        self.tidal.tick(self.global_tick)

        # Reset per-tick reward tracking
        for lob in self.lobsters.values():
            lob.reward_this_tick = 0.0

        # 1. Process movements
        self._process_movements()

        # 2. Process sounds (before rift rewards so we know who spoke)
        self._process_sounds()

        # 2b. Eruption processing (before rift rewards so multiplier applies)
        self._eruption_events = []
        ecfg = self.config.get("eruptions", {})
        if ecfg.get("enabled", False):
            lobster_positions = self._get_lobster_positions()
            eruption_events = self.rift_mgr.tick_eruptions(self.tick, lobster_positions)
            self._eruption_events = eruption_events
            # Count new eruptions for metrics
            for ev in eruption_events:
                if ev["phase"] == "start":
                    self._eruption_count += 1
            # Spawn predators at erupting rifts past delay
            delay = ecfg.get("predator_attract_delay", 10)
            attract_radius = ecfg.get("predator_attract_radius", 8)
            for rift in self.rift_mgr.get_erupting_needing_predator(delay):
                # Spawn a shark near the erupting rift
                offset_x = random.randint(-attract_radius, attract_radius)
                offset_y = random.randint(-attract_radius, attract_radius)
                pred = self.predator_mgr.spawn_at(
                    "shark", rift.x + offset_x, rift.y + offset_y, self.size
                )
                rift.eruption_predator_spawned = True
                if pred:
                    print(f"[ERUPTION] Predator {pred.id} attracted to erupting {rift.id} at ({rift.x},{rift.y})")

        # 3. Rift rewards (group bonus)
        self._process_rift_rewards()
        # Tidal accumulator: credits earned this tick
        for lob in self.lobsters.values():
            self._tidal_credits_earned += lob.reward_this_tick

        # 3a. Pheromone deposits from rift rewards — path-based (ACO-style)
        food_scale = self.config.get("pheromones", {}).get("food_deposit_scale", 0.02)
        trail_reinforce = self.config.get("pheromones", {}).get("trail_reinforce", 0.05)
        for lob in self.lobsters.values():
            if lob.alive and lob.reward_this_tick > 0:
                # Deposit along recent path (last 10 positions) — ACO-style trail
                path = lob.path_history[-10:]
                if path:
                    total_deposit = lob.reward_this_tick * food_scale
                    deposit_per_cell = min(total_deposit / len(path), 1.0)
                    if deposit_per_cell > 0.001:
                        for px, py in path:
                            self.pheromone_map.deposit(px, py, "food", deposit_per_cell)
            # Trail reinforcement: positive feedback when following strong food trails
            if lob.alive and trail_reinforce > 0:
                food_here = self.pheromone_map.food_trails.get((lob.x, lob.y), 0)
                if food_here > 0.5:
                    self.pheromone_map.deposit(lob.x, lob.y, "food", trail_reinforce)

        # 3a2. No-entry pheromone: deposit when agent is at depleted rift (within rift radius)
        ne_deposit = self.config.get("pheromones", {}).get("noentry_deposit", 1.5)
        ne_threshold = self.config.get("pheromones", {}).get("noentry_richness_threshold", 0.10)
        rift_radius = self.config["rifts"]["radius"]
        for rift in self.rift_mgr.active_rifts:
            max_richness = self.rift_mgr._richness_for_type(rift.rift_type)
            if max_richness <= 0:
                continue
            pct = rift.richness / max_richness
            if pct < ne_threshold:
                self.pheromone_map.deposit(rift.x, rift.y, "noentry", ne_deposit)
                for lob in self.lobsters.values():
                    if lob.alive and abs(lob.x - rift.x) <= rift_radius and abs(lob.y - rift.y) <= rift_radius:
                        self.pheromone_map.deposit(lob.x, lob.y, "noentry", ne_deposit)

        # 3b. Decrement grace ticks & increment age
        turnover_enabled = self.config.get("ablation", {}).get("turnover", True)
        agent_lifetime = self.config.get("agents", {}).get("agent_lifetime", 3000)
        for lob in self.lobsters.values():
            if lob.alive and lob.grace_ticks > 0:
                lob.grace_ticks -= 1
            lob.age += 1
            # Turnover: mark retired agents
            if turnover_enabled and lob.age >= agent_lifetime and not lob.retired:
                lob.retired = True
                print(f"[RETIRE] {lob.id} ({lob.name}) retired at tick {self.tick} (age={lob.age})")

        # 3c. Track CSR: agents that heard sounds — record positions for move-toward-rift check
        self._track_csr_heard()

        # 4. Predator spawns & movement (with danger pheromone attraction)
        rift_positions = [(r.x, r.y) for r in self.rift_mgr.active_rifts]
        alive_count = sum(1 for l in self.lobsters.values() if l.alive)
        spawned = self.predator_mgr.process_tick(
            self._get_lobster_positions(), self.size, rift_positions,
            pheromone_map=self.pheromone_map,
            spawn_rate_multiplier=self.tidal.get_predator_spawn_multiplier(),
            alive_count=alive_count,
        )
        # Tidal accumulator: predator spawns by phase
        if self.tidal.day_night_phase > 0.5:
            self._tidal_predators_spawned_night += spawned
        else:
            self._tidal_predators_spawned_day += spawned

        # 5. Predator kills
        kill_positions = self._process_predator_kills()
        # Tidal accumulator: deaths by phase
        if kill_positions:
            if self.tidal.day_night_phase > 0.5:
                self._tidal_deaths_night += len(kill_positions)
            else:
                self._tidal_deaths_day += len(kill_positions)

        # 5b. Danger pheromone at kill sites
        danger_amt = self.config.get("pheromones", {}).get("danger_deposit", 3.0)
        for kx, ky in kill_positions:
            self.pheromone_map.deposit(kx, ky, "danger", danger_amt)

        # 5c. No-entry at dying rifts (richness <= 0): strong mark
        for rift in self.rift_mgr.active_rifts:
            if rift.richness <= 0:
                self.pheromone_map.deposit(rift.x, rift.y, "noentry", ne_deposit * 2)

        # 6. Rift depletion & respawn
        rift_count_before = len(self.rift_mgr.active_rifts)
        self.rift_mgr.tick_rifts(self.tick, tidal_offset=self.tidal.get_rift_offset())
        if len(self.rift_mgr.active_rifts) > rift_count_before:
            self._ticks_since_new_rift = 0  # reset scout stimulus

        # 7. Death timers
        self._process_death_timers()

        # 7b. Pheromone decay + diffusion
        self.pheromone_map.tick()

        # 7c. Colony detection
        rift_positions = [(r.x, r.y, r.id) for r in self.rift_mgr.active_rifts]
        self.colony_mgr.tick(self.lobsters, rift_positions, self.tick)

        # 7c2. Colony cache: forward dissolutions, tick dormant TTL
        if self.colony_cache_mgr.enabled:
            for cid, cx, cy in self.colony_mgr._last_dissolved:
                self.colony_cache_mgr.on_colony_dissolved(cid, cx, cy, self.tick)
            active_ids = set(self.colony_mgr.colonies.keys())
            self.colony_cache_mgr.tick(self.tick, active_ids)

        # 7d. Colony scent deposits: members deposit at position + 4 cardinal neighbors
        cs_deposit = self.config.get("pheromones", {}).get("colony_scent_deposit", 0.5)
        cs_neighbor = self.config.get("pheromones", {}).get("colony_scent_neighbor_deposit", 0.2)
        for colony in self.colony_mgr.colonies.values():
            for member_id in colony.members:
                lob = self.lobsters.get(member_id)
                if not lob or not lob.alive:
                    continue
                # Center cell
                self.pheromone_map.deposit_colony_scent(lob.x, lob.y, colony.id, cs_deposit)
                # Cardinal neighbors
                for nx, ny in [(lob.x+1, lob.y), (lob.x-1, lob.y),
                               (lob.x, lob.y+1), (lob.x, lob.y-1)]:
                    self.pheromone_map.deposit_colony_scent(nx, ny, colony.id, cs_neighbor)

        # 7e. Phase 2: Cultural transmission bottleneck
        self._process_bottleneck()

        # 7f. Phase 2: Compute task stimuli for specialization
        self._compute_task_stimuli()

        # 7g. Track deaths for stimuli (rolling window)
        if kill_positions:
            for _ in kill_positions:
                self._recent_deaths_window.append(self.tick)
        lookback = self.config.get("specialization", {}).get("tasks", {}).get(
            "guard", {}).get("lookback_ticks", 50)
        self._recent_deaths_window = [
            t for t in self._recent_deaths_window if self.tick - t <= lookback
        ]

        # 7h. Track new rift discovery for scout stimulus
        self._ticks_since_new_rift += 1

        # 8. Check epoch end
        if self.epoch_mgr.tick():
            self._end_epoch()

        # Clear pending actions
        self.pending_actions.clear()

        # Log every 100 ticks
        if self.tick % 100 == 0:
            self._log_stats()
            # Colony/pheromone stats for metrics
            n_colonies = len(self.colony_mgr.colonies)
            avg_col_size = (
                sum(len(c.members) for c in self.colony_mgr.colonies.values()) / n_colonies
                if n_colonies > 0 else 0
            )
            # Snapshot predator counts by type
            self._predators_by_type = self.predator_mgr.get_counts_by_type()
            self.latest_metrics = self.metrics_logger.compute(
                self.tick, self.epoch_mgr.epoch, self.lobsters,
                self._csr_heard_events, self._csr_successes,
                self._csr_social_heard, self._csr_social_successes,
                self._pca_events, self._pca_successes,
                self._cic_heard_events, self._cic_heard_moved,
                self._cic_silent_events, self._cic_silent_moved,
                colony_count=n_colonies,
                avg_colony_size=avg_col_size,
                food_trail_cells=len(self.pheromone_map.food_trails),
                danger_trail_cells=len(self.pheromone_map.danger_trails),
                noentry_trail_cells=len(self.pheromone_map.noentry_trails),
                colony_scent_cells=len(self.pheromone_map.colony_scent),
                tidal_pred_day=self._tidal_predators_spawned_day,
                tidal_pred_night=self._tidal_predators_spawned_night,
                tidal_credits=round(self._tidal_credits_earned, 2),
                tidal_deaths_day=self._tidal_deaths_day,
                tidal_deaths_night=self._tidal_deaths_night,
                seasonal_mult=round(self.tidal.get_seasonal_multiplier(), 3),
                predators_by_type=dict(self._predators_by_type),
                deaths_by_predator_type=dict(self._deaths_by_predator_type),
                pheromone_clear_events=self._pheromone_clear_events,
                # Phase 2 metrics
                bottleneck_retirements=self._bottleneck_retirements,
                task_stimuli=list(self._current_stimuli),
                eruption_count=self._eruption_count,
            )
            # Reset tidal accumulators
            self._tidal_predators_spawned_day = 0
            self._tidal_predators_spawned_night = 0
            self._tidal_credits_earned = 0.0
            self._tidal_deaths_day = 0
            self._tidal_deaths_night = 0
            # Reset predator mosaic accumulators
            self._deaths_by_predator_type = {}
            self._pheromone_clear_events = 0
            # Reset bottleneck accumulator
            self._bottleneck_retirements = 0
            # Reset eruption counter (was previously in _end_epoch — too early)
            self._eruption_count = 0
            # Reset CSR/PCA/CIC counters (were never reset — metrics were cumulative!)
            self._csr_heard_events = 0
            self._csr_successes = 0
            self._csr_social_heard = 0
            self._csr_social_successes = 0
            self._pca_events = 0
            self._pca_successes = 0
            self._cic_heard_events = 0
            self._cic_heard_moved = 0
            self._cic_silent_events = 0
            self._cic_silent_moved = 0

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
            # Track path history for trail-based pheromone deposit
            lob.path_history.append((lob.x, lob.y))
            if len(lob.path_history) > 50:
                lob.path_history = lob.path_history[-50:]

    def _process_sounds(self):
        sound_cost_enabled = self.config.get("ablation", {}).get("sound_cost", True)
        sound_cost = self.config["economy"]["sound_credit_cost"] if sound_cost_enabled else 0
        valid_sounds = set(self.sounds)
        rift_radius = self.config["rifts"]["radius"]

        for agent_id, actions in self.pending_actions.items():
            lob = self.lobsters.get(agent_id)
            if not lob or not lob.alive:
                continue
            raw_sounds = actions.get("speak", [])
            if not raw_sounds:
                lob.speaking = []
                continue

            # Validate and limit to max_message_length
            max_msg = self.config.get("communication", {}).get("max_message_length", 2)
            sounds = [s for s in raw_sounds if s in valid_sounds][:max_msg]
            lob.speaking = sounds

            if sounds:
                # Superlinear cost: 2-sound messages cost more than 2×single
                if len(sounds) == 1:
                    cost = self.config["economy"]["sound_credit_cost"]
                elif len(sounds) >= 2:
                    cost = self.config["economy"].get("sound_cost_2", 5)
                else:
                    cost = 0
                lob.credits_spent += cost
                self.current_sounds.append({
                    "from": agent_id,
                    "sounds": sounds,
                    "pos": [lob.x, lob.y],
                })

                # Track sound-context correlations (server-side)
                contexts = self._get_speaker_context(lob, rift_radius)
                for snd in sounds:
                    if snd not in self.sound_correlations:
                        self.sound_correlations[snd] = {}
                    for ctx in contexts:
                        self.sound_correlations[snd][ctx] = (
                            self.sound_correlations[snd].get(ctx, 0) + 1
                        )

                # Record for metrics
                ctx_key = self._get_speaker_context_key(lob, rift_radius)
                reward = lob.credits_earned - lob.credits_spent
                self.metrics_logger.record(sounds, ctx_key, reward,
                                           agent_type=lob.agent_type, agent_age=lob.age)

    def _process_rift_rewards(self):
        seasonal_mult = self.tidal.get_seasonal_multiplier()
        for rift in self.rift_mgr.active_rifts:
            nearby = self._lobsters_near(rift.x, rift.y, self.config["rifts"]["radius"])
            if not nearby:
                continue
            n = len(nearby)
            reward = self.rift_mgr.calc_group_reward(n) * seasonal_mult
            # Apply eruption multiplier
            if rift.erupting:
                reward *= rift.eruption_multiplier
            # Per-type depletion: scale drain by rift type's depletion_rate / base 0.02
            depletion_rate = self.rift_mgr.depletion_rate_for_type(rift.rift_type)
            drain_multiplier = depletion_rate / 0.02
            total_drain = n * reward * drain_multiplier
            if total_drain > rift.richness:
                reward = rift.richness / (n * drain_multiplier) if n > 0 else 0
                total_drain = rift.richness
            feedback_frac = self.config.get("communication", {}).get("listener_feedback_frac", 0.0)
            for lob in nearby:
                # Newcomer discovery bonus: 2x rewards for first 50 ticks (Game Designer rec)
                bonus = 2.0 if lob.age < 50 else 1.0
                effective_reward = reward * bonus
                # Colony bonus: members get extra rift rewards
                colony = self.colony_mgr.get_colony_for(lob.id)
                if colony:
                    effective_reward *= self.colony_mgr.reward_bonus
                    colony.total_reward += effective_reward
                lob.credits_earned += effective_reward
                lob.reward_this_tick += effective_reward
                if n >= 2:
                    lob.group_hits += 1
                # Listener feedback: if this lobster heard a sound recently, reward the speaker
                if feedback_frac > 0 and lob.id in self._heard_from:
                    speaker_id, heard_tick = self._heard_from[lob.id]
                    if self.tick - heard_tick <= 10:
                        speaker = self.lobsters.get(speaker_id)
                        if speaker and speaker.alive:
                            speaker.credits_earned += reward * bonus * feedback_frac
            rift.richness -= total_drain

    def _process_predator_kills(self) -> list[tuple[int, int]]:
        """Process predator kills and return list of kill positions for danger pheromones."""
        kill_radius = self.config["predators"]["kill_radius"]
        death_timeout = self.config["economy"]["death_timeout"]
        death_penalty = self.config["economy"]["death_credit_penalty"]
        confusion_enabled = self.config["predators"].get("confusion_enabled", True)
        failed_penalty = self.config["predators"].get("failed_attack_penalty", 5)
        kill_positions: list[tuple[int, int]] = []

        # Octopus pheromone clear config
        type_configs = self.config["predators"].get("types", {})
        octopus_clear_radius = type_configs.get("octopus", {}).get("pheromone_clear_radius", 3)

        for pred in self.predator_mgr.active_predators:
            # Eel: skip if already attacked (waiting for cooldown respawn)
            if pred.predator_type == "eel" and pred.has_attacked:
                continue

            # Per-type kill radius: eel uses attack_radius from config, others use global
            effective_kill_radius = kill_radius
            if pred.predator_type == "eel":
                effective_kill_radius = type_configs.get("eel", {}).get("attack_radius", kill_radius)

            # Find all targets in effective kill radius
            targets = [
                lob for lob in self.lobsters.values()
                if lob.alive and lob.grace_ticks <= 0
                and math.sqrt((lob.x - pred.x) ** 2 + (lob.y - pred.y) ** 2) <= effective_kill_radius
            ]
            if not targets:
                continue

            # Pick random target from those in range (dilution)
            target = random.choice(targets)

            # Find group: all alive lobsters within radius 3 of target
            group = [
                lob for lob in self.lobsters.values()
                if lob.alive and abs(lob.x - target.x) <= 3 and abs(lob.y - target.y) <= 3
            ]
            group_size = max(1, len(group))

            # Per-type lethality instead of global kill_prob_base
            kill_prob = pred.lethality
            if confusion_enabled:
                kill_prob = pred.lethality / math.sqrt(group_size)

            # Social bonus: 3+ social agents in group → extra 0.8x reduction
            social_count = sum(1 for lob in group if lob.agent_type == "social")
            if social_count >= 3:
                kill_prob *= 0.8

            if random.random() < kill_prob:
                # Kill succeeds
                kill_positions.append((target.x, target.y))
                target.alive = False
                target.death_timer = death_timeout
                target.deaths_this_epoch += 1
                penalty = target.credits_earned * death_penalty
                target.credits_earned = max(0, target.credits_earned - penalty)
                # Track deaths by predator type
                self._deaths_by_predator_type[pred.predator_type] = (
                    self._deaths_by_predator_type.get(pred.predator_type, 0) + 1
                )
            else:
                # Failed attack: predator loses lifetime
                pred.lifetime_remaining -= failed_penalty

            # Eel: after any attack (hit or miss), enter cooldown
            if pred.predator_type == "eel":
                pred.has_attacked = True
                pred.attack_cooldown = pred.respawn_delay

            # Octopus: on contact with targets, clear food pheromone in radius
            if pred.predator_type == "octopus":
                r = octopus_clear_radius
                px, py = int(pred.x), int(pred.y)
                for key in list(self.pheromone_map.food_trails.keys()):
                    if abs(key[0] - px) <= r and abs(key[1] - py) <= r:
                        del self.pheromone_map.food_trails[key]
                self._pheromone_clear_events += 1

        return kill_positions

    def _process_death_timers(self):
        grace = self.config["predators"].get("grace_period", 60)
        for lob in self.lobsters.values():
            if not lob.alive:
                lob.death_timer -= 1
                if lob.death_timer <= 0:
                    lob.alive = True
                    lob.grace_ticks = grace
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
        # {type: {credits: [], rewards: [], group_hits: [], deaths: []}}
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
                type_stats[agent_type] = {"credits": [], "rewards": [], "group_hits": [], "deaths": []}
            type_stats[agent_type]["credits"].append(lob.credits_earned - lob.credits_spent)
            type_stats[agent_type]["rewards"].append(reward)
            type_stats[agent_type]["group_hits"].append(lob.group_hits)
            type_stats[agent_type]["deaths"].append(lob.deaths_this_epoch)

            # Reset credits, group_hits, deaths for new epoch
            lob.credits_earned = 0.0
            lob.credits_spent = 0.0
            lob.group_hits = 0
            lob.deaths_this_epoch = 0

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
        epoch_summary: dict[str, dict] = {}
        for atype in ("social", "greedy", "random"):
            stats = type_stats.get(atype)
            if not stats or not stats["credits"]:
                continue
            avg_credits = sum(stats["credits"]) / len(stats["credits"])
            total_reward = sum(stats["rewards"])
            total_group_hits = sum(stats["group_hits"])
            total_deaths = sum(stats["deaths"])
            count = len(stats["credits"])
            epoch_summary[atype] = {
                "count": count,
                "avg_credits": round(avg_credits, 1),
                "total_reward": round(total_reward),
                "group_hits": total_group_hits,
                "deaths": total_deaths,
            }
            print(
                f"[EPOCH {epoch_num}] {atype}: "
                f"avg_credits={avg_credits:.1f} "
                f"total_reward={total_reward:,.0f} "
                f"group_hits={total_group_hits} "
                f"deaths={total_deaths}"
            )
        print()

        # Save epoch history for viewer
        self.epoch_history.append({
            "epoch": epoch_num,
            "types": epoch_summary,
        })

        # Decay cultural cache (global + colony caches)
        self.cultural_cache.epoch_decay()
        if self.colony_cache_mgr.enabled:
            self.colony_cache_mgr.epoch_decay()

        # Note: eruption_count reset moved to after metrics logging (tick % 100 block)

    # ----- Phase 2: Cultural Transmission Bottleneck -----

    def _process_bottleneck(self):
        """Every retirement_interval ticks, retire lowest earner and replace with newborn."""
        cfg = self.config.get("bottleneck", {})
        if not cfg.get("enabled", False):
            return
        interval = cfg.get("retirement_interval", 200)
        if self.tick % interval != 0:
            return
        min_agents = cfg.get("min_agents_for_retirement", 25)
        if len(self.lobsters) < min_agents:
            return

        # Find lowest-earning social agent (skip very young, skip non-social)
        candidates = [
            lob for lob in self.lobsters.values()
            if lob.alive and lob.age > 100 and lob.agent_type == "social"
        ]
        if not candidates:
            return

        worst = min(candidates, key=lambda a: a.credits_earned)

        # Deposit knowledge to cultural cache before retirement
        # (the agent-side on_pre_retire handles this; server marks retired)
        worst.retired = True
        self._bottleneck_retirements += 1
        print(f"[BOTTLENECK] Retired {worst.id} ({worst.name}) at tick {self.tick} "
              f"(credits_earned={worst.credits_earned:.0f}, age={worst.age})")

    # ----- Phase 2: Task Stimuli Computation -----

    def _compute_task_stimuli(self):
        """Compute global task stimuli for Bonabeau response threshold model."""
        cfg = self.config.get("specialization", {})
        if not cfg.get("enabled", False):
            self._current_stimuli = [1.0, 1.0, 1.0, 1.0]
            return

        tasks_cfg = cfg.get("tasks", {})
        alive_agents = [a for a in self.lobsters.values() if a.alive]
        if not alive_agents:
            self._current_stimuli = [1.0, 1.0, 1.0, 1.0]
            return

        # FORAGE stimulus: inverse of average credits earned per tick
        avg_income = sum(
            a.credits_earned / max(a.age, 1) for a in alive_agents
        ) / len(alive_agents)
        forage_scale = tasks_cfg.get("forage", {}).get("stimulus_scale", 100.0)
        forage_stim = min(forage_scale / max(avg_income, 0.1), 10.0)

        # SCOUT stimulus: ticks since any new rift discovered
        scout_scale = tasks_cfg.get("scout", {}).get("stimulus_scale", 200.0)
        scout_stim = min(self._ticks_since_new_rift / max(scout_scale, 1.0), 10.0)

        # GUARD stimulus: recent deaths in lookback window
        guard_scale = tasks_cfg.get("guard", {}).get("stimulus_scale", 10.0)
        guard_stim = min(len(self._recent_deaths_window) / max(guard_scale, 1.0), 10.0)

        # TEACH stimulus: number of newborn agents in learning period
        # Newborns are agents with age < learning_period from bottleneck config
        bn_cfg = self.config.get("bottleneck", {})
        learning_period = bn_cfg.get("learning_period", 300)
        newborn_count = sum(1 for a in alive_agents if a.age < learning_period)
        teach_scale = tasks_cfg.get("teach", {}).get("stimulus_scale", 5.0)
        teach_stim = min(newborn_count / max(teach_scale, 1.0), 10.0)

        self._current_stimuli = [forage_stim, scout_stim, guard_stim, teach_stim]

    # ----- CSR (Communication Success Rate) -----

    def _track_csr_heard(self):
        """Track agents that heard sounds this tick for CSR measurement.
        Success = agent heard a sound and moved within rift radius within 10 ticks.
        Also tracks social-only CSR and PCA (rift type matching)."""
        rift_radius = self.config["rifts"]["radius"]

        # Check pending CSR events for success (moved near rift within 10 ticks)
        resolved = []
        for agent_id, (heard_tick, hx, hy, speaker_type, listener_type) in self._csr_pending.items():
            if self.tick - heard_tick > 10:
                resolved.append(agent_id)
                continue
            lob = self.lobsters.get(agent_id)
            if not lob or not lob.alive:
                resolved.append(agent_id)
                continue
            # Check if agent is now near any rift
            for rift in self.rift_mgr.active_rifts:
                if abs(lob.x - rift.x) <= rift_radius and abs(lob.y - rift.y) <= rift_radius:
                    self._csr_successes += 1
                    if speaker_type == "social" and listener_type == "social":
                        self._csr_social_successes += 1
                    resolved.append(agent_id)
                    break
        for aid in resolved:
            self._csr_pending.pop(aid, None)

        # Check pending PCA events (social→social with rift type matching)
        pca_resolved = []
        for agent_id, (heard_tick, speaker_rift_type) in self._pca_pending.items():
            if self.tick - heard_tick > 10:
                pca_resolved.append(agent_id)
                continue
            lob = self.lobsters.get(agent_id)
            if not lob or not lob.alive:
                pca_resolved.append(agent_id)
                continue
            for rift in self.rift_mgr.active_rifts:
                if abs(lob.x - rift.x) <= rift_radius and abs(lob.y - rift.y) <= rift_radius:
                    # Check rift type match
                    if rift.rift_type == speaker_rift_type:
                        self._pca_successes += 1
                    pca_resolved.append(agent_id)
                    break
        for aid in pca_resolved:
            self._pca_pending.pop(aid, None)

        # CIC: resolve pending heard/silent events (did agent move toward rift within 5 ticks?)
        self._resolve_cic_events()

        # Speaker influence: resolve pending and apply sliding window decay
        self._resolve_speaker_influence()
        self._decay_speaker_influence()

        # Register new heard events from this tick
        # Also track which social agents heard sounds (for CIC)
        social_heard_this_tick: set[str] = set()
        for snd in self.current_sounds:
            speaker_id = snd["from"]
            speaker_lob = self.lobsters.get(speaker_id)
            if not speaker_lob:
                continue
            speaker_type = speaker_lob.agent_type
            sx, sy = snd["pos"]

            # Determine speaker's nearest rift type for PCA
            speaker_rift_type = ""
            if speaker_type == "social":
                for rift in self.rift_mgr.active_rifts:
                    if abs(speaker_lob.x - rift.x) <= rift_radius and abs(speaker_lob.y - rift.y) <= rift_radius:
                        speaker_rift_type = rift.rift_type
                        break

            for lob in self.lobsters.values():
                if lob.id == speaker_id or not lob.alive:
                    continue
                # Skip very young agents for CSR (noisy data)
                if lob.age <= 100:
                    continue
                tier_info = self.get_tier_info(self.get_tier(lob))
                sound_range = int(tier_info["sound_range"] * self.tidal.get_sound_range_multiplier())
                if abs(lob.x - sx) <= sound_range and abs(lob.y - sy) <= sound_range:
                    # CSR: don't overwrite if already pending
                    if lob.id not in self._csr_pending:
                        self._csr_heard_events += 1
                        if speaker_type == "social" and lob.agent_type == "social":
                            self._csr_social_heard += 1
                        self._csr_pending[lob.id] = (self.tick, lob.x, lob.y, speaker_type, lob.agent_type)

                    # PCA: social→social only, speaker must be near rift
                    if (speaker_type == "social" and lob.agent_type == "social"
                            and speaker_rift_type and lob.id not in self._pca_pending):
                        self._pca_events += 1
                        self._pca_pending[lob.id] = (self.tick, speaker_rift_type)

                    # Listener feedback: record most recent speaker for this listener
                    self._heard_from[lob.id] = (speaker_id, self.tick)

                    # Speaker influence: record pending for per-speaker CIC attribution
                    if (speaker_type == "social" and lob.agent_type == "social"
                            and lob.id not in self._cic_speaker_pending):
                        self._cic_speaker_pending[lob.id] = (speaker_id, self.tick, lob.x, lob.y)

                    # CIC: track social listeners that heard sounds
                    if lob.agent_type == "social" and lob.id not in self._cic_heard_pending:
                        social_heard_this_tick.add(lob.id)

        # CIC: register heard social agents and silent social agents
        for lob in self.lobsters.values():
            if not lob.alive or lob.agent_type != "social" or lob.age <= 100:
                continue
            if lob.id in social_heard_this_tick and lob.id not in self._cic_heard_pending:
                self._cic_heard_pending[lob.id] = (self.tick, lob.x, lob.y)
            elif lob.id not in social_heard_this_tick and lob.id not in self._cic_silent_pending:
                # Only sample ~10% of silent agents to keep counts balanced
                if random.random() < 0.10:
                    self._cic_silent_pending[lob.id] = (self.tick, lob.x, lob.y)

    def _resolve_cic_events(self):
        """Resolve pending CIC events: did agent move closer to any rift within 5 ticks?"""
        rift_radius = self.config["rifts"]["radius"]

        def _moved_toward_rift(agent_id: str, orig_x: int, orig_y: int) -> bool:
            lob = self.lobsters.get(agent_id)
            if not lob or not lob.alive:
                return False
            for rift in self.rift_mgr.active_rifts:
                old_dist = abs(orig_x - rift.x) + abs(orig_y - rift.y)
                new_dist = abs(lob.x - rift.x) + abs(lob.y - rift.y)
                if new_dist < old_dist and new_dist <= rift_radius:
                    return True
            return False

        # Resolve heard events
        resolved = []
        for agent_id, (heard_tick, ox, oy) in self._cic_heard_pending.items():
            if self.tick - heard_tick >= 5:
                self._cic_heard_events += 1
                if _moved_toward_rift(agent_id, ox, oy):
                    self._cic_heard_moved += 1
                resolved.append(agent_id)
        for aid in resolved:
            self._cic_heard_pending.pop(aid, None)

        # Resolve silent events
        resolved = []
        for agent_id, (tick0, ox, oy) in self._cic_silent_pending.items():
            if self.tick - tick0 >= 5:
                self._cic_silent_events += 1
                if _moved_toward_rift(agent_id, ox, oy):
                    self._cic_silent_moved += 1
                resolved.append(agent_id)
        for aid in resolved:
            self._cic_silent_pending.pop(aid, None)

    def _resolve_speaker_influence(self):
        """Resolve per-speaker CIC: did listener move closer to a rift within 5 ticks?
        Updates _speaker_influence[speaker_id] = [heard_count, moved_count, last_decay_tick]."""
        resolved = []
        for listener_id, (speaker_id, heard_tick, ox, oy) in self._cic_speaker_pending.items():
            if self.tick - heard_tick < 5:
                continue
            resolved.append(listener_id)
            # Ensure speaker entry exists
            if speaker_id not in self._speaker_influence:
                self._speaker_influence[speaker_id] = [0, 0, self.tick]
            self._speaker_influence[speaker_id][0] += 1  # heard_count
            # Check if listener moved closer to any rift
            lob = self.lobsters.get(listener_id)
            if not lob or not lob.alive:
                continue
            for rift in self.rift_mgr.active_rifts:
                old_dist = abs(ox - rift.x) + abs(oy - rift.y)
                new_dist = abs(lob.x - rift.x) + abs(lob.y - rift.y)
                if new_dist < old_dist:
                    self._speaker_influence[speaker_id][1] += 1  # moved_count
                    break
        for lid in resolved:
            self._cic_speaker_pending.pop(lid, None)

    def _decay_speaker_influence(self):
        """Sliding window decay: every 50 ticks, multiply both counters by 0.5."""
        to_remove = []
        for speaker_id, counters in self._speaker_influence.items():
            last_decay = counters[2]
            if self.tick - last_decay >= 50:
                counters[0] *= 0.5  # heard_count
                counters[1] *= 0.5  # moved_count
                counters[2] = self.tick
                # Remove if both counters negligible
                if counters[0] < 0.1 and counters[1] < 0.1:
                    to_remove.append(speaker_id)
        for sid in to_remove:
            self._speaker_influence.pop(sid, None)

    # ----- Helpers -----

    def _get_speaker_context(self, lob: Lobster, rift_radius: int) -> set[str]:
        """Determine context for a speaking lobster (server-side ground truth)."""
        contexts: set[str] = set()

        # Near rift? — include rift type for richer vocabulary
        for rift in self.rift_mgr.active_rifts:
            if abs(lob.x - rift.x) <= rift_radius and abs(lob.y - rift.y) <= rift_radius:
                contexts.add("near_rift")
                contexts.add(f"near_{rift.rift_type}")  # near_gold, near_silver, near_copper
                break

        # Near predator?
        for pred in self.predator_mgr.active_predators:
            if abs(lob.x - pred.x) <= 5 and abs(lob.y - pred.y) <= 5:
                contexts.add("danger")
                break

        # Crowded?
        nearby_count = sum(
            1 for other in self.lobsters.values()
            if other.id != lob.id and other.alive
            and abs(other.x - lob.x) <= 3 and abs(other.y - lob.y) <= 3
        )
        if nearby_count >= 3:
            contexts.add("crowded")

        return contexts if contexts else {"open_water"}

    @staticmethod
    def _heading(dx: int, dy: int) -> int:
        """Convert relative (dx, dy) to compass heading 0-7, 8=at target."""
        if dx == 0 and dy == 0:
            return 8
        angle = math.atan2(dy, dx)
        return int((angle + math.pi) / (2 * math.pi) * 8) % 8

    def _get_speaker_context_key(self, lob: Lobster, rift_radius: int) -> tuple:
        """Build a discrete context key tuple for metrics (mirrors agent ContextDiscoverer dims)."""
        # dim 0: distance to nearest rift
        rifts = self.rift_mgr.active_rifts
        if rifts:
            closest = min(rifts, key=lambda r: abs(lob.x - r.x) + abs(lob.y - r.y))
            d0 = abs(lob.x - closest.x) + abs(lob.y - closest.y)
            max_richness = self.rift_mgr._richness_for_type(closest.rift_type)
            d1 = closest.richness / max_richness if max_richness > 0 else 0
            d2 = {"gold": 3, "silver": 2, "copper": 1}.get(closest.rift_type, 0)
            d5 = self._heading(closest.x - lob.x, closest.y - lob.y)
        else:
            d0, d1, d2, d5 = 99, 0, 0, 8

        # dim 3: nearby lobster count
        d3 = sum(
            1 for other in self.lobsters.values()
            if other.id != lob.id and other.alive
            and abs(other.x - lob.x) <= 5 and abs(other.y - lob.y) <= 5
        )
        # dim 4: nearby predator count
        d4 = sum(
            1 for pred in self.predator_mgr.active_predators
            if abs(pred.x - lob.x) <= 5 and abs(pred.y - lob.y) <= 5
        )

        # Bin: simple 4-bin quantization
        def _bin(val, lo, hi, bins=4):
            if hi <= lo:
                return 0
            return min(int((val - lo) / (hi - lo) * bins), bins - 1)

        return (
            _bin(d0, 0, 50),
            _bin(d1, 0, 1),
            d2,
            _bin(d3, 0, 10),
            _bin(d4, 0, 5),
            d5,  # heading to nearest rift (0-7 compass, 8=at target)
        )

    def _build_emergent_dictionary(self) -> list[dict]:
        """Build dictionary from accumulated sound-context correlations."""
        dictionary = []
        for sound, contexts in self.sound_correlations.items():
            total = sum(contexts.values())
            if total < 5:
                continue
            best_ctx = max(contexts, key=contexts.get)
            confidence = contexts[best_ctx] / total
            if confidence > 0.3:
                dictionary.append({
                    "sound": sound,
                    "meaning": best_ctx,
                    "confidence": round(confidence, 2),
                    "observations": total,
                })
        dictionary.sort(key=lambda x: (-x["confidence"], -x["observations"]))
        return dictionary

    def _build_sound_lines(self) -> list[dict]:
        """Build communication lines: speaker → each listener in range."""
        lines = []
        for snd in self.current_sounds:
            speaker = self.lobsters.get(snd["from"])
            if not speaker:
                continue
            tier_info = self.get_tier_info(self.get_tier(speaker))
            sound_range = tier_info["sound_range"]
            for lob in self.lobsters.values():
                if lob.id == speaker.id or not lob.alive:
                    continue
                if abs(lob.x - speaker.x) <= sound_range and abs(lob.y - speaker.y) <= sound_range:
                    lines.append({
                        "from": [speaker.x, speaker.y],
                        "to": [lob.x, lob.y],
                    })
        return lines

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
                max_richness = self.rift_mgr._richness_for_type(rift.rift_type)
                pct = rift.richness / max_richness if max_richness > 0 else 0
                nearby_rifts.append({
                    "id": rift.id,
                    "position": [rift.x, rift.y],
                    "richness_pct": round(pct, 2),
                    "relative": [dx, dy],
                    "rift_type": rift.rift_type,
                    "erupting": rift.erupting,
                    "eruption_multiplier": rift.eruption_multiplier if rift.erupting else 1.0,
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
                    "type": pred.predator_type,
                })

        # Nearby pheromones (within vision radius)
        nearby_food_trails = self.pheromone_map.read(lob.x, lob.y, "food", vision)
        nearby_danger_trails = self.pheromone_map.read(lob.x, lob.y, "danger", vision)
        nearby_noentry_trails = self.pheromone_map.read(lob.x, lob.y, "noentry", vision)
        colony = self.colony_mgr.get_colony_for(agent_id)
        in_colony = colony is not None
        my_colony_id = colony.id if colony else None
        same_w = self.config.get("pheromones", {}).get("same_colony_weight", 1.0)
        other_w = self.config.get("pheromones", {}).get("other_colony_weight", 0.3)
        nearby_colony_scent = self.pheromone_map.read_colony_scent(
            lob.x, lob.y, vision, my_colony_id, same_w, other_w
        )

        # Sounds heard (within sound_range) — with channel noise
        # Tidal: night reduces effective sound range
        effective_sound_range = int(sound_range * self.tidal.get_sound_range_multiplier())
        noise_rate = self.config.get("communication", {}).get("channel_noise_rate", 0.0)
        sounds_heard = []
        for snd in self.current_sounds:
            if snd["from"] == agent_id:
                continue
            sx, sy = snd["pos"]
            dist = max(abs(sx - lob.x), abs(sy - lob.y))
            if dist <= effective_sound_range:
                # Channel noise: each sound has noise_rate chance of corruption
                if noise_rate > 0:
                    noisy = [
                        random.choice(self.sounds) if random.random() < noise_rate else s
                        for s in snd["sounds"]
                    ]
                else:
                    noisy = snd["sounds"]
                sounds_heard.append({
                    "from": snd["from"],
                    "sounds": noisy,
                    "distance": dist,
                    "tick": self.tick,
                })

        net_credits = lob.credits_earned - lob.credits_spent

        # Factored context (Phase 2)
        fc = self._get_factored_context_for_agent(lob)

        # Speaker influence score: moved/heard ratio for this agent as speaker
        inf = self._speaker_influence.get(agent_id)
        influence_score = 0.0
        if inf and inf[0] > 0.5:  # need at least ~1 heard event (after decay)
            influence_score = inf[1] / inf[0]

        state_dict = {
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
            "retired": lob.retired,
            "group_hits": lob.group_hits,
            "deaths_this_epoch": lob.deaths_this_epoch,
            "nearby_food_trails": nearby_food_trails,
            "nearby_danger_trails": nearby_danger_trails,
            "nearby_noentry_trails": nearby_noentry_trails,
            "nearby_colony_scent": nearby_colony_scent,
            "my_colony": in_colony,
            "my_colony_id": my_colony_id,
            "tidal": self.tidal.get_state_for_agent(),
            "factored_context": fc,
            "is_novel_context": fc in self._novelty_holdout_set if fc is not None else False,
            "task_stimuli": self._current_stimuli,
            "influence_score": round(influence_score, 3),
        }

        return state_dict

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
                "grace": lob.grace_ticks > 0,
                "colony": self.colony_mgr.get_colony_for(lob.id) is not None,
                "role": lob.current_role,
            })

        rifts = []
        for rift in self.rift_mgr.active_rifts:
            max_richness = self.rift_mgr._richness_for_type(rift.rift_type)
            pct = rift.richness / max_richness if max_richness > 0 else 0
            rifts.append({
                "id": rift.id,
                "pos": [rift.x, rift.y],
                "richness_pct": round(pct, 2),
                "rift_type": rift.rift_type,
                "erupting": rift.erupting,
                "eruption_mult": rift.eruption_multiplier if rift.erupting else 1.0,
                "pressure": round(rift.pressure, 3),
            })

        predators = []
        for pred in self.predator_mgr.active_predators:
            predators.append({
                "id": pred.id,
                "pos": [pred.x, pred.y],
                "type": pred.predator_type,
                "cooldown": pred.attack_cooldown > 0,
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
            "sound_lines": self._build_sound_lines(),
            "pheromones": self.pheromone_map.get_viewer_data(zone_min, zone_max),
            "colonies": self.colony_mgr.get_viewer_data(),
            "tidal": self.tidal.get_viewer_data(),
            "emergent_dictionary": self._build_emergent_dictionary(),
            "eruption_events": self._eruption_events,
            "epoch_history": self.epoch_history,
            "metrics": self.latest_metrics,
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
        self.epoch_history.clear()
        self.sound_correlations.clear()
        self.pending_actions.clear()
        self.current_sounds.clear()
        self.rift_mgr.reset()
        self.predator_mgr.reset()
        self.epoch_mgr.reset()
        self.metrics_logger.reset()
        self.latest_metrics = {}
        self._csr_pending.clear()
        self._csr_heard_events = 0
        self._csr_successes = 0
        self._csr_social_heard = 0
        self._csr_social_successes = 0
        self._pca_pending.clear()
        self._pca_events = 0
        self._pca_successes = 0
        self._cic_heard_pending.clear()
        self._cic_silent_pending.clear()
        self._cic_heard_events = 0
        self._cic_heard_moved = 0
        self._cic_silent_events = 0
        self._cic_silent_moved = 0
        self._heard_from.clear()
        self._cic_speaker_pending.clear()
        # Note: _speaker_influence NOT cleared — uses sliding window decay across epochs
        self.cultural_cache = CulturalCache(self.config)
        self.colony_cache_mgr = ColonyCacheManager(self.config, CulturalCache)
        self.pheromone_map = PheromoneMap(self.config)
        self.colony_mgr = ColonyManager(self.config)
        self._eruption_count = 0
        self._eruption_events = []
        self._recalculate_active_zone()
        zone_min, zone_max = self._active_zone_bounds()
        self.rift_mgr.spawn_epoch_rifts(self.epoch_mgr.epoch, zone_min, self.active_zone_size)
