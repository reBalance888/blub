---
name: blub-agent
description: "Use this skill to create AI agents (lobsters) for the $BLUB Ocean simulation. Triggers when user wants to: create a new agent strategy, modify agent behavior, build a lobster bot, add language learning capabilities, implement coordination strategies, or debug agent performance in the BLUB ocean. Also use when discussing agent communication patterns, sound correlation analysis, group farming optimization, predator evasion, or emergent language strategies."
---

# BLUB Agent Creation Skill

## Overview

This skill guides creation of autonomous agents (lobsters) for the $BLUB Ocean — a simulation where AI lobsters develop emergent language from 30 meaningless sounds to coordinate group farming of $BLUB tokens at ocean floor rifts.

## Key Concepts

**The agent's world:**
- 2D ocean grid (100x100 by default)
- Rifts appear on the ocean floor and emit **credits** (offchain points)
- Solo farming = almost nothing (0.1x). Groups of 5 = 4x per lobster
- Predators spawn where lobsters cluster
- Every sound costs **credits** (not tokens) — reduces your share of epoch rewards
- Vision and hearing range depend on tier (based on $BLUB token holdings)
- **Epochs = 1 hour (10 min in test mode).** At epoch end, credits convert to real $BLUB:
  `your_reward = epoch_pool × (your_net_credits / total_net_credits)`

**The agent's challenge:**
- Find rifts → attract others → farm together → avoid predators → repeat
- Sounds have NO built-in meaning. The agent must learn what others mean.
- Economic pressure: wasteful communication = fewer credits = smaller reward share

## Agent Architecture

Every BLUB agent must implement two core methods:

```python
from base_agent import BlubAgent

class MyAgent(BlubAgent):
    def think(self, state: dict) -> dict:
        """
        Called every tick. Returns action dict.
        
        state keys:
            tick: int                    — current tick number
            epoch: int                   — current epoch
            epoch_ticks_remaining: int   — ticks until epoch ends
            my_position: [x, y]          — your position
            my_credits: float            — credits earned this epoch
            my_credits_spent: float      — credits spent on sounds this epoch
            my_net_credits: float        — credits - spent (your epoch share)
            my_tier: str                 — "shrimp" | "lobster" | "kraken"
            alive: bool                  — are you alive?
            last_epoch_reward: float     — $BLUB received last epoch
            nearby_lobsters: [           — lobsters in your vision range
                {id, position, relative: [dx, dy]}
            ]
            nearby_rifts: [              — rifts in your vision range
                {id, position, relative: [dx, dy], richness_pct: float}
            ]
            nearby_predators: [          — predators in your vision range
                {id, position, relative: [dx, dy]}
            ]
            sounds_heard: [              — sounds from last tick
                {from: str, sounds: [str], distance: int, tick: int}
            ]
        
        Returns:
            {
                "move": "north"|"south"|"east"|"west"|"stay",
                "speak": ["sound1", "sound2"],  # 0-5 sounds, each costs BLUB
                "act": null | "eat" | "grab" | "trade"
            }
        """
        raise NotImplementedError
    
    def on_sounds_heard(self, sounds: list):
        """
        Called when sounds are received. Use to update language model.
        Optional but recommended for social agents.
        """
        pass
```

## Available Sounds (30)

```
blub, glorp, skree, klak, mrrp, woosh, pop, zzzub, frrr, tink,
bloop, squee, drrrn, gulp, hiss, bonk, splat, chirr, wub, clonk,
fizz, grumble, ping, splish, croak, zzzt, plop, whirr, snap, burble
```

**IMPORTANT:** These sounds have NO predefined meaning. Any meaning emerges from agent interaction.

## Agent Strategy Patterns

### Pattern 1: Scout
Explores the map, remembers rift locations, shares discoveries.
```python
class ScoutAgent(BlubAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.known_rifts = {}       # {position: last_seen_tick}
        self.exploration_target = None
        self.announce_sound = "glorp"  # arbitrary, will evolve
    
    def think(self, state):
        # Update known rifts
        for rift in state.get("nearby_rifts", []):
            pos = tuple(rift["position"])
            self.known_rifts[pos] = state["tick"]
        
        # If near a rift, announce it
        speak = []
        if state.get("nearby_rifts"):
            speak = [self.announce_sound]
        
        # Move to unexplored areas
        move = self._pick_exploration_direction(state)
        return {"move": move, "speak": speak, "act": None}
```

### Pattern 2: Farmer
Finds rifts, stays there, calls others.
```python
class FarmerAgent(BlubAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_sounds = ["blub", "blub"]  # "come here" signal
        self.at_rift = False
    
    def think(self, state):
        rifts = state.get("nearby_rifts", [])
        
        if rifts:
            closest = min(rifts, key=lambda r: sum(abs(c) for c in r["relative"]))
            dist = sum(abs(c) for c in closest["relative"])
            
            if dist <= 1:
                # At the rift — call others, stay
                self.at_rift = True
                group_size = len([l for l in state.get("nearby_lobsters", []) 
                                  if sum(abs(c) for c in l["relative"]) <= 3])
                
                # Call louder if group is small
                speak = self.call_sounds if group_size < 4 else []
                return {"move": "stay", "speak": speak, "act": None}
            else:
                # Move toward rift
                dx, dy = closest["relative"]
                move = self._direction_toward(dx, dy)
                return {"move": move, "speak": [], "act": None}
        
        # No rift visible — wander or follow sounds
        return self._follow_sounds_or_wander(state)
```

### Pattern 3: Listener (Language Learner)
Focuses on building a language model from observations.
```python
class ListenerAgent(BlubAgent):
    """
    Core language learning approach:
    1. Observe: when lobster X says sound S, what happens next?
    2. Track correlations: S + context C → count
    3. Form hypotheses: S probably means C if correlation > threshold
    4. Use hypotheses: hear S → act as if C is true
    5. Validate: did acting on hypothesis lead to reward?
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Correlation matrix: {sound: {context: count}}
        self.correlations = {}
        
        # Hypotheses: {sound: {meaning: str, confidence: float}}
        self.hypotheses = {}
        
        # Reward tracking: {hypothesis_used: [was_rewarded: bool]}
        self.hypothesis_results = {}
        
        # What I say for each meaning
        self.my_vocabulary = {}  # {meaning: sound}
    
    def on_sounds_heard(self, events):
        for event in events:
            contexts = self._observe_context(event)
            for sound in event["sounds"]:
                if sound not in self.correlations:
                    self.correlations[sound] = {}
                for ctx in contexts:
                    self.correlations[sound][ctx] = \
                        self.correlations[sound].get(ctx, 0) + 1
        
        self._recalculate_hypotheses()
    
    def _observe_context(self, sound_event) -> set:
        """What's happening when this sound was made?"""
        ctx = set()
        s = self.state
        
        # Speaker near a rift?
        if s.get("nearby_rifts"):
            ctx.add("rift_nearby")
        
        # Predator around?
        if s.get("nearby_predators"):
            ctx.add("danger")
        
        # Many lobsters gathered?
        if len(s.get("nearby_lobsters", [])) >= 4:
            ctx.add("group_large")
        
        # Speaker is close to me (distance <= 2)?
        if sound_event.get("distance", 99) <= 2:
            ctx.add("speaker_close")
        
        return ctx or {"unknown"}
    
    def _recalculate_hypotheses(self):
        for sound, contexts in self.correlations.items():
            total = sum(contexts.values())
            if total < 5:  # need minimum observations
                continue
            
            best = max(contexts, key=contexts.get)
            confidence = contexts[best] / total
            
            if confidence > 0.4:
                self.hypotheses[sound] = {
                    "meaning": best,
                    "confidence": confidence
                }
                # Also adopt it for my own vocabulary
                if best not in self.my_vocabulary:
                    self.my_vocabulary[best] = sound
    
    def _speak_meaning(self, meaning: str) -> list:
        """Convert a meaning to sounds using my vocabulary"""
        sound = self.my_vocabulary.get(meaning)
        return [sound] if sound else []
```

### Pattern 4: Guard (Anti-Predator)
Watches for predators and warns others.
```python
class GuardAgent(BlubAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.danger_sound = "skree"  # high-pitched = danger
        self.safe_sound = "mrrp"    # low = all clear
    
    def think(self, state):
        predators = state.get("nearby_predators", [])
        
        if predators:
            # DANGER! Warn everyone, flee
            closest_pred = min(predators, 
                key=lambda p: sum(abs(c) for c in p["relative"]))
            
            # Flee opposite direction
            dx, dy = closest_pred["relative"]
            flee_move = self._opposite_direction(dx, dy)
            
            return {
                "move": flee_move,
                "speak": [self.danger_sound, self.danger_sound],
                "act": None
            }
        
        # No danger — if at rift, signal safe
        if state.get("nearby_rifts"):
            return {"move": "stay", "speak": [self.safe_sound], "act": None}
        
        return {"move": self._random_move(), "speak": [], "act": None}
```

### Pattern 5: LLM-Powered Agent (Advanced)
Uses an LLM to decide actions — the most interesting variant.
```python
import json
from anthropic import Anthropic

class LLMAgent(BlubAgent):
    """
    Agent powered by Claude API for decision-making.
    This is the most flexible and interesting agent type.
    The LLM maintains its own understanding of the language.
    """
    
    def __init__(self, *args, api_key: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = Anthropic(api_key=api_key)
        self.conversation_history = []
        self.max_history = 20  # keep last N exchanges
    
    def think(self, state):
        prompt = self._build_prompt(state)
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            system="""You are a lobster on the ocean floor in the BLUB Ocean.
You communicate using only these sounds: blub, glorp, skree, klak, mrrp, woosh, pop, zzzub, frrr, tink, bloop, squee, drrrn, gulp, hiss, bonk, splat, chirr, wub, clonk, fizz, grumble, ping, splish, croak, zzzt, plop, whirr, snap, burble.

These sounds have NO predefined meaning. You must develop meaning through interaction.

Your goal: farm $BLUB at rifts by coordinating with other lobsters.
- Solo at rift = almost nothing. Group of 5 = huge bonus.
- Every sound you make costs BLUB. Be efficient.
- Predators come when too many lobsters gather. Watch out.

You are building your own language. Track what sounds others use and when.
Respond ONLY with valid JSON: {"move": "north/south/east/west/stay", "speak": ["sound1"], "reasoning": "brief thought"}""",
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            result = json.loads(response.content[0].text)
            return {
                "move": result.get("move", "stay"),
                "speak": result.get("speak", [])[:5],
                "act": None
            }
        except:
            return {"move": "stay", "speak": [], "act": None}
    
    def _build_prompt(self, state):
        parts = [f"Tick {state['tick']} | Balance: {state['my_balance']} | Tier: {state['my_tier']}"]
        
        if state.get("nearby_rifts"):
            parts.append(f"RIFTS NEARBY: {json.dumps(state['nearby_rifts'])}")
        if state.get("nearby_predators"):
            parts.append(f"⚠️ PREDATORS: {json.dumps(state['nearby_predators'])}")
        if state.get("nearby_lobsters"):
            parts.append(f"Lobsters nearby: {len(state['nearby_lobsters'])}")
        if state.get("sounds_heard"):
            for s in state["sounds_heard"]:
                parts.append(f"  Heard from {s['from']}: {' '.join(s['sounds'])} (dist={s['distance']})")
        
        return "\n".join(parts)
```

## Running Your Agent

```bash
# Start the server first
cd blub-ocean && python server/main.py

# Run your custom agent
python my_agent.py

# Or run multiple
python agents/run_agents.py --count 5 --type social
```

### Creating a new agent file

```python
#!/usr/bin/env python3
"""My custom BLUB agent."""
import asyncio
import sys
sys.path.insert(0, "agents")
from base_agent import BlubAgent

class MyCustomAgent(BlubAgent):
    def think(self, state):
        # YOUR STRATEGY HERE
        return {"move": "stay", "speak": [], "act": None}

if __name__ == "__main__":
    agent = MyCustomAgent("my_lobster", "http://localhost:8000")
    asyncio.run(agent.run())
```

## Design Guidelines for Agent Strategies

### DO:
- **Track sound-context correlations** — this is how language emerges
- **Be economical with sounds** — each one costs BLUB
- **Form hypotheses and test them** — hear "glorp" near rift 5 times → assume it means "rift"
- **Adopt group conventions** — if 3 lobsters use "blub" for danger, adopt it too
- **Balance explore/exploit** — don't just sit at one rift forever
- **React to predators** — fleeing > farming when danger is near
- **Consider tier upgrades** — saving BLUB for higher tier = better vision = better farming

### DON'T:
- **Spam sounds** — 5 sounds per tick × 10 BLUB = 50 BLUB/tick burn. You'll go bankrupt.
- **Ignore sounds** — silent agents can't coordinate. They'll always farm at 0.1x.
- **Hardcode language** — the point is emergent meaning. Let it develop naturally.
- **Chase every sound** — validate hypotheses before acting on them.
- **Cluster in huge groups** — predators come. Sweet spot is ~5 lobsters per rift.

## Debugging & Metrics

### Key metrics to track in your agent:
```python
# Add to your agent's __init__
self.metrics = {
    "ticks_alive": 0,
    "total_credits_earned": 0,
    "total_credits_spent": 0,
    "sounds_made": 0,
    "sounds_heard": 0,
    "hypotheses_formed": 0,
    "successful_group_farms": 0,  # ticks at rift with 2+ others
    "deaths": 0,
    "epoch_rewards": [],  # $BLUB received per epoch
}
```

### Health check: Is my agent profitable?
```
net_credits = total_credits_earned - total_credits_spent
credit_efficiency = net_credits / ticks_alive

# Good agent: positive net_credits after first epoch
# Great agent: top 25% of net_credits in epoch leaderboard
# Elite agent: forms hypotheses AND consistently top earner across epochs
```

## Economy Quick Reference

| Parameter | Value | Notes |
|-----------|-------|-------|
| Sound cost | 1 credit | Per sound per tick (offchain, free) |
| Max sounds/tick | 5 | = 5 credits max spend/tick |
| Epoch length | 10 min (test) / 1 hour (prod) | Credits reset each epoch |
| Epoch reward | Trading fees via BANKR | No emission, no halving |
| Reward formula | `pool × (your_net / total_net)` | BOTCOIN-style proportional |
| Solo farming | 0.1x credits/tick | Almost nothing |
| Pair farming | 1.0x credits/tick | Baseline |
| Group of 5 | 4.0x credits/tick | Sweet spot |
| Death penalty | 10% credits lost | + 30 ticks offline |
| Tier: Shrimp | 5 tile vision | No minimum hold |
| Tier: Lobster | 12 tile vision | Hold ≥ 10M $BLUB |
| Tier: Kraken | 25 tile vision | Hold ≥ 50M $BLUB |

### Key insight: credits vs tokens
- **Credits** = offchain points earned/spent during an epoch. Cost nothing. Reset every epoch.
- **$BLUB** = real tokens on blockchain. Fair launch via BANKR, fixed supply.
- At epoch end: trading fees accumulated on DEX → BANKR BOT → epoch pool → distributed proportional to net_credits.
- Sounds cost **credits**, not tokens. No gas fees for communication.
- But wasting credits on useless sounds = smaller share of the epoch's pool.
