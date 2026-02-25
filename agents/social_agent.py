"""
social_agent.py — Agent that listens, builds language hypotheses, coordinates.
"""
from __future__ import annotations

import random
from base_agent import BlubAgent, SOUNDS


class SocialAgent(BlubAgent):
    """
    Builds a language model through observation.

    Strategy:
    1. Observe: when lobster X says sound S, what context is active?
    2. Track correlations: S + context C -> count
    3. Form hypotheses: S probably means C if correlation > threshold
    4. Use hypotheses: say S when context C is true to attract others
    5. Adapt: if hypothesis fails, weaken the link
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # {sound: {context: count}}
        self.correlations: dict[str, dict[str, int]] = {}
        # {sound: {meaning, confidence}}
        self.hypotheses: dict[str, dict] = {}
        # {lobster_id: (x, y)} for tracking movements
        self.last_positions: dict[str, tuple[int, int]] = {}

    def _get_context(self, state: dict, speaker_id: str) -> set[str]:
        """Determine context when a sound was made."""
        contexts: set[str] = set()

        if state.get("nearby_rifts"):
            contexts.add("near_rift")

        if state.get("nearby_predators"):
            contexts.add("near_predator")

        if len(state.get("nearby_lobsters", [])) >= 3:
            contexts.add("crowded")

        for lob in state.get("nearby_lobsters", []):
            if lob["id"] == speaker_id:
                last = self.last_positions.get(speaker_id)
                if last:
                    dx = lob["position"][0] - last[0]
                    dy = lob["position"][1] - last[1]
                    if dx > 0:
                        contexts.add("moving_east")
                    if dx < 0:
                        contexts.add("moving_west")
                    if dy > 0:
                        contexts.add("moving_south")
                    if dy < 0:
                        contexts.add("moving_north")
                self.last_positions[speaker_id] = tuple(lob["position"])

        return contexts if contexts else {"no_context"}

    def on_sounds_heard(self, sounds_events: list):
        """Update correlations when sounds are received."""
        for event in sounds_events:
            speaker = event["from"]
            contexts = self._get_context(self.state, speaker)

            for sound in event["sounds"]:
                if sound not in self.correlations:
                    self.correlations[sound] = {}
                for ctx in contexts:
                    self.correlations[sound][ctx] = (
                        self.correlations[sound].get(ctx, 0) + 1
                    )

        self._update_hypotheses()

    def _update_hypotheses(self):
        """Recalculate hypotheses about sound meanings."""
        for sound, contexts in self.correlations.items():
            if not contexts:
                continue
            total = sum(contexts.values())
            best_ctx = max(contexts, key=contexts.get)
            confidence = contexts[best_ctx] / total
            if confidence > 0.4 and total >= 3:
                self.hypotheses[sound] = {
                    "meaning": best_ctx,
                    "confidence": confidence,
                }

    def think(self, state: dict) -> dict:
        # Log hypotheses every 100 ticks
        if state["tick"] % 100 == 0:
            if self.hypotheses:
                hyp_summary = {s: f'{h["meaning"]}({h["confidence"]:.0%})' for s, h in self.hypotheses.items()}
                print(f"[{self.name}] tick={state['tick']} Hypotheses: {hyp_summary}")
            else:
                top_corr = {}
                for sound, ctxs in list(self.correlations.items())[:5]:
                    if ctxs:
                        best = max(ctxs, key=ctxs.get)
                        top_corr[sound] = f"{best}:{ctxs[best]}"
                print(f"[{self.name}] tick={state['tick']} No hypotheses yet | Top correlations: {top_corr}")

        speak: list[str] = []

        # If near rift — ALWAYS speak to create data for learning
        if state.get("nearby_rifts"):
            rift_sound = self._sound_for("near_rift")
            if rift_sound:
                speak = [rift_sound]
            else:
                # No hypothesis yet — say a random sound (others will correlate it with rift)
                speak = [random.choice(SOUNDS)]

        # If predator nearby — warn and flee
        if state.get("nearby_predators"):
            danger_sound = self._sound_for("near_predator")
            if danger_sound:
                speak = [danger_sound]
            else:
                speak = [random.choice(SOUNDS)]
            pred = state["nearby_predators"][0]
            dx, dy = pred["relative"]
            if dx > 0:
                move = "west"
            elif dx < 0:
                move = "east"
            elif dy > 0:
                move = "north"
            else:
                move = "south"
            return {"move": move, "speak": speak, "act": None}

        # If heard a "near_rift" sound — go toward the speaker
        for event in state.get("sounds_heard", []):
            for sound in event["sounds"]:
                hyp = self.hypotheses.get(sound)
                if hyp and hyp["meaning"] == "near_rift" and hyp["confidence"] > 0.5:
                    for lob in state.get("nearby_lobsters", []):
                        if lob["id"] == event["from"]:
                            dx, dy = lob["relative"]
                            if dx > 0:
                                move = "east"
                            elif dx < 0:
                                move = "west"
                            elif dy > 0:
                                move = "south"
                            elif dy < 0:
                                move = "north"
                            else:
                                move = "stay"
                            return {"move": move, "speak": speak, "act": None}

        # Default: move toward rift or wander
        rifts = state.get("nearby_rifts", [])
        if rifts:
            closest = min(
                rifts,
                key=lambda r: abs(r["relative"][0]) + abs(r["relative"][1]),
            )
            dx, dy = closest["relative"]
            if dx > 0:
                move = "east"
            elif dx < 0:
                move = "west"
            elif dy > 0:
                move = "south"
            elif dy < 0:
                move = "north"
            else:
                move = "stay"
        else:
            move = random.choice(["north", "south", "east", "west"])

        # Speak random sound 20% of the time even without context — generates data
        if not speak and random.random() < 0.2:
            speak = [random.choice(SOUNDS)]

        return {"move": move, "speak": speak, "act": None}

    def _sound_for(self, meaning: str) -> str | None:
        """Find a sound with the given meaning."""
        for sound, hyp in self.hypotheses.items():
            if hyp["meaning"] == meaning:
                return sound
        return None


if __name__ == "__main__":
    import asyncio

    agent = SocialAgent("social_solo", "http://localhost:8000")
    asyncio.run(agent.run())
