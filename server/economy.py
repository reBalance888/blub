"""
economy.py — Credit balances, burn mechanics, tier calculation.
"""
from __future__ import annotations


class EconomyManager:
    """Handles economy constants and tier logic."""

    def __init__(self, config: dict):
        self.config = config
        self.total_supply = config["economy"]["total_supply"]
        self.epoch_pool = config["economy"]["simulated_epoch_pool"]
        self.sound_cost = config["economy"]["sound_credit_cost"]

    def get_tier(self, balance: float) -> str:
        """Determine tier based on balance (MVP: internal balance)."""
        tiers = self.config["tiers"]
        current = "shrimp"
        for name, info in tiers.items():
            if balance >= info["min_hold"]:
                current = name
        return current

    def get_tier_info(self, tier: str) -> dict:
        return self.config["tiers"].get(tier, self.config["tiers"]["shrimp"])

    def calc_epoch_rewards(self, agents: list[dict]) -> list[dict]:
        """
        Calculate epoch rewards for all agents.
        agents: list of {id, credits_earned, credits_spent}
        Returns: list of {id, reward, net_credits}
        """
        pool = self.epoch_pool

        results = []
        total_net = 0.0
        nets = []
        for a in agents:
            net = max(0, a["credits_earned"] - a["credits_spent"])
            nets.append(net)
            total_net += net

        for a, net in zip(agents, nets):
            if total_net > 0:
                reward = pool * (net / total_net)
            else:
                reward = 0
            results.append({
                "id": a["id"],
                "reward": round(reward, 2),
                "net_credits": round(net, 2),
            })

        return results
