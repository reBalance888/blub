"""
epoch.py — Epoch management: tracking, transitions.
"""
from __future__ import annotations


class EpochManager:
    def __init__(self, config: dict):
        self.config = config
        self.epoch_length = config["economy"]["epoch_length_ticks"]
        self.epoch: int = 1
        self.tick_in_epoch: int = 0

    def tick(self) -> bool:
        """Advance one tick. Returns True if epoch just ended."""
        self.tick_in_epoch += 1
        if self.tick_in_epoch >= self.epoch_length:
            self.epoch += 1
            self.tick_in_epoch = 0
            return True
        return False

    def ticks_remaining(self) -> int:
        return self.epoch_length - self.tick_in_epoch

    def reset(self):
        self.epoch = 1
        self.tick_in_epoch = 0
