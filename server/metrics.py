"""
metrics.py — Language emergence metrics: TopSim, PosDis, MI, vocabulary size, economic delta.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path


class MetricsLogger:
    def __init__(self, log_file: str = "metrics_log.jsonl"):
        self.log_file = Path(log_file)
        # Short buffer: per-epoch observations for PosDis/BosDis (cleared each epoch)
        self.observations: list[tuple[tuple, tuple, float]] = []
        # Long buffer: rolling window for TopSim/MI (keeps ~3 epochs ≈ 5000 obs)
        self.long_observations: list[tuple[tuple, tuple, float]] = []
        # Social-only observations (social agents age>100)
        self.social_observations: list[tuple[tuple, tuple, float]] = []
        self.latest: dict = {}

    def record(self, sound_seq: list[str], context_key: tuple, reward: float,
               agent_type: str = "", agent_age: int = 0):
        """Called each tick for each sound event with server-side context."""
        obs = (tuple(sound_seq), context_key, reward)
        self.observations.append(obs)
        # Long buffer: only include experienced agents (age>100) to avoid
        # MI dilution from naive reborn agents producing random sounds
        if agent_age > 100:
            self.long_observations.append(obs)
        if agent_type == "social" and agent_age > 100:
            self.social_observations.append(obs)

    def compute(self, tick: int, epoch: int, lobsters,
                csr_heard: int = 0, csr_successes: int = 0,
                csr_social_heard: int = 0, csr_social_successes: int = 0,
                pca_events: int = 0, pca_successes: int = 0,
                cic_heard_events: int = 0, cic_heard_moved: int = 0,
                cic_silent_events: int = 0, cic_silent_moved: int = 0,
                colony_count: int = 0, avg_colony_size: float = 0,
                food_trail_cells: int = 0, danger_trail_cells: int = 0,
                noentry_trail_cells: int = 0, colony_scent_cells: int = 0,
                **extra) -> dict:
        """Compute all metrics. Called every 100 ticks."""
        csr = round(csr_successes / csr_heard, 4) if csr_heard > 0 else 0.0
        social_csr = round(csr_social_successes / csr_social_heard, 4) if csr_social_heard > 0 else 0.0
        pca = round(pca_successes / pca_events, 4) if pca_events > 0 else 0.0

        # CIC: behavioral delta = P(move_toward_rift | heard) - P(move_toward_rift | silent)
        rate_heard = cic_heard_moved / cic_heard_events if cic_heard_events > 0 else 0.0
        rate_silent = cic_silent_moved / cic_silent_events if cic_silent_events > 0 else 0.0
        cic = round(rate_heard - rate_silent, 4)

        result = {
            "tick": tick,
            "epoch": epoch,
            "vocabulary_size": self._vocab_size(),
            "vocab_argmax": self._vocab_size_argmax(),
            "top_sim": self._topographic_similarity(use_long=True),
            "pos_dis": self._positional_disentanglement(),
            "bos_dis": self._bag_of_symbols_disentanglement(),
            "mutual_info": self._mutual_information(use_long=True),
            "economic_delta": self._economic_delta(lobsters),
            "csr": csr,
            "csr_heard": csr_heard,
            "csr_successes": csr_successes,
            "social_csr": social_csr,
            "social_csr_heard": csr_social_heard,
            "social_csr_successes": csr_social_successes,
            "pca": pca,
            "pca_events": pca_events,
            "pca_successes": pca_successes,
            "cic": cic,
            "cic_heard_events": cic_heard_events,
            "cic_heard_moved": cic_heard_moved,
            "cic_silent_events": cic_silent_events,
            "cic_silent_moved": cic_silent_moved,
            "social_mi": self._social_mutual_information(),
            "per_type_credits": self._per_type_credits(lobsters),
            "colony_count": colony_count,
            "avg_colony_size": round(avg_colony_size, 1),
            "food_trail_cells": food_trail_cells,
            "danger_trail_cells": danger_trail_cells,
            "noentry_trail_cells": noentry_trail_cells,
            "colony_scent_cells": colony_scent_cells,
        }
        if extra:
            result.update(extra)
        self.latest = result

        # Append to JSONL file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        # Short buffer: per-epoch reset for PosDis/BosDis
        self.observations.clear()
        self.social_observations.clear()
        # Long buffer: rolling window ~5000 for TopSim/MI (~5 epochs)
        max_long = 5000
        if len(self.long_observations) > max_long:
            self.long_observations = self.long_observations[-max_long:]

        return result

    def _vocab_size(self) -> int:
        """Count distinct sound sequences that appear with >60% consistency for one context."""
        if not self.observations:
            return 0

        # sound_seq -> {context_key: count}
        seq_ctx: dict[tuple, dict[tuple, int]] = defaultdict(lambda: defaultdict(int))
        for seq, ctx, _ in self.observations:
            seq_ctx[seq][ctx] += 1

        vocab = 0
        for seq, ctx_counts in seq_ctx.items():
            total = sum(ctx_counts.values())
            if total < 3:
                continue
            best_count = max(ctx_counts.values())
            if best_count / total > 0.6:
                vocab += 1
        return vocab

    def _vocab_size_argmax(self) -> int:
        """Count unique most-probable messages per context (argmax vocabulary).
        Uses long observations for stability."""
        obs = self.long_observations if self.long_observations else self.observations
        if not obs:
            return 0
        # context → {message_seq: count}
        ctx_best: dict[tuple, dict[tuple, int]] = {}
        for seq, ctx, _ in obs:
            if ctx not in ctx_best:
                ctx_best[ctx] = {}
            ctx_best[ctx][seq] = ctx_best[ctx].get(seq, 0) + 1
        vocab = set()
        for ctx, seq_counts in ctx_best.items():
            best_seq = max(seq_counts, key=seq_counts.get)
            vocab.add(best_seq)
        return len(vocab)

    def _topographic_similarity(self, use_long: bool = False) -> float:
        """Spearman correlation between meaning distances and signal distances.
        Sample up to 200 (meaning, signal) pairs."""
        obs = self.long_observations if use_long else self.observations
        if len(obs) < 10:
            return 0.0

        # Build (context_key, sound_seq) pairs — deduplicate by most common mapping
        ctx_to_seq: dict[tuple, dict[tuple, int]] = defaultdict(lambda: defaultdict(int))
        for seq, ctx, _ in obs:
            ctx_to_seq[ctx][seq] += 1

        pairs = []
        for ctx, seqs in ctx_to_seq.items():
            best_seq = max(seqs, key=seqs.get)
            pairs.append((ctx, best_seq))

        if len(pairs) < 4:
            return 0.0

        # Sample up to 200 pairs
        if len(pairs) > 200:
            pairs = random.sample(pairs, 200)

        # Compute all pairwise distances
        meaning_dists = []
        signal_dists = []
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                md = _hamming(pairs[i][0], pairs[j][0])
                sd = _levenshtein(pairs[i][1], pairs[j][1])
                meaning_dists.append(md)
                signal_dists.append(sd)

        if not meaning_dists:
            return 0.0

        return _spearman(meaning_dists, signal_dists)

    def _positional_disentanglement(self) -> float:
        """For multi-sound messages: how well each position encodes a single dimension."""
        if len(self.observations) < 20:
            return 0.0

        # Only use multi-sound sequences
        multi = [(seq, ctx) for seq, ctx, _ in self.observations if len(seq) > 1]
        if len(multi) < 10:
            return 0.0

        max_len = max(len(seq) for seq, _ in multi)
        if max_len < 2:
            return 0.0

        n_dims = len(multi[0][1]) if multi else 0
        if n_dims == 0:
            return 0.0

        scores = []
        for j in range(min(max_len, 5)):  # up to 5 positions
            # Collect (sound_at_j, context) pairs
            pos_data = [(seq[j], ctx) for seq, ctx in multi if len(seq) > j]
            if len(pos_data) < 5:
                continue

            # H(sound_j)
            h_sj = _entropy_list([s for s, _ in pos_data])
            if h_sj < 0.01:
                continue

            # MI(sound_j, dim_k) for each k
            mis = []
            for k in range(n_dims):
                mi = _mi_two_lists(
                    [s for s, _ in pos_data],
                    [c[k] for _, c in pos_data],
                )
                mis.append(mi)

            mis.sort(reverse=True)
            if len(mis) >= 2 and h_sj > 0:
                gap = (mis[0] - mis[1]) / h_sj
                scores.append(max(0.0, min(1.0, gap)))

        return sum(scores) / len(scores) if scores else 0.0

    def _bag_of_symbols_disentanglement(self) -> float:
        """BosDis (Chaabouni 2020): for each context dimension, how well is it
        predicted by the BAG (unordered set) of symbols in the message?
        Complementary to PosDis — measures order-independent encoding."""
        if len(self.observations) < 20:
            return 0.0

        multi = [(seq, ctx) for seq, ctx, _ in self.observations if len(seq) > 1]
        if len(multi) < 10:
            return 0.0

        n_dims = len(multi[0][1]) if multi else 0
        if n_dims == 0:
            return 0.0

        scores = []
        for k in range(n_dims):
            # Build bag-of-symbols representation for each message
            bags = [frozenset(seq) for seq, _ in multi]
            dims_vals = [ctx[k] for _, ctx in multi]

            if len(set(dims_vals)) < 2:
                continue

            # H(bag)
            h_bag = _entropy_list(bags)
            if h_bag < 0.01:
                continue

            # MI(bag, dim_k)
            mi_k = _mi_two_lists(bags, dims_vals)

            # MI(bag, dim_j) for all other dims j != k
            other_mis = []
            for j in range(n_dims):
                if j == k:
                    continue
                other_vals = [ctx[j] for _, ctx in multi]
                other_mis.append(_mi_two_lists(bags, other_vals))

            if other_mis and h_bag > 0:
                max_other = max(other_mis)
                gap = (mi_k - max_other) / h_bag
                scores.append(max(0.0, min(1.0, gap)))

        return round(sum(scores) / len(scores), 4) if scores else 0.0

    def _mutual_information(self, use_long: bool = False) -> float:
        """MI(signal; context) from joint frequency table."""
        obs = self.long_observations if use_long else self.observations
        if len(obs) < 10:
            return 0.0

        joint: dict[tuple, int] = defaultdict(int)
        sig_count: dict[tuple, int] = defaultdict(int)
        ctx_count: dict[tuple, int] = defaultdict(int)
        total = 0

        for seq, ctx, _ in obs:
            joint[(seq, ctx)] += 1
            sig_count[seq] += 1
            ctx_count[ctx] += 1
            total += 1

        if total == 0:
            return 0.0

        mi = 0.0
        for (s, c), count in joint.items():
            p_sc = count / total
            p_s = sig_count[s] / total
            p_c = ctx_count[c] / total
            if p_s > 0 and p_c > 0 and p_sc > 0:
                mi += p_sc * math.log2(p_sc / (p_s * p_c))

        return round(mi, 4)

    def _social_mutual_information(self) -> float:
        """MI(signal; context) only from social agents with age>100."""
        if len(self.social_observations) < 10:
            return 0.0

        joint: dict[tuple, int] = defaultdict(int)
        sig_count: dict[tuple, int] = defaultdict(int)
        ctx_count: dict[tuple, int] = defaultdict(int)
        total = 0

        for seq, ctx, _ in self.social_observations:
            joint[(seq, ctx)] += 1
            sig_count[seq] += 1
            ctx_count[ctx] += 1
            total += 1

        if total == 0:
            return 0.0

        mi = 0.0
        for (s, c), count in joint.items():
            p_sc = count / total
            p_s = sig_count[s] / total
            p_c = ctx_count[c] / total
            if p_s > 0 and p_c > 0 and p_sc > 0:
                mi += p_sc * math.log2(p_sc / (p_s * p_c))

        return round(mi, 4)

    def _economic_delta(self, lobsters) -> float:
        """Compare avg credits of lobsters that spoke recently vs those that didn't."""
        if not lobsters:
            return 0.0

        # Track who spoke in recent observations (last 100)
        recent_speakers = set()
        for seq, ctx, _ in self.observations[-100:]:
            pass  # We don't store speaker IDs in observations

        # Use lobster state directly: those with speaking > 0 credits
        speakers = [l for l in lobsters.values() if l.credits_earned > l.credits_spent and l.agent_type == "social"]
        silent = [l for l in lobsters.values() if l.agent_type in ("greedy", "random")]

        if not speakers or not silent:
            return 0.0

        avg_speakers = sum(l.credits_earned - l.credits_spent for l in speakers) / len(speakers)
        avg_silent = sum(l.credits_earned - l.credits_spent for l in silent) / len(silent)

        return round(avg_speakers - avg_silent, 2)

    def _per_type_credits(self, lobsters) -> dict[str, float]:
        """Average net credits per agent type."""
        if not lobsters:
            return {}
        by_type: dict[str, list[float]] = {}
        for lob in lobsters.values():
            atype = lob.agent_type or "unknown"
            if atype not in by_type:
                by_type[atype] = []
            by_type[atype].append(lob.credits_earned - lob.credits_spent)
        return {
            atype: round(sum(vals) / len(vals), 1)
            for atype, vals in by_type.items()
            if vals
        }

    def compute_zero_shot_accuracy(self, novelty_set: set[tuple]) -> float:
        """For novel contexts (held-out combos): check if message is compositionally correct.
        Correct = same position-0 as other messages with same Feature 1 value,
        AND same position-1 as other messages with same Feature 2 value.
        Returns fraction correct out of novel encounters."""
        if not novelty_set or not self.long_observations:
            return 0.0

        # Build feature→position mapping from NON-novel observations
        # feature_val_to_sound[pos][feature_val] → most common sound at that position
        pos0_map: dict[int, dict[int, int]] = {}  # sit → {sound: count}
        pos1_map: dict[int, dict[int, int]] = {}  # tgt → {sound: count}

        novel_encounters = []
        for seq, ctx, _ in self.long_observations:
            if len(seq) < 2 or len(ctx) < 3:
                continue
            fc = (ctx[0], ctx[1], ctx[2])  # factored context in first 3 dims
            # Convert scaled values back to discrete indices
            sit = round(fc[0] * 3.0 / 8.0) if fc[0] > 0 else 0
            tgt = round(fc[1] * 4.0 / 8.0) if fc[1] > 0 else 0
            urg = round(fc[2] * 2.0 / 8.0) if fc[2] > 0 else 0
            fc_discrete = (sit, tgt, urg)

            if fc_discrete in novelty_set:
                novel_encounters.append((seq, fc_discrete))
            else:
                # Track training data for pos→feature mapping
                s0 = seq[0] if isinstance(seq[0], int) else 0
                s1 = seq[1] if isinstance(seq[1], int) else 0
                if sit not in pos0_map:
                    pos0_map[sit] = {}
                pos0_map[sit][s0] = pos0_map[sit].get(s0, 0) + 1
                if tgt not in pos1_map:
                    pos1_map[tgt] = {}
                pos1_map[tgt][s1] = pos1_map[tgt].get(s1, 0) + 1

        if not novel_encounters or not pos0_map or not pos1_map:
            return 0.0

        # Expected sound per feature value
        expected_pos0 = {sit: max(counts, key=counts.get) for sit, counts in pos0_map.items()}
        expected_pos1 = {tgt: max(counts, key=counts.get) for tgt, counts in pos1_map.items()}

        correct = 0
        total = 0
        for seq, (sit, tgt, urg) in novel_encounters:
            s0 = seq[0] if isinstance(seq[0], int) else 0
            s1 = seq[1] if isinstance(seq[1], int) else 0
            p0_ok = expected_pos0.get(sit) == s0
            p1_ok = expected_pos1.get(tgt) == s1
            if p0_ok and p1_ok:
                correct += 1
            total += 1

        return round(correct / total, 4) if total > 0 else 0.0

    def reset(self):
        self.observations.clear()
        self.long_observations.clear()
        self.social_observations.clear()
        self.latest = {}


# --- Helper functions ---

def _hamming(a: tuple, b: tuple) -> int:
    """Hamming distance between two tuples."""
    return sum(1 for x, y in zip(a, b) if x != y)


def _levenshtein(a: tuple, b: tuple) -> int:
    """Levenshtein distance between two sequences."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def _rank(values: list[float]) -> list[float]:
    """Assign ranks to values (average rank for ties)."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _spearman(x: list, y: list) -> float:
    """Spearman rank correlation."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    rx = _rank([float(v) for v in x])
    ry = _rank([float(v) for v in y])
    n = len(rx)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))
    if den_x == 0 or den_y == 0:
        return 0.0
    return round(num / (den_x * den_y), 4)


def _entropy_list(values: list) -> float:
    """Shannon entropy of a list of discrete values."""
    counts: dict = defaultdict(int)
    for v in values:
        counts[v] += 1
    total = len(values)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def _mi_two_lists(xs: list, ys: list) -> float:
    """Mutual information between two aligned lists of discrete values."""
    n = len(xs)
    if n == 0:
        return 0.0
    joint: dict[tuple, int] = defaultdict(int)
    x_count: dict = defaultdict(int)
    y_count: dict = defaultdict(int)
    for x, y in zip(xs, ys):
        joint[(x, y)] += 1
        x_count[x] += 1
        y_count[y] += 1
    mi = 0.0
    for (x, y), c in joint.items():
        p_xy = c / n
        p_x = x_count[x] / n
        p_y = y_count[y] / n
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * math.log2(p_xy / (p_x * p_y))
    return mi
