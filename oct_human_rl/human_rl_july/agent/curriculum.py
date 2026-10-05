"""
PartnerCurriculum — pick which partner the agent trains against next.

Uniform sampling spends as much budget on partners the agent already handles
perfectly as on the one it keeps failing. This weights towards the ones it is
currently worst at, scored by a running estimate of coordination efficiency.

Prioritising the hardest scenario is known to backfire if left unchecked — the
schedule collapses onto it and the policy regresses everywhere else. Two guards
prevent that: sampling stays uniform until every partner has been seen `warmup`
times, and no partner's probability may fall below `min_share` of uniform, so a
mastered partner still appears and cannot be forgotten.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np


class PartnerCurriculum:

    def __init__(self, pool: Sequence[str], temperature: float = 0.35,
                 ema: float = 0.1, warmup: int = 5, min_share: float = 0.4):
        """
        temperature  softmax temperature over negative CES. Observed CES spans
                     roughly 0.40 to 0.74; 0.35 turns that into about a 2x
                     over-sampling of the weakest partner, while 0.15 pushes it
                     past 60% of all episodes.
        min_share    floor on each partner's probability, as a fraction of uniform
        """
        if not pool:
            raise ValueError("curriculum needs a non-empty partner pool")
        self.pool: List[str] = list(pool)
        self.temperature = max(temperature, 1e-6)
        self.ema = ema
        self.warmup = warmup
        self.min_share = min_share
        self.score: Dict[str, Optional[float]] = {p: None for p in self.pool}
        self.counts: Dict[str, int] = {p: 0 for p in self.pool}

    def update(self, profile: str, ces: float):
        if profile not in self.score:
            return                      # e.g. a combined label for two humans
        self.counts[profile] += 1
        previous = self.score[profile]
        self.score[profile] = ces if previous is None else (
            (1.0 - self.ema) * previous + self.ema * ces)

    def weights(self) -> np.ndarray:
        n = len(self.pool)
        if any(c < self.warmup for c in self.counts.values()):
            return np.full(n, 1.0 / n)

        logits = -np.array([self.score[p] for p in self.pool]) / self.temperature
        w = np.exp(logits - logits.max())
        w /= w.sum()

        floor = self.min_share / n
        if w.min() < floor:
            deficit = np.maximum(floor - w, 0.0).sum()
            surplus = np.maximum(w - floor, 0.0)
            w = np.maximum(w, floor) - surplus / surplus.sum() * deficit
            w /= w.sum()
        return w

    def sample(self, rng: np.random.Generator) -> str:
        return str(rng.choice(self.pool, p=self.weights()))

    def summary(self) -> str:
        return "  ".join(
            f"{name} {weight * 100:.0f}%" +
            (f"/CES {self.score[name]:.2f}" if self.score[name] is not None else "")
            for name, weight in zip(self.pool, self.weights()))
