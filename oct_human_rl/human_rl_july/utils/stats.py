"""
Bootstrap confidence intervals and paired significance tests.

`evaluate_policy` gives every policy the same episode seeds, so episode i is
the same warehouse and the same partner for all of them. Comparisons are
therefore paired, which is both correct and more sensitive than treating the
samples as independent — `paired_diff` relies on the two sequences lining up
element-for-element and refuses mismatched lengths.

Bootstrap rather than a t-test because CES and episode reward are bounded and
skewed. Seeds are explicit so a reported interval can be reproduced.
"""

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from utils.metrics import CoordinationMetrics

RESAMPLES = 10_000


def significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


@dataclass
class Estimate:
    mean: float
    lo:   float
    hi:   float
    n:    int

    def as_dict(self) -> dict:
        return {"mean": self.mean, "ci_lo": self.lo, "ci_hi": self.hi, "n": self.n}


@dataclass
class Comparison:
    """mean(a) - mean(b) over paired episodes."""
    label_a: str
    label_b: str
    mean_a:  float
    mean_b:  float
    diff:    float
    lo:      float
    hi:      float
    p_value: float
    n:       int

    @property
    def significant(self) -> bool:
        return self.lo > 0.0 or self.hi < 0.0

    def verdict(self) -> str:
        if not self.significant:
            return "no significant difference"
        return "significantly better" if self.diff > 0 else "significantly worse"

    def __str__(self) -> str:
        return (f"{self.label_a} - {self.label_b} = {self.diff:+.4f} "
                f"[{self.lo:+.4f}, {self.hi:+.4f}]  p={self.p_value:.4f}"
                f" {significance_stars(self.p_value)}".rstrip())

    def as_dict(self) -> dict:
        return {"a": self.label_a, "b": self.label_b,
                "mean_a": self.mean_a, "mean_b": self.mean_b,
                "diff": self.diff, "ci_lo": self.lo, "ci_hi": self.hi,
                "p_value": self.p_value, "n": self.n,
                "significant": self.significant}


def _resample_means(values: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(RESAMPLES, values.size))
    return values[idx].mean(axis=1)


def bootstrap_ci(values: Sequence[float], level: float = 0.95,
                 seed: int = 0) -> Estimate:
    """Percentile bootstrap interval for the mean."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return Estimate(0.0, 0.0, 0.0, 0)
    mean = float(arr.mean())
    if arr.size == 1:
        return Estimate(mean, mean, mean, 1)

    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(_resample_means(arr, seed), [alpha, 1.0 - alpha])
    return Estimate(mean, float(lo), float(hi), arr.size)


def paired_diff(a: Sequence[float], b: Sequence[float],
                label_a: str = "a", label_b: str = "b",
                level: float = 0.95, seed: int = 0) -> Comparison:
    arr_a = np.asarray(list(a), dtype=float)
    arr_b = np.asarray(list(b), dtype=float)
    if arr_a.size != arr_b.size:
        raise ValueError(f"paired comparison needs aligned samples, got "
                         f"{arr_a.size} and {arr_b.size} — the two policies "
                         "were not evaluated on the same episodes")
    if arr_a.size == 0:
        return Comparison(label_a, label_b, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0)

    deltas = arr_a - arr_b
    observed = float(deltas.mean())
    if deltas.size == 1:
        return Comparison(label_a, label_b, float(arr_a[0]), float(arr_b[0]),
                          observed, observed, observed, 1.0, 1)

    boot = _resample_means(deltas, seed)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(boot, [alpha, 1.0 - alpha])

    # Two-sided test of "the mean difference is zero": recentre the bootstrap
    # distribution on zero and count resamples at least as extreme. The +1
    # keeps p strictly positive.
    extreme = int(np.sum(np.abs(boot - observed) >= abs(observed)))
    p_value = min((extreme + 1) / (RESAMPLES + 1), 1.0)

    return Comparison(label_a, label_b, float(arr_a.mean()), float(arr_b.mean()),
                      observed, float(lo), float(hi), float(p_value), deltas.size)


def episode_ces(metrics: CoordinationMetrics) -> List[float]:
    """Per-episode CES in recording order, so two policies line up for pairing."""
    return [CoordinationMetrics.ces(s) for s in metrics.history]
