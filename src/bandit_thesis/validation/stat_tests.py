from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BootstrapCI:
    diff_mean: float
    ci_low: float
    ci_high: float


def paired_bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapCI:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must have the same paired shape.")

    rng = np.random.default_rng(seed)
    diffs = a - b
    n = diffs.size
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(np.mean(diffs[idx]))

    return BootstrapCI(
        diff_mean=float(np.mean(diffs)),
        ci_low=float(np.quantile(boot, alpha / 2)),
        ci_high=float(np.quantile(boot, 1 - alpha / 2)),
    )


def paired_cohens_dz(a: np.ndarray, b: np.ndarray) -> float:
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sd = float(np.std(diffs, ddof=1)) if diffs.size > 1 else 0.0
    if sd == 0.0:
        return float("nan")
    return float(np.mean(diffs) / sd)


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * p[idx]
        running = max(running, value)
        adjusted[idx] = min(running, 1.0)
    return adjusted
