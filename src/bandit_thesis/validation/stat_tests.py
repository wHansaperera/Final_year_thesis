from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class BootstrapCI:
    diff_mean: float
    ci_low: float
    ci_high: float


def paired_bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapCI:
    """
    Paired bootstrap over seeds:
    - a[i], b[i] are metric for same seed i
    Returns CI for mean(a-b).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must have same shape (paired).")

    rng = np.random.default_rng(seed)
    n = a.size
    diffs = a - b
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(np.mean(diffs[idx]))

    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return BootstrapCI(diff_mean=float(np.mean(diffs)), ci_low=lo, ci_high=hi)
