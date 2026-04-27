from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import rankdata, shapiro, skew


@dataclass(frozen=True)
class BootstrapCI:
    diff_mean: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class NormalityCheck:
    n: int
    shapiro_stat: float
    shapiro_p: float
    skewness: float
    approximately_normal: bool


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


def check_paired_normality(diffs: np.ndarray, alpha: float = 0.05) -> NormalityCheck:
    values = np.asarray(diffs, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)

    if n < 3:
        return NormalityCheck(
            n=n,
            shapiro_stat=float("nan"),
            shapiro_p=float("nan"),
            skewness=float("nan"),
            approximately_normal=False,
        )

    if np.allclose(values, values[0]):
        return NormalityCheck(
            n=n,
            shapiro_stat=float("nan"),
            shapiro_p=float("nan"),
            skewness=0.0,
            approximately_normal=False,
        )

    shapiro_stat, shapiro_p = shapiro(values)
    skewness = float(skew(values, bias=False))
    return NormalityCheck(
        n=n,
        shapiro_stat=float(shapiro_stat),
        shapiro_p=float(shapiro_p),
        skewness=skewness,
        approximately_normal=bool(np.isfinite(shapiro_p) and shapiro_p >= alpha),
    )


def paired_rank_biserial(diffs: np.ndarray) -> float:
    values = np.asarray(diffs, dtype=float)
    values = values[np.isfinite(values)]
    values = values[~np.isclose(values, 0.0)]
    if values.size == 0:
        return float("nan")

    ranks = rankdata(np.abs(values), method="average")
    pos_rank_sum = float(np.sum(ranks[values > 0]))
    neg_rank_sum = float(np.sum(ranks[values < 0]))
    total_rank_sum = pos_rank_sum + neg_rank_sum
    if total_rank_sum == 0.0:
        return float("nan")
    return float((pos_rank_sum - neg_rank_sum) / total_rank_sum)


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
