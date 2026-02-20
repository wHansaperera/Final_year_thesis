from __future__ import annotations
import numpy as np


def ctr_overall(rewards: np.ndarray) -> float:
    rewards = np.asarray(rewards, dtype=float)
    return float(np.mean(rewards)) if rewards.size > 0 else 0.0


def ctr_window(rewards: np.ndarray, window: int = 500) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=float)
    if rewards.size == 0:
        return rewards
    out = np.zeros_like(rewards, dtype=float)
    for t in range(rewards.size):
        lo = max(0, t - window + 1)
        out[t] = float(np.mean(rewards[lo : t + 1]))
    return out
