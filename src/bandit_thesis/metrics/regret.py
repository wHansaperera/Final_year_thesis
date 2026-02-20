from __future__ import annotations
import numpy as np


def cumulative_pseudo_regret(p_opt: np.ndarray, p_chosen: np.ndarray) -> np.ndarray:
    p_opt = np.asarray(p_opt, dtype=float)
    p_chosen = np.asarray(p_chosen, dtype=float)
    return np.cumsum(p_opt - p_chosen)


def cumulative_dynamic_regret(p_opt: np.ndarray, p_chosen: np.ndarray) -> np.ndarray:
    # In this simple simulator, dynamic regret equals pseudo-regret because oracle changes with time.
    return cumulative_pseudo_regret(p_opt, p_chosen)
