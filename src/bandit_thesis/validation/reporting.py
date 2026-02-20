from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd

from .stat_tests import paired_bootstrap_ci


@dataclass
class ExperimentSummary:
    table: pd.DataFrame


def summarize_experiment(
    metrics_by_agent: Dict[str, Dict[str, List[float]]],
    seed_count: int,
) -> ExperimentSummary:
    """
    metrics_by_agent example:
      {
        "random": {"ctr": [...], "regret": [...]},
        "logistic_ts": {"ctr": [...], "regret": [...]},
        "bayes_fm_ts": {"ctr": [...], "regret": [...]},
      }
    """
    rows = []
    for agent, m in metrics_by_agent.items():
        ctr = np.array(m["ctr"], dtype=float)
        reg = np.array(m["regret"], dtype=float)
        rows.append(
            {
                "agent": agent,
                "ctr_mean": float(np.mean(ctr)),
                "ctr_std": float(np.std(ctr, ddof=1)),
                "regret_mean": float(np.mean(reg)),
                "regret_std": float(np.std(reg, ddof=1)),
                "n_seeds": seed_count,
            }
        )
    df = pd.DataFrame(rows).sort_values("ctr_mean", ascending=False)
    return ExperimentSummary(table=df)
