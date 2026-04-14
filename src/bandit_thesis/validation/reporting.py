from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class ExperimentSummary:
    table: pd.DataFrame


def summarize_experiment(
    metrics_by_agent: Dict[str, Dict[str, List[float]]],
    seed_count: int,
) -> ExperimentSummary:
    rows = []
    for agent, metric_map in metrics_by_agent.items():
        row = {"agent": agent, "n_seeds": seed_count}
        for metric_name, values in metric_map.items():
            arr = np.asarray(values, dtype=float)
            row[f"{metric_name}_mean"] = float(np.nanmean(arr))
            row[f"{metric_name}_std"] = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0
        rows.append(row)

    table = pd.DataFrame(rows)
    if "ctr_mean" in table.columns:
        table = table.sort_values("ctr_mean", ascending=False)
    return ExperimentSummary(table=table)
