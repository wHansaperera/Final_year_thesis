# # from __future__ import annotations
# # from dataclasses import dataclass
# # from typing import Dict, List
# # import numpy as np
# # import pandas as pd

# # from .stat_tests import paired_bootstrap_ci


# # @dataclass
# # class ExperimentSummary:
# #     table: pd.DataFrame


# # def summarize_experiment(
# #     metrics_by_agent: Dict[str, Dict[str, List[float]]],
# #     seed_count: int,
# # ) -> ExperimentSummary:
# #     """
# #     metrics_by_agent example:
# #       {
# #         "random": {"ctr": [...], "regret": [...]},
# #         "logistic_ts": {"ctr": [...], "regret": [...]},
# #         "bayes_fm_ts": {"ctr": [...], "regret": [...]},
# #       }
# #     """
# #     rows = []
# #     for agent, m in metrics_by_agent.items():
# #         ctr = np.array(m["ctr"], dtype=float)
# #         reg = np.array(m["regret"], dtype=float)
# #         rows.append(
# #             {
# #                 "agent": agent,
# #                 "ctr_mean": float(np.mean(ctr)),
# #                 "ctr_std": float(np.std(ctr, ddof=1)),
# #                 "regret_mean": float(np.mean(reg)),
# #                 "regret_std": float(np.std(reg, ddof=1)),
# #                 ""
# #                 "n_seeds": seed_count,
# #             }
# #         )
# #     df = pd.DataFrame(rows).sort_values("ctr_mean", ascending=False)
# #     return ExperimentSummary(table=df)

# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Dict, List
# import numpy as np
# import pandas as pd


# @dataclass
# class ExperimentSummary:
#     table: pd.DataFrame


# def summarize_experiment(
#     metrics_by_agent: Dict[str, Dict[str, List[float]]],
#     seed_count: int,
# ) -> ExperimentSummary:
#     """
#     Generic experiment summary.
#     metrics_by_agent example:
#     {
#       "random": {"ctr": [...], "regret": [...], "time": [...], "ctr_after_shift": [...]},
#       "hybrid": {"ctr": [...], "regret": [...], ...}
#     }

#     Produces columns: <metric>_mean and <metric>_std for every metric.
#     """
#     rows = []

#     for agent, m in metrics_by_agent.items():
#         row = {"agent": agent, "n_seeds": seed_count}

#         for metric_name, values in m.items():
#             arr = np.array(values, dtype=float)
#             row[f"{metric_name}_mean"] = float(np.nanmean(arr))
#             row[f"{metric_name}_std"] = float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0

#         rows.append(row)

#     df = pd.DataFrame(rows)

#     if "ctr_mean" in df.columns:
#         df = df.sort_values("ctr_mean", ascending=False)

#     return ExperimentSummary(table=df)

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd

@dataclass
class ExperimentSummary:
    table: pd.DataFrame

def summarize_experiment(metrics_by_agent: Dict[str, Dict[str, List[float]]], seed_count: int) -> ExperimentSummary:
    rows = []
    for agent, m in metrics_by_agent.items():
        row = {"agent": agent, "n_seeds": seed_count}
        for metric_name, values in m.items():
            arr = np.array(values, dtype=float)
            row[f"{metric_name}_mean"] = float(np.nanmean(arr))
            row[f"{metric_name}_std"] = float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0
        rows.append(row)

    df = pd.DataFrame(rows)
    if "ctr_mean" in df.columns:
        df = df.sort_values("ctr_mean", ascending=False)
    return ExperimentSummary(table=df)