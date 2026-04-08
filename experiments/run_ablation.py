from __future__ import annotations

import copy
import yaml
import pandas as pd
from typing import Dict, List

from experiments.run_stationary import run_one_seed as run_stationary_seed
from experiments.run_nonstationary import run_one_seed as run_nonstat_seed
from bandit_thesis.validation.reporting import summarize_experiment
from bandit_thesis.utils.io import write_csv


# ---------------------------
# CONFIG
# ---------------------------

BASE_CONFIG_PATH = "configs/stationary.yaml"  # change to nonstationary.yaml if needed
MODE = "stationary"  # "stationary" or "nonstationary"

WARMUP_VALUES = [10, 30, 50]
WINDOW_VALUES = [100, 200]  # only used for nonstationary


# ---------------------------
# MAIN ABLATION LOOP
# ---------------------------

def run_ablation():

    with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    seeds = int(base_cfg["experiment"]["seeds"])

    all_results = []

    for warmup in WARMUP_VALUES:

        for window in (WINDOW_VALUES if MODE == "nonstationary" else [None]):

            print(f"\n===== Running warmup={warmup}, window={window} =====")

            cfg = copy.deepcopy(base_cfg)
            cfg["models"]["hybrid"]["warmup_impressions"] = warmup

            if MODE == "nonstationary" and window is not None:
                cfg["models"]["probit"]["window"] = window

            metrics_by_agent = {
                "hybrid": {
                    "ctr": [],
                    "regret": [],
                    "time": []
                }
            }

            for s in range(seeds):

                if MODE == "stationary":
                    out = run_stationary_seed(seed=s, cfg=cfg)
                else:
                    out = run_nonstat_seed(seed=s, cfg=cfg)

                hybrid = out["hybrid"]

                metrics_by_agent["hybrid"]["ctr"].append(hybrid["ctr"])
                metrics_by_agent["hybrid"]["regret"].append(hybrid["final_regret"])
                metrics_by_agent["hybrid"]["time"].append(hybrid["time_sec"])

            summary = summarize_experiment(metrics_by_agent, seed_count=seeds)

            row = summary.table.iloc[0].to_dict()
            row["warmup"] = warmup
            row["window"] = window

            all_results.append(row)

    df = pd.DataFrame(all_results)

    print("\n==== ABLATION SUMMARY ====")
    print(df)

    write_csv("results/tables/hybrid_ablation.csv", df)


if __name__ == "__main__":
    run_ablation()