from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import yaml

from bandit_thesis.validation.reporting import summarize_experiment
from bandit_thesis.utils.io import write_csv
from experiments.run_nonstationary import run_one_seed as run_nonstationary_seed
from experiments.run_stationary import run_one_seed as run_stationary_seed


def run_ablation(config_path: str, mode: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    seeds = int(base_cfg["experiment"]["seeds"])
    warmup_values = list(base_cfg.get("ablation", {}).get("warmup_impressions", [base_cfg["models"]["hybrid"]["warmup_impressions"]]))
    recovery_values = list(base_cfg.get("ablation", {}).get("recovery_steps", [base_cfg["models"]["hybrid"].get("recovery_steps", 0)]))

    all_rows = []
    for warmup in warmup_values:
        for recovery_steps in (recovery_values if mode == "nonstationary" else [0]):
            cfg = copy.deepcopy(base_cfg)
            cfg["models"]["hybrid"]["warmup_impressions"] = int(warmup)
            cfg["models"]["hybrid"]["recovery_steps"] = int(recovery_steps)

            output_root = str(Path("results", "ablation", mode, f"warmup_{warmup}_recovery_{recovery_steps}"))
            metrics_by_agent = {"hybrid": {"ctr": [], "final_regret": [], "time_sec": []}}

            for seed in range(seeds):
                if mode == "stationary":
                    out = run_stationary_seed(seed=seed, cfg=cfg, output_root=output_root, save_raw=False)
                else:
                    out = run_nonstationary_seed(seed=seed, cfg=cfg, output_root=output_root, save_raw=False)

                hybrid_metrics = out["hybrid"]
                metrics_by_agent["hybrid"]["ctr"].append(hybrid_metrics["ctr"])
                metrics_by_agent["hybrid"]["final_regret"].append(hybrid_metrics["final_regret"])
                metrics_by_agent["hybrid"]["time_sec"].append(hybrid_metrics["time_sec"])

            summary = summarize_experiment(metrics_by_agent, seed_count=seeds).table.iloc[0].to_dict()
            summary["warmup_impressions"] = int(warmup)
            summary["recovery_steps"] = int(recovery_steps)
            all_rows.append(summary)

    df = pd.DataFrame(all_rows)
    out_path = str(Path("results", "tables", f"{mode}_hybrid_ablation.csv"))
    write_csv(out_path, df)
    print(df)
    print("Saved:", out_path)


def main() -> None:
    run_ablation("configs/stationary.yaml", mode="stationary")
    run_ablation("configs/nonstationary.yaml", mode="nonstationary")


if __name__ == "__main__":
    main()
