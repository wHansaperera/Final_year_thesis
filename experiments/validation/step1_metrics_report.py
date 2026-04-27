from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from bandit_thesis.metrics.cold_start import (
    cold_start_ctr_per_user,
    cold_start_regret_per_user,
    ctr_after_shift,
    ctr_at_n,
    ctr_last_w,
    regret_at_n,
)
from bandit_thesis.utils.io import read_jsonl, write_csv


@dataclass
class ReportPaths:
    out_dir: str = "results/validation/step1_metrics"
    tables_dir: str = "results/tables"


class MetricsReporter:
    def __init__(self, paths: ReportPaths) -> None:
        self.paths = paths
        os.makedirs(self.paths.out_dir, exist_ok=True)
        os.makedirs(self.paths.tables_dir, exist_ok=True)

    def run(self) -> None:
        for mode in ["stationary", "nonstationary"]:
            report = self._build_mode_report(mode)

            canonical_path = os.path.join(self.paths.tables_dir, f"{mode}_summary.csv")
            report_path = os.path.join(self.paths.out_dir, f"{mode}_metrics.csv")

            write_csv(canonical_path, report)
            write_csv(report_path, report)
            print("Saved:", canonical_path)
            print("Saved:", report_path)

        with open(os.path.join(self.paths.out_dir, "README.txt"), "w", encoding="utf-8") as f:
            f.write(
                "Step 1 Metrics Report\n"
                "- Rebuilds clean summary tables directly from raw per-seed logs.\n"
                "- Refreshes canonical summaries under results/tables/.\n"
                "- Exports thesis-facing copies under results/validation/step1_metrics/.\n"
                "- Runtime columns are carried over from the existing summary files when available.\n"
            )

    def _build_mode_report(self, mode: str) -> pd.DataFrame:
        config = self._load_config(mode)
        time_columns = self._load_runtime_columns(mode)
        rows = []

        for agent in self._agent_names(mode):
            seed_paths = sorted(glob.glob(os.path.join("results", "raw", mode, agent, "seed_*.jsonl")))
            metric_values: Dict[str, List[float]] = {metric: [] for metric in self._metrics_for_mode(mode)}

            for path in seed_paths:
                seed_rows = read_jsonl(path)
                metrics = self._compute_metrics(mode, seed_rows, config)
                for metric_name, value in metrics.items():
                    metric_values[metric_name].append(value)

            row = {"agent": agent, "n_seeds": len(seed_paths)}
            for metric_name, values in metric_values.items():
                arr = np.asarray(values, dtype=float)
                row[f"{metric_name}_mean"] = float(np.nanmean(arr))
                row[f"{metric_name}_std"] = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0

            runtime = time_columns.get(agent)
            if runtime is not None:
                row["time_sec_mean"] = runtime["time_sec_mean"]
                row["time_sec_std"] = runtime["time_sec_std"]

            rows.append(row)

        report = pd.DataFrame(rows)
        if "ctr_mean" in report.columns:
            report = report.sort_values("ctr_mean", ascending=False).reset_index(drop=True)

        numeric_cols = report.select_dtypes(include="number").columns
        report[numeric_cols] = report[numeric_cols].round(6)
        return report

    def _load_config(self, mode: str) -> dict:
        manifest_path = os.path.join("results", "manifests", f"{mode}.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)["config"]

    def _load_runtime_columns(self, mode: str) -> Dict[str, Dict[str, float]]:
        summary_path = os.path.join(self.paths.tables_dir, f"{mode}_summary.csv")
        if not os.path.exists(summary_path):
            return {}

        summary = pd.read_csv(summary_path)
        if not {"agent", "time_sec_mean", "time_sec_std"}.issubset(summary.columns):
            return {}

        runtime = {}
        for _, row in summary.iterrows():
            runtime[str(row["agent"])] = {
                "time_sec_mean": float(row["time_sec_mean"]),
                "time_sec_std": float(row["time_sec_std"]),
            }
        return runtime

    def _agent_names(self, mode: str) -> List[str]:
        summary_path = os.path.join(self.paths.tables_dir, f"{mode}_summary.csv")
        if os.path.exists(summary_path):
            summary = pd.read_csv(summary_path)
            if "agent" in summary.columns:
                return [str(agent) for agent in summary["agent"].tolist()]

        paths = glob.glob(os.path.join("results", "raw", mode, "*"))
        return sorted(os.path.basename(path) for path in paths if os.path.isdir(path))

    def _metrics_for_mode(self, mode: str) -> List[str]:
        if mode == "stationary":
            return ["ctr", "final_regret", "ctr_early", "regret_early", "cold_ctr", "cold_regret"]
        if mode == "nonstationary":
            return ["ctr", "final_regret", "cold_ctr", "cold_regret", "ctr_after_shift", "ctr_last_w"]
        raise ValueError(f"Unknown mode: {mode}")

    def _compute_metrics(self, mode: str, rows: List[dict], config: dict) -> Dict[str, float]:
        rewards = np.asarray([int(row["reward"]) for row in rows], dtype=float)
        regrets = np.asarray([float(row["regret"]) for row in rows], dtype=float)
        metrics_cfg = config.get("metrics", {})

        if mode == "stationary":
            cold_m = int(metrics_cfg.get("cold_m", 20))
            early_n = int(metrics_cfg.get("early_n", 1000))
            return {
                "ctr": float(np.mean(rewards)) if rewards.size else float("nan"),
                "final_regret": float(np.sum(regrets)) if regrets.size else float("nan"),
                "ctr_early": float(ctr_at_n(rows, n=early_n)),
                "regret_early": float(regret_at_n(rows, n=early_n)),
                "cold_ctr": float(cold_start_ctr_per_user(rows, m=cold_m)),
                "cold_regret": float(cold_start_regret_per_user(rows, m=cold_m)),
            }

        if mode == "nonstationary":
            cold_m = int(metrics_cfg.get("cold_m", 20))
            adapt_w = int(metrics_cfg.get("adapt_w", 500))
            shift_time = int(config["experiment"]["shift_time"])
            return {
                "ctr": float(np.mean(rewards)) if rewards.size else float("nan"),
                "final_regret": float(np.sum(regrets)) if regrets.size else float("nan"),
                "cold_ctr": float(cold_start_ctr_per_user(rows, m=cold_m)),
                "cold_regret": float(cold_start_regret_per_user(rows, m=cold_m)),
                "ctr_after_shift": float(ctr_after_shift(rows, shift_time=shift_time, w=adapt_w)),
                "ctr_last_w": float(ctr_last_w(rows, w=adapt_w)),
            }

        raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    MetricsReporter(ReportPaths()).run()


if __name__ == "__main__":
    main()
