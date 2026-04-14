from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from bandit_thesis.metrics.cold_start import (
    cold_start_ctr_per_user,
    cold_start_regret_per_user,
    ctr_after_shift,
    ctr_last_w,
)
from bandit_thesis.validation.stat_tests import holm_adjust, paired_bootstrap_ci


@dataclass
class SigConfig:
    mode: str
    out_dir: str = "results/validation/step2_significance"
    n_boot: int = 5000
    alpha: float = 0.05


class RawLoader:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.base = os.path.join("results", "raw", mode)
        self.manifest_path = os.path.join("results", "manifests", f"{mode}.json")
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

    @property
    def config(self) -> dict:
        return self.manifest["config"]

    def list_seeds(self, agent: str) -> List[int]:
        paths = glob.glob(os.path.join(self.base, agent, "seed_*.jsonl"))
        return sorted(int(os.path.basename(path).split("_")[1].split(".")[0]) for path in paths)

    def load_rows(self, agent: str, seed: int) -> List[dict]:
        path = os.path.join(self.base, agent, f"seed_{seed}.jsonl")
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        return rows


class MetricComputer:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

    def compute(self, metric: str, rows: List[dict]) -> float:
        if metric == "ctr":
            return float(np.mean([row["reward"] for row in rows])) if rows else float("nan")
        if metric == "final_regret":
            return float(np.sum([float(row["regret"]) for row in rows])) if rows else float("nan")
        if metric == "cold_ctr":
            return float(cold_start_ctr_per_user(rows, m=int(self.cfg["metrics"]["cold_m"])))
        if metric == "cold_regret":
            return float(cold_start_regret_per_user(rows, m=int(self.cfg["metrics"]["cold_m"])))
        if metric == "ctr_after_shift":
            return float(
                ctr_after_shift(
                    rows,
                    shift_time=int(self.cfg["experiment"]["shift_time"]),
                    w=int(self.cfg["metrics"]["adapt_w"]),
                )
            )
        if metric == "ctr_last_w":
            return float(ctr_last_w(rows, w=int(self.cfg["metrics"]["adapt_w"])))
        raise ValueError(f"Unknown metric: {metric}")


class SignificanceRunner:
    def __init__(self, cfg: SigConfig) -> None:
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)
        self.loader = RawLoader(cfg.mode)
        self.metric_computer = MetricComputer(self.loader.config)

    def _comparisons(self) -> List[Tuple[str, str, str, bool]]:
        if self.cfg.mode == "stationary":
            return [
                ("ctr", "hybrid", "probit_ts", True),
                ("final_regret", "hybrid", "probit_ts", True),
                ("cold_ctr", "hybrid", "probit_ts", True),
                ("cold_regret", "hybrid", "probit_ts", True),
                ("ctr", "hybrid", "bayes_fm_ts", True),
                ("final_regret", "hybrid", "bayes_fm_ts", True),
                ("cold_ctr", "hybrid", "bayes_fm_ts", True),
                ("cold_regret", "hybrid", "bayes_fm_ts", True),
                ("ctr", "hybrid", "random", False),
                ("final_regret", "hybrid", "random", False),
            ]
        return [
            ("ctr", "hybrid", "probit_ts", True),
            ("final_regret", "hybrid", "probit_ts", True),
            ("cold_ctr", "hybrid", "probit_ts", True),
            ("cold_regret", "hybrid", "probit_ts", True),
            ("ctr_after_shift", "hybrid", "probit_ts", True),
            ("ctr_last_w", "hybrid", "probit_ts", True),
            ("ctr", "hybrid", "bayes_fm_ts", True),
            ("final_regret", "hybrid", "bayes_fm_ts", True),
            ("cold_ctr", "hybrid", "bayes_fm_ts", True),
            ("cold_regret", "hybrid", "bayes_fm_ts", True),
            ("ctr", "hybrid", "random", False),
            ("final_regret", "hybrid", "random", False),
        ]

    def run(self) -> None:
        rows_out = [self._compare(*comparison) for comparison in self._comparisons()]
        df = pd.DataFrame(rows_out)

        primary_mask = df["primary"].astype(bool).to_numpy()
        adjusted = np.full(len(df), np.nan, dtype=float)
        valid_primary = primary_mask & np.isfinite(df["ttest_p"].to_numpy())
        if np.any(valid_primary):
            adjusted_primary = holm_adjust(df.loc[valid_primary, "ttest_p"].to_numpy())
            adjusted[valid_primary] = adjusted_primary
        df["holm_ttest_p"] = adjusted

        out_path = os.path.join(self.cfg.out_dir, f"{self.cfg.mode}_significance.csv")
        df.to_csv(out_path, index=False)
        print("Saved:", out_path)
        print(df.round(6).to_string(index=False))

    def _compare(self, metric: str, agent_a: str, agent_b: str, primary: bool) -> dict:
        seeds_a = set(self.loader.list_seeds(agent_a))
        seeds_b = set(self.loader.list_seeds(agent_b))
        seeds = sorted(seeds_a.intersection(seeds_b))

        xa = []
        xb = []
        for seed in seeds:
            rows_a = self.loader.load_rows(agent_a, seed)
            rows_b = self.loader.load_rows(agent_b, seed)
            xa.append(self.metric_computer.compute(metric, rows_a))
            xb.append(self.metric_computer.compute(metric, rows_b))

        xa_arr = np.asarray(xa, dtype=float)
        xb_arr = np.asarray(xb, dtype=float)
        diff = xa_arr - xb_arr

        if int(np.sum(np.isfinite(diff))) < 2:
            ttest_p = float("nan")
            wilcoxon_p = float("nan")
        else:
            ttest_p = float(ttest_rel(xa_arr, xb_arr, nan_policy="omit").pvalue)
            if np.allclose(diff, 0.0, equal_nan=True):
                wilcoxon_p = float("nan")
            else:
                wilcoxon_p = float(wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)

        ci = paired_bootstrap_ci(
            xa_arr,
            xb_arr,
            n_boot=self.cfg.n_boot,
            alpha=self.cfg.alpha,
            seed=0,
        )

        return {
            "mode": self.cfg.mode,
            "metric": metric,
            "A": agent_a,
            "B": agent_b,
            "primary": primary,
            "mean_A": float(np.nanmean(xa_arr)),
            "mean_B": float(np.nanmean(xb_arr)),
            "mean_diff(A-B)": float(np.nanmean(diff)),
            "ci_low": float(ci.ci_low),
            "ci_high": float(ci.ci_high),
            "ttest_p": ttest_p,
            "wilcoxon_p": wilcoxon_p,
            "n_seeds": int(np.sum(np.isfinite(diff))),
        }


def main() -> None:
    for mode in ["stationary", "nonstationary"]:
        SignificanceRunner(SigConfig(mode=mode)).run()


if __name__ == "__main__":
    main()
