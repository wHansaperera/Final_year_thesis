from __future__ import annotations
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from bandit_thesis.validation.stat_tests import paired_bootstrap_ci


@dataclass
class SigConfig:
    mode: str  # "stationary" or "nonstationary"
    out_dir: str = "results/validation/step2_significance"
    n_boot: int = 5000
    alpha: float = 0.05


class RawLoader:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.base = os.path.join("results", "raw", mode)

    def list_seeds(self, agent: str) -> List[int]:
        paths = glob.glob(os.path.join(self.base, agent, "seed_*.jsonl"))
        seeds = []
        for p in paths:
            s = int(os.path.basename(p).split("_")[1].split(".")[0])
            seeds.append(s)
        return sorted(seeds)

    def load_rows(self, agent: str, seed: int) -> List[dict]:
        path = os.path.join(self.base, agent, f"seed_{seed}.jsonl")
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        return rows


class MetricComputer:
    @staticmethod
    def ctr(rows: List[dict]) -> float:
        if not rows:
            return float("nan")
        return float(np.mean([r["reward"] for r in rows]))

    @staticmethod
    def final_regret(rows: List[dict]) -> float:
        # cumulative pseudo regret sum(p_opt - p_chosen)
        if not rows:
            return float("nan")
        return float(np.sum([float(r["p_opt"]) - float(r["p_chosen"]) for r in rows]))

    @staticmethod
    def ctr_after_shift(rows: List[dict], shift_time: int, w: int) -> float:
        post = [r for r in rows if int(r["t"]) >= shift_time][:w]
        if not post:
            return float("nan")
        return float(np.mean([r["reward"] for r in post]))

    @staticmethod
    def ctr_last_w(rows: List[dict], w: int) -> float:
        tail = rows[-w:] if len(rows) >= 1 else []
        if not tail:
            return float("nan")
        return float(np.mean([r["reward"] for r in tail]))


class SignificanceRunner:
    def __init__(self, cfg: SigConfig) -> None:
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)
        self.loader = RawLoader(cfg.mode)

    def run(self) -> None:
        if self.cfg.mode == "stationary":
            comparisons = [
                ("ctr", "hybrid", "random"),
                ("ctr", "hybrid", "probit_ts"),
                ("final_regret", "hybrid", "random"),
                ("final_regret", "hybrid", "probit_ts"),
                ("ctr", "probit_ts", "random"),
            ]
        else:
            # assumes nonstationary shift_time and adapt_w same as your config defaults
            comparisons = [
                ("ctr", "hybrid", "random"),
                ("ctr", "hybrid", "probit_ts"),
                ("final_regret", "hybrid", "random"),
                ("final_regret", "hybrid", "probit_ts"),
                ("ctr_after_shift", "hybrid", "probit_ts"),
                ("ctr_last_w", "hybrid", "probit_ts"),
            ]

        rows_out = []
        for metric, A, B in comparisons:
            rows_out.append(self._compare(metric, A, B))

        df = pd.DataFrame(rows_out)
        out_path = os.path.join(self.cfg.out_dir, f"{self.cfg.mode}_significance.csv")
        df.to_csv(out_path, index=False)
        print("Saved:", out_path)
        print(df.round(6).to_string(index=False))

    def _compare(self, metric: str, A: str, B: str) -> dict:
        seeds = self.loader.list_seeds(A)
        seeds_b = self.loader.list_seeds(B)
        seeds = [s for s in seeds if s in set(seeds_b)]

        xa, xb = [], []
        for s in seeds:
            ra = self.loader.load_rows(A, s)
            rb = self.loader.load_rows(B, s)
            xa.append(self._metric(metric, ra))
            xb.append(self._metric(metric, rb))

        xa = np.array(xa, dtype=float)
        xb = np.array(xb, dtype=float)

        # paired t-test
        t_stat, p_value = ttest_rel(xa, xb, nan_policy="omit")

        # paired bootstrap CI for mean diff
        # paired bootstrap CI for mean diff
        ci = paired_bootstrap_ci(xa, xb, n_boot=self.cfg.n_boot, alpha=self.cfg.alpha)

        # support both naming styles: (low/high) or (ci_low/ci_high)
        ci_low = float(getattr(ci, "low", getattr(ci, "ci_low")))
        ci_high = float(getattr(ci, "high", getattr(ci, "ci_high")))

        diff = xa - xb
        return {
            "mode": self.cfg.mode,
            "metric": metric,
            "A": A,
            "B": B,
            "mean_A": float(np.nanmean(xa)),
            "mean_B": float(np.nanmean(xb)),
            "mean_diff(A-B)": float(np.nanmean(diff)),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "p_value": float(p_value),
            "n_seeds": int(np.sum(np.isfinite(diff))),
        }

    def _metric(self, metric: str, rows: List[dict]) -> float:
        if metric == "ctr":
            return MetricComputer.ctr(rows)
        if metric == "final_regret":
            return MetricComputer.final_regret(rows)
        if metric == "ctr_after_shift":
            # You can set these via env config later; keep simple defaults
            return MetricComputer.ctr_after_shift(rows, shift_time=5000, w=500)
        if metric == "ctr_last_w":
            return MetricComputer.ctr_last_w(rows, w=500)
        raise ValueError(f"Unknown metric: {metric}")


def main() -> None:
    for mode in ["stationary", "nonstationary"]:
        SignificanceRunner(SigConfig(mode=mode)).run()


if __name__ == "__main__":
    main()