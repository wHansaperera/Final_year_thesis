from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from bandit_thesis.metrics.cold_start import (
    cold_start_ctr_per_user,
    cold_start_regret_per_user,
    ctr_after_shift,
    ctr_last_w,
)
from bandit_thesis.validation.stat_tests import paired_cohens_dz


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

    def list_seeds(self, agent: str):
        paths = glob.glob(os.path.join(self.base, agent, "seed_*.jsonl"))
        return sorted(int(os.path.basename(path).split("_")[1].split(".")[0]) for path in paths)

    def load_rows(self, agent: str, seed: int):
        rows = []
        with open(os.path.join(self.base, agent, f"seed_{seed}.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        return rows


class MetricComputer:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

    def compute(self, metric: str, rows):
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


class EffectSizeRunner:
    def __init__(self, mode: str, out_dir: str = "results/validation/step3_effect_size") -> None:
        self.mode = mode
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.loader = RawLoader(mode)
        self.metric_computer = MetricComputer(self.loader.config)
        self.sig_csv = os.path.join("results", "validation", "step2_significance", f"{mode}_significance.csv")

    def run(self) -> None:
        df = pd.read_csv(self.sig_csv)
        effect_sizes = []
        for _, row in df.iterrows():
            agent_a = str(row["A"])
            agent_b = str(row["B"])
            metric = str(row["metric"])

            seeds = sorted(set(self.loader.list_seeds(agent_a)).intersection(self.loader.list_seeds(agent_b)))
            xa = []
            xb = []
            for seed in seeds:
                rows_a = self.loader.load_rows(agent_a, seed)
                rows_b = self.loader.load_rows(agent_b, seed)
                xa.append(self.metric_computer.compute(metric, rows_a))
                xb.append(self.metric_computer.compute(metric, rows_b))

            effect_sizes.append(paired_cohens_dz(np.asarray(xa, dtype=float), np.asarray(xb, dtype=float)))

        df["cohens_dz"] = effect_sizes
        out_path = os.path.join(self.out_dir, f"{self.mode}_effect_size.csv")
        df.to_csv(out_path, index=False)
        print("Saved:", out_path)
        print(df.round(6).to_string(index=False))


def main() -> None:
    for mode in ["stationary", "nonstationary"]:
        EffectSizeRunner(mode=mode).run()


if __name__ == "__main__":
    main()
