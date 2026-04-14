from __future__ import annotations

import glob
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "bandit_thesis_mpl"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class PlotCfg:
    mode: str
    out_dir: str = "results/figures"
    recovery_bin: int = 100


class PlotBuilder:
    def __init__(self, cfg: PlotCfg) -> None:
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)
        manifest_path = os.path.join("results", "manifests", f"{cfg.mode}.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        self.config = self.manifest["config"]
        self.base = os.path.join("results", "raw", cfg.mode)
        self.agents = self.manifest["agents"]

    def run(self) -> None:
        self._plot_ctr_over_time()
        self._plot_regret_over_time()
        self._plot_cold_start_curve()
        if self.cfg.mode == "nonstationary":
            self._plot_post_shift_recovery()
        self._plot_seed_boxplots()
        self._plot_runtime_bars()

    def _seed_paths(self, agent: str) -> List[str]:
        return sorted(glob.glob(os.path.join(self.base, agent, "seed_*.jsonl")))

    def _load_rows(self, path: str) -> List[dict]:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        return rows

    def _mean_ci(self, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = np.nanmean(arr, axis=0)
        if arr.shape[0] <= 1:
            return mean, mean, mean
        se = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
        band = 1.96 * se
        return mean, mean - band, mean + band

    def _plot_ctr_over_time(self) -> None:
        plt.figure(figsize=(10, 6))
        for agent in self.agents:
            series = []
            for path in self._seed_paths(agent):
                rewards = np.array([int(row["reward"]) for row in self._load_rows(path)], dtype=float)
                series.append(np.cumsum(rewards) / np.arange(1, rewards.size + 1))
            arr = np.vstack(series)
            mean, low, high = self._mean_ci(arr)
            x = np.arange(1, mean.size + 1)
            plt.plot(x, mean, label=agent)
            plt.fill_between(x, low, high, alpha=0.15)
        plt.xlabel("Time step")
        plt.ylabel("Cumulative CTR")
        plt.title(f"{self.cfg.mode.title()} CTR Over Time")
        plt.legend()
        plt.tight_layout()
        path = os.path.join(self.cfg.out_dir, f"{self.cfg.mode}_ctr_over_time.png")
        plt.savefig(path, dpi=200)
        plt.close()
        print("Saved:", path)

    def _plot_regret_over_time(self) -> None:
        plt.figure(figsize=(10, 6))
        for agent in self.agents:
            series = []
            for path in self._seed_paths(agent):
                regrets = np.array([float(row["regret"]) for row in self._load_rows(path)], dtype=float)
                series.append(np.cumsum(regrets))
            arr = np.vstack(series)
            mean, low, high = self._mean_ci(arr)
            x = np.arange(1, mean.size + 1)
            plt.plot(x, mean, label=agent)
            plt.fill_between(x, low, high, alpha=0.15)
        plt.xlabel("Time step")
        plt.ylabel("Cumulative regret")
        plt.title(f"{self.cfg.mode.title()} Regret Over Time")
        plt.legend()
        plt.tight_layout()
        path = os.path.join(self.cfg.out_dir, f"{self.cfg.mode}_regret_over_time.png")
        plt.savefig(path, dpi=200)
        plt.close()
        print("Saved:", path)

    def _plot_cold_start_curve(self) -> None:
        cold_m = int(self.config["metrics"]["cold_m"])
        plt.figure(figsize=(10, 6))
        x = np.arange(cold_m)
        for agent in self.agents:
            curves = []
            for path in self._seed_paths(agent):
                rows = self._load_rows(path)
                values = []
                for age in x:
                    age_rows = [int(row["reward"]) for row in rows if int(row["user_history_len"]) == int(age)]
                    values.append(float(np.mean(age_rows)) if age_rows else np.nan)
                curves.append(values)
            arr = np.asarray(curves, dtype=float)
            mean, low, high = self._mean_ci(arr)
            plt.plot(x + 1, mean, marker="o", label=agent)
            plt.fill_between(x + 1, low, high, alpha=0.15)
        plt.xlabel("User impression index")
        plt.ylabel("CTR")
        plt.title(f"{self.cfg.mode.title()} Cold-Start CTR")
        plt.legend()
        plt.tight_layout()
        path = os.path.join(self.cfg.out_dir, f"{self.cfg.mode}_cold_start_ctr.png")
        plt.savefig(path, dpi=200)
        plt.close()
        print("Saved:", path)

    def _plot_post_shift_recovery(self) -> None:
        shift_time = int(self.config["experiment"]["shift_time"])
        adapt_w = int(self.config["metrics"]["adapt_w"])
        bin_width = int(self.cfg.recovery_bin)
        bins = np.arange(0, adapt_w, bin_width)

        plt.figure(figsize=(10, 6))
        for agent in self.agents:
            curves = []
            for path in self._seed_paths(agent):
                rows = self._load_rows(path)
                post = rows[shift_time : shift_time + adapt_w]
                values = []
                for start in bins:
                    chunk = post[start : start + bin_width]
                    values.append(float(np.mean([int(row["reward"]) for row in chunk])) if chunk else np.nan)
                curves.append(values)
            arr = np.asarray(curves, dtype=float)
            mean, low, high = self._mean_ci(arr)
            x = bins + 1
            plt.plot(x, mean, marker="o", label=agent)
            plt.fill_between(x, low, high, alpha=0.15)
        plt.xlabel("Steps after shift")
        plt.ylabel("Bin CTR")
        plt.title("Post-Shift Recovery CTR")
        plt.legend()
        plt.tight_layout()
        path = os.path.join(self.cfg.out_dir, f"{self.cfg.mode}_post_shift_recovery.png")
        plt.savefig(path, dpi=200)
        plt.close()
        print("Saved:", path)

    def _plot_seed_boxplots(self) -> None:
        ctr_data: Dict[str, List[float]] = {}
        regret_data: Dict[str, List[float]] = {}
        for agent in self.agents:
            ctr_values = []
            regret_values = []
            for path in self._seed_paths(agent):
                rows = self._load_rows(path)
                ctr_values.append(float(np.mean([int(row["reward"]) for row in rows])))
                regret_values.append(float(np.sum([float(row["regret"]) for row in rows])))
            ctr_data[agent] = ctr_values
            regret_data[agent] = regret_values

        for metric_name, metric_data in [("final_ctr", ctr_data), ("final_regret", regret_data)]:
            plt.figure(figsize=(10, 6))
            labels = list(metric_data.keys())
            values = [metric_data[label] for label in labels]
            plt.boxplot(values, tick_labels=labels, vert=True)
            plt.ylabel(metric_name.replace("_", " ").title())
            plt.title(f"{self.cfg.mode.title()} {metric_name.replace('_', ' ').title()} Across Seeds")
            plt.tight_layout()
            path = os.path.join(self.cfg.out_dir, f"{self.cfg.mode}_{metric_name}_boxplot.png")
            plt.savefig(path, dpi=200)
            plt.close()
            print("Saved:", path)

    def _plot_runtime_bars(self) -> None:
        summary_path = os.path.join("results", "tables", f"{self.cfg.mode}_summary.csv")
        df = pd.read_csv(summary_path)
        if "time_sec_mean" not in df.columns:
            return
        plt.figure(figsize=(8, 5))
        plt.bar(df["agent"], df["time_sec_mean"])
        plt.ylabel("Mean runtime (s)")
        plt.title(f"{self.cfg.mode.title()} Runtime")
        plt.tight_layout()
        path = os.path.join(self.cfg.out_dir, f"{self.cfg.mode}_runtime.png")
        plt.savefig(path, dpi=200)
        plt.close()
        print("Saved:", path)


def main() -> None:
    for mode in ["stationary", "nonstationary"]:
        PlotBuilder(PlotCfg(mode=mode)).run()


if __name__ == "__main__":
    main()
