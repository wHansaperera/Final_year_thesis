from __future__ import annotations

import glob
import json
import os
import tempfile
from dataclasses import dataclass
from typing import List

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "bandit_thesis_mpl"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import probplot, ttest_rel, wilcoxon

from bandit_thesis.metrics.cold_start import (
    cold_start_ctr_per_user,
    cold_start_regret_per_user,
    ctr_after_shift,
    ctr_last_w,
)
from bandit_thesis.validation.stat_tests import (
    check_paired_normality,
    holm_adjust,
    paired_bootstrap_ci,
    paired_cohens_dz,
    paired_rank_biserial,
)


@dataclass(frozen=True)
class ComparisonSpec:
    metric: str
    agent_a: str
    agent_b: str
    primary: bool = True


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
        self.diagnostics_dir = os.path.join(cfg.out_dir, "diagnostics", cfg.mode)
        os.makedirs(self.diagnostics_dir, exist_ok=True)
        self.loader = RawLoader(cfg.mode)
        self.metric_computer = MetricComputer(self.loader.config)

    def _comparisons(self) -> List[ComparisonSpec]:
        return [
            ComparisonSpec(metric="ctr", agent_a="hybrid", agent_b="probit_ts"),
            ComparisonSpec(metric="final_regret", agent_a="hybrid", agent_b="probit_ts"),
            ComparisonSpec(metric="ctr", agent_a="hybrid", agent_b="bayes_fm_ts"),
            ComparisonSpec(metric="final_regret", agent_a="hybrid", agent_b="bayes_fm_ts"),
        ]

    def run(self) -> None:
        rows_out = [self._compare(spec) for spec in self._comparisons()]
        df = pd.DataFrame(rows_out)

        primary_mask = df["primary"].astype(bool).to_numpy()
        adjusted = np.full(len(df), np.nan, dtype=float)
        valid_primary = primary_mask & np.isfinite(df["primary_p"].to_numpy())
        if np.any(valid_primary):
            adjusted_primary = holm_adjust(df.loc[valid_primary, "primary_p"].to_numpy())
            adjusted[valid_primary] = adjusted_primary
        df["holm_primary_p"] = adjusted

        out_path = os.path.join(self.cfg.out_dir, f"{self.cfg.mode}_significance.csv")
        df.to_csv(out_path, index=False)
        print("Saved:", out_path)
        print(df.round(6).to_string(index=False))

    def _compare(self, spec: ComparisonSpec) -> dict:
        seeds_a = set(self.loader.list_seeds(spec.agent_a))
        seeds_b = set(self.loader.list_seeds(spec.agent_b))
        seeds = sorted(seeds_a.intersection(seeds_b))

        xa = []
        xb = []
        for seed in seeds:
            rows_a = self.loader.load_rows(spec.agent_a, seed)
            rows_b = self.loader.load_rows(spec.agent_b, seed)
            xa.append(self.metric_computer.compute(spec.metric, rows_a))
            xb.append(self.metric_computer.compute(spec.metric, rows_b))

        xa_arr = np.asarray(xa, dtype=float)
        xb_arr = np.asarray(xb, dtype=float)
        diff = xa_arr - xb_arr
        finite_diff = diff[np.isfinite(diff)]

        normality = check_paired_normality(finite_diff, alpha=self.cfg.alpha)
        ttest_p, wilcoxon_p = self._run_tests(xa_arr, xb_arr, finite_diff)
        ci = paired_bootstrap_ci(
            xa_arr,
            xb_arr,
            n_boot=self.cfg.n_boot,
            alpha=self.cfg.alpha,
            seed=0,
        )

        cohens_dz = paired_cohens_dz(xa_arr, xb_arr)
        rank_biserial = paired_rank_biserial(finite_diff)
        if normality.approximately_normal:
            primary_test = "paired_ttest"
            primary_p = ttest_p
            robustness_test = "wilcoxon"
            robustness_p = wilcoxon_p
            primary_effect_name = "cohens_dz"
            primary_effect_size = cohens_dz
        else:
            primary_test = "wilcoxon"
            primary_p = wilcoxon_p
            robustness_test = "paired_ttest"
            robustness_p = ttest_p
            primary_effect_name = "rank_biserial"
            primary_effect_size = rank_biserial

        self._save_normality_plot(spec, finite_diff, normality)

        return {
            "mode": self.cfg.mode,
            "metric": spec.metric,
            "A": spec.agent_a,
            "B": spec.agent_b,
            "primary": spec.primary,
            "mean_A": float(np.nanmean(xa_arr)),
            "mean_B": float(np.nanmean(xb_arr)),
            "mean_diff(A-B)": float(np.nanmean(diff)),
            "ci_low": float(ci.ci_low),
            "ci_high": float(ci.ci_high),
            "shapiro_stat": normality.shapiro_stat,
            "shapiro_p": normality.shapiro_p,
            "diff_skewness": normality.skewness,
            "approximately_normal": normality.approximately_normal,
            "ttest_p": ttest_p,
            "wilcoxon_p": wilcoxon_p,
            "primary_test": primary_test,
            "primary_p": primary_p,
            "robustness_test": robustness_test,
            "robustness_p": robustness_p,
            "cohens_dz": cohens_dz,
            "rank_biserial": rank_biserial,
            "primary_effect_size_name": primary_effect_name,
            "primary_effect_size": primary_effect_size,
            "n_seeds": int(np.sum(np.isfinite(diff))),
        }

    def _run_tests(
        self,
        xa_arr: np.ndarray,
        xb_arr: np.ndarray,
        finite_diff: np.ndarray,
    ) -> tuple[float, float]:
        if finite_diff.size < 2:
            return float("nan"), float("nan")

        ttest_p = float(ttest_rel(xa_arr, xb_arr, nan_policy="omit").pvalue)
        if finite_diff.size == 0 or np.allclose(finite_diff, 0.0):
            wilcoxon_p = float("nan")
        else:
            try:
                wilcoxon_p = float(
                    wilcoxon(finite_diff, zero_method="wilcox", alternative="two-sided").pvalue
                )
            except ValueError:
                wilcoxon_p = float("nan")
        return ttest_p, wilcoxon_p

    def _save_normality_plot(
        self,
        spec: ComparisonSpec,
        diff: np.ndarray,
        normality,
    ) -> None:
        if diff.size == 0:
            return

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        bins = min(10, max(5, int(np.sqrt(diff.size))))
        axes[0].hist(diff, bins=bins, edgecolor="black", alpha=0.75)
        axes[0].axvline(np.mean(diff), color="tab:red", linestyle="--", linewidth=1.2)
        axes[0].set_title("Paired-Difference Histogram")
        axes[0].set_xlabel("A - B difference")
        axes[0].set_ylabel("Frequency")

        probplot(diff, dist="norm", plot=axes[1])
        axes[1].set_title("Q-Q Plot")

        fig.suptitle(
            f"{self.cfg.mode}: {spec.metric} ({spec.agent_a} - {spec.agent_b})\n"
            f"Shapiro p = {normality.shapiro_p:.4f} | skew = {normality.skewness:.4f}",
            fontsize=11,
        )
        fig.tight_layout()
        path = os.path.join(
            self.diagnostics_dir,
            f"{self.cfg.mode}_{spec.metric}_{spec.agent_a}_vs_{spec.agent_b}_normality.png",
        )
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print("Saved:", path)


def main() -> None:
    for mode in ["stationary", "nonstationary"]:
        SignificanceRunner(SigConfig(mode=mode)).run()


if __name__ == "__main__":
    main()
