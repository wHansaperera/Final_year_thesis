from __future__ import annotations

import os

import pandas as pd


EXPECTED_COLUMNS = [
    "mode",
    "metric",
    "A",
    "B",
    "mean_A",
    "mean_B",
    "mean_diff(A-B)",
    "ci_low",
    "ci_high",
    "shapiro_p",
    "diff_skewness",
    "approximately_normal",
    "primary_test",
    "primary_p",
    "holm_primary_p",
    "robustness_test",
    "robustness_p",
    "primary_effect_size_name",
    "primary_effect_size",
    "cohens_dz",
    "rank_biserial",
    "n_seeds",
]


class EffectSizeReporter:
    def __init__(
        self,
        mode: str,
        sig_dir: str = "results/validation/step2_significance",
        out_dir: str = "results/validation/step3_effect_size",
    ) -> None:
        self.mode = mode
        self.sig_csv = os.path.join(sig_dir, f"{mode}_significance.csv")
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def run(self) -> None:
        df = pd.read_csv(self.sig_csv)
        missing = [column for column in EXPECTED_COLUMNS if column not in df.columns]
        if missing:
            raise SystemExit(
                "Step 3 expects the updated step 2 significance output. "
                f"Missing columns: {missing}. Re-run step2_significance_tests.py first."
            )

        report = df[EXPECTED_COLUMNS].copy()

        out_path = os.path.join(self.out_dir, f"{self.mode}_effect_size.csv")
        report.to_csv(out_path, index=False)
        print("Saved:", out_path)
        print(report.round(6).to_string(index=False))


def main() -> None:
    for mode in ["stationary", "nonstationary"]:
        EffectSizeReporter(mode=mode).run()


if __name__ == "__main__":
    main()
