from __future__ import annotations
import os
import pandas as pd
import numpy as np


class EffectSize:
    @staticmethod
    def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
        d = a - b
        sd = np.std(d, ddof=1) if len(d) > 1 else 0.0
        return float(np.mean(d) / sd) if sd > 0 else float("nan")


class EffectSizeRunner:
    def __init__(self, sig_csv: str, out_dir: str = "results/validation/step3_effect_size") -> None:
        self.sig_csv = sig_csv
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def run(self) -> None:
        df = pd.read_csv(self.sig_csv)
        # Cohen d from mean_diff / std_diff is not possible without raw diffs, so use approximation:
        # We'll compute: d ≈ mean_diff / ( (ci_high - ci_low)/(2*1.96) )
        # This is a simple publishable approximation.
        approx_sd = (df["ci_high"] - df["ci_low"]) / (2 * 1.96)
        df["cohens_d_approx"] = df["mean_diff(A-B)"] / approx_sd.replace(0, np.nan)
        out_path = os.path.join(self.out_dir, os.path.basename(self.sig_csv).replace(".csv", "_effect.csv"))
        df.to_csv(out_path, index=False)
        print("Saved:", out_path)
        print(df.round(6).to_string(index=False))


def main() -> None:
    EffectSizeRunner("results/validation/step2_significance/stationary_significance.csv").run()
    EffectSizeRunner("results/validation/step2_significance/nonstationary_significance.csv").run()


if __name__ == "__main__":
    main()