from __future__ import annotations
import os
from dataclasses import dataclass
import pandas as pd


@dataclass
class ReportPaths:
    out_dir: str = "results/validation/step1_metrics"


class MetricsReporter:
    def __init__(self, paths: ReportPaths) -> None:
        self.paths = paths
        os.makedirs(self.paths.out_dir, exist_ok=True)

    def run(self) -> None:
        self._copy_and_clean(
            "results/tables/stationary_summary.csv",
            os.path.join(self.paths.out_dir, "stationary_metrics.csv"),
        )
        self._copy_and_clean(
            "results/tables/nonstationary_summary.csv",
            os.path.join(self.paths.out_dir, "nonstationary_metrics.csv"),
        )
        with open(os.path.join(self.paths.out_dir, "README.txt"), "w", encoding="utf-8") as f:
            f.write(
                "Step 1 Metrics Report\n"
                "- stationary_metrics.csv: CTR, regret, cold-start, early metrics, time\n"
                "- nonstationary_metrics.csv: CTR, dynamic regret, ctr_after_shift, ctr_last_w, time\n"
            )

    def _copy_and_clean(self, src: str, dst: str) -> None:
        df = pd.read_csv(src)
        # keep numeric columns rounded for readability
        num_cols = df.select_dtypes(include="number").columns
        df[num_cols] = df[num_cols].round(6)
        df.to_csv(dst, index=False)
        print("Saved:", dst)


def main() -> None:
    MetricsReporter(ReportPaths()).run()


if __name__ == "__main__":
    main()