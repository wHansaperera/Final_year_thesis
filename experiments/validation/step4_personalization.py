from __future__ import annotations
import glob
import json
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, chi2_contingency
from sklearn.metrics import mutual_info_score


@dataclass
class PersonalizationCfg:
    mode: str
    out_dir: str = "results/validation/step4_personalization"


class PersonalizationValidator:
    """
    Personalization validation (publishable + simple):
    1) ANOVA on segment-wise CTR (shows environment has segment structure)
    2) Chi-square independence test: segment vs arm_group (policy-level personalization)
    3) Mutual information: MI(segment, arm_group) (effect size-ish signal)
    """

    def __init__(self, cfg: PersonalizationCfg) -> None:
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)
        self.base = os.path.join("results", "raw", cfg.mode)

    def run(self) -> None:
        agents = [d for d in os.listdir(self.base) if os.path.isdir(os.path.join(self.base, d))]
        out_rows = []
        for agent in agents:
            df = self._load_agent_all_seeds(agent)
            if df.empty:
                continue
            out_rows.append(self._agent_stats(agent, df))

        out = pd.DataFrame(out_rows)
        path = os.path.join(self.cfg.out_dir, f"{self.cfg.mode}_personalization.csv")
        out.to_csv(path, index=False)
        print("Saved:", path)
        print(out.round(6).to_string(index=False))

    def _load_agent_all_seeds(self, agent: str) -> pd.DataFrame:
        paths = glob.glob(os.path.join(self.base, agent, "seed_*.jsonl"))
        rows = []
        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    ctx = r.get("context", {})
                    rows.append(
                        {
                            "segment_id": int(ctx.get("segment_id", -1)),
                            "arm_group": int(r.get("arm_group", -1)),  # ✅ uses env-defined group from logs
                            "reward": int(r.get("reward", 0)),
                        }
                    )
        df = pd.DataFrame(rows)
        # drop invalid
        df = df[(df["segment_id"] >= 0) & (df["arm_group"] >= 0)]
        return df

    def _agent_stats(self, agent: str, df: pd.DataFrame) -> dict:
        # 1) ANOVA: segment-wise CTR differs?
        seg_groups = [g["reward"].values for _, g in df.groupby("segment_id") if len(g) > 10]
        anova_p = float(f_oneway(*seg_groups).pvalue) if len(seg_groups) >= 2 else float("nan")

        # 2) Chi-square: is arm_group independent of segment?
        ct = pd.crosstab(df["segment_id"], df["arm_group"])
        chi2_p = float(chi2_contingency(ct)[1]) if ct.shape[0] > 1 and ct.shape[1] > 1 else float("nan")

        # 3) Mutual information between segment and arm_group
        mi = float(mutual_info_score(df["segment_id"], df["arm_group"]))

        return {
            "mode": self.cfg.mode,
            "agent": agent,
            "anova_p_segment_ctr": anova_p,
            "chi2_p_segment_armgroup": chi2_p,
            "mutual_info(segment,arm_group)": mi,
            "n_rows": int(len(df)),
            "n_groups": int(df["arm_group"].nunique()),
        }


def main() -> None:
    for mode in ["stationary", "nonstationary"]:
        PersonalizationValidator(PersonalizationCfg(mode=mode)).run()


if __name__ == "__main__":
    main()