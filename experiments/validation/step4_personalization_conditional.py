from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd


@dataclass
class CondCfg:
    mode: str
    out_dir: str = "results/validation/step4_personalization"


class ConditionalPersonalization:
    """
    Conditional selection-rate diagnostic:
    when a group is available, how often does each segment choose it?
    """

    def __init__(self, cfg: CondCfg) -> None:
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)
        self.base = os.path.join("results", "raw", cfg.mode)

    def run(self) -> None:
        agents = sorted(
            d for d in os.listdir(self.base)
            if os.path.isdir(os.path.join(self.base, d))
        )

        outputs = []
        for agent in agents:
            df = self._load_agent(agent)
            if df.empty:
                continue
            rates = self._compute_rates(df)
            rates.insert(0, "agent", agent)
            rates.insert(0, "mode", self.cfg.mode)
            outputs.append(rates)

        if not outputs:
            print(f"No usable personalization rows for mode={self.cfg.mode}.")
            return

        result = pd.concat(outputs, ignore_index=True)
        long_path = os.path.join(self.cfg.out_dir, f"{self.cfg.mode}_conditional_personalization.csv")
        result.to_csv(long_path, index=False)
        print("Saved:", long_path)

        for agent in result["agent"].unique():
            pivot = result[result["agent"] == agent].pivot_table(
                index="segment_id",
                columns="group_id",
                values="selection_rate",
                aggfunc="mean",
            )
            pivot_path = os.path.join(
                self.cfg.out_dir,
                f"{self.cfg.mode}_{agent}_segment_x_group_rates.csv",
            )
            pivot.to_csv(pivot_path)
            print("Saved:", pivot_path)

        print(result.head(20).to_string(index=False))

    def _load_agent(self, agent: str) -> pd.DataFrame:
        paths = glob.glob(os.path.join(self.base, agent, "seed_*.jsonl"))
        rows = []
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    rows.append(
                        {
                            "segment_id": int(row["segment_id"]),
                            "chosen_group": int(row["chosen_group"]),
                            "candidate_groups": list(row["candidate_groups"]),
                        }
                    )
        return pd.DataFrame(rows, columns=["segment_id", "chosen_group", "candidate_groups"])

    def _compute_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        available: Dict[Tuple[int, int], int] = {}
        chosen: Dict[Tuple[int, int], int] = {}

        for _, row in df.iterrows():
            segment_id = int(row["segment_id"])
            chosen_group = int(row["chosen_group"])
            candidate_groups = [int(group) for group in row["candidate_groups"]]

            for group in candidate_groups:
                key = (segment_id, group)
                available[key] = available.get(key, 0) + 1

            if chosen_group in candidate_groups:
                chosen_key = (segment_id, chosen_group)
                chosen[chosen_key] = chosen.get(chosen_key, 0) + 1

        output_rows = []
        for segment_id, group_id in sorted(available):
            available_count = available[(segment_id, group_id)]
            chosen_count = chosen.get((segment_id, group_id), 0)
            output_rows.append(
                {
                    "segment_id": segment_id,
                    "group_id": group_id,
                    "available_count": available_count,
                    "chosen_count": chosen_count,
                    "selection_rate": chosen_count / available_count if available_count else float("nan"),
                }
            )
        return pd.DataFrame(output_rows)


def main() -> None:
    for mode in ["stationary", "nonstationary"]:
        ConditionalPersonalization(CondCfg(mode=mode)).run()


if __name__ == "__main__":
    main()
