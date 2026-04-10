from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class CondCfg:
    mode: str  # "stationary" or "nonstationary"
    out_dir: str = "results/validation/step4_personalization_conditional"


class ConditionalPersonalization:
    """
    Correct personalization test under candidate-set constraint:

    For each segment s and group g:
        available_count = number of steps where g was in candidate_groups
        chosen_count    = number of steps where chosen_group == g AND g was available
        rate            = chosen_count / available_count

    This directly answers:
        "When group g is available, does segment s pick it more often?"
    """

    def __init__(self, cfg: CondCfg) -> None:
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)
        self.base = os.path.join("results", "raw", cfg.mode)

    def run(self) -> None:
        agents = [d for d in os.listdir(self.base) if os.path.isdir(os.path.join(self.base, d))]
        all_rows = []

        for agent in agents:
            df = self._load_agent(agent)
            if df.empty:
                continue
            out = self._compute_rates(df)
            out.insert(0, "agent", agent)
            out.insert(0, "mode", self.cfg.mode)
            all_rows.append(out)

        if not all_rows:
            print("No data found. Did you rerun experiments after adding candidate_groups?")
            return

        res = pd.concat(all_rows, ignore_index=True)

        # save long-format table
        out_path = os.path.join(self.cfg.out_dir, f"{self.cfg.mode}_conditional_personalization.csv")
        res.to_csv(out_path, index=False)
        print("Saved:", out_path)

        # also save a pivot view (segment x group rates) per agent for quick reading
        for agent in res["agent"].unique():
            sub = res[res["agent"] == agent]
            pivot = sub.pivot_table(index="segment_id", columns="group_id", values="selection_rate", aggfunc="mean")
            pivot_path = os.path.join(self.cfg.out_dir, f"{self.cfg.mode}_{agent}_segment_x_group_rates.csv")
            pivot.to_csv(pivot_path)
            print("Saved:", pivot_path)

        # print a compact view
        print("\nSample (first 20 rows):")
        print(res.head(20).to_string(index=False))

    # def _load_agent(self, agent: str) -> pd.DataFrame:
    #     paths = glob.glob(os.path.join(self.base, agent, "seed_*.jsonl"))
    #     rows = []
    #     for p in paths:
    #         with open(p, "r", encoding="utf-8") as f:
    #             for line in f:
    #                 r = json.loads(line)
    #                 ctx = r.get("context", {})
    #                 # require new logging fields
    #                 if "candidate_groups" not in r or "chosen_group" not in r:
    #                     continue
    #                 rows.append(
    #                     {
    #                         "segment_id": int(ctx.get("segment_id", -1)),
    #                         "chosen_group": int(r.get("chosen_group", -1)),
    #                         "candidate_groups": list(r.get("candidate_groups", [])),
    #                     }
    #                 )
    #     df = pd.DataFrame(rows)
    #     df = df[df["segment_id"] >= 0]
    #     df = df[df["chosen_group"] >= 0]
    #     return df

    def _load_agent(self, agent: str) -> pd.DataFrame:
        paths = glob.glob(os.path.join(self.base, agent, "seed_*.jsonl"))
        rows = []
        skipped_missing = 0
        skipped_badctx = 0

        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    ctx = r.get("context", {})

                    # must exist (new logging)
                    if "candidate_groups" not in r or "chosen_group" not in r:
                        skipped_missing += 1
                        continue

                    if "segment_id" not in ctx:
                        skipped_badctx += 1
                        continue

                    rows.append(
                        {
                            "segment_id": int(ctx["segment_id"]),
                            "chosen_group": int(r["chosen_group"]),
                            "candidate_groups": list(r["candidate_groups"]),
                        }
                    )

        # ✅ always create DF with expected columns
        df = pd.DataFrame(rows, columns=["segment_id", "chosen_group", "candidate_groups"])

        if df.empty:
            print(
                f"[WARN] No usable rows for agent='{agent}'. "
                f"paths={len(paths)}, skipped_missing={skipped_missing}, skipped_badctx={skipped_badctx}\n"
                f"      → Did you rerun experiments after adding chosen_group + candidate_groups logging?"
            )
            return df

        df = df[df["segment_id"] >= 0]
        df = df[df["chosen_group"] >= 0]
        return df

    def _compute_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        # counts[(segment, group)] = [available, chosen]
        available: Dict[Tuple[int, int], int] = {}
        chosen: Dict[Tuple[int, int], int] = {}

        for _, row in df.iterrows():
            s = int(row["segment_id"])
            cg = int(row["chosen_group"])
            cands = row["candidate_groups"]

            # availability count
            for g in cands:
                key = (s, int(g))
                available[key] = available.get(key, 0) + 1

            # chosen count (only meaningful if chosen group was available)
            if cg in cands:
                keyc = (s, cg)
                chosen[keyc] = chosen.get(keyc, 0) + 1

        out_rows = []
        keys = sorted(available.keys())
        for (s, g) in keys:
            av = available.get((s, g), 0)
            ch = chosen.get((s, g), 0)
            rate = (ch / av) if av > 0 else float("nan")
            out_rows.append(
                {
                    "segment_id": s,
                    "group_id": g,
                    "available_count": av,
                    "chosen_count": ch,
                    "selection_rate": rate,
                }
            )

        return pd.DataFrame(out_rows)


def main() -> None:
    for mode in ["stationary", "nonstationary"]:
        ConditionalPersonalization(CondCfg(mode=mode)).run()


if __name__ == "__main__":
    main()