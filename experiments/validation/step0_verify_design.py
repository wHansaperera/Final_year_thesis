from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import List


@dataclass
class CheckResult:
    ok: bool
    message: str


class ExperimentDesignVerifier:
    def __init__(self, repo_root: str = ".") -> None:
        self.repo_root = repo_root

    def _check_exists(self, path: str) -> CheckResult:
        exists = os.path.exists(path)
        return CheckResult(exists, f"{'OK' if exists else 'MISSING'}: {path}")

    def verify(self) -> List[CheckResult]:
        results: List[CheckResult] = []
        for mode in ["stationary", "nonstationary"]:
            results.extend(self._verify_mode(mode))
        return results

    def _verify_mode(self, mode: str) -> List[CheckResult]:
        out: List[CheckResult] = []
        manifest_path = os.path.join(self.repo_root, "results", "manifests", f"{mode}.json")
        summary_path = os.path.join(self.repo_root, "results", "tables", f"{mode}_summary.csv")
        raw_base = os.path.join(self.repo_root, "results", "raw", mode)

        out.append(self._check_exists(manifest_path))
        out.append(self._check_exists(summary_path))

        if not os.path.isdir(raw_base):
            out.append(CheckResult(False, f"MISSING raw folder: {raw_base}"))
            return out

        agents = sorted(
            d for d in os.listdir(raw_base)
            if os.path.isdir(os.path.join(raw_base, d))
        )
        if not agents:
            out.append(CheckResult(False, f"No agent folders found in {raw_base}"))
            return out

        seed_sets = {}
        for agent in agents:
            paths = glob.glob(os.path.join(raw_base, agent, "seed_*.jsonl"))
            seeds = sorted(int(os.path.basename(path).split("_")[1].split(".")[0]) for path in paths)
            seed_sets[agent] = seeds

        expected = None
        for agent, seeds in seed_sets.items():
            if expected is None:
                expected = seeds
            elif seeds != expected:
                out.append(CheckResult(False, f"NOT PAIRED in {mode}: {seed_sets}"))
                break
        else:
            out.append(CheckResult(True, f"OK paired seeds in {mode}: {seed_sets}"))

        sample_paths = glob.glob(os.path.join(raw_base, agents[0], "seed_*.jsonl"))
        if sample_paths:
            with open(sample_paths[0], "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
            if first_line:
                row = json.loads(first_line)
                required_fields = {
                    "schema_version",
                    "seed",
                    "agent",
                    "t",
                    "user_id",
                    "user_history_len",
                    "candidate_arms",
                    "decision_mode",
                    "shift_applied",
                }
                missing = sorted(required_fields.difference(row))
                if missing:
                    out.append(CheckResult(False, f"BAD schema in {mode}: missing {missing}"))
                else:
                    out.append(CheckResult(True, f"OK raw schema in {mode}: required fields present"))

        return out


def main() -> None:
    verifier = ExperimentDesignVerifier(".")
    results = verifier.verify()
    for result in results:
        print(result.message)

    failed = [result for result in results if not result.ok]
    if failed:
        raise SystemExit("\nDesign verification FAILED. Fix items above.")
    print("\nDesign verification PASSED.")


if __name__ == "__main__":
    main()
