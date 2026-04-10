from __future__ import annotations
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import yaml


@dataclass
class CheckResult:
    ok: bool
    message: str


class ExperimentDesignVerifier:
    def __init__(self, repo_root: str = ".") -> None:
        self.repo_root = repo_root

    def _load_yaml(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _check_exists(self, path: str) -> CheckResult:
        return CheckResult(os.path.exists(path), f"{'OK' if os.path.exists(path) else 'MISSING'}: {path}")

    def verify(self) -> List[CheckResult]:
        results: List[CheckResult] = []

        # configs
        stat_cfg_path = os.path.join(self.repo_root, "configs", "stationary.yaml")
        non_cfg_path = os.path.join(self.repo_root, "configs", "nonstationary.yaml")
        results.append(self._check_exists(stat_cfg_path))
        results.append(self._check_exists(non_cfg_path))

        if os.path.exists(stat_cfg_path):
            cfg = self._load_yaml(stat_cfg_path)
            results.extend(self._verify_stationary_cfg(cfg))

        if os.path.exists(non_cfg_path):
            cfg = self._load_yaml(non_cfg_path)
            results.extend(self._verify_nonstationary_cfg(cfg))

        # tables
        results.append(self._check_exists(os.path.join(self.repo_root, "results", "tables", "stationary_summary.csv")))
        results.append(self._check_exists(os.path.join(self.repo_root, "results", "tables", "nonstationary_summary.csv")))

        # raw logs presence
        results.extend(self._verify_raw_logs("stationary"))
        results.extend(self._verify_raw_logs("nonstationary"))

        return results

    def _verify_stationary_cfg(self, cfg: dict) -> List[CheckResult]:
        req = [("experiment", ["T", "seeds"]), ("env", ["n_arms", "n_candidates"]), ("models", ["fm", "probit", "hybrid"])]
        out: List[CheckResult] = []
        for section, keys in req:
            if section not in cfg:
                out.append(CheckResult(False, f"MISSING section: {section}"))
                continue
            for k in keys:
                if k not in cfg[section]:
                    out.append(CheckResult(False, f"MISSING key: {section}.{k}"))
        # metrics optional but recommended
        if "metrics" not in cfg:
            out.append(CheckResult(False, "RECOMMENDED: add metrics.cold_m and metrics.early_n in stationary.yaml"))
        return out

    def _verify_nonstationary_cfg(self, cfg: dict) -> List[CheckResult]:
        out = []
        if "experiment" not in cfg:
            return [CheckResult(False, "MISSING section: experiment")]
        T = int(cfg["experiment"].get("T", 0))
        shift = int(cfg["experiment"].get("shift_time", -1))
        if shift >= T:
            out.append(CheckResult(False, f"BAD: shift_time ({shift}) must be < T ({T}) for ctr_after_shift to work"))
        else:
            out.append(CheckResult(True, f"OK: shift_time ({shift}) < T ({T})"))
        if "metrics" not in cfg or "adapt_w" not in cfg.get("metrics", {}):
            out.append(CheckResult(False, "RECOMMENDED: add metrics.adapt_w in nonstationary.yaml"))
        return out

    def _verify_raw_logs(self, mode: str) -> List[CheckResult]:
        base = os.path.join(self.repo_root, "results", "raw", mode)
        if not os.path.isdir(base):
            return [CheckResult(False, f"MISSING raw folder: {base}")]
        agents = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        out: List[CheckResult] = []
        if not agents:
            out.append(CheckResult(False, f"No agents found in {base}"))
            return out

        counts = {}
        for a in agents:
            files = glob.glob(os.path.join(base, a, "seed_*.jsonl"))
            counts[a] = len(files)
        # paired fairness check: same number of seed files
        unique = set(counts.values())
        if len(unique) != 1:
            out.append(CheckResult(False, f"NOT PAIRED: seed counts differ across agents: {counts}"))
        else:
            out.append(CheckResult(True, f"OK paired seeds: {counts}"))
        return out


def main() -> None:
    v = ExperimentDesignVerifier(".")
    res = v.verify()
    for r in res:
        print(r.message)

    failed = [r for r in res if not r.ok]
    if failed:
        raise SystemExit("\nDesign verification FAILED. Fix items above.")
    print("\nDesign verification PASSED.")


if __name__ == "__main__":
    main()