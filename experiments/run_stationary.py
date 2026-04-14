from __future__ import annotations

import yaml

from bandit_thesis.experiment import run_experiment as _run_experiment
from bandit_thesis.experiment import run_one_seed as _run_one_seed


def run_one_seed(seed: int, cfg: dict, output_root: str = "results", save_raw: bool = True):
    return _run_one_seed(
        seed=seed,
        cfg=cfg,
        mode="stationary",
        output_root=output_root,
        save_raw=save_raw,
    )


def main() -> None:
    with open("configs/stationary.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _run_experiment(config=cfg, mode="stationary", output_root="results")


if __name__ == "__main__":
    main()
