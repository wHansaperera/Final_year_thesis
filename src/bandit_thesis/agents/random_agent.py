from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class RandomAgent:
    rng: np.random.Generator

    def select_arm(self, candidate_arms: np.ndarray, context=None) -> int:
        del context
        return int(self.rng.choice(candidate_arms))

    def update(self, *args, **kwargs) -> None:
        del args, kwargs
        return
