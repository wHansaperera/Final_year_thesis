from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..models.bayesian_fm import BayesianFM


@dataclass
class ThompsonAgent:
    model: BayesianFM

    def score_candidates(self, X_cand: np.ndarray) -> np.ndarray:
        return self.model.sampled_probabilities(X_cand)

    def select_arm(self, X_cand: np.ndarray, candidate_arms: np.ndarray) -> int:
        scores = self.score_candidates(X_cand)
        return int(candidate_arms[int(np.argmax(scores))])

    def update(self, x: np.ndarray, reward: int) -> None:
        self.model.update(x, reward)
