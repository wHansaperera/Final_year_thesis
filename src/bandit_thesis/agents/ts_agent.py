from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..models.bayesian_fm import BayesianFM


@dataclass
class ThompsonAgent:
    model: BayesianFM

    def select_arm(self, X_cand: np.ndarray, candidate_arms: np.ndarray) -> int:
        # X_cand: (n_candidates, p)
        scores = np.array([self.model.thompson_score(X_cand[i]) for i in range(X_cand.shape[0])])
        return int(candidate_arms[int(np.argmax(scores))])

    def update(self, x: np.ndarray, reward: int) -> None:
        self.model.update(x, reward)
