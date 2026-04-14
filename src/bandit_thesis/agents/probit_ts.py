from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bandit_thesis.models.bayes_probit import BayesianProbitPosterior


@dataclass
class ProbitTSAgent:
    model: BayesianProbitPosterior
    update_every: int = 50

    def __post_init__(self) -> None:
        self._t = 0

    def reset(self) -> None:
        self._t = 0

    def score_candidates(self, X_cand: np.ndarray) -> np.ndarray:
        beta = self.model.sample_beta()
        return np.array(
            [self.model.probit_prob(X_cand[i], beta) for i in range(X_cand.shape[0])],
            dtype=float,
        )

    def select_arm(self, X_cand: np.ndarray, candidate_arms: np.ndarray) -> int:
        scores = self.score_candidates(X_cand)
        return int(candidate_arms[int(np.argmax(scores))])

    def update(self, x: np.ndarray, reward: int) -> None:
        self._t += 1
        self.model.add_observation(x, reward)
        if self._t % self.update_every == 0:
            self.model.update()

    def flush(self) -> None:
        if self._t % self.update_every != 0:
            self.model.update()
