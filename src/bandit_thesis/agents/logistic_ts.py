from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


@dataclass
class LogisticTSConfig:
    p: int
    lr: float = 0.1
    prior_var: float = 1.0
    drift_var: float = 0.0


class LogisticTSAgent:
    """
    Simple baseline: Bayesian-ish logistic regression with diagonal uncertainty.
    - sample w ~ N(mu, diag(var)) for TS
    - update mu using gradient
    - shrink var over time, inflate by drift_var for nonstationarity
    """

    def __init__(self, cfg: LogisticTSConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng
        self.mu = np.zeros(cfg.p, dtype=float)
        self.var = np.full(cfg.p, cfg.prior_var, dtype=float)

    def score(self, x: np.ndarray) -> float:
        if self.cfg.drift_var > 0:
            self.var += self.cfg.drift_var
        w = self.rng.normal(self.mu, np.sqrt(np.maximum(self.var, 1e-9)))
        return sigmoid(float(np.dot(w, x)))

    def select_arm(self, X_cand: np.ndarray, candidate_arms: np.ndarray) -> int:
        # X_cand shape: (n_candidates, p)
        scores = np.array([self.score(X_cand[i]) for i in range(X_cand.shape[0])])
        return int(candidate_arms[int(np.argmax(scores))])

    def update(self, x: np.ndarray, reward: int) -> None:
        p = sigmoid(float(np.dot(self.mu, x)))
        err = (p - int(reward))

        prior_prec = 1.0 / max(self.cfg.prior_var, 1e-9)
        self.mu -= self.cfg.lr * (err * x + prior_prec * self.mu)

        self.var *= 0.999
