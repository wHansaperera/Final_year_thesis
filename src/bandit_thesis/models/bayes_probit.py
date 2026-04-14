from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
from scipy.stats import norm, truncnorm


@dataclass
class ProbitConfig:
    p: int
    prior_var: float = 5.0
    window: int = 1000
    gibbs_steps: int = 3
    ridge: float = 1e-6

    def __post_init__(self) -> None:
        self.prior_var = float(self.prior_var)
        self.window = int(self.window)
        self.gibbs_steps = int(self.gibbs_steps)
        self.ridge = float(self.ridge)


class BayesianProbitPosterior:
    """
    Bayesian probit regression with Albert-Chib augmentation and a sliding window.
    """

    def __init__(self, cfg: ProbitConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng

        self.mu0 = np.zeros(cfg.p, dtype=float)
        self.Sigma0_inv = np.eye(cfg.p, dtype=float) / cfg.prior_var

        self.mu = self.mu0.copy()
        self.Sigma = np.eye(cfg.p, dtype=float) * cfg.prior_var

        self._X: Deque[np.ndarray] = deque(maxlen=cfg.window)
        self._y: Deque[int] = deque(maxlen=cfg.window)

    def add_observation(self, x: np.ndarray, y: int) -> None:
        self._X.append(np.asarray(x, dtype=float))
        self._y.append(int(y))

    def sample_beta(self) -> np.ndarray:
        return self.rng.multivariate_normal(self.mu, self.Sigma)

    def probit_prob(self, x: np.ndarray, beta: np.ndarray) -> float:
        return float(norm.cdf(float(np.dot(x, beta))))

    def update(self) -> None:
        if len(self._X) == 0:
            return

        X = np.vstack(list(self._X))
        y = np.array(list(self._y), dtype=int)
        n, p = X.shape

        XtX = X.T @ X
        Sigma_inv = self.Sigma0_inv + XtX + np.eye(p, dtype=float) * self.cfg.ridge
        if not np.all(np.isfinite(Sigma_inv)):
            return

        beta = self.mu.copy()
        Sigma = np.linalg.inv(Sigma_inv)

        for _ in range(self.cfg.gibbs_steps):
            mean = np.clip(X @ beta, -15.0, 15.0)
            z = np.empty(n, dtype=float)

            mask_pos = y == 1
            mask_neg = ~mask_pos

            if np.any(mask_pos):
                loc = mean[mask_pos]
                a = 0.0 - loc
                b = np.full_like(loc, np.inf, dtype=float)
                z[mask_pos] = truncnorm.rvs(a, b, loc=loc, scale=1.0, random_state=self.rng)

            if np.any(mask_neg):
                loc = mean[mask_neg]
                a = np.full_like(loc, -np.inf, dtype=float)
                b = 0.0 - loc
                z[mask_neg] = truncnorm.rvs(a, b, loc=loc, scale=1.0, random_state=self.rng)

            if not np.all(np.isfinite(z)):
                return

            rhs = X.T @ z
            beta = Sigma @ rhs

        if np.all(np.isfinite(beta)) and np.all(np.isfinite(Sigma)):
            self.mu = beta
            self.Sigma = Sigma
