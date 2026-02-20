from __future__ import annotations
from dataclasses import dataclass
from typing import Deque
from collections import deque

import numpy as np
from scipy.stats import norm
from scipy.stats import truncnorm

@dataclass
class ProbitConfig:
    p: int
    prior_var: float = 5.0
    window: int = 1000
    gibbs_steps: int = 3
    ridge: float = 1e-6


def sample_trunc_normal(rng: np.random.Generator, mu: float, lower: float, upper: float) -> float:
    """
    Sample from N(mu, 1) truncated to (lower, upper).
    Uses scipy.stats.truncnorm for numerical stability.
    """
    a = (lower - mu) if np.isfinite(lower) else -np.inf
    b = (upper - mu) if np.isfinite(upper) else np.inf

    # If bounds collapse or are invalid, just return a safe value near mu
    if not np.isfinite(mu) or a >= b:
        return float(np.clip(mu if np.isfinite(mu) else 0.0, lower if np.isfinite(lower) else -1e6, upper if np.isfinite(upper) else 1e6))

    return float(truncnorm.rvs(a, b, loc=mu, scale=1.0, random_state=rng))

class BayesianProbitPosterior:
    """
    Bayesian probit regression with Albert–Chib augmentation:
      z_i | beta ~ N(x_i^T beta, 1)
      y_i = 1[z_i > 0]

    Prior: beta ~ N(0, prior_var I)

    Sliding window: keep last W (x,y).
    """

    def __init__(self, cfg: ProbitConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng

        self.mu = np.zeros(cfg.p, dtype=float)
        self.Sigma = np.eye(cfg.p, dtype=float) * cfg.prior_var

        self._X: Deque[np.ndarray] = deque(maxlen=cfg.window)
        self._y: Deque[int] = deque(maxlen=cfg.window)

        self.Sigma0_inv = np.eye(cfg.p) / cfg.prior_var
        self.mu0 = np.zeros(cfg.p, dtype=float)

    def add_observation(self, x: np.ndarray, y: int) -> None:
        self._X.append(np.asarray(x, dtype=float))
        self._y.append(int(y))

    def sample_beta(self) -> np.ndarray:
        return self.rng.multivariate_normal(self.mu, self.Sigma)

    def probit_prob(self, x: np.ndarray, beta: np.ndarray) -> float:
        # P(y=1|x) = Phi(x^T beta)
        return float(norm.cdf(float(np.dot(x, beta))))

    def update(self) -> None:
        """
        Gibbs steps:
          z | beta, y  (truncated normal)
          beta | z     (Gaussian closed form)
        """
        if len(self._X) == 0:
            return

        X = np.vstack(list(self._X))               # (n,p)
        y = np.array(list(self._y), dtype=int)     # (n,)
        n, p = X.shape

        beta = self.mu.copy()

        for _ in range(self.cfg.gibbs_steps):
            # Step A: sample latent z
            z = np.zeros(n, dtype=float)
            mean = np.clip(X @ beta, -15.0, 15.0)  # avoid extreme truncnorm params
            for i in range(n):
                if y[i] == 1:
                    z[i] = sample_trunc_normal(self.rng, mu=float(mean[i]), lower=0.0, upper=np.inf)
                else:
                    z[i] = sample_trunc_normal(self.rng, mu=float(mean[i]), lower=-np.inf, upper=0.0)

            # Step B: Gaussian posterior for beta given z
            # Sigma^{-1} = Sigma0^{-1} + X^T X
            # mu = Sigma (Sigma0^{-1} mu0 + X^T z)
            Sigma_inv = self.Sigma0_inv + X.T @ X
            Sigma_inv = Sigma_inv + np.eye(p) * self.cfg.ridge

            Sigma = np.linalg.inv(Sigma_inv)
            mu = Sigma @ (self.Sigma0_inv @ self.mu0 + X.T @ z)

            self.Sigma = Sigma
            self.mu = mu
            beta = mu
