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


# def sample_trunc_normal(rng: np.random.Generator, mu: float, lower: float, upper: float) -> float:
#     """
#     Sample from N(mu, 1) truncated to (lower, upper).
#     Uses scipy.stats.truncnorm for numerical stability.
#     """
#     a = (lower - mu) if np.isfinite(lower) else -np.inf
#     b = (upper - mu) if np.isfinite(upper) else np.inf

#     # If bounds collapse or are invalid, just return a safe value near mu
#     if not np.isfinite(mu) or a >= b:
#         return float(np.clip(mu if np.isfinite(mu) else 0.0, lower if np.isfinite(lower) else -1e6, upper if np.isfinite(upper) else 1e6))

#     return float(truncnorm.rvs(a, b, loc=mu, scale=1.0, random_state=rng))

# def sample_trunc_normal(rng: np.random.Generator, mu: float, lower: float, upper: float) -> float:
#     """
#     Sample from N(mu,1) truncated to (lower, upper) using scipy truncnorm (stable).
#     """
#     # Convert to standard truncnorm parameters (a,b) in standard normal space
#     a = (lower - mu) if np.isfinite(lower) else -np.inf
#     b = (upper - mu) if np.isfinite(upper) else np.inf

#     # Safety: if mu is invalid or bounds invalid, return a clipped fallback
#     if not np.isfinite(mu) or a >= b:
#         base = 0.0 if not np.isfinite(mu) else mu
#         return float(np.clip(base, -10.0, 10.0))

#     return float(truncnorm.rvs(a, b, loc=mu, scale=1.0, random_state=rng))

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

    # def update(self) -> None:
    #     """
    #     Gibbs steps:
    #       z | beta, y  (truncated normal)
    #       beta | z     (Gaussian closed form)
    #     """
    #     if len(self._X) == 0:
    #         return

    #     X = np.vstack(list(self._X))               # (n,p)
    #     y = np.array(list(self._y), dtype=int)     # (n,)
    #     n, p = X.shape

    #     beta = self.mu.copy()

    #     for _ in range(self.cfg.gibbs_steps):
    #         # Step A: sample latent z
    #         z = np.zeros(n, dtype=float)
    #         mean = np.clip(X @ beta, -15.0, 15.0)  # avoid extreme truncnorm params
    #         for i in range(n):
    #             if y[i] == 1:
    #                 z[i] = sample_trunc_normal(self.rng, mu=float(mean[i]), lower=0.0, upper=np.inf)
    #             else:
    #                 z[i] = sample_trunc_normal(self.rng, mu=float(mean[i]), lower=-np.inf, upper=0.0)

    #         # Step B: Gaussian posterior for beta given z
    #         # Sigma^{-1} = Sigma0^{-1} + X^T X
    #         # mu = Sigma (Sigma0^{-1} mu0 + X^T z)
    #         Sigma_inv = self.Sigma0_inv + X.T @ X
    #         Sigma_inv = Sigma_inv + np.eye(p) * self.cfg.ridge

    #         Sigma = np.linalg.inv(Sigma_inv)
    #         mu = Sigma @ (self.Sigma0_inv @ self.mu0 + X.T @ z)

            

    #         self.Sigma = Sigma
    #         self.mu = mu
    #         beta = mu

    # def update(self) -> None:
    #     """
    #     Gibbs steps:
    #     z | beta, y  (truncated normal)
    #     beta | z     (Gaussian closed form)
    #     """
    #     if len(self._X) == 0:
    #         return

    #     X = np.vstack(list(self._X))               # (n,p)
    #     y = np.array(list(self._y), dtype=int)     # (n,)
    #     n, p = X.shape

    #     beta = self.mu.copy()

    #     for _ in range(self.cfg.gibbs_steps):
    #         # Step A: sample latent z
    #         z = np.zeros(n, dtype=float)
    #         mean = np.clip(X @ beta, -15.0, 15.0)  # avoid extreme truncnorm params

    #         for i in range(n):
    #             if y[i] == 1:
    #                 z[i] = sample_trunc_normal(self.rng, mu=float(mean[i]), lower=0.0, upper=np.inf)
    #             else:
    #                 z[i] = sample_trunc_normal(self.rng, mu=float(mean[i]), lower=-np.inf, upper=0.0)

    #          # guard z
    #         if not np.all(np.isfinite(z)):
    #             self.mu = self.mu0.copy()
    #             self.Sigma = np.eye(self.cfg.p) * self.cfg.prior_var
    #             return

    #         # Step B: Gaussian posterior for beta given z
    #         Sigma_inv = self.Sigma0_inv + X.T @ X
    #         Sigma_inv = Sigma_inv + np.eye(p) * self.cfg.ridge

    #         # guard Sigma_inv
    #         if not np.all(np.isfinite(Sigma_inv)):
    #             self.mu = self.mu0.copy()
    #             self.Sigma = np.eye(self.cfg.p) * self.cfg.prior_var
    #             return

    #         rhs = (self.Sigma0_inv @ self.mu0) + (X.T @ z)

    #         # solve for mu (stable)
    #         mu = np.linalg.solve(Sigma_inv, rhs)

    #         # compute Sigma for sampling (ok after solve)
    #         Sigma = np.linalg.inv(Sigma_inv)

    #         # final guard
    #         if (not np.all(np.isfinite(mu))) or (not np.all(np.isfinite(Sigma))):
    #             self.mu = self.mu0.copy()
    #             self.Sigma = np.eye(self.cfg.p) * self.cfg.prior_var
    #             return

    #         self.mu = mu
    #         self.Sigma = Sigma
    #         beta = mu

    def update(self) -> None:
        """
        Gibbs steps:
        z | beta, y  (truncated normal)
            beta | z     (Gaussian closed form)

        Optimized Plan A:
        - vectorized z sampling
        - precompute X^T X and Sigma_inv once per update() call
        """
        if len(self._X) == 0:
            return

        X = np.vstack(list(self._X))               # (n,p)
        y = np.array(list(self._y), dtype=int)     # (n,)
        n, p = X.shape

        beta = self.mu.copy()

         # ---- Precompute once (huge speed win) ----
        XtX = X.T @ X
        Sigma_inv = self.Sigma0_inv + XtX + np.eye(p) * self.cfg.ridge

        if not np.all(np.isfinite(Sigma_inv)):
            self.mu = self.mu0.copy()
            self.Sigma = np.eye(self.cfg.p) * self.cfg.prior_var
            return

         # ---- Gibbs steps (usually 1 for speed) ----
        for _ in range(self.cfg.gibbs_steps):
            mean = np.clip(X @ beta, -15.0, 15.0)

            z = np.empty(n, dtype=float)
            mask1 = (y == 1)
            mask0 = ~mask1

            # y=1: z ~ N(mean,1) truncated to (0, inf)
            if np.any(mask1):
                mu1 = mean[mask1]
                a1 = (0.0 - mu1)                   # lower - mu
                b1 = np.full_like(mu1, np.inf)     # upper - mu
                z[mask1] = truncnorm.rvs(a1, b1, loc=mu1, scale=1.0, random_state=self.rng)

            # y=0: z ~ N(mean,1) truncated to (-inf, 0)
            if np.any(mask0):
                mu0 = mean[mask0]
                a0 = np.full_like(mu0, -np.inf)
                b0 = (0.0 - mu0)
                z[mask0] = truncnorm.rvs(a0, b0, loc=mu0, scale=1.0, random_state=self.rng)

            # guard z
            if not np.all(np.isfinite(z)):
                self.mu = self.mu0.copy()
                self.Sigma = np.eye(self.cfg.p) * self.cfg.prior_var
                return

            rhs = (self.Sigma0_inv @ self.mu0) + (X.T @ z)

            # solve for mu (stable)
            mu = np.linalg.solve(Sigma_inv, rhs)

            # compute Sigma for sampling
            Sigma = np.linalg.inv(Sigma_inv)

            if (not np.all(np.isfinite(mu))) or (not np.all(np.isfinite(Sigma))):
                self.mu = self.mu0.copy()
                self.Sigma = np.eye(self.cfg.p) * self.cfg.prior_var
                return

            self.mu = mu
            self.Sigma = Sigma
            beta = mu