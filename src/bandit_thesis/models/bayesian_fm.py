from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


@dataclass
class BayesianFMConfig:
    p: int
    k: int = 8
    lr: float = 0.05
    prior_var: float = 1.0
    drift_var: float = 0.0  # add to parameter variance each step (nonstationarity)


class BayesianFM:
    """
    Bayesian Factorization Machine (approx).
    We keep:
      - parameters (w0, w, V)
      - diagonal variances for (w0, w, V) to sample with Thompson Sampling

    Update uses one-step gradient on logloss + light variance update.
    """

    def __init__(self, cfg: BayesianFMConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng

        p, k = cfg.p, cfg.k
        self.w0 = 0.0
        self.w = np.zeros(p, dtype=float)
        self.V = self.rng.normal(0.0, 0.1, size=(p, k))

        # diagonal variances (start from prior)
        self.var_w0 = float(cfg.prior_var)
        self.var_w = np.full(p, cfg.prior_var, dtype=float)
        self.var_V = np.full((p, k), cfg.prior_var, dtype=float)

    def _fm_logit(self, x: np.ndarray, w0: float, w: np.ndarray, V: np.ndarray) -> float:
        # linear
        lin = w0 + float(np.dot(w, x))
        # pairwise interactions: 0.5 * sum_f ((sum_i v_if x_i)^2 - sum_i (v_if^2 x_i^2))
        XV = x[:, None] * V
        sum_f = np.sum(XV, axis=0)
        sum_f_sq = np.sum(XV * XV, axis=0)
        inter = 0.5 * float(np.sum(sum_f * sum_f - sum_f_sq))
        return lin + inter

    def predict_proba(self, x: np.ndarray) -> float:
        return sigmoid(self._fm_logit(x, self.w0, self.w, self.V))

    def sample_params(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """Sample parameters for Thompson Sampling."""
        # apply drift (nonstationary) by inflating variances
        if self.cfg.drift_var > 0:
            self.var_w0 += self.cfg.drift_var
            self.var_w += self.cfg.drift_var
            self.var_V += self.cfg.drift_var

        w0_s = float(self.rng.normal(self.w0, np.sqrt(max(self.var_w0, 1e-9))))
        w_s = self.rng.normal(self.w, np.sqrt(np.maximum(self.var_w, 1e-9)))
        V_s = self.rng.normal(self.V, np.sqrt(np.maximum(self.var_V, 1e-9)))
        return w0_s, w_s, V_s

    def thompson_score(self, x: np.ndarray) -> float:
        w0_s, w_s, V_s = self.sample_params()
        return sigmoid(self._fm_logit(x, w0_s, w_s, V_s))

    def update(self, x: np.ndarray, y: int) -> None:
        """
        One-step update with logistic loss gradient:
          y in {0,1}
        """
        y = int(y)
        p = self.predict_proba(x)
        err = (p - y)  # d/dlogit of logloss

        # gradients
        # w0
        g_w0 = err
        # w
        g_w = err * x

        # V gradient for FM (standard FM gradient)
        # for each factor f: dlogit/dv_if = x_i * (sum_j v_jf x_j - v_if x_i)
        XV = x[:, None] * self.V
        sum_f = np.sum(XV, axis=0)  # (k,)
        g_V = np.zeros_like(self.V)
        for i in range(self.cfg.p):
            if x[i] == 0:
                continue
            g_V[i, :] = err * (x[i] * (sum_f - self.V[i, :] * x[i]))

        # MAP-style gradient step + L2 prior regularization
        lr = self.cfg.lr
        prior_prec = 1.0 / max(self.cfg.prior_var, 1e-9)

        self.w0 -= lr * (g_w0 + prior_prec * self.w0)
        self.w -= lr * (g_w + prior_prec * self.w)
        self.V -= lr * (g_V + prior_prec * self.V)

        # simple variance shrink update: more data -> lower variance
        # (not exact Bayes, but consistent monotonic uncertainty reduction)
        shrink = 0.999
        self.var_w0 *= shrink
        self.var_w *= shrink
        self.var_V *= shrink
