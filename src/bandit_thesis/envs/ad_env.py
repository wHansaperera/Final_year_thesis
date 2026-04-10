from __future__ import annotations
from dataclasses import dataclass
# from multiprocessing import context
from typing import Any, Dict, Optional

# from matplotlib.style import context
import numpy as np

from .shifts import AbruptShift


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


@dataclass(frozen=True)
class BanditStep:
    t: int
    context: Dict[str, Any]
    candidate_arms: np.ndarray
    chosen_arm: int
    reward: int
    p_chosen: float
    p_opt: float
    opt_arm: int


@dataclass
class AdEnvConfig:
    n_arms: int = 20
    n_candidates: int = 10
    cold_start_user_prob: float = 0.02
    n_user_segments: int = 6
    bias: float = -2.0


class AdPersonalizationEnv:
    """
    Environment for contextual bandit simulation.
    - sample_context(): creates a user context
    - candidate_set(): shows a subset of arms
    - expected_reward(): true P(click|x,a)
    - step(a): returns BanditStep + oracle info for regret
    """

    def __init__(
        self,
        cfg: AdEnvConfig,
        rng: np.random.Generator,
        nonstationarity: Optional[AbruptShift] = None,
    ) -> None:
        self.cfg = cfg
        self.rng = rng
        self.nonstationarity = nonstationarity
        self.t = 0

        self.G = max(3, int(np.sqrt(cfg.n_arms)))
        self.arm_group = self.rng.integers(0, self.G, size=cfg.n_arms)
        self.next_user_id = 0

        self._init_true_params()

    def _init_true_params(self) -> None:
        cfg = self.cfg
        self.true_w_segment = self.rng.normal(0.0, 1.0, size=cfg.n_user_segments)
        self.true_w_arm = self.rng.normal(0.0, 1.0, size=cfg.n_arms)
        self.true_W_inter = self.rng.normal(0.0, 1.2, size=(cfg.n_user_segments, self.G))
        self.true_w_device = float(self.rng.normal(0.0, 0.3))
        self.true_w_weekend = float(self.rng.normal(0.0, 0.3))
        self.true_w_hour = self.rng.normal(0.0, 0.2, size=6)

    def reset(self) -> None:
        self.t = 0
        self.next_user_id = 0

    def sample_context(self, t: int) -> Dict[str, Any]:
        cfg = self.cfg

        if self.rng.random() < cfg.cold_start_user_prob:
            user_id = self.next_user_id
            self.next_user_id += 1
        else:
            if self.next_user_id == 0:
                user_id = self.next_user_id
                self.next_user_id += 1
            else:
                user_id = int(self.rng.integers(0, self.next_user_id))

        segment_id = int(self.rng.integers(0, cfg.n_user_segments))
        device = int(self.rng.integers(0, 2))
        hour_bucket = int(self.rng.integers(0, 6))
        is_weekend = int(self.rng.integers(0, 2))

        return {
            "user_id": user_id,
            "segment_id": segment_id,
            "device": device,
            "hour_bucket": hour_bucket,
            "is_weekend": is_weekend,
        }

    def candidate_set(self, context: Dict[str, Any]) -> np.ndarray:
        n = min(self.cfg.n_candidates, self.cfg.n_arms)
        return self.rng.choice(self.cfg.n_arms, size=n, replace=False).astype(int)

    

    def expected_reward(self, context: Dict[str, Any], arm: int) -> float:
        cfg = self.cfg
        s = int(context["segment_id"])
        g = int(self.arm_group[int(arm)])

        # 1) Make interaction dominate (this is what FM learns best)
        interaction = 4.0 * float(self.true_W_inter[s, g])   # BIG

        # 2) Make linear terms smaller (logistic baseline can't fully solve)
        seg_base = 0.1 * float(self.true_w_segment[s])
        arm_base = 0.05 * float(self.true_w_arm[int(arm)])

        # 3) Keep nuisance context small
        nuisance = (
            0.05 * self.true_w_device * float(context["device"])
            + 0.05 * self.true_w_weekend * float(context["is_weekend"])
            + 0.05 * float(self.true_w_hour[int(context["hour_bucket"])])
        )

        logit = cfg.bias + seg_base + arm_base + interaction + nuisance
        return sigmoid(logit)



    def oracle(self, context: Dict[str, Any], cand: np.ndarray) -> tuple[int, float]:
        ps = np.array([self.expected_reward(context, int(a)) for a in cand], dtype=float)
        idx = int(np.argmax(ps))
        return int(cand[idx]), float(ps[idx])

    def draw_reward(self, p: float) -> int:
        p = float(np.clip(p, 0.0, 1.0))
        return int(self.rng.random() < p)

    def step(self, context: Dict[str, Any], candidate_arms: np.ndarray, chosen_arm: int) -> BanditStep:
        t = self.t

        if self.nonstationarity is not None:
            self.nonstationarity.apply(self, t)

        chosen_arm = int(chosen_arm)
        if chosen_arm not in set(map(int, candidate_arms)):
            raise ValueError(f"chosen_arm {chosen_arm} not in candidate set")

        opt_arm, p_opt = self.oracle(context, candidate_arms)
        p_chosen = float(self.expected_reward(context, chosen_arm))
        r = self.draw_reward(p_chosen)

        self.t += 1
        return BanditStep(
            t=t,
            context=context,
            candidate_arms=candidate_arms,
            chosen_arm=chosen_arm,
            reward=r,
            p_chosen=p_chosen,
            p_opt=p_opt,
            opt_arm=opt_arm,
        )

    # shift hooks (called by AbruptShift)
    def shift_preferences(self, strength: float = 1.2) -> None:
        cfg = self.cfg
        self.true_w_segment = self.rng.normal(0.0, strength, size=cfg.n_user_segments)
        self.true_W_inter = self.rng.normal(0.0, 1.2 * strength, size=(cfg.n_user_segments, self.G))
