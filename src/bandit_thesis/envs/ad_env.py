from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

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
    shift_applied: bool


@dataclass
class AdEnvConfig:
    n_arms: int = 20
    n_candidates: int = 10
    cold_start_user_prob: float = 0.02
    max_users: int = 160
    n_user_segments: int = 6
    bias: float = -2.0


class AdPersonalizationEnv:
    """
    Synthetic ad-personalization environment with persistent users.

    Observed context:
    - user_id
    - stable user segment
    - transient device / hour / weekend features

    Hidden preference structure:
    - user-specific affinity over arm groups
    - segment-level affinity over arm groups

    This keeps the simulator a contextual bandit while making cold start
    and repeated-user personalization meaningful.
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

        self._init_true_params()
        self.reset()

    def _init_true_params(self) -> None:
        cfg = self.cfg

        self.user_segment = self.rng.integers(0, cfg.n_user_segments, size=cfg.max_users)

        self.base_arm_bias = self.rng.normal(0.0, 0.35, size=cfg.n_arms)
        self.base_user_bias = self.rng.normal(0.0, 0.25, size=cfg.max_users)
        self.base_segment_group = self.rng.normal(0.0, 0.65, size=(cfg.n_user_segments, self.G))
        self.base_user_group = self.rng.normal(0.0, 1.0, size=(cfg.max_users, self.G))
        self.base_device_w = float(self.rng.normal(0.0, 0.20))
        self.base_weekend_w = float(self.rng.normal(0.0, 0.15))
        self.base_hour_w = self.rng.normal(0.0, 0.10, size=6)

    def reset(self) -> None:
        self.t = 0
        self.next_user_id = 0
        self.user_impressions = np.zeros(self.cfg.max_users, dtype=int)

        self.true_arm_bias = self.base_arm_bias.copy()
        self.true_user_bias = self.base_user_bias.copy()
        self.true_segment_group = self.base_segment_group.copy()
        self.true_user_group = self.base_user_group.copy()
        self.true_device_w = float(self.base_device_w)
        self.true_weekend_w = float(self.base_weekend_w)
        self.true_hour_w = self.base_hour_w.copy()

        if self.nonstationarity is not None:
            self.nonstationarity.reset()

    def sample_context(self, t: int) -> Dict[str, Any]:
        del t

        if self.next_user_id == 0:
            user_id = 0
            self.next_user_id = 1
        else:
            introduce_new = (
                self.next_user_id < self.cfg.max_users
                and self.rng.random() < self.cfg.cold_start_user_prob
            )
            if introduce_new:
                user_id = self.next_user_id
                self.next_user_id += 1
            else:
                user_id = int(self.rng.integers(0, self.next_user_id))

        history_len = int(self.user_impressions[user_id])
        segment_id = int(self.user_segment[user_id])

        return {
            "user_id": user_id,
            "segment_id": segment_id,
            "user_history_len": history_len,
            "is_new_user": int(history_len == 0),
            "device": int(self.rng.integers(0, 2)),
            "hour_bucket": int(self.rng.integers(0, 6)),
            "is_weekend": int(self.rng.integers(0, 2)),
        }

    def candidate_set(self, context: Dict[str, Any]) -> np.ndarray:
        del context
        n = min(self.cfg.n_candidates, self.cfg.n_arms)
        return self.rng.choice(self.cfg.n_arms, size=n, replace=False).astype(int)

    def expected_reward(self, context: Dict[str, Any], arm: int) -> float:
        cfg = self.cfg
        user_id = int(context["user_id"])
        segment_id = int(context["segment_id"])
        group_id = int(self.arm_group[int(arm)])

        interaction_user = 1.40 * float(self.true_user_group[user_id, group_id])
        interaction_segment = 0.80 * float(self.true_segment_group[segment_id, group_id])
        user_base = 0.30 * float(self.true_user_bias[user_id])
        arm_base = 0.20 * float(self.true_arm_bias[int(arm)])
        nuisance = (
            0.10 * self.true_device_w * float(context["device"])
            + 0.10 * self.true_weekend_w * float(context["is_weekend"])
            + 0.10 * float(self.true_hour_w[int(context["hour_bucket"])])
        )

        logit = cfg.bias + interaction_user + interaction_segment + user_base + arm_base + nuisance
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
        shift_applied = False

        if self.nonstationarity is not None:
            shift_applied = self.nonstationarity.apply(self, t)

        chosen_arm = int(chosen_arm)
        if chosen_arm not in set(map(int, candidate_arms)):
            raise ValueError(f"chosen_arm {chosen_arm} not in candidate set")

        opt_arm, p_opt = self.oracle(context, candidate_arms)
        p_chosen = float(self.expected_reward(context, chosen_arm))
        reward = self.draw_reward(p_chosen)

        user_id = int(context["user_id"])
        self.user_impressions[user_id] += 1
        self.t += 1

        return BanditStep(
            t=t,
            context=context,
            candidate_arms=candidate_arms,
            chosen_arm=chosen_arm,
            reward=reward,
            p_chosen=p_chosen,
            p_opt=p_opt,
            opt_arm=opt_arm,
            shift_applied=shift_applied,
        )

    def shift_preferences(self, strength: float = 1.0) -> None:
        cfg = self.cfg
        self.true_segment_group = (
            0.35 * self.true_segment_group
            + self.rng.normal(0.0, 0.75 * strength, size=(cfg.n_user_segments, self.G))
        )
        self.true_user_group = (
            0.35 * self.true_user_group
            + self.rng.normal(0.0, strength, size=(cfg.max_users, self.G))
        )
        self.true_arm_bias = (
            0.50 * self.true_arm_bias
            + self.rng.normal(0.0, 0.20 * strength, size=cfg.n_arms)
        )
