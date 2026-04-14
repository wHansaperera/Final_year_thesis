from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from bandit_thesis.agents.probit_ts import ProbitTSAgent
from bandit_thesis.agents.ts_agent import ThompsonAgent


@dataclass
class HybridColdStartAgent:
    probit: ProbitTSAgent
    fm: ThompsonAgent
    warmup_impressions: int = 10
    blend_span: int = 40
    max_fm_weight: float = 0.45
    fm_override_margin: float = 0.03
    max_probit_drop: float = 0.06
    recovery_steps: int = 500

    def __post_init__(self) -> None:
        self.recovery_until_t = -1
        self.last_mode = "probit_cold_start"

    def reset(self) -> None:
        self.recovery_until_t = -1
        self.last_mode = "probit_cold_start"
        if hasattr(self.probit, "reset"):
            self.probit.reset()

    def on_shift(self, next_t: int) -> None:
        self.recovery_until_t = int(next_t) + int(self.recovery_steps)

    def _fm_weight(self, user_history_len: int) -> float:
        if user_history_len < self.warmup_impressions:
            return 0.0
        progress = (user_history_len - self.warmup_impressions) / max(self.blend_span, 1)
        progress = float(np.clip(progress, 0.0, 1.0))
        return float(self.max_fm_weight * progress)

    def _allow_fm_override(
        self,
        probit_scores: np.ndarray,
        fm_scores: np.ndarray,
        probit_idx: int,
        fm_idx: int,
        fm_weight: float,
    ) -> bool:
        if fm_idx == probit_idx:
            return True
        if fm_weight < 0.5 * self.max_fm_weight:
            return False

        fm_margin = float(fm_scores[fm_idx] - fm_scores[probit_idx])
        probit_drop = float(probit_scores[probit_idx] - probit_scores[fm_idx])
        return fm_margin >= self.fm_override_margin and probit_drop <= self.max_probit_drop

    def select_arm(self, X_cand: np.ndarray, candidate_arms: np.ndarray, ctx: Dict[str, Any]) -> int:
        t = int(ctx.get("t", -1))
        user_history_len = int(ctx.get("user_history_len", 0))

        probit_scores = self.probit.score_candidates(X_cand)

        if t >= 0 and t < self.recovery_until_t:
            self.last_mode = "probit_recovery"
            return int(candidate_arms[int(np.argmax(probit_scores))])

        fm_weight = self._fm_weight(user_history_len)
        if fm_weight <= 0.0:
            self.last_mode = "probit_cold_start"
            return int(candidate_arms[int(np.argmax(probit_scores))])

        fm_scores = self.fm.score_candidates(X_cand)
        probit_idx = int(np.argmax(probit_scores))
        fm_idx = int(np.argmax(fm_scores))
        scores = (1.0 - fm_weight) * probit_scores + fm_weight * fm_scores
        blended_idx = int(np.argmax(scores))

        if not self._allow_fm_override(probit_scores, fm_scores, probit_idx, fm_idx, fm_weight):
            self.last_mode = "probit_guardrail"
            return int(candidate_arms[probit_idx])

        if fm_idx == probit_idx:
            if fm_weight >= self.max_fm_weight:
                self.last_mode = "consensus_personalized"
            else:
                self.last_mode = "consensus_transition"
            return int(candidate_arms[blended_idx])

        if fm_weight >= self.max_fm_weight:
            self.last_mode = "blended_personalized"
        else:
            self.last_mode = "blended_transition"
        return int(candidate_arms[blended_idx])

    def update(self, x_chosen: np.ndarray, reward: int, ctx: Dict[str, Any]) -> None:
        del ctx
        self.probit.update(x_chosen, reward)
        self.fm.update(x_chosen, reward)

    def flush(self) -> None:
        if hasattr(self.probit, "flush"):
            self.probit.flush()
