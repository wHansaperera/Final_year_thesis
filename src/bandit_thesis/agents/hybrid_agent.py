# from __future__ import annotations
# from dataclasses import dataclass, field
# from typing import Dict, Any
# import numpy as np

# from bandit_thesis.agents.probit_ts import ProbitTSAgent
# from bandit_thesis.agents.ts_agent import ThompsonAgent


# @dataclass
# class HybridColdStartAgent:
#     """
#     Hybrid policy:
#       - Use Probit TS for cold-start (first M impressions per user)
#       - Switch to FM Thompson Sampling after M impressions

#     This is a POLICY hybrid, not a new reward model.
#     """
#     probit: ProbitTSAgent
#     fm: ThompsonAgent
#     warmup_impressions: int = 30
#     user_impressions: Dict[int, int] = field(default_factory=dict)

#     def _user_id(self, ctx: Dict[str, Any]) -> int:
#         # adjust if your context uses another key
#         return int(ctx.get("user_id", ctx.get("uid", 0)))

#     def select_arm(self, X_cand: np.ndarray, candidate_arms: np.ndarray, ctx: Dict[str, Any]) -> int:
#         uid = self._user_id(ctx)
#         n = self.user_impressions.get(uid, 0)

#         if n < self.warmup_impressions:
#             # cold-start: Bayesian Probit TS
#             return int(self.probit.select_arm(X_cand, candidate_arms))
#         else:
#             # warm-start: FM TS (interaction personalization)
#             return int(self.fm.select_arm(X_cand, candidate_arms))

#     def update(self, x_chosen: np.ndarray, reward: int, ctx: Dict[str, Any]) -> None:
#         uid = self._user_id(ctx)
#         self.user_impressions[uid] = self.user_impressions.get(uid, 0) + 1

#         # update BOTH so FM is learning in background during cold-start too
#         self.probit.update(x_chosen, reward)
#         self.fm.update(x_chosen, reward)


# from dataclasses import dataclass, field
# from typing import Dict, Any
# import numpy as np

# from bandit_thesis.agents.probit_ts import ProbitTSAgent
# from bandit_thesis.agents.ts_agent import ThompsonAgent


# @dataclass
# class HybridColdStartAgent:
#     probit: ProbitTSAgent
#     fm: ThompsonAgent
#     warmup_impressions: int = 30
#     update_fm_during_warmup: bool = True  # <--- add this switch
#     user_impressions: Dict[int, int] = field(default_factory=dict)

#     def _user_id(self, ctx: Dict[str, Any]) -> int:
#         return int(ctx.get("user_id", ctx.get("uid", 0)))

#     def reset(self) -> None:
#         self.user_impressions.clear()

#     def select_arm(self, X_cand: np.ndarray, candidate_arms: np.ndarray, ctx: Dict[str, Any]) -> int:
#         uid = self._user_id(ctx)
#         n = self.user_impressions.get(uid, 0)

#         if n < self.warmup_impressions:
#             return int(self.probit.select_arm(X_cand, candidate_arms))
#         return int(self.fm.select_arm(X_cand, candidate_arms))

#     def update(self, x_chosen: np.ndarray, reward: int, ctx: Dict[str, Any]) -> None:
#         uid = self._user_id(ctx)
#         n = self.user_impressions.get(uid, 0)
#         self.user_impressions[uid] = n + 1

#         # always update probit
#         self.probit.update(x_chosen, reward)

#         # update FM only if allowed during warmup or if warmup completed
#         if self.update_fm_during_warmup or n >= self.warmup_impressions:
#             self.fm.update(x_chosen, reward)

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np

from bandit_thesis.agents.probit_ts import ProbitTSAgent
from bandit_thesis.agents.ts_agent import ThompsonAgent


@dataclass
class HybridColdStartAgent:
    probit: ProbitTSAgent
    fm: ThompsonAgent
    warmup_impressions: int = 30

    # ✅ Option B: recovery window after shift
    recovery_steps: int = 500
    recovery_until_t: int = -1  # if t < recovery_until_t => force probit

    user_impressions: Dict[int, int] = field(default_factory=dict)

    def _user_id(self, ctx: Dict[str, Any]) -> int:
        return int(ctx.get("user_id", 0))

    def reset(self) -> None:
        self.user_impressions.clear()
        self.recovery_until_t = -1
        if hasattr(self.probit, "reset"):
            self.probit.reset()

    def on_shift(self, t: int) -> None:
        """Call when environment shifts. Force probit decisions for next recovery_steps."""
        self.recovery_until_t = int(t) + int(self.recovery_steps)

    def select_arm(self, X_cand: np.ndarray, candidate_arms: np.ndarray, ctx: Dict[str, Any]) -> int:
        uid = self._user_id(ctx)
        n = self.user_impressions.get(uid, 0)

        # current global time (we will add ctx["t"] in experiments)
        t = int(ctx.get("t", -1))

        # ✅ If in recovery window, force probit
        if t >= 0 and t < self.recovery_until_t:
            return int(self.probit.select_arm(X_cand, candidate_arms))

        # cold-start by user impressions
        if n < self.warmup_impressions:
            return int(self.probit.select_arm(X_cand, candidate_arms))

        return int(self.fm.select_arm(X_cand, candidate_arms))

    def update(self, x_chosen: np.ndarray, reward: int, ctx: Dict[str, Any]) -> None:
        uid = self._user_id(ctx)
        self.user_impressions[uid] = self.user_impressions.get(uid, 0) + 1

        # update both (good performance)
        self.probit.update(x_chosen, reward)
        self.fm.update(x_chosen, reward)

    def flush(self) -> None:
    # only flush if there is leftover batch
        if self._t % self.update_every != 0:
            self.model.update()