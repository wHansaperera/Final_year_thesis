# from __future__ import annotations
# from dataclasses import dataclass
# import numpy as np

# from bandit_thesis.models.bayes_probit import BayesianProbitPosterior


# # @dataclass
# # class ProbitTSAgent:
# #     model: BayesianProbitPosterior

# #     def select_arm(self, X_cand: np.ndarray, candidate_arms: np.ndarray) -> int:
# #         beta = self.model.sample_beta()
# #         # score by probit probability Phi(x^T beta)
# #         scores = np.array([self.model.probit_prob(X_cand[i], beta) for i in range(X_cand.shape[0])])
# #         return int(candidate_arms[int(np.argmax(scores))])

# #     def update(self, x: np.ndarray, reward: int) -> None:
# #         self.model.add_observation(x, reward)
# #         self.model.update()


# @dataclass
# class ProbitTSAgent:
#     model: BayesianProbitPosterior
#     update_every: int = 10  # <- update posterior every N steps

#     def __post_init__(self) -> None:
#         self._t = 0

#     def select_arm(self, X_cand: np.ndarray, candidate_arms: np.ndarray) -> int:
#         beta = self.model.sample_beta()
#         scores = np.array([self.model.probit_prob(X_cand[i], beta) for i in range(X_cand.shape[0])])
#         return int(candidate_arms[int(np.argmax(scores))])

#     def update(self, x: np.ndarray, reward: int) -> None:
#         self._t += 1
#         self.model.add_observation(x, reward)
#         if self._t % self.update_every == 0:
#             self.model.update()


from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from bandit_thesis.models.bayes_probit import BayesianProbitPosterior


@dataclass
class ProbitTSAgent:
    model: BayesianProbitPosterior
    update_every: int = 10

    def __post_init__(self) -> None:
        self._t = 0

    def reset(self) -> None:
        self._t = 0

    def select_arm(self, X_cand: np.ndarray, candidate_arms: np.ndarray) -> int:
        beta = self.model.sample_beta()
        scores = np.array([self.model.probit_prob(X_cand[i], beta) for i in range(X_cand.shape[0])])
        return int(candidate_arms[int(np.argmax(scores))])

    def update(self, x: np.ndarray, reward: int) -> None:
        self._t += 1
        self.model.add_observation(x, reward)
        if self._t % self.update_every == 0:
            self.model.update()

    # def flush(self) -> None:
    #     # ensure last partial batch is reflected in posterior
    #     self.model.update()

    def flush(self) -> None:
        if len(self.model._X) > 0:
            self.model.update()