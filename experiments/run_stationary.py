# from __future__ import annotations
# import os
# from typing import Dict, List
# import time
# from matplotlib.pyplot import step
# import numpy as np
# import yaml

# from bandit_thesis.envs.ad_env import AdEnvConfig, AdPersonalizationEnv
# from bandit_thesis.features.featurizer import SimpleFeaturizer
# from bandit_thesis.models.bayesian_fm import BayesianFM, BayesianFMConfig
# from bandit_thesis.agents.random_agent import RandomAgent
# # from src.bandit_thesis.agents.logistic_ts import LogisticTSAgent, LogisticTSConfig

# from bandit_thesis.models.bayes_probit import BayesianProbitPosterior, ProbitConfig
# from bandit_thesis.agents.probit_ts import ProbitTSAgent
# from bandit_thesis.agents.hybrid_agent import HybridColdStartAgent

# from bandit_thesis.agents.ts_agent import ThompsonAgent
# from bandit_thesis.metrics.ctr import ctr_overall
# from bandit_thesis.metrics.regret import cumulative_pseudo_regret
# from bandit_thesis.validation.protocols import make_rng_streams
# from bandit_thesis.validation.reporting import summarize_experiment
# from bandit_thesis.utils.io import write_jsonl, write_csv


# def run_one_seed(seed: int, cfg: dict) -> Dict[str, Dict[str, float]]:
#     streams = make_rng_streams(seed)

#     env_cfg = AdEnvConfig(**cfg["env"])
#     env = AdPersonalizationEnv(env_cfg, rng=streams.rng_env)
#     env.reset()

#     featurizer = SimpleFeaturizer(n_segments=env_cfg.n_user_segments, n_arms=env_cfg.n_arms)

#     T = int(cfg["experiment"]["T"])
#     log_every = int(cfg["experiment"].get("log_every", 1000))  # add to yaml or default 1000

#     # agents
#     agent_random = RandomAgent(rng=streams.rng_baseline)

#     # log_cfg = LogisticTSConfig(p=featurizer.dim, **cfg["models"]["logistic"])
#     # agent_log = LogisticTSAgent(log_cfg, rng=streams.rng_agent)
#     # probit_cfg = ProbitConfig(p=featurizer.dim, prior_var=5.0, window=300, gibbs_steps=2)
#     # probit_model = BayesianProbitPosterior(probit_cfg, rng=streams.rng_agent)
#     # agent_log = ProbitTSAgent(probit_model)

#     pcfg = cfg["models"]["probit"]
#     probit_cfg = ProbitConfig(
#     p=featurizer.dim,
#     prior_var=pcfg["prior_var"],
#     window=pcfg["window"],
#     gibbs_steps=pcfg["gibbs_steps"],
# )
#     probit_model = BayesianProbitPosterior(probit_cfg, rng=streams.rng_agent)
#     agent_probit = ProbitTSAgent(probit_model, update_every=pcfg.get("update_every", 10))
#     hybrid_agent = HybridColdStartAgent(
#         probit=agent_probit,
#         fm=agent_fm,
#         warmup_impressions=int(cfg["models"]["hybrid"]["warmup_impressions"]),
#     )


#     fm_cfg = BayesianFMConfig(p=featurizer.dim, **cfg["models"]["fm"])
#     fm = BayesianFM(fm_cfg, rng=streams.rng_agent)
#     agent_fm = ThompsonAgent(model=fm)

#     agents = {
#         "random": agent_random,
#         # "logistic_ts": agent_log,
#         "probit_ts": agent_probit,
#         "bayes_fm_ts": agent_fm,
#         "hybrid": hybrid_agent,
#     }

#     results: Dict[str, Dict[str, float]] = {}

#     # seed timer
#     seed_start = time.perf_counter()

#     for name, agent in agents.items():

#         model_start = time.perf_counter()

#         env.reset()
#         rewards: List[int] = []
#         p_opt: List[float] = []
#         p_chosen: List[float] = []
#         rows = []

#         for _ in range(T):
#             # Build candidate feature matrix
#             ctx = env.sample_context(env.t)
#             cand = env.candidate_set(ctx)
#             X_cand = np.vstack([featurizer.transform(ctx, int(a)) for a in cand])

#             # select action
#             if name == "random":
#                 chosen = agent.select_arm(cand, ctx)
#             elif name == "hybrid":
#                 chosen = agent.select_arm(X_cand, cand, ctx)
#             else:
#                 chosen = agent.select_arm(X_cand, cand)

#             # env step
#             step = env.step(ctx, cand, int(chosen))
#             x_chosen = featurizer.transform(step.context, step.chosen_arm)

#             # update model
#             if name == "random":
#                 pass
#             elif name == "hybrid":
#                 agent.update(x_chosen, step.reward, step.context)
#             else:
#                 agent.update(x_chosen, step.reward)

#             rewards.append(step.reward)
#             p_opt.append(step.p_opt)
#             p_chosen.append(step.p_chosen)

#             rows.append(
#                 {
#                     "t": step.t,
#                     "chosen_arm": step.chosen_arm,
#                     "reward": step.reward,
#                     "p_opt": step.p_opt,
#                     "p_chosen": step.p_chosen,
#                     "opt_arm": step.opt_arm,
#                 }
#             )

#             # progress log
#             if (_ + 1) % log_every == 0:
#                 elapsed = time.perf_counter() - model_start
#                 rate = (_ + 1) / max(elapsed, 1e-9)
#                 print(
#                     f"[seed {seed:02d}] [{name:10s}] t={_+1:>6}/{T} "
#                     f"elapsed={elapsed:6.1f}s  rate={rate:7.1f} it/s"
#                 )

#         # save raw
#         raw_path = f"results/raw/stationary/{name}/seed_{seed}.jsonl"
#         write_jsonl(raw_path, rows)

#         cum_reg = cumulative_pseudo_regret(np.array(p_opt), np.array(p_chosen))
#         results[name] = {
#             "ctr": float(ctr_overall(np.array(rewards))),
#             "final_regret": float(cum_reg[-1]),
#             "time_sec": float(time.perf_counter() - model_start), 
#         }

#         print(f"[seed {seed:02d}] [{name}] DONE in {results[name]['time_sec']:.2f}s")

#     seed_time = time.perf_counter() - seed_start
#     print(f"[seed {seed:02d}] ALL MODELS DONE in {seed_time:.2f}s\n")

#     return results


# # def main() -> None:
# #     with open("configs/stationary.yaml", "r", encoding="utf-8") as f:
# #         cfg = yaml.safe_load(f)

# #     seeds = int(cfg["experiment"]["seeds"])
# #     metrics_by_agent = {
# #         "random": {"ctr": [], "regret": []},
# #         "probit_ts": {"ctr": [], "regret": []},
# #         "bayes_fm_ts": {"ctr": [], "regret": []},
# #     }

# #     for s in range(seeds):
# #         out = run_one_seed(seed=s, cfg=cfg)
# #         for agent, vals in out.items():
# #             metrics_by_agent[agent]["ctr"].append(vals["ctr"])
# #             metrics_by_agent[agent]["regret"].append(vals["final_regret"])

# #     summary = summarize_experiment(metrics_by_agent, seed_count=seeds)
# #     write_csv("results/tables/stationary_summary.csv", summary.table)
# #     print(summary.table)

# def main() -> None:
#     with open("configs/stationary.yaml", "r", encoding="utf-8") as f:
#         cfg = yaml.safe_load(f)

#     seeds = int(cfg["experiment"]["seeds"])

#     metrics_by_agent = {
#         "random": {"ctr": [], "regret": [], "time": []},
#         "probit_ts": {"ctr": [], "regret": [], "time": []},
#         "bayes_fm_ts": {"ctr": [], "regret": [], "time": []},
#     }

#     overall_start = time.perf_counter()

#     for s in range(seeds):
#         out = run_one_seed(seed=s, cfg=cfg)
#         for agent, vals in out.items():
#             metrics_by_agent[agent]["ctr"].append(vals["ctr"])
#             metrics_by_agent[agent]["regret"].append(vals["final_regret"])
#             metrics_by_agent[agent]["time"].append(vals.get("time_sec", 0.0))

#     overall_time = time.perf_counter() - overall_start

#     summary = summarize_experiment(metrics_by_agent, seed_count=seeds)
#     write_csv("results/tables/stationary_summary.csv", summary.table)
#     print(summary.table)

#     # time report
#     print("\nTiming (avg seconds per seed):")
#     for agent in metrics_by_agent:
#         t = np.array(metrics_by_agent[agent]["time"], dtype=float)
#         print(f"  {agent:10s}: mean={t.mean():.2f}s  std={t.std(ddof=1) if len(t)>1 else 0.0:.2f}s")

#     print(f"\nTOTAL WALL TIME: {overall_time:.2f}s")



# if __name__ == "__main__":
#     main()



#--------------------------#

from __future__ import annotations

from typing import Dict, List
import time

import numpy as np
import yaml

from bandit_thesis.envs.ad_env import AdEnvConfig, AdPersonalizationEnv
from bandit_thesis.features.featurizer import SimpleFeaturizer
from bandit_thesis.models.bayesian_fm import BayesianFM, BayesianFMConfig

from bandit_thesis.agents.random_agent import RandomAgent
from bandit_thesis.agents.ts_agent import ThompsonAgent
from bandit_thesis.agents.probit_ts import ProbitTSAgent
from bandit_thesis.agents.hybrid_agent import HybridColdStartAgent

from bandit_thesis.models.bayes_probit import BayesianProbitPosterior, ProbitConfig

from bandit_thesis.metrics.ctr import ctr_overall
from bandit_thesis.metrics.regret import cumulative_pseudo_regret
from bandit_thesis.metrics.cold_start import (
    cold_start_ctr_per_user,
    cold_start_regret_per_user,
    ctr_at_n,
    regret_at_n,
)

from bandit_thesis.validation.protocols import make_rng_streams
from bandit_thesis.validation.reporting import summarize_experiment
from bandit_thesis.utils.io import write_jsonl, write_csv


def run_one_seed(seed: int, cfg: dict) -> Dict[str, Dict[str, float]]:
    streams = make_rng_streams(seed)

    env_cfg = AdEnvConfig(**cfg["env"])
    env = AdPersonalizationEnv(env_cfg, rng=streams.rng_env)
    env.reset()

    featurizer = SimpleFeaturizer(n_segments=env_cfg.n_user_segments, n_arms=env_cfg.n_arms)

    T = int(cfg["experiment"]["T"])
    log_every = int(cfg["experiment"].get("log_every", 1000))

    # metric params (configurable)
    m_cold = int(cfg["metrics"].get("cold_m", 30))
    n_early = int(cfg["metrics"].get("early_n", 1000))

    # --- agents ---
    agent_random = RandomAgent(rng=streams.rng_baseline)

    # FM TS
    fm_cfg = BayesianFMConfig(p=featurizer.dim, **cfg["models"]["fm"])
    fm = BayesianFM(fm_cfg, rng=streams.rng_agent)
    agent_fm = ThompsonAgent(model=fm)

    # Probit TS
    pcfg = cfg["models"]["probit"]
    probit_cfg = ProbitConfig(
        p=featurizer.dim,
        prior_var=pcfg["prior_var"],
        window=pcfg["window"],
        gibbs_steps=pcfg["gibbs_steps"],
    )
    probit_model = BayesianProbitPosterior(probit_cfg, rng=streams.rng_agent)
    agent_probit = ProbitTSAgent(probit_model, update_every=pcfg.get("update_every", 10))

    # Hybrid: Probit (cold start) -> FM
    hcfg = cfg["models"]["hybrid"]
    hybrid_agent = HybridColdStartAgent(
        probit=agent_probit,
        fm=agent_fm,
        warmup_impressions=int(hcfg["warmup_impressions"]),
    )

    agents = {
        "random": agent_random,
        "probit_ts": agent_probit,
        "bayes_fm_ts": agent_fm,
        "hybrid": hybrid_agent,
    }

    results: Dict[str, Dict[str, float]] = {}

    seed_start = time.perf_counter()

    for name, agent in agents.items():
        if hasattr(agent, "reset"):
            agent.reset()  # reset any internal state if needed
        

        rewards: List[int] = []
        p_opt: List[float] = []
        p_chosen: List[float] = []
        rows = []

        model_start = time.perf_counter()

        for i in range(T):


            ctx = env.sample_context(env.t)
            ctx["t"] = env.t  # add time to context for featurization if needed
            cand = env.candidate_set(ctx)
            X_cand = np.vstack([featurizer.transform(ctx, int(a)) for a in cand])

            # select
            if name == "random":
                chosen = agent.select_arm(cand, ctx)  # type: ignore
            elif name == "hybrid":
                chosen = agent.select_arm(X_cand, cand, ctx)  # type: ignore
            else:
                chosen = agent.select_arm(X_cand, cand)  # type: ignore

            # step
            step = env.step(ctx, cand, int(chosen))
            x_chosen = featurizer.transform(step.context, step.chosen_arm)

            # update
            if name == "random":
                pass
            elif name == "hybrid":
                agent.update(x_chosen, step.reward, step.context)  # type: ignore
            else:
                agent.update(x_chosen, step.reward)  # type: ignore

            rewards.append(step.reward)
            p_opt.append(step.p_opt)
            p_chosen.append(step.p_chosen)

            rows.append(
                {
                    "t": step.t,
                    "context": step.context,
                    "chosen_arm": step.chosen_arm,
                    "reward": step.reward,
                    "p_opt": step.p_opt,
                    "p_chosen": step.p_chosen,
                    "opt_arm": step.opt_arm,
                }
            )

            if (i + 1) % log_every == 0:
                elapsed = time.perf_counter() - model_start
                rate = (i + 1) / max(elapsed, 1e-9)
                print(
                    f"[seed {seed:02d}] [{name:10s}] t={i+1:>6}/{T} "
                    f"elapsed={elapsed:6.1f}s  rate={rate:7.1f} it/s"
                )

        if hasattr(agent, "flush"):
            agent.flush()  # for any final updates after the loop

        # save raw logs
        write_jsonl(f"results/raw/stationary/{name}/seed_{seed}.jsonl", rows)

        # metrics
        cum_reg = cumulative_pseudo_regret(np.array(p_opt), np.array(p_chosen))

        ctr_early = ctr_at_n(rows, n=n_early)
        reg_early = regret_at_n(rows, n=n_early)
        cold_ctr = cold_start_ctr_per_user(rows, m=m_cold)
        cold_reg = cold_start_regret_per_user(rows, m=m_cold)

        results[name] = {
            "ctr": float(ctr_overall(np.array(rewards))),
            "final_regret": float(cum_reg[-1]),
            "ctr_early": float(ctr_early),
            "regret_early": float(reg_early),
            "cold_ctr": float(cold_ctr),
            "cold_regret": float(cold_reg),
            "time_sec": float(time.perf_counter() - model_start),
        }

        print(f"[seed {seed:02d}] [{name}] DONE in {results[name]['time_sec']:.2f}s")

    seed_time = time.perf_counter() - seed_start
    print(f"[seed {seed:02d}] ALL MODELS DONE in {seed_time:.2f}s\n")

    return results


def main() -> None:
    with open("configs/stationary.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seeds = int(cfg["experiment"]["seeds"])

    # IMPORTANT: include all metrics you return
    metrics_by_agent = {
        "random": {"ctr": [], "regret": [], "ctr_early": [], "regret_early": [], "cold_ctr": [], "cold_regret": [], "time": []},
        "probit_ts": {"ctr": [], "regret": [], "ctr_early": [], "regret_early": [], "cold_ctr": [], "cold_regret": [], "time": []},
        "bayes_fm_ts": {"ctr": [], "regret": [], "ctr_early": [], "regret_early": [], "cold_ctr": [], "cold_regret": [], "time": []},
        "hybrid": {"ctr": [], "regret": [], "ctr_early": [], "regret_early": [], "cold_ctr": [], "cold_regret": [], "time": []},
    }

    overall_start = time.perf_counter()

    for s in range(seeds):
        out = run_one_seed(seed=s, cfg=cfg)
        for agent, vals in out.items():
            metrics_by_agent[agent]["ctr"].append(vals["ctr"])
            metrics_by_agent[agent]["regret"].append(vals["final_regret"])
            metrics_by_agent[agent]["ctr_early"].append(vals["ctr_early"])
            metrics_by_agent[agent]["regret_early"].append(vals["regret_early"])
            metrics_by_agent[agent]["cold_ctr"].append(vals["cold_ctr"])
            metrics_by_agent[agent]["cold_regret"].append(vals["cold_regret"])
            metrics_by_agent[agent]["time"].append(vals["time_sec"])

    overall_time = time.perf_counter() - overall_start

    summary = summarize_experiment(metrics_by_agent, seed_count=seeds)
    write_csv("results/tables/stationary_summary.csv", summary.table)
    print(summary.table)

    print(f"\nTOTAL WALL TIME: {overall_time:.2f}s")


if __name__ == "__main__":
    main()