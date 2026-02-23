# from __future__ import annotations
# import numpy as np
# import yaml

# from src.bandit_thesis.envs.ad_env import AdEnvConfig, AdPersonalizationEnv
# from src.bandit_thesis.envs.shifts import AbruptShift
# from src.bandit_thesis.features.featurizer import SimpleFeaturizer
# from src.bandit_thesis.models.bayesian_fm import BayesianFM, BayesianFMConfig
# from src.bandit_thesis.agents.random_agent import RandomAgent
# from src.bandit_thesis.agents.logistic_ts import LogisticTSAgent, LogisticTSConfig
# from src.bandit_thesis.agents.ts_agent import ThompsonAgent
# from src.bandit_thesis.metrics.ctr import ctr_overall
# from src.bandit_thesis.metrics.regret import cumulative_dynamic_regret
# from src.bandit_thesis.validation.protocols import make_rng_streams
# from src.bandit_thesis.validation.reporting import summarize_experiment
# from src.bandit_thesis.utils.io import write_jsonl, write_csv


# def run_one_seed(seed: int, cfg: dict) -> dict:
#     streams = make_rng_streams(seed)

#     shift_time = int(cfg["experiment"]["shift_time"])
#     env_cfg = AdEnvConfig(**cfg["env"])
#     env = AdPersonalizationEnv(
#         env_cfg,
#         rng=streams.rng_env,
#         nonstationarity=AbruptShift(shift_time=shift_time, on_shift=lambda e: e.shift_preferences()),
#     )
#     env.reset()

#     featurizer = SimpleFeaturizer(n_segments=env_cfg.n_user_segments, n_arms=env_cfg.n_arms)
#     T = int(cfg["experiment"]["T"])

#     agent_random = RandomAgent(rng=streams.rng_baseline)

#     log_cfg = LogisticTSConfig(p=featurizer.dim, **cfg["models"]["logistic"])
#     agent_log = LogisticTSAgent(log_cfg, rng=streams.rng_agent)

#     fm_cfg = BayesianFMConfig(p=featurizer.dim, **cfg["models"]["fm"])
#     fm = BayesianFM(fm_cfg, rng=streams.rng_agent)
#     agent_fm = ThompsonAgent(model=fm)

#     agents = {
#         "random": agent_random,
#         "logistic_ts": agent_log,
#         "bayes_fm_ts": agent_fm,
#     }

#     results = {}
#     for name, agent in agents.items():
#         env.reset()
#         rewards, p_opt, p_chosen = [], [], []
#         rows = []

#         for _ in range(T):
#             ctx = env.sample_context(env.t)
#             cand = env.candidate_set(ctx)
#             X_cand = np.vstack([featurizer.transform(ctx, int(a)) for a in cand])

#             if name == "random":
#                 chosen = agent.select_arm(cand, ctx)  # type: ignore
#             else:
#                 chosen = agent.select_arm(X_cand, cand)  # type: ignore

#             step = env.step(ctx, cand, int(chosen))
#             x_chosen = featurizer.transform(step.context, step.chosen_arm)

#             if name != "random":
#                 agent.update(x_chosen, step.reward)  # type: ignore

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

#         write_jsonl(f"results/raw/nonstationary/{name}/seed_{seed}.jsonl", rows)

#         cum_reg = cumulative_dynamic_regret(np.array(p_opt), np.array(p_chosen))
#         results[name] = {
#             "ctr": float(ctr_overall(np.array(rewards))),
#             "final_regret": float(cum_reg[-1]),
#         }

#     return results


# def main() -> None:
#     with open("configs/nonstationary.yaml", "r", encoding="utf-8") as f:
#         cfg = yaml.safe_load(f)

#     seeds = int(cfg["experiment"]["seeds"])
#     metrics_by_agent = {
#         "random": {"ctr": [], "regret": []},
#         "logistic_ts": {"ctr": [], "regret": []},
#         "bayes_fm_ts": {"ctr": [], "regret": []},
#     }

#     for s in range(seeds):
#         out = run_one_seed(s, cfg)
#         for agent, vals in out.items():
#             metrics_by_agent[agent]["ctr"].append(vals["ctr"])
#             metrics_by_agent[agent]["regret"].append(vals["final_regret"])

#     summary = summarize_experiment(metrics_by_agent, seed_count=seeds)
#     write_csv("results/tables/nonstationary_summary.csv", summary.table)
#     print(summary.table)


# if __name__ == "__main__":
#     main()

#----------------------------------------------------------------------------#

# experiments/run_nonstationary.py
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import yaml

from bandit_thesis.envs.ad_env import AdEnvConfig, AdPersonalizationEnv
from bandit_thesis.envs.shifts import AbruptShift
from bandit_thesis.features.featurizer import SimpleFeaturizer

from bandit_thesis.models.bayesian_fm import BayesianFM, BayesianFMConfig
from bandit_thesis.models.bayes_probit import BayesianProbitPosterior, ProbitConfig

from bandit_thesis.agents.random_agent import RandomAgent
from bandit_thesis.agents.ts_agent import ThompsonAgent
from bandit_thesis.agents.probit_ts import ProbitTSAgent
from bandit_thesis.agents.hybrid_agent import HybridColdStartAgent

from bandit_thesis.metrics.ctr import ctr_overall
from bandit_thesis.metrics.regret import cumulative_dynamic_regret
from bandit_thesis.metrics.cold_start import ctr_after_shift, ctr_last_w
from bandit_thesis.validation.protocols import make_rng_streams
from bandit_thesis.validation.reporting import summarize_experiment
from bandit_thesis.utils.io import write_jsonl, write_csv


def run_one_seed(seed: int, cfg: dict) -> Dict[str, Dict[str, float]]:
    streams = make_rng_streams(seed)

    shift_time = int(cfg["experiment"]["shift_time"])
    T = int(cfg["experiment"]["T"])
    log_every = int(cfg["experiment"].get("log_every", 1000))

    env_cfg = AdEnvConfig(**cfg["env"])
    env = AdPersonalizationEnv(
        env_cfg,
        rng=streams.rng_env,
        nonstationarity=AbruptShift(
            shift_time=shift_time,
            on_shift=lambda e: e.shift_preferences(),
        ),
    )
    env.reset()

    featurizer = SimpleFeaturizer(n_segments=env_cfg.n_user_segments, n_arms=env_cfg.n_arms)

    # --- agents ---
    agent_random = RandomAgent(rng=streams.rng_baseline)

    # 1) FM TS
    fm_cfg = BayesianFMConfig(p=featurizer.dim, **cfg["models"]["fm"])
    fm = BayesianFM(fm_cfg, rng=streams.rng_agent)
    agent_fm = ThompsonAgent(model=fm)

    # 2) Probit TS
    pcfg = cfg["models"]["probit"]
    probit_cfg = ProbitConfig(
        p=featurizer.dim,
        prior_var=pcfg["prior_var"],
        window=pcfg["window"],
        gibbs_steps=pcfg["gibbs_steps"],
    )
    probit_model = BayesianProbitPosterior(probit_cfg, rng=streams.rng_agent)
    agent_probit = ProbitTSAgent(probit_model, update_every=pcfg.get("update_every", 10))

    # 3) Hybrid (Probit cold-start -> FM warm-start)
    hcfg = cfg["models"]["hybrid"]
    agent_hybrid = HybridColdStartAgent(
        probit=agent_probit,
        fm=agent_fm,
        warmup_impressions=int(hcfg["warmup_impressions"]),
        recovery_steps=int(hcfg.get("recovery_steps", 500)),
    )

    agents = {
        "random": agent_random,
        "probit_ts": agent_probit,
        "bayes_fm_ts": agent_fm,
        "hybrid": agent_hybrid,
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
            if env.nonstationarityis not None:
                env.nonstationarity.apply(env, env.t)


            # ✅ if shift just happened, tell hybrid
            if env.t == shift_time and name == "hybrid" and hasattr(agent, "on_shift"):
                agent.on_shift(env.t)

            ctx = env.sample_context(env.t)
            ctx["t"] = env.t  # add time to context for featurization if needed
            cand = env.candidate_set(ctx)
            X_cand = np.vstack([featurizer.transform(ctx, int(a)) for a in cand])

            # select action
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
                    f"[nonstat][seed {seed:02d}] [{name:10s}] t={i+1:>6}/{T} "
                    f"elapsed={elapsed:6.1f}s  rate={rate:7.1f} it/s"
                )

            if hasattr(agent,'flush'):
                agent.flush()  # ensure any pending updates are applied

        shift_time = int(cfg["experiment"]["shift_time"])
        w_adapt = int(cfg.get("metrics", {}).get("adapt_w", 1000))

        ctr_post_shift = ctr_after_shift(rows, shift_time=shift_time, w=w_adapt)
        ctr_tail = ctr_last_w(rows, w=w_adapt)

        # save raw logs
        write_jsonl(f"results/raw/nonstationary/{name}/seed_{seed}.jsonl", rows)

        # dynamic regret
        cum_reg = cumulative_dynamic_regret(np.array(p_opt), np.array(p_chosen))

        results[name] = {
            "ctr": float(ctr_overall(np.array(rewards))),
            "final_regret": float(cum_reg[-1]),
            "ctr_after_shift": float(ctr_post_shift),
            "ctr_last_w": float(ctr_tail),
            "time_sec": float(time.perf_counter() - model_start),
        }

        print(f"[nonstat][seed {seed:02d}] [{name}] DONE in {results[name]['time_sec']:.2f}s")

    seed_time = time.perf_counter() - seed_start
    print(f"[nonstat][seed {seed:02d}] ALL MODELS DONE in {seed_time:.2f}s\n")

    return results


def main() -> None:
    with open("configs/nonstationary.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seeds = int(cfg["experiment"]["seeds"])

    metrics_by_agent = {
            "random": {"ctr": [], "regret": [], "ctr_after_shift": [], "ctr_last_w": [], "time": []},
            "probit_ts": {"ctr": [], "regret": [], "ctr_after_shift": [], "ctr_last_w": [], "time": []},
            "bayes_fm_ts": {"ctr": [], "regret": [], "ctr_after_shift": [], "ctr_last_w": [], "time": []},
            "hybrid": {"ctr": [], "regret": [], "ctr_after_shift": [], "ctr_last_w": [], "time": []},
    }

    overall_start = time.perf_counter()

    for s in range(seeds):
        out = run_one_seed(seed=s, cfg=cfg)
        for agent, vals in out.items():
            metrics_by_agent[agent]["ctr"].append(vals["ctr"])
            metrics_by_agent[agent]["regret"].append(vals["final_regret"])
            metrics_by_agent[agent]["time"].append(vals.get("time_sec", 0.0))
            metrics_by_agent[agent]["ctr_after_shift"].append(vals["ctr_after_shift"])
            metrics_by_agent[agent]["ctr_last_w"].append(vals["ctr_last_w"])


    overall_time = time.perf_counter() - overall_start

    summary = summarize_experiment(metrics_by_agent, seed_count=seeds)
    write_csv("results/tables/nonstationary_summary.csv", summary.table)
    print(summary.table)

    print("\nTiming (avg seconds per seed):")
    for agent in metrics_by_agent:
        t = np.array(metrics_by_agent[agent]["time"], dtype=float)
        print(f"  {agent:10s}: mean={t.mean():.2f}s  std={t.std(ddof=1) if len(t)>1 else 0.0:.2f}s")

    print(f"\nTOTAL WALL TIME: {overall_time:.2f}s")


if __name__ == "__main__":
    main()
