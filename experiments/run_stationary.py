
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import yaml

from bandit_thesis.envs.ad_env import AdEnvConfig, AdPersonalizationEnv
from bandit_thesis.features.featurizer import SimpleFeaturizer

from bandit_thesis.models.bayesian_fm import BayesianFM, BayesianFMConfig
from bandit_thesis.models.bayes_probit import BayesianProbitPosterior, ProbitConfig

from bandit_thesis.agents.random_agent import RandomAgent
from bandit_thesis.agents.ts_agent import ThompsonAgent
from bandit_thesis.agents.probit_ts import ProbitTSAgent
from bandit_thesis.agents.hybrid_agent import HybridColdStartAgent

from bandit_thesis.metrics.ctr import ctr_overall
from bandit_thesis.metrics.regret import cumulative_pseudo_regret
from bandit_thesis.metrics.cold_start import (
    cold_start_ctr_per_user,
    cold_start_regret_per_user,
    ctr_at_n,
    regret_at_n,
)

from bandit_thesis.validation.reporting import summarize_experiment
from bandit_thesis.utils.io import write_jsonl, write_csv


def _stable_seed(base_seed: int, tag: str) -> int:
    """Deterministic per-tag seed (stable across runs)."""
    h = 0
    for i, ch in enumerate(tag):
        h = (h * 131 + ord(ch) + i * 17) & 0xFFFFFFFF
    return (base_seed * 1000003 + h) & 0xFFFFFFFF


def _make_agent(
    name: str,
    featurizer_dim: int,
    cfg: dict,
    rng_agent: np.random.Generator,
    rng_baseline: np.random.Generator,
):
    """Build a fresh agent instance (ablation-ready)."""
    if name == "random":
        return RandomAgent(rng=rng_baseline)

    # FM TS model
    fmcfg = cfg["models"]["fm"]
    fm_cfg = BayesianFMConfig(p=featurizer_dim, **fmcfg)
    fm = BayesianFM(fm_cfg, rng=rng_agent)
    agent_fm = ThompsonAgent(model=fm)

    if name == "bayes_fm_ts":
        return agent_fm

    # Probit TS model
    pcfg = cfg["models"]["probit"]
    probit_cfg = ProbitConfig(
        p=featurizer_dim,
        prior_var=pcfg["prior_var"],
        window=pcfg["window"],
        gibbs_steps=pcfg["gibbs_steps"],
        ridge=float(pcfg.get("ridge", 1e-6)),
    )
    probit_model = BayesianProbitPosterior(probit_cfg, rng=rng_agent)
    agent_probit = ProbitTSAgent(probit_model, update_every=pcfg.get("update_every", 50))

    if name == "probit_ts":
        return agent_probit

    # Hybrid (stationary: warmup only; recovery not used)
    hcfg = cfg["models"]["hybrid"]
    return HybridColdStartAgent(
        probit=agent_probit,
        fm=agent_fm,
        warmup_impressions=int(hcfg["warmup_impressions"]),
        recovery_steps=int(hcfg.get("recovery_steps", 0)),  # safe default
    )


def run_one_seed(seed: int, cfg: dict) -> Dict[str, Dict[str, float]]:
    T = int(cfg["experiment"]["T"])
    log_every = int(cfg["experiment"].get("log_every", 1000))

    # metrics config
    mcfg = cfg.get("metrics", {})
    m_cold = int(mcfg.get("cold_m", 30))
    n_early = int(mcfg.get("early_n", 1000))

    env_cfg = AdEnvConfig(**cfg["env"])
    featurizer = SimpleFeaturizer(n_segments=env_cfg.n_user_segments, n_arms=env_cfg.n_arms)

    agents_order = ["random", "probit_ts", "bayes_fm_ts", "hybrid"]
    results: Dict[str, Dict[str, float]] = {}

    seed_start = time.perf_counter()

    for name in agents_order:
        # paired fairness: deterministic RNGs per (seed, agent)
        rng_env = np.random.default_rng(_stable_seed(seed, f"env::{name}"))
        rng_agent = np.random.default_rng(_stable_seed(seed, f"agent::{name}"))
        rng_baseline = np.random.default_rng(_stable_seed(seed, f"baseline::{name}"))

        env = AdPersonalizationEnv(env_cfg, rng=rng_env)
        env.reset()

        agent = _make_agent(
            name=name,
            featurizer_dim=featurizer.dim,
            cfg=cfg,
            rng_agent=rng_agent,
            rng_baseline=rng_baseline,
        )

        if hasattr(agent, "reset"):
            agent.reset()  # type: ignore

        rewards: List[int] = []
        p_opt: List[float] = []
        p_chosen: List[float] = []
        rows = []

        model_start = time.perf_counter()

        for i in range(T):
            ctx = env.sample_context(env.t)
            ctx["t"] = env.t

            cand = env.candidate_set(ctx)
            X_cand = np.vstack([featurizer.transform(ctx, int(a)) for a in cand])

            # select action
            if name == "random":
                chosen = agent.select_arm(cand, ctx)  # type: ignore
            elif name == "hybrid":
                chosen = agent.select_arm(X_cand, cand, ctx)  # type: ignore
            else:
                chosen = agent.select_arm(X_cand, cand)  # type: ignore

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
                    f"[stat][seed {seed:02d}] [{name:10s}] t={i+1:>6}/{T} "
                    f"elapsed={elapsed:6.1f}s  rate={rate:7.1f} it/s"
                )

        # flush once after the run
        if hasattr(agent, "flush"):
            agent.flush()  # type: ignore

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

        print(f"[stat][seed {seed:02d}] [{name}] DONE in {results[name]['time_sec']:.2f}s")

    seed_time = time.perf_counter() - seed_start
    print(f"[stat][seed {seed:02d}] ALL MODELS DONE in {seed_time:.2f}s\n")
    return results


def main() -> None:
    with open("configs/stationary.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seeds = int(cfg["experiment"]["seeds"])

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