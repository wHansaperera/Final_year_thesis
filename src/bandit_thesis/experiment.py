from __future__ import annotations

from pathlib import Path
import time
from typing import Dict, List

import numpy as np

from bandit_thesis.agents.hybrid_agent import HybridColdStartAgent
from bandit_thesis.agents.probit_ts import ProbitTSAgent
from bandit_thesis.agents.random_agent import RandomAgent
from bandit_thesis.agents.ts_agent import ThompsonAgent
from bandit_thesis.envs.ad_env import AdEnvConfig, AdPersonalizationEnv
from bandit_thesis.envs.shifts import AbruptShift
from bandit_thesis.features.featurizer import SimpleFeaturizer
from bandit_thesis.metrics.cold_start import (
    cold_start_ctr_per_user,
    cold_start_regret_per_user,
    ctr_after_shift,
    ctr_at_n,
    ctr_last_w,
    regret_at_n,
)
from bandit_thesis.metrics.ctr import ctr_overall
from bandit_thesis.metrics.regret import cumulative_dynamic_regret, cumulative_pseudo_regret
from bandit_thesis.models.bayes_probit import BayesianProbitPosterior, ProbitConfig
from bandit_thesis.models.bayesian_fm import BayesianFM, BayesianFMConfig
from bandit_thesis.utils.io import write_csv, write_json, write_jsonl
from bandit_thesis.validation.reporting import summarize_experiment

AGENT_ORDER = ["random", "probit_ts", "bayes_fm_ts", "hybrid"]
SCHEMA_VERSION = 2


def stable_seed(base_seed: int, tag: str) -> int:
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
    if name == "random":
        return RandomAgent(rng=rng_baseline)

    fm_cfg = BayesianFMConfig(p=featurizer_dim, **cfg["models"]["fm"])
    fm = BayesianFM(fm_cfg, rng=rng_agent)
    fm_agent = ThompsonAgent(model=fm)
    if name == "bayes_fm_ts":
        return fm_agent

    probit_cfg_raw = cfg["models"]["probit"]
    probit_cfg = ProbitConfig(
        p=featurizer_dim,
        prior_var=probit_cfg_raw["prior_var"],
        window=probit_cfg_raw["window"],
        gibbs_steps=probit_cfg_raw["gibbs_steps"],
        ridge=float(probit_cfg_raw.get("ridge", 1e-6)),
    )
    probit_model = BayesianProbitPosterior(probit_cfg, rng=rng_agent)
    probit_agent = ProbitTSAgent(
        model=probit_model,
        update_every=int(probit_cfg_raw.get("update_every", 50)),
    )
    if name == "probit_ts":
        return probit_agent

    hybrid_cfg = cfg["models"]["hybrid"]
    return HybridColdStartAgent(
        probit=probit_agent,
        fm=fm_agent,
        warmup_impressions=int(hybrid_cfg["warmup_impressions"]),
        blend_span=int(hybrid_cfg.get("blend_span", 40)),
        max_fm_weight=float(hybrid_cfg.get("max_fm_weight", 0.45)),
        fm_override_margin=float(hybrid_cfg.get("fm_override_margin", 0.03)),
        max_probit_drop=float(hybrid_cfg.get("max_probit_drop", 0.06)),
        recovery_steps=int(hybrid_cfg.get("recovery_steps", 0)),
    )


def _make_env(mode: str, env_cfg: AdEnvConfig, cfg: dict, rng_env: np.random.Generator) -> AdPersonalizationEnv:
    if mode == "nonstationary":
        shift_time = int(cfg["experiment"]["shift_time"])
        shift_strength = float(cfg["experiment"].get("shift_strength", 1.0))
        return AdPersonalizationEnv(
            env_cfg,
            rng=rng_env,
            nonstationarity=AbruptShift(
                shift_time=shift_time,
                on_shift=lambda env: env.shift_preferences(strength=shift_strength),
            ),
        )
    return AdPersonalizationEnv(env_cfg, rng=rng_env)


def _results_path(output_root: str, *parts: str) -> str:
    return str(Path(output_root, *parts))


def _manifest(output_root: str, mode: str, cfg: dict) -> None:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "agents": AGENT_ORDER,
        "config": cfg,
        "raw_row_fields": [
            "schema_version",
            "seed",
            "agent",
            "t",
            "mode",
            "user_id",
            "segment_id",
            "user_history_len",
            "is_new_user",
            "device",
            "hour_bucket",
            "is_weekend",
            "candidate_arms",
            "candidate_groups",
            "chosen_arm",
            "chosen_group",
            "opt_arm",
            "opt_group",
            "reward",
            "p_chosen",
            "p_opt",
            "regret",
            "shift_applied",
            "decision_mode",
        ],
    }
    write_json(_results_path(output_root, "manifests", f"{mode}.json"), manifest)


def _stationary_metrics(rows: List[dict], rewards: np.ndarray, p_opt: np.ndarray, p_chosen: np.ndarray, cfg: dict) -> Dict[str, float]:
    metrics_cfg = cfg.get("metrics", {})
    cold_m = int(metrics_cfg.get("cold_m", 20))
    early_n = int(metrics_cfg.get("early_n", 1000))
    cum_reg = cumulative_pseudo_regret(p_opt, p_chosen)
    return {
        "ctr": float(ctr_overall(rewards)),
        "final_regret": float(cum_reg[-1]),
        "ctr_early": float(ctr_at_n(rows, n=early_n)),
        "regret_early": float(regret_at_n(rows, n=early_n)),
        "cold_ctr": float(cold_start_ctr_per_user(rows, m=cold_m)),
        "cold_regret": float(cold_start_regret_per_user(rows, m=cold_m)),
    }


def _nonstationary_metrics(rows: List[dict], rewards: np.ndarray, p_opt: np.ndarray, p_chosen: np.ndarray, cfg: dict) -> Dict[str, float]:
    metrics_cfg = cfg.get("metrics", {})
    cold_m = int(metrics_cfg.get("cold_m", 20))
    adapt_w = int(metrics_cfg.get("adapt_w", 500))
    shift_time = int(cfg["experiment"]["shift_time"])
    cum_reg = cumulative_dynamic_regret(p_opt, p_chosen)
    return {
        "ctr": float(ctr_overall(rewards)),
        "final_regret": float(cum_reg[-1]),
        "cold_ctr": float(cold_start_ctr_per_user(rows, m=cold_m)),
        "cold_regret": float(cold_start_regret_per_user(rows, m=cold_m)),
        "ctr_after_shift": float(ctr_after_shift(rows, shift_time=shift_time, w=adapt_w)),
        "ctr_last_w": float(ctr_last_w(rows, w=adapt_w)),
    }


def run_one_seed(
    seed: int,
    cfg: dict,
    mode: str,
    output_root: str = "results",
    save_raw: bool = True,
) -> Dict[str, Dict[str, float]]:
    T = int(cfg["experiment"]["T"])
    log_every = int(cfg["experiment"].get("log_every", 1000))

    env_cfg = AdEnvConfig(**cfg["env"])
    featurizer = SimpleFeaturizer(
        n_segments=env_cfg.n_user_segments,
        n_users=env_cfg.max_users,
        n_arms=env_cfg.n_arms,
    )

    seed_results: Dict[str, Dict[str, float]] = {}
    seed_start = time.perf_counter()

    env_seed = stable_seed(seed, "env")

    for agent_name in AGENT_ORDER:
        rng_env = np.random.default_rng(env_seed)
        rng_agent = np.random.default_rng(stable_seed(seed, f"agent::{agent_name}"))
        rng_baseline = np.random.default_rng(stable_seed(seed, f"baseline::{agent_name}"))

        env = _make_env(mode=mode, env_cfg=env_cfg, cfg=cfg, rng_env=rng_env)
        agent = _make_agent(
            name=agent_name,
            featurizer_dim=featurizer.dim,
            cfg=cfg,
            rng_agent=rng_agent,
            rng_baseline=rng_baseline,
        )
        if hasattr(agent, "reset"):
            agent.reset()  # type: ignore[attr-defined]

        rows: List[dict] = []
        rewards: List[int] = []
        p_opt: List[float] = []
        p_chosen: List[float] = []

        model_start = time.perf_counter()

        for i in range(T):
            ctx = env.sample_context(env.t)
            ctx["t"] = env.t

            candidate_arms = env.candidate_set(ctx)
            X_cand = np.vstack([featurizer.transform(ctx, int(a)) for a in candidate_arms])

            decision_mode = agent_name
            if agent_name == "random":
                chosen_arm = agent.select_arm(candidate_arms, ctx)  # type: ignore[attr-defined]
            elif agent_name == "hybrid":
                chosen_arm = agent.select_arm(X_cand, candidate_arms, ctx)  # type: ignore[attr-defined]
                decision_mode = getattr(agent, "last_mode", "hybrid")
            else:
                chosen_arm = agent.select_arm(X_cand, candidate_arms)  # type: ignore[attr-defined]

            step = env.step(ctx, candidate_arms, int(chosen_arm))
            x_chosen = featurizer.transform(step.context, step.chosen_arm)

            if agent_name == "hybrid":
                agent.update(x_chosen, step.reward, step.context)  # type: ignore[attr-defined]
            elif agent_name != "random":
                agent.update(x_chosen, step.reward)  # type: ignore[attr-defined]

            if step.shift_applied and hasattr(agent, "on_shift"):
                agent.on_shift(step.t + 1)  # type: ignore[attr-defined]

            chosen_group = int(env.arm_group[int(step.chosen_arm)])
            opt_group = int(env.arm_group[int(step.opt_arm)])
            candidate_groups = sorted(set(int(env.arm_group[int(a)]) for a in candidate_arms))

            row = {
                "schema_version": SCHEMA_VERSION,
                "seed": seed,
                "agent": agent_name,
                "mode": mode,
                "t": step.t,
                "context": step.context,
                "user_id": int(step.context["user_id"]),
                "segment_id": int(step.context["segment_id"]),
                "user_history_len": int(step.context["user_history_len"]),
                "is_new_user": int(step.context["is_new_user"]),
                "device": int(step.context["device"]),
                "hour_bucket": int(step.context["hour_bucket"]),
                "is_weekend": int(step.context["is_weekend"]),
                "candidate_arms": [int(a) for a in candidate_arms.tolist()],
                "candidate_groups": candidate_groups,
                "chosen_arm": int(step.chosen_arm),
                "chosen_group": chosen_group,
                "opt_arm": int(step.opt_arm),
                "opt_group": opt_group,
                "reward": int(step.reward),
                "p_chosen": float(step.p_chosen),
                "p_opt": float(step.p_opt),
                "regret": float(step.p_opt - step.p_chosen),
                "shift_applied": bool(step.shift_applied),
                "decision_mode": decision_mode,
            }
            rows.append(row)

            rewards.append(step.reward)
            p_opt.append(step.p_opt)
            p_chosen.append(step.p_chosen)

            if (i + 1) % log_every == 0:
                elapsed = time.perf_counter() - model_start
                rate = (i + 1) / max(elapsed, 1e-9)
                print(
                    f"[{mode}][seed {seed:02d}] [{agent_name:10s}] "
                    f"t={i+1:>6}/{T} elapsed={elapsed:6.1f}s rate={rate:7.1f} it/s"
                )

        if hasattr(agent, "flush"):
            agent.flush()  # type: ignore[attr-defined]

        if save_raw:
            write_jsonl(_results_path(output_root, "raw", mode, agent_name, f"seed_{seed}.jsonl"), rows)

        rewards_arr = np.asarray(rewards, dtype=float)
        p_opt_arr = np.asarray(p_opt, dtype=float)
        p_chosen_arr = np.asarray(p_chosen, dtype=float)

        if mode == "stationary":
            metrics = _stationary_metrics(rows, rewards_arr, p_opt_arr, p_chosen_arr, cfg)
        elif mode == "nonstationary":
            metrics = _nonstationary_metrics(rows, rewards_arr, p_opt_arr, p_chosen_arr, cfg)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        metrics["time_sec"] = float(time.perf_counter() - model_start)
        seed_results[agent_name] = metrics
        print(f"[{mode}][seed {seed:02d}] [{agent_name}] DONE in {metrics['time_sec']:.2f}s")

    seed_time = time.perf_counter() - seed_start
    print(f"[{mode}][seed {seed:02d}] ALL MODELS DONE in {seed_time:.2f}s\n")
    return seed_results


def run_experiment(config: dict, mode: str, output_root: str = "results") -> None:
    seeds = int(config["experiment"]["seeds"])
    _manifest(output_root=output_root, mode=mode, cfg=config)

    metric_names = {
        "stationary": ["ctr", "final_regret", "ctr_early", "regret_early", "cold_ctr", "cold_regret", "time_sec"],
        "nonstationary": ["ctr", "final_regret", "cold_ctr", "cold_regret", "ctr_after_shift", "ctr_last_w", "time_sec"],
    }[mode]

    metrics_by_agent: Dict[str, Dict[str, List[float]]] = {
        agent: {metric: [] for metric in metric_names}
        for agent in AGENT_ORDER
    }

    overall_start = time.perf_counter()
    for seed in range(seeds):
        out = run_one_seed(seed=seed, cfg=config, mode=mode, output_root=output_root, save_raw=True)
        for agent_name, metrics in out.items():
            for metric_name in metric_names:
                metrics_by_agent[agent_name][metric_name].append(metrics[metric_name])

    summary = summarize_experiment(metrics_by_agent=metrics_by_agent, seed_count=seeds)
    write_csv(_results_path(output_root, "tables", f"{mode}_summary.csv"), summary.table)

    total_time = time.perf_counter() - overall_start
    print(summary.table)
    print(f"\nTOTAL WALL TIME ({mode}): {total_time:.2f}s")
