# Bayesian Hybrid Contextual Bandit Thesis

Simulation study for a personalized contextual multi-armed bandit in ad personalization.

Active agents:
- `random`
- `probit_ts`
- `bayes_fm_ts`
- `hybrid` (proposed)

The simulator now uses persistent users, user-level cold start, and abrupt preference shifts.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## Main Runs

```bash
python -m experiments.run_stationary
python -m experiments.run_nonstationary
```

## Validation

```bash
python experiments/validation/step0_verify_design.py
python experiments/validation/step1_metrics_report.py
python experiments/validation/step2_significance_tests.py
python experiments/validation/step3_effect_size.py
python experiments/validation/step4_personalization_conditional.py
python experiments/validation/step5_plots.py
```

`step2_significance_tests.py` is the main inferential workflow. It:
- computes paired seed-wise differences for the main comparisons,
- checks normality with Shapiro-Wilk,
- saves histogram and Q-Q diagnostics,
- chooses the primary paired test (`paired_ttest` or `wilcoxon`) from the normality result,
- reports bootstrap confidence intervals and effect sizes,
- and applies Holm correction to the primary p-values.

`step1_metrics_report.py` rebuilds the thesis-facing summary tables directly from raw per-seed logs and refreshes the canonical summaries under `results/tables/`.

`step3_effect_size.py` exports a compact thesis-ready summary table from the step 2 results.

Outputs are written under `results/`:
- `results/raw/`
- `results/tables/`
- `results/validation/`
- `results/figures/`
- `results/manifests/`
