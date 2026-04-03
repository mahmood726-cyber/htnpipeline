"""Counterfactual treatment coverage engine.

For each MCMC draw, computes counterfactual treatment coverage under
user-defined shift or target scenarios, then aggregates to posterior
summaries (mean, 2.5th, 97.5th percentile) per country-year cell.

Scenarios schema (list of dicts):
  {"name": str, "shift_pp": float}     -- add shift_pp to observed, cap at 100
  {"name": str, "target_pct": float}   -- take max(observed, target_pct), cap at 100
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_DEFAULT_SCENARIOS: List[Dict] = [
    {"name": "plus_25pp", "shift_pp": 25},
    {"name": "target_80", "target_pct": 80},
]


def compute_counterfactuals(
    result: Any,
    scenarios: Optional[List[Dict]] = None,
) -> pd.DataFrame:
    """Compute counterfactual treatment coverage per scenario and MCMC draw.

    Parameters
    ----------
    result:
        A GibbsResult instance.
    scenarios:
        List of scenario dicts.  Each dict must have "name" and either
        "shift_pp" (additive percentage-point shift) or "target_pct"
        (floor target).  If None, uses the two default scenarios.

    Returns
    -------
    pd.DataFrame with columns:
        iso3c, year,
        treatment_mean, treatment_low95, treatment_high95,  (posterior for observed)
        cf_treatment_<name>, cf_treatment_<name>_low95, cf_treatment_<name>_high95,
        cf_treatment_<name>_delta   (per scenario)
    """
    if scenarios is None:
        scenarios = _DEFAULT_SCENARIOS

    panel = result.panel.reset_index(drop=True)
    n_obs = len(panel)
    n_draws = result.latent_treat_draws.shape[0]

    # Baseline treatment draws: (n_draws, n_obs)
    # We use latent_treat_draws as the posterior treatment for each obs
    baseline_draws = result.latent_treat_draws  # (n_draws, n_obs)

    # Summarise baseline
    baseline_mean = np.mean(baseline_draws, axis=0)
    baseline_low95 = np.quantile(baseline_draws, 0.025, axis=0)
    baseline_high95 = np.quantile(baseline_draws, 0.975, axis=0)

    out = panel[["iso3c", "year"]].copy()
    out["treatment_mean"] = baseline_mean
    out["treatment_low95"] = baseline_low95
    out["treatment_high95"] = baseline_high95

    # Observed baseline (from panel) -- used for shift/target logic
    # The counterfactual operates on the observed coverage, then we add
    # the posterior uncertainty from the latent state draws.
    obs_treatment = panel["htn_treatment_pct"].values  # (n_obs,)

    for scenario in scenarios:
        name = scenario["name"]

        # For each draw, compute per-observation counterfactual coverage
        if "shift_pp" in scenario:
            shift = float(scenario["shift_pp"])
            # Counterfactual baseline (obs + shift, capped at 100)
            cf_obs = np.clip(obs_treatment + shift, 0.0, 100.0)  # (n_obs,)
            # Posterior: shift the entire draw distribution by the same amount
            cf_draws = np.clip(baseline_draws + shift, 0.0, 100.0)

        elif "target_pct" in scenario:
            target = float(scenario["target_pct"])
            # Counterfactual baseline: take max(obs, target), cap at 100
            cf_obs = np.clip(np.maximum(obs_treatment, target), 0.0, 100.0)
            # Posterior: for each obs, if observed < target, shift draws up by the gap
            gap = np.maximum(target - obs_treatment, 0.0)  # (n_obs,)
            cf_draws = np.clip(baseline_draws + gap[np.newaxis, :], 0.0, 100.0)

        else:
            raise ValueError(
                f"Scenario '{name}' must specify either 'shift_pp' or 'target_pct'."
            )

        cf_mean = np.mean(cf_draws, axis=0)
        cf_low95 = np.quantile(cf_draws, 0.025, axis=0)
        cf_high95 = np.quantile(cf_draws, 0.975, axis=0)
        cf_delta = cf_mean - baseline_mean

        out[f"cf_treatment_{name}"] = cf_mean
        out[f"cf_treatment_{name}_low95"] = cf_low95
        out[f"cf_treatment_{name}_high95"] = cf_high95
        out[f"cf_treatment_{name}_delta"] = cf_delta

    return out.reset_index(drop=True)
