"""Convergence diagnostics: split-chain Rhat and bulk ESS via autocorrelation.

Implements:
  - compute_rhat(draws)       -- split-chain potential scale reduction factor
  - compute_ess(draws)        -- bulk ESS via autocorrelation sum
  - run_diagnostics(result)   -- runs Rhat+ESS on all GibbsResult parameters
  - save_diagnostics(diag, path) -- writes diagnostics dict to a JSON file
  - _get_draws_by_name(result, name) -- maps parameter name -> draws array
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def compute_rhat(draws: np.ndarray) -> float:
    """Split-chain Rhat.

    Splits a single chain of length N into two halves of length m = N // 2.
    Computes within-chain variance W and between-chain variance B, then
    returns sqrt(var_hat / W).  Returns 1.0 when both B and W are near zero
    (perfectly flat chains).
    """
    draws = np.asarray(draws, dtype=float)
    n = len(draws)
    if n < 4:
        return 1.0

    m = n // 2
    chain1 = draws[:m]
    chain2 = draws[n - m:]   # take last m so we cover the full range

    mean1 = np.mean(chain1)
    mean2 = np.mean(chain2)
    overall_mean = 0.5 * (mean1 + mean2)

    # Within-chain variance (average of per-chain sample variances)
    var1 = np.var(chain1, ddof=1)
    var2 = np.var(chain2, ddof=1)
    W = 0.5 * (var1 + var2)

    # Between-chain variance (scaled)
    B = m * 0.5 * ((mean1 - overall_mean) ** 2 + (mean2 - overall_mean) ** 2)

    # Marginal posterior variance estimator
    var_hat = (m - 1) / m * W + B / m

    if W < 1e-12 and B < 1e-12:
        return 1.0

    if W < 1e-12:
        # Degenerate: both chains constant but at different values
        return float("inf")

    return float(np.sqrt(var_hat / W))


def compute_ess(draws: np.ndarray, max_lag: int = 100) -> float:
    """Bulk ESS via autocorrelation.

    Computes autocorrelation at lags 1..max_lag, sums rho until rho < 0.05,
    and returns n / (1 + 2 * sum_rho).
    """
    draws = np.asarray(draws, dtype=float)
    n = len(draws)
    if n < 4:
        return float(n)

    mu = np.mean(draws)
    var = np.var(draws, ddof=1)
    if var < 1e-16:
        return float(n)

    centered = draws - mu

    # Compute autocorrelations via FFT for efficiency
    # Use numpy correlate for simplicity and correctness across all lags
    max_lag_eff = min(max_lag, n - 1)
    sum_rho = 0.0
    for lag in range(1, max_lag_eff + 1):
        rho = float(np.mean(centered[lag:] * centered[:n - lag]) / var)
        if rho < 0.05:
            break
        sum_rho += rho

    ess = n / (1.0 + 2.0 * sum_rho)
    return float(np.clip(ess, 1.0, n))


# ---------------------------------------------------------------------------
# Parameter name -> draws mapping
# ---------------------------------------------------------------------------

def _get_draws_by_name(result: Any, name: str) -> Optional[np.ndarray]:
    """Map a parameter name from parameter_summary() to its draws array.

    Handles scalar parameters (alpha, sigma_*, phi_*) and vector parameters
    (beta_prev_*, beta_treat_*, gamma_*, delta_*).
    """
    # Scalar draws
    scalar_map = {
        "alpha": result.alpha_draws,
        "sigma_prev2": result.sigma_prev2_draws,
        "sigma_treat2": result.sigma_treat2_draws,
        "sigma_obs_prev2": result.sigma_obs_prev2_draws,
        "sigma_obs_treat2": result.sigma_obs_treat2_draws,
        "sigma_u_prev2": result.sigma_u_prev2_draws,
        "sigma_u_treat2": result.sigma_u_treat2_draws,
        "phi_prev": getattr(result, "phi_prev_draws", None),
        "phi_treat": getattr(result, "phi_treat_draws", None),
        "sigma_ihd2": getattr(result, "sigma_ihd2_draws", None),
        "sigma_stroke2": getattr(result, "sigma_stroke2_draws", None),
        "sigma_v_ihd2": getattr(result, "sigma_v_ihd2_draws", None),
        "sigma_v_stroke2": getattr(result, "sigma_v_stroke2_draws", None),
    }
    if name in scalar_map:
        draws = scalar_map[name]
        if draws is not None and len(draws) > 0:
            return draws
        return None

    # Vector beta coefficients: beta_prev_<covariate> or beta_treat_<covariate>
    covariate_names = ["intercept", "log_gdp_z", "health_exp_z", "urban_z", "year_z"]
    for prefix, draws_matrix in [
        ("beta_prev_", result.beta_prev_draws),
        ("beta_treat_", result.beta_treat_draws),
    ]:
        if name.startswith(prefix):
            cname = name[len(prefix):]
            if cname in covariate_names:
                j = covariate_names.index(cname)
                if j < draws_matrix.shape[1]:
                    return draws_matrix[:, j]

    # Task 5A: CVD coefficient vectors gamma_<name>, delta_<name>
    cvd_coef_names = ["intercept", "prev", "treat", "gdp"]
    for prefix, attr_name in [
        ("gamma_", "gamma_draws"),
        ("delta_", "delta_draws"),
    ]:
        if name.startswith(prefix):
            cname = name[len(prefix):]
            if cname in cvd_coef_names:
                draws_matrix = getattr(result, attr_name, None)
                if draws_matrix is not None and draws_matrix.ndim == 2:
                    j = cvd_coef_names.index(cname)
                    if j < draws_matrix.shape[1]:
                        return draws_matrix[:, j]

    return None


# ---------------------------------------------------------------------------
# Run diagnostics on a full GibbsResult
# ---------------------------------------------------------------------------

def run_diagnostics(result: Any) -> Dict[str, Any]:
    """Run Rhat + ESS on all parameters from a GibbsResult.

    Returns a dict with:
      "parameters": {name -> {rhat, ess, status}}
      "overall_pass": bool
      "overall_status": "pass" | "warn" | "fail"
      "n_warn": int
      "n_fail": int

    Thresholds:
      - FAIL  if Rhat > 1.10
      - WARN  if Rhat > 1.05 OR ESS < 400
      - PASS  otherwise
    """
    summary = result.parameter_summary()
    param_names = summary["parameter"].tolist()

    params_out: Dict[str, Dict[str, Any]] = {}
    n_warn = 0
    n_fail = 0

    for name in param_names:
        draws = _get_draws_by_name(result, name)
        if draws is None:
            continue

        rhat = compute_rhat(draws)
        ess = compute_ess(draws)

        if rhat > 1.10:
            status = "fail"
            n_fail += 1
        elif rhat > 1.05 or ess < 400:
            status = "warn"
            n_warn += 1
        else:
            status = "pass"

        params_out[name] = {
            "rhat": round(float(rhat), 6),
            "ess": round(float(ess), 2),
            "status": status,
        }

    overall_pass = n_fail == 0
    if n_fail > 0:
        overall_status = "fail"
    elif n_warn > 0:
        overall_status = "warn"
    else:
        overall_status = "pass"

    return {
        "parameters": params_out,
        "overall_pass": overall_pass,
        "overall_status": overall_status,
        "n_warn": n_warn,
        "n_fail": n_fail,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_diagnostics(diag: Dict[str, Any], path: "str | Path") -> None:
    """Write diagnostics dict to a JSON file at *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(diag, fh, indent=2)
