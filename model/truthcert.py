"""TruthCert bundle generator.

Produces a self-contained audit bundle of 7 files:
  1. parameter_summary.csv
  2. posterior_draws.parquet
  3. counterfactual.csv
  4. convergence.json
  5. panel_fitted.csv
  6. coverage_summary.csv
  7. provenance.json

Usage:
    from model.truthcert import generate_bundle
    bundle = generate_bundle(result, raw_panel, bundle_dir="out/bundle", seed=42)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from model.counterfactual import compute_counterfactuals
from model.diagnostics import run_diagnostics, save_diagnostics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sha256_dataframe(df: pd.DataFrame) -> str:
    """Return SHA-256 hex digest of the canonical CSV representation of df."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def _build_posterior_draws_df(result: Any) -> pd.DataFrame:
    """Collect all scalar draws as columns in a single DataFrame."""
    n_draws = result.alpha_draws.shape[0]
    data: Dict[str, np.ndarray] = {}

    # Scalar chains
    data["alpha"] = result.alpha_draws
    data["sigma_prev2"] = result.sigma_prev2_draws
    data["sigma_treat2"] = result.sigma_treat2_draws
    data["sigma_obs_prev2"] = result.sigma_obs_prev2_draws
    data["sigma_obs_treat2"] = result.sigma_obs_treat2_draws
    data["sigma_u_prev2"] = result.sigma_u_prev2_draws
    data["sigma_u_treat2"] = result.sigma_u_treat2_draws

    # Beta coefficient chains
    covariate_names = ["intercept", "log_gdp_z", "health_exp_z", "urban_z", "year_z"]
    p_prev = result.beta_prev_draws.shape[1]
    p_treat = result.beta_treat_draws.shape[1]

    for j in range(p_prev):
        cname = covariate_names[j] if j < len(covariate_names) else f"x{j}"
        data[f"beta_prev_{cname}"] = result.beta_prev_draws[:, j]

    for j in range(p_treat):
        cname = covariate_names[j] if j < len(covariate_names) else f"x{j}"
        data[f"beta_treat_{cname}"] = result.beta_treat_draws[:, j]

    # Task 5B: AR(1) phi parameters
    if hasattr(result, "phi_prev_draws") and len(result.phi_prev_draws) > 0:
        data["phi_prev"] = result.phi_prev_draws
        data["phi_treat"] = result.phi_treat_draws

    # Task 5A: CVD parameters
    if getattr(result, "has_cvd", False):
        cvd_coef_names = ["intercept", "prev", "treat", "gdp"]
        for j, cname in enumerate(cvd_coef_names):
            data[f"gamma_{cname}"] = result.gamma_draws[:, j]
            data[f"delta_{cname}"] = result.delta_draws[:, j]
        data["sigma_ihd2"] = result.sigma_ihd2_draws
        data["sigma_stroke2"] = result.sigma_stroke2_draws
        data["sigma_v_ihd2"] = result.sigma_v_ihd2_draws
        data["sigma_v_stroke2"] = result.sigma_v_stroke2_draws

    data["draw_index"] = np.arange(n_draws)
    return pd.DataFrame(data)


def _build_panel_fitted(result: Any) -> pd.DataFrame:
    """Attach posterior means for latent prevalence and treatment to the panel."""
    panel = result.panel.copy()
    panel["latent_prev_mean"] = np.mean(result.latent_prev_draws, axis=0)
    panel["latent_prev_low95"] = np.quantile(result.latent_prev_draws, 0.025, axis=0)
    panel["latent_prev_high95"] = np.quantile(result.latent_prev_draws, 0.975, axis=0)
    panel["latent_treat_mean"] = np.mean(result.latent_treat_draws, axis=0)
    panel["latent_treat_low95"] = np.quantile(result.latent_treat_draws, 0.025, axis=0)
    panel["latent_treat_high95"] = np.quantile(result.latent_treat_draws, 0.975, axis=0)
    return panel.reset_index(drop=True)


def _build_coverage_summary(raw_panel: pd.DataFrame) -> pd.DataFrame:
    """Compute data completeness metrics per column."""
    rows = []
    n_total = len(raw_panel)
    for col in raw_panel.columns:
        n_non_null = int(raw_panel[col].notna().sum())
        rows.append({
            "column": col,
            "n_total": n_total,
            "n_non_null": n_non_null,
            "n_missing": n_total - n_non_null,
            "pct_complete": round(100.0 * n_non_null / max(n_total, 1), 2),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_bundle(
    result: Any,
    raw_panel: pd.DataFrame,
    bundle_dir: "str | Path",
    seed: int,
    scenarios: Optional[List[Dict]] = None,
) -> Path:
    """Generate a TruthCert bundle and write all 7 files to *bundle_dir*.

    Parameters
    ----------
    result:
        GibbsResult from StateSpaceGibbs.fit().
    raw_panel:
        The original (pre-fit) panel DataFrame — used for provenance hash
        and coverage summary.
    bundle_dir:
        Directory where bundle files are written (created if absent).
    seed:
        The PRNG seed used for the Gibbs run (recorded in provenance).
    scenarios:
        Optional counterfactual scenarios.  If None, defaults are used.

    Returns
    -------
    Path to bundle_dir.
    """
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. parameter_summary.csv
    # ------------------------------------------------------------------
    param_summary = result.parameter_summary()
    param_summary.to_csv(bundle_dir / "parameter_summary.csv", index=False)

    # ------------------------------------------------------------------
    # 2. posterior_draws.parquet
    # ------------------------------------------------------------------
    draws_df = _build_posterior_draws_df(result)
    draws_df.to_parquet(bundle_dir / "posterior_draws.parquet", index=False)

    # ------------------------------------------------------------------
    # 3. counterfactual.csv
    # ------------------------------------------------------------------
    cf_df = compute_counterfactuals(result, scenarios=scenarios)
    cf_df.to_csv(bundle_dir / "counterfactual.csv", index=False)

    # ------------------------------------------------------------------
    # 4. convergence.json
    # ------------------------------------------------------------------
    diag = run_diagnostics(result)
    save_diagnostics(diag, bundle_dir / "convergence.json")

    # ------------------------------------------------------------------
    # 5. panel_fitted.csv
    # ------------------------------------------------------------------
    fitted_df = _build_panel_fitted(result)
    fitted_df.to_csv(bundle_dir / "panel_fitted.csv", index=False)

    # ------------------------------------------------------------------
    # 6. coverage_summary.csv
    # ------------------------------------------------------------------
    coverage_df = _build_coverage_summary(raw_panel)
    coverage_df.to_csv(bundle_dir / "coverage_summary.csv", index=False)

    # ------------------------------------------------------------------
    # 7. provenance.json
    # ------------------------------------------------------------------
    data_hash = _sha256_dataframe(raw_panel)
    n_draws = result.alpha_draws.shape[0]
    unique_countries = raw_panel["iso3c"].nunique() if "iso3c" in raw_panel.columns else None
    unique_years = raw_panel["year"].nunique() if "year" in raw_panel.columns else None

    provenance = {
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_hash": data_hash,
        "n_countries": int(unique_countries) if unique_countries is not None else None,
        "n_years": int(unique_years) if unique_years is not None else None,
        "n_draws": int(n_draws),
        "convergence_status": diag["overall_status"],
        "n_warn": diag["n_warn"],
        "n_fail": diag["n_fail"],
    }

    with open(bundle_dir / "provenance.json", "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2)

    return bundle_dir
