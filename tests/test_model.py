"""Tests for the Bayesian state-space Gibbs sampler (model/gibbs.py).

7 tests covering: reproducibility (P0-2), variance sampling (P0-3),
link sign (P0-5), 95% CrI (P1-2), uniform thinning (P1-3).
"""

import numpy as np
import pandas as pd
import pytest

from model.gibbs import StateSpaceGibbs


def make_synthetic_panel(n_countries=5, n_years=5, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    true_beta_prev = np.array([30.0, -1.5, -0.5, 0.8, -0.2])
    true_beta_treat = np.array([40.0, 3.0, 1.5, 0.5, 0.5])
    true_alpha = 0.4
    countries = [f"C{i:02d}" for i in range(n_countries)]
    for c_idx, iso in enumerate(countries):
        u_prev = rng.normal(0, 2)
        u_treat = rng.normal(0, 3)
        for y in range(n_years):
            gdp_z = rng.normal(0, 1)
            health_z = rng.normal(0, 1)
            urban_z = rng.normal(0, 1)
            year_z = (y - n_years / 2) / max(n_years, 1)
            x = np.array([1.0, gdp_z, health_z, urban_z, year_z])
            prev_true = x @ true_beta_prev + u_prev + rng.normal(0, 1.5)
            treat_true = (
                x @ true_beta_treat
                + true_alpha * prev_true
                + u_treat
                + rng.normal(0, 2.0)
            )
            rows.append({
                "iso3c": iso,
                "year": 2015 + y,
                "country_id": c_idx,
                "log_gdp_z": gdp_z,
                "health_exp_z": health_z,
                "urban_z": urban_z,
                "year_z": year_z,
                "htn_prevalence_pct": prev_true + rng.normal(0, 1),
                "htn_treatment_pct": treat_true + rng.normal(0, 1),
            })
    return pd.DataFrame(rows)


class TestGibbsReproducibility:
    def test_same_seed_identical_output(self):
        df = make_synthetic_panel()
        g1 = StateSpaceGibbs(seed=2026, n_iter=100, burn=30, thin=2)
        g2 = StateSpaceGibbs(seed=2026, n_iter=100, burn=30, thin=2)
        r1 = g1.fit(df)
        r2 = g2.fit(df)
        np.testing.assert_array_equal(r1.beta_prev_draws, r2.beta_prev_draws)

    def test_different_seeds_different_output(self):
        df = make_synthetic_panel()
        g1 = StateSpaceGibbs(seed=1, n_iter=100, burn=30, thin=2)
        g2 = StateSpaceGibbs(seed=2, n_iter=100, burn=30, thin=2)
        r1 = g1.fit(df)
        r2 = g2.fit(df)
        assert not np.array_equal(r1.beta_prev_draws, r2.beta_prev_draws)


class TestGibbsVarianceSampling:
    def test_variance_params_change_across_draws(self):
        df = make_synthetic_panel()
        g = StateSpaceGibbs(seed=42, n_iter=200, burn=50, thin=1)
        r = g.fit(df)
        assert r.sigma_prev2_draws.std() > 0.01

    def test_variance_mean_reasonable(self):
        df = make_synthetic_panel(n_countries=10, n_years=5)
        g = StateSpaceGibbs(seed=42, n_iter=500, burn=200, thin=1)
        r = g.fit(df)
        assert 0.1 < r.sigma_prev2_draws.mean() < 50.0


class TestGibbsLinkSign:
    def test_positive_link_on_positive_data(self):
        df = make_synthetic_panel(n_countries=10, n_years=5, seed=42)
        g = StateSpaceGibbs(seed=42, n_iter=500, burn=200, thin=1)
        r = g.fit(df)
        assert r.alpha_draws.mean() > 0.0


class TestGibbsCrI:
    def test_95_percent_cri(self):
        df = make_synthetic_panel()
        g = StateSpaceGibbs(seed=42, n_iter=200, burn=50, thin=1)
        r = g.fit(df)
        summary = r.parameter_summary()
        assert "low95" in summary.columns
        assert "high95" in summary.columns


class TestGibbsUniformThinning:
    def test_all_draw_arrays_same_length(self):
        df = make_synthetic_panel()
        g = StateSpaceGibbs(seed=42, n_iter=200, burn=50, thin=2)
        r = g.fit(df)
        n = len(r.beta_prev_draws)
        assert len(r.beta_treat_draws) == n
        assert len(r.alpha_draws) == n
        assert len(r.sigma_prev2_draws) == n
        assert len(r.latent_prev_draws) == n


# ---------------------------------------------------------------------------
# Task 6: Convergence diagnostics
# ---------------------------------------------------------------------------

from model.diagnostics import compute_rhat, compute_ess, run_diagnostics
from model.counterfactual import compute_counterfactuals
from model.truthcert import generate_bundle


class TestRhat:
    def test_good_chain_near_one(self):
        rng = np.random.default_rng(42)
        draws = rng.normal(0, 1, 1000)
        assert abs(compute_rhat(draws) - 1.0) < 0.05

    def test_bad_chain_above_threshold(self):
        draws = np.concatenate([np.random.default_rng(1).normal(-5, 0.1, 500),
                                np.random.default_rng(2).normal(5, 0.1, 500)])
        assert compute_rhat(draws) > 1.1


class TestESS:
    def test_iid_draws_high_ess(self):
        rng = np.random.default_rng(42)
        draws = rng.normal(0, 1, 1000)
        assert compute_ess(draws) > 800

    def test_autocorrelated_draws_low_ess(self):
        rng = np.random.default_rng(42)
        draws = np.zeros(1000)
        draws[0] = rng.normal()
        for i in range(1, 1000):
            draws[i] = 0.95 * draws[i-1] + rng.normal(0, 0.1)
        assert compute_ess(draws) < 200


class TestDiagnostics:
    def test_run_diagnostics_returns_dict(self):
        df = make_synthetic_panel()
        g = StateSpaceGibbs(seed=42, n_iter=200, burn=50, thin=1)
        result = g.fit(df)
        diag = run_diagnostics(result)
        assert "parameters" in diag
        assert "overall_pass" in diag


class TestCounterfactual:
    def _get_result(self):
        df = make_synthetic_panel(n_countries=5, n_years=3)
        g = StateSpaceGibbs(seed=42, n_iter=200, burn=50, thin=1)
        return g.fit(df)

    def test_treatment_capped_at_100(self):
        result = self._get_result()
        cf = compute_counterfactuals(result, scenarios=[{"name": "plus_25pp", "shift_pp": 25}])
        assert (cf["cf_treatment_plus_25pp"] <= 100.0).all()

    def test_cri_width_positive(self):
        result = self._get_result()
        cf = compute_counterfactuals(result, scenarios=[{"name": "plus_25pp", "shift_pp": 25}])
        assert (cf["cf_treatment_plus_25pp_high95"] >= cf["cf_treatment_plus_25pp_low95"]).all()

    def test_custom_shift_works(self):
        result = self._get_result()
        cf = compute_counterfactuals(result, scenarios=[{"name": "custom_10", "shift_pp": 10}])
        assert "cf_treatment_custom_10" in cf.columns

    def test_target_80_scenario(self):
        result = self._get_result()
        cf = compute_counterfactuals(result, scenarios=[{"name": "target_80", "target_pct": 80}])
        assert "cf_treatment_target_80" in cf.columns


class TestTruthCertBundle:
    def test_all_files_present(self, tmp_path):
        df = make_synthetic_panel()
        g = StateSpaceGibbs(seed=42, n_iter=100, burn=30, thin=1)
        result = g.fit(df)
        generate_bundle(result, df, bundle_dir=tmp_path, seed=42)
        expected_files = [
            "parameter_summary.csv", "posterior_draws.parquet",
            "counterfactual.csv", "convergence.json",
            "provenance.json", "panel_fitted.csv", "coverage_summary.csv",
        ]
        for fname in expected_files:
            assert (tmp_path / fname).exists(), f"Missing: {fname}"

    def test_provenance_has_required_fields(self, tmp_path):
        df = make_synthetic_panel()
        g = StateSpaceGibbs(seed=42, n_iter=100, burn=30, thin=1)
        result = g.fit(df)
        generate_bundle(result, df, bundle_dir=tmp_path, seed=42)
        import json
        with open(tmp_path / "provenance.json") as f:
            prov = json.load(f)
        assert "seed" in prov
        assert "timestamp" in prov
        assert "data_hash" in prov
        assert "n_countries" in prov
        assert prov["seed"] == 42
