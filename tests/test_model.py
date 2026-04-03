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
