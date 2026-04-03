"""Bayesian state-space Gibbs sampler with full conditionals.

Fixes: P0-1 (latent refresh), P0-2 (seeded RNG), P0-3 (variance sampling),
       P0-5 (link sign), P1-2 (95% CrI), P1-3 (uniform thinning),
       P1-6 (GDP prior direction).
Tasks: 5A (CVD outcome sub-model), 5B (AR(1) phi via MH).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class GibbsResult:
    """Container for posterior draws from the Gibbs sampler."""

    beta_prev_draws: np.ndarray          # (n_draws, p)
    beta_treat_draws: np.ndarray         # (n_draws, p)
    alpha_draws: np.ndarray              # (n_draws,)
    sigma_prev2_draws: np.ndarray        # (n_draws,)
    sigma_treat2_draws: np.ndarray       # (n_draws,)
    sigma_obs_prev2_draws: np.ndarray    # (n_draws,)
    sigma_obs_treat2_draws: np.ndarray   # (n_draws,)
    sigma_u_prev2_draws: np.ndarray      # (n_draws,)
    sigma_u_treat2_draws: np.ndarray     # (n_draws,)
    latent_prev_draws: np.ndarray        # (n_draws, n_obs)
    latent_treat_draws: np.ndarray       # (n_draws, n_obs)
    country_ids: np.ndarray
    iso3c_map: Dict[int, str]
    panel: pd.DataFrame

    # Task 5B: AR(1) autoregressive coefficients
    phi_prev_draws: np.ndarray = field(default_factory=lambda: np.array([]))
    phi_treat_draws: np.ndarray = field(default_factory=lambda: np.array([]))

    # Task 5A: CVD outcome sub-model
    has_cvd: bool = False
    gamma_draws: np.ndarray = field(default_factory=lambda: np.array([]))   # (n_draws, 4)
    delta_draws: np.ndarray = field(default_factory=lambda: np.array([]))   # (n_draws, 4)
    sigma_ihd2_draws: np.ndarray = field(default_factory=lambda: np.array([]))
    sigma_stroke2_draws: np.ndarray = field(default_factory=lambda: np.array([]))
    sigma_v_ihd2_draws: np.ndarray = field(default_factory=lambda: np.array([]))
    sigma_v_stroke2_draws: np.ndarray = field(default_factory=lambda: np.array([]))

    def parameter_summary(self) -> pd.DataFrame:
        """Return DataFrame with mean, median, sd, low95, high95 for each parameter."""
        rows = []

        def _add_scalar(name: str, draws: np.ndarray) -> None:
            rows.append({
                "parameter": name,
                "mean": float(np.mean(draws)),
                "median": float(np.median(draws)),
                "sd": float(np.std(draws, ddof=1)),
                "low95": float(np.quantile(draws, 0.025)),
                "high95": float(np.quantile(draws, 0.975)),
            })

        # Beta prevalence coefficients
        p = self.beta_prev_draws.shape[1]
        covariate_names = ["intercept", "log_gdp_z", "health_exp_z", "urban_z", "year_z"][:p]
        for j, cname in enumerate(covariate_names):
            _add_scalar(f"beta_prev_{cname}", self.beta_prev_draws[:, j])
        for j, cname in enumerate(covariate_names):
            _add_scalar(f"beta_treat_{cname}", self.beta_treat_draws[:, j])

        _add_scalar("alpha", self.alpha_draws)
        _add_scalar("sigma_prev2", self.sigma_prev2_draws)
        _add_scalar("sigma_treat2", self.sigma_treat2_draws)
        _add_scalar("sigma_obs_prev2", self.sigma_obs_prev2_draws)
        _add_scalar("sigma_obs_treat2", self.sigma_obs_treat2_draws)
        _add_scalar("sigma_u_prev2", self.sigma_u_prev2_draws)
        _add_scalar("sigma_u_treat2", self.sigma_u_treat2_draws)

        # Task 5B: AR(1) phi parameters
        if len(self.phi_prev_draws) > 0:
            _add_scalar("phi_prev", self.phi_prev_draws)
            _add_scalar("phi_treat", self.phi_treat_draws)

        # Task 5A: CVD parameters (only if CVD data was available)
        if self.has_cvd:
            cvd_coef_names = ["intercept", "prev", "treat", "gdp"]
            for j, cname in enumerate(cvd_coef_names):
                _add_scalar(f"gamma_{cname}", self.gamma_draws[:, j])
            for j, cname in enumerate(cvd_coef_names):
                _add_scalar(f"delta_{cname}", self.delta_draws[:, j])
            _add_scalar("sigma_ihd2", self.sigma_ihd2_draws)
            _add_scalar("sigma_stroke2", self.sigma_stroke2_draws)
            _add_scalar("sigma_v_ihd2", self.sigma_v_ihd2_draws)
            _add_scalar("sigma_v_stroke2", self.sigma_v_stroke2_draws)

        return pd.DataFrame(rows)


def _compute_ar_residuals(
    latent: np.ndarray,
    country_id: np.ndarray,
    phi: float,
    mu: np.ndarray,
    n_country: int,
) -> np.ndarray:
    """Compute AR(1) residuals for each country's time series (Task 5B)."""
    resid = np.empty_like(latent)
    for c in range(n_country):
        idx = np.where(country_id == c)[0]
        resid[idx[0]] = latent[idx[0]] - mu[idx[0]]
        for i in range(1, len(idx)):
            resid[idx[i]] = latent[idx[i]] - mu[idx[i]] - phi * latent[idx[i - 1]]
    return resid


def _compute_ar_loglik(resid: np.ndarray, sigma2: float) -> float:
    """Log-likelihood of AR(1) residuals under N(0, sigma2)."""
    n = len(resid)
    return -0.5 * n * np.log(2 * np.pi * sigma2) - 0.5 * np.sum(resid**2) / sigma2


class StateSpaceGibbs:
    """Bayesian state-space Gibbs sampler for HTN prevalence-treatment model.

    All random draws use a seeded numpy Generator (P0-2).
    All variances are sampled every iteration from InvGamma (P0-3).
    Latent states use precision-weighted full conditionals (P0-1).
    Alpha prior is positive 0.3 (P0-5).
    95% CrI at 0.025/0.975 quantiles (P1-2).
    Uniform thinning across all draw types (P1-3).
    GDP beta prior centred near 0 (P1-6).
    Tasks 5A (CVD outcome sub-model), 5B (AR(1) phi via MH).
    """

    def __init__(
        self,
        seed: int = 2026,
        n_iter: int = 3000,
        burn: int = 1000,
        thin: int = 2,
    ) -> None:
        self.seed = seed
        self.n_iter = n_iter
        self.burn = burn
        self.thin = thin

    def fit(self, df: pd.DataFrame) -> GibbsResult:
        """Run the Gibbs sampler on a harmonised panel DataFrame."""
        rng = np.random.default_rng(self.seed)

        # ------------------------------------------------------------------
        # 1. Prepare data
        # ------------------------------------------------------------------
        required_cols = [
            "iso3c", "htn_prevalence_pct", "htn_treatment_pct",
            "log_gdp_z", "health_exp_z", "year_z",
        ]
        df_work = df.dropna(subset=[c for c in required_cols if c in df.columns]).copy()
        df_work = df_work.reset_index(drop=True)

        n_obs = len(df_work)
        y_prev_obs = df_work["htn_prevalence_pct"].values.astype(np.float64)
        y_treat_obs = df_work["htn_treatment_pct"].values.astype(np.float64)

        # Design matrix X: intercept + covariates
        has_urban = "urban_z" in df_work.columns and df_work["urban_z"].notna().all()
        x_cols = ["log_gdp_z", "health_exp_z"]
        if has_urban:
            x_cols.append("urban_z")
        x_cols.append("year_z")

        X = np.column_stack([
            np.ones(n_obs),
            *[df_work[c].values.astype(np.float64) for c in x_cols],
        ])
        p = X.shape[1]

        # Country IDs
        iso_codes = df_work["iso3c"].values
        unique_iso, country_id = np.unique(iso_codes, return_inverse=True)
        n_country = len(unique_iso)
        iso3c_map = {i: str(c) for i, c in enumerate(unique_iso)}

        # Task 5A: Check for CVD outcome columns
        has_cvd = (
            "ihd_mort_rate" in df_work.columns
            and "stroke_mort_rate" in df_work.columns
            and df_work["ihd_mort_rate"].notna().any()
            and df_work["stroke_mort_rate"].notna().any()
        )
        if has_cvd:
            y_ihd = df_work["ihd_mort_rate"].values.astype(np.float64)
            y_stroke = df_work["stroke_mort_rate"].values.astype(np.float64)
            # log_gdp_z is already in df_work
            log_gdp_z_vec = df_work["log_gdp_z"].values.astype(np.float64)

        # ------------------------------------------------------------------
        # 2. Initialise
        # ------------------------------------------------------------------
        # Prior means for beta_prev (P1-6: GDP prior near 0)
        beta_prev_prior_mean = np.zeros(p)
        beta_prev_prior_var = np.zeros(p)
        _prev_defaults = [
            (30.0, 10.0**2),   # intercept
            (0.0, 3.0**2),     # log_gdp_z (P1-6: near zero)
            (-0.5, 3.0**2),    # health_exp_z
            (1.0, 3.0**2),     # urban_z
            (-0.1, 2.0**2),    # year_z
        ]
        # If no urban_z, skip index 3
        if has_urban:
            idx_map = list(range(p))
        else:
            idx_map = [0, 1, 2, 4]  # skip urban_z (index 3)
        for j_local, j_full in enumerate(idx_map[:p]):
            beta_prev_prior_mean[j_local] = _prev_defaults[j_full][0]
            beta_prev_prior_var[j_local] = _prev_defaults[j_full][1]
        Sigma_beta_prev_prior_inv = np.diag(1.0 / beta_prev_prior_var)

        # Prior means for beta_treat
        beta_treat_prior_mean = np.zeros(p)
        beta_treat_prior_var = np.zeros(p)
        _treat_defaults = [
            (40.0, 12.0**2),
            (2.0, 5.0**2),
            (1.0, 5.0**2),
            (0.5, 5.0**2),
            (0.5, 2.0**2),
        ]
        for j_local, j_full in enumerate(idx_map[:p]):
            beta_treat_prior_mean[j_local] = _treat_defaults[j_full][0]
            beta_treat_prior_var[j_local] = _treat_defaults[j_full][1]
        Sigma_beta_treat_prior_inv = np.diag(1.0 / beta_treat_prior_var)

        # Alpha prior (P0-5: positive)
        alpha_prior_mean = 0.3
        alpha_prior_var = 0.5**2

        # InvGamma hyperparams
        a0 = 3.0
        b0 = 1.0

        # Current state
        beta_prev = beta_prev_prior_mean.copy()
        beta_treat = beta_treat_prior_mean.copy()
        alpha = 0.3

        sigma_prev2 = 4.0
        sigma_treat2 = 6.0
        sigma_obs_prev2 = 2.0
        sigma_obs_treat2 = 2.0
        sigma_u_prev2 = 4.0
        sigma_u_treat2 = 4.0

        # Country random effects
        u_prev = np.zeros(n_country)
        u_treat = np.zeros(n_country)

        # Latent states initialised to observed (P0-1)
        latent_prev = y_prev_obs.copy()
        latent_treat = y_treat_obs.copy()

        # Task 5B: AR(1) phi parameters initialised to 0
        phi_prev = 0.0
        phi_treat = 0.0

        # Task 5A: CVD sub-model initialisation
        if has_cvd:
            gamma = np.zeros(4)   # [intercept, prev, treat, gdp]
            delta = np.zeros(4)
            v_ihd = np.zeros(n_country)
            v_stroke = np.zeros(n_country)
            sigma_ihd2 = 1.0
            sigma_stroke2 = 1.0
            sigma_v_ihd2 = 1.0
            sigma_v_stroke2 = 1.0
            # Weakly informative prior for gamma/delta: MVN(0, diag(5^2))
            cvd_prior_prec = np.diag(np.full(4, 1.0 / 25.0))

        # ------------------------------------------------------------------
        # 3. Storage for draws (uniform thinning, P1-3)
        # ------------------------------------------------------------------
        n_keep = 0
        for it in range(self.n_iter):
            if it >= self.burn and (it - self.burn) % self.thin == 0:
                n_keep += 1

        beta_prev_draws = np.empty((n_keep, p))
        beta_treat_draws = np.empty((n_keep, p))
        alpha_draws = np.empty(n_keep)
        sigma_prev2_draws = np.empty(n_keep)
        sigma_treat2_draws = np.empty(n_keep)
        sigma_obs_prev2_draws = np.empty(n_keep)
        sigma_obs_treat2_draws = np.empty(n_keep)
        sigma_u_prev2_draws = np.empty(n_keep)
        sigma_u_treat2_draws = np.empty(n_keep)
        latent_prev_draws = np.empty((n_keep, n_obs))
        latent_treat_draws = np.empty((n_keep, n_obs))

        # Task 5B: AR(1) phi draws
        phi_prev_draws = np.empty(n_keep)
        phi_treat_draws = np.empty(n_keep)

        # Task 5A: CVD draws
        if has_cvd:
            gamma_draws = np.empty((n_keep, 4))
            delta_draws = np.empty((n_keep, 4))
            sigma_ihd2_draws = np.empty(n_keep)
            sigma_stroke2_draws = np.empty(n_keep)
            sigma_v_ihd2_draws = np.empty(n_keep)
            sigma_v_stroke2_draws = np.empty(n_keep)

        draw_idx = 0

        # Precompute X'X
        XtX = X.T @ X

        # ------------------------------------------------------------------
        # 4. Main Gibbs loop
        # ------------------------------------------------------------------
        for it in range(self.n_iter):

            # (a) Sample country effects u_prev, u_treat
            for c in range(n_country):
                mask_c = country_id == c
                n_c = mask_c.sum()

                # u_prev
                resid_prev_c = latent_prev[mask_c] - X[mask_c] @ beta_prev
                prec_u_prev = n_c / sigma_prev2 + 1.0 / sigma_u_prev2
                var_u_prev = 1.0 / prec_u_prev
                mean_u_prev = var_u_prev * (resid_prev_c.sum() / sigma_prev2)
                u_prev[c] = rng.normal(mean_u_prev, np.sqrt(var_u_prev))

                # u_treat
                resid_treat_c = (
                    latent_treat[mask_c]
                    - X[mask_c] @ beta_treat
                    - alpha * latent_prev[mask_c]
                    - 0  # u_treat not yet updated for this c
                )
                prec_u_treat = n_c / sigma_treat2 + 1.0 / sigma_u_treat2
                var_u_treat = 1.0 / prec_u_treat
                mean_u_treat = var_u_treat * (resid_treat_c.sum() / sigma_treat2)
                u_treat[c] = rng.normal(mean_u_treat, np.sqrt(var_u_treat))

            # (b) Sample beta_prev
            y_prev_adj = latent_prev - u_prev[country_id]
            post_prec_prev = Sigma_beta_prev_prior_inv + XtX / sigma_prev2
            post_cov_prev = np.linalg.inv(post_prec_prev)
            post_mean_prev = post_cov_prev @ (
                Sigma_beta_prev_prior_inv @ beta_prev_prior_mean
                + X.T @ y_prev_adj / sigma_prev2
            )
            beta_prev = rng.multivariate_normal(post_mean_prev, post_cov_prev)

            # (c) Sample beta_treat
            y_treat_adj = latent_treat - alpha * latent_prev - u_treat[country_id]
            post_prec_treat = Sigma_beta_treat_prior_inv + XtX / sigma_treat2
            post_cov_treat = np.linalg.inv(post_prec_treat)
            post_mean_treat = post_cov_treat @ (
                Sigma_beta_treat_prior_inv @ beta_treat_prior_mean
                + X.T @ y_treat_adj / sigma_treat2
            )
            beta_treat = rng.multivariate_normal(post_mean_treat, post_cov_treat)

            # (d) Sample alpha (scalar normal)
            x_alpha = latent_prev
            y_alpha = latent_treat - X @ beta_treat - u_treat[country_id]
            xtx_alpha = np.dot(x_alpha, x_alpha)
            xty_alpha = np.dot(x_alpha, y_alpha)
            prec_alpha = 1.0 / alpha_prior_var + xtx_alpha / sigma_treat2
            var_alpha = 1.0 / prec_alpha
            mean_alpha = var_alpha * (
                alpha_prior_mean / alpha_prior_var + xty_alpha / sigma_treat2
            )
            alpha = rng.normal(mean_alpha, np.sqrt(var_alpha))

            # (e) Sample all 6 variances from InvGamma (P0-3)
            def _sample_inv_gamma(resid: np.ndarray) -> float:
                n = len(resid)
                a_post = a0 + n / 2.0
                b_post = b0 + np.sum(resid**2) / 2.0
                # InvGamma: draw gamma from Gamma(a_post, 1/b_post), invert
                return 1.0 / rng.gamma(a_post, 1.0 / b_post)

            # sigma_prev2
            resid_prev = latent_prev - X @ beta_prev - u_prev[country_id]
            sigma_prev2 = _sample_inv_gamma(resid_prev)

            # sigma_treat2
            resid_treat = (
                latent_treat - X @ beta_treat - alpha * latent_prev
                - u_treat[country_id]
            )
            sigma_treat2 = _sample_inv_gamma(resid_treat)

            # sigma_obs_prev2
            resid_obs_prev = y_prev_obs - latent_prev
            sigma_obs_prev2 = _sample_inv_gamma(resid_obs_prev)

            # sigma_obs_treat2
            resid_obs_treat = y_treat_obs - latent_treat
            sigma_obs_treat2 = _sample_inv_gamma(resid_obs_treat)

            # sigma_u_prev2
            sigma_u_prev2 = _sample_inv_gamma(u_prev)

            # sigma_u_treat2
            sigma_u_treat2 = _sample_inv_gamma(u_treat)

            # (f) Sample latent states (P0-1: precision-weighted full conditional)
            # Latent prevalence — include AR(1) lag term (Task 5B)
            mu_struct_prev = X @ beta_prev + u_prev[country_id]
            # Add phi_prev * lag for each country (except first obs)
            prev_lag = np.zeros(n_obs)
            for c in range(n_country):
                idx_c = np.where(country_id == c)[0]
                for i in range(1, len(idx_c)):
                    prev_lag[idx_c[i]] = latent_prev[idx_c[i - 1]]
            mu_struct_prev = mu_struct_prev + phi_prev * prev_lag

            prec_obs_prev = 1.0 / sigma_obs_prev2
            prec_struct_prev = 1.0 / sigma_prev2
            var_latent_prev = 1.0 / (prec_obs_prev + prec_struct_prev)
            mean_latent_prev = var_latent_prev * (
                prec_obs_prev * y_prev_obs + prec_struct_prev * mu_struct_prev
            )
            latent_prev = rng.normal(mean_latent_prev, np.sqrt(var_latent_prev))

            # Latent treatment — include AR(1) lag term + alpha*latent_prev
            mu_struct_treat = (
                X @ beta_treat + alpha * latent_prev + u_treat[country_id]
            )
            treat_lag = np.zeros(n_obs)
            for c in range(n_country):
                idx_c = np.where(country_id == c)[0]
                for i in range(1, len(idx_c)):
                    treat_lag[idx_c[i]] = latent_treat[idx_c[i - 1]]
            mu_struct_treat = mu_struct_treat + phi_treat * treat_lag

            prec_obs_treat = 1.0 / sigma_obs_treat2
            prec_struct_treat = 1.0 / sigma_treat2
            var_latent_treat = 1.0 / (prec_obs_treat + prec_struct_treat)
            mean_latent_treat = var_latent_treat * (
                prec_obs_treat * y_treat_obs + prec_struct_treat * mu_struct_treat
            )
            latent_treat = rng.normal(mean_latent_treat, np.sqrt(var_latent_treat))

            # (f2) Task 5B: MH step for phi_prev
            mu_base_prev = X @ beta_prev + u_prev[country_id]
            phi_prev_prop = rng.normal(phi_prev, 0.05)
            if -0.95 < phi_prev_prop < 0.95:
                resid_cur = _compute_ar_residuals(
                    latent_prev, country_id, phi_prev, mu_base_prev, n_country,
                )
                resid_prop = _compute_ar_residuals(
                    latent_prev, country_id, phi_prev_prop, mu_base_prev, n_country,
                )
                ll_cur = _compute_ar_loglik(resid_cur, sigma_prev2)
                ll_prop = _compute_ar_loglik(resid_prop, sigma_prev2)
                if np.log(rng.uniform()) < (ll_prop - ll_cur):
                    phi_prev = phi_prev_prop

            # MH step for phi_treat
            mu_base_treat = (
                X @ beta_treat + alpha * latent_prev + u_treat[country_id]
            )
            phi_treat_prop = rng.normal(phi_treat, 0.05)
            if -0.95 < phi_treat_prop < 0.95:
                resid_cur_t = _compute_ar_residuals(
                    latent_treat, country_id, phi_treat, mu_base_treat, n_country,
                )
                resid_prop_t = _compute_ar_residuals(
                    latent_treat, country_id, phi_treat_prop, mu_base_treat, n_country,
                )
                ll_cur_t = _compute_ar_loglik(resid_cur_t, sigma_treat2)
                ll_prop_t = _compute_ar_loglik(resid_prop_t, sigma_treat2)
                if np.log(rng.uniform()) < (ll_prop_t - ll_cur_t):
                    phi_treat = phi_treat_prop

            # (f3) Task 5A: CVD outcome sub-model sampling
            if has_cvd:
                # Build CVD design matrix: [ones, latent_prev, latent_treat, log_gdp_z]
                X_cvd = np.column_stack([
                    np.ones(n_obs), latent_prev, latent_treat, log_gdp_z_vec,
                ])
                XcXc = X_cvd.T @ X_cvd

                # --- IHD: sample gamma ---
                y_ihd_adj = y_ihd - v_ihd[country_id]
                post_prec_gamma = cvd_prior_prec + XcXc / sigma_ihd2
                post_cov_gamma = np.linalg.inv(post_prec_gamma)
                post_mean_gamma = post_cov_gamma @ (
                    X_cvd.T @ y_ihd_adj / sigma_ihd2
                    # prior mean is 0, so prior_prec @ prior_mean = 0
                )
                gamma = rng.multivariate_normal(post_mean_gamma, post_cov_gamma)

                # Sample v_ihd[c] (country RE for IHD)
                for c in range(n_country):
                    mask_c = country_id == c
                    n_c = mask_c.sum()
                    resid_ihd_c = y_ihd[mask_c] - X_cvd[mask_c] @ gamma
                    prec_v = n_c / sigma_ihd2 + 1.0 / sigma_v_ihd2
                    var_v = 1.0 / prec_v
                    mean_v = var_v * (resid_ihd_c.sum() / sigma_ihd2)
                    v_ihd[c] = rng.normal(mean_v, np.sqrt(var_v))

                # sigma_ihd2
                resid_ihd = y_ihd - X_cvd @ gamma - v_ihd[country_id]
                sigma_ihd2 = _sample_inv_gamma(resid_ihd)

                # sigma_v_ihd2
                sigma_v_ihd2 = _sample_inv_gamma(v_ihd)

                # --- Stroke: sample delta ---
                y_stroke_adj = y_stroke - v_stroke[country_id]
                post_prec_delta = cvd_prior_prec + XcXc / sigma_stroke2
                post_cov_delta = np.linalg.inv(post_prec_delta)
                post_mean_delta = post_cov_delta @ (
                    X_cvd.T @ y_stroke_adj / sigma_stroke2
                )
                delta = rng.multivariate_normal(post_mean_delta, post_cov_delta)

                # Sample v_stroke[c]
                for c in range(n_country):
                    mask_c = country_id == c
                    n_c = mask_c.sum()
                    resid_stroke_c = y_stroke[mask_c] - X_cvd[mask_c] @ delta
                    prec_v = n_c / sigma_stroke2 + 1.0 / sigma_v_stroke2
                    var_v = 1.0 / prec_v
                    mean_v = var_v * (resid_stroke_c.sum() / sigma_stroke2)
                    v_stroke[c] = rng.normal(mean_v, np.sqrt(var_v))

                # sigma_stroke2
                resid_stroke = y_stroke - X_cvd @ delta - v_stroke[country_id]
                sigma_stroke2 = _sample_inv_gamma(resid_stroke)

                # sigma_v_stroke2
                sigma_v_stroke2 = _sample_inv_gamma(v_stroke)

            # (g) Store draws: after burn-in, every thin iterations (P1-3)
            if it >= self.burn and (it - self.burn) % self.thin == 0:
                beta_prev_draws[draw_idx] = beta_prev
                beta_treat_draws[draw_idx] = beta_treat
                alpha_draws[draw_idx] = alpha
                sigma_prev2_draws[draw_idx] = sigma_prev2
                sigma_treat2_draws[draw_idx] = sigma_treat2
                sigma_obs_prev2_draws[draw_idx] = sigma_obs_prev2
                sigma_obs_treat2_draws[draw_idx] = sigma_obs_treat2
                sigma_u_prev2_draws[draw_idx] = sigma_u_prev2
                sigma_u_treat2_draws[draw_idx] = sigma_u_treat2
                latent_prev_draws[draw_idx] = latent_prev
                latent_treat_draws[draw_idx] = latent_treat
                phi_prev_draws[draw_idx] = phi_prev
                phi_treat_draws[draw_idx] = phi_treat
                if has_cvd:
                    gamma_draws[draw_idx] = gamma
                    delta_draws[draw_idx] = delta
                    sigma_ihd2_draws[draw_idx] = sigma_ihd2
                    sigma_stroke2_draws[draw_idx] = sigma_stroke2
                    sigma_v_ihd2_draws[draw_idx] = sigma_v_ihd2
                    sigma_v_stroke2_draws[draw_idx] = sigma_v_stroke2
                draw_idx += 1

        # ------------------------------------------------------------------
        # 5. Return result
        # ------------------------------------------------------------------
        # CVD draws: real arrays if has_cvd, NaN-filled otherwise
        if has_cvd:
            cvd_kwargs = dict(
                has_cvd=True,
                gamma_draws=gamma_draws,
                delta_draws=delta_draws,
                sigma_ihd2_draws=sigma_ihd2_draws,
                sigma_stroke2_draws=sigma_stroke2_draws,
                sigma_v_ihd2_draws=sigma_v_ihd2_draws,
                sigma_v_stroke2_draws=sigma_v_stroke2_draws,
            )
        else:
            cvd_kwargs = dict(
                has_cvd=False,
                gamma_draws=np.full((n_keep, 4), np.nan),
                delta_draws=np.full((n_keep, 4), np.nan),
                sigma_ihd2_draws=np.full(n_keep, np.nan),
                sigma_stroke2_draws=np.full(n_keep, np.nan),
                sigma_v_ihd2_draws=np.full(n_keep, np.nan),
                sigma_v_stroke2_draws=np.full(n_keep, np.nan),
            )

        return GibbsResult(
            beta_prev_draws=beta_prev_draws,
            beta_treat_draws=beta_treat_draws,
            alpha_draws=alpha_draws,
            sigma_prev2_draws=sigma_prev2_draws,
            sigma_treat2_draws=sigma_treat2_draws,
            sigma_obs_prev2_draws=sigma_obs_prev2_draws,
            sigma_obs_treat2_draws=sigma_obs_treat2_draws,
            sigma_u_prev2_draws=sigma_u_prev2_draws,
            sigma_u_treat2_draws=sigma_u_treat2_draws,
            latent_prev_draws=latent_prev_draws,
            latent_treat_draws=latent_treat_draws,
            country_ids=country_id,
            iso3c_map=iso3c_map,
            panel=df_work,
            phi_prev_draws=phi_prev_draws,
            phi_treat_draws=phi_treat_draws,
            **cvd_kwargs,
        )
