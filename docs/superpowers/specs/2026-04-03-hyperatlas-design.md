# HyperAtlas Design Spec
## Global Hypertension Pipeline v4 -- "HyperAtlas"
### Date: 2026-04-03

---

## 1. Purpose

A publish-ready, three-layer pipeline that:
1. Fetches and harmonizes global hypertension data from WHO GHO + World Bank
2. Fits a Bayesian state-space model estimating HTN prevalence, treatment coverage, and CVD mortality (IHD + stroke split)
3. Generates HEARTS-aligned counterfactual scenarios with full uncertainty propagation
4. Presents results in an interactive single-file HTML dashboard with world map + country explorer

**Target outlets:** Lancet Global Health (dashboard + methods), F1000 (tool paper), or supplementary engine for a cardiology meta-analysis manuscript.

**Success criteria:**
- All 7 P0 + 10 P1 review findings resolved
- Gibbs sampler produces valid posteriors (Rhat < 1.05, ESS > 400)
- Reproducible: same seed -> bit-exact output
- TruthCert bundle with provenance chain
- ~50 tests, all passing
- Interactive dashboard loads < 3s, works on mobile

---

## 2. Architecture

Three independent layers, each testable in isolation:

| Layer | Directory | Responsibility | Output |
|-------|-----------|---------------|--------|
| Data Engine | `data/` | Fetch, filter, harmonize, validate | Clean country-year panel (CSV + Parquet) |
| Bayesian Engine | `model/` | State-space Gibbs sampler, diagnostics, counterfactuals | TruthCert bundle (draws, summaries, provenance) |
| Dashboard | `dashboard/` | Interactive visualization | Single HTML file |

Communication between layers is via files (CSV/JSON/Parquet). No runtime coupling.

---

## 3. Data Engine

### 3.1 Indicator Registry

Extensible decorator pattern:

```python
@register_indicator("gdp_pc", source="worldbank", code="NY.GDP.PCAP.CD")
def fetch_gdp(start_year, end_year, session):
    ...
```

Each decorated function handles its own:
- API fetch logic
- Fixture fallback (deterministic, seeded)
- Harmonization rules (sex filter, age filter, unit conversion)

Adding a new covariate = one decorated function, zero changes elsewhere.

### 3.2 Ship-Ready Indicators (8)

| Source | Code | Column Name | Purpose |
|--------|------|-------------|---------|
| World Bank | NY.GDP.PCAP.CD | gdp_pc_usd | Economic covariate |
| World Bank | SH.XPD.CHEX.GD.ZS | health_exp_pct_gdp | Health system covariate |
| World Bank | SP.URB.TOTL.IN.ZS | urban_pct | Proximal risk factor |
| WHO GHO | NCD_HYP_PREVALENCE_A | htn_prevalence_pct | Primary outcome (BP >= 140/90, age-std, 30-79, BTSX) |
| WHO GHO | NCD_HYP_TREATMENT_A | htn_treatment_pct | Primary outcome (30-79, BTSX) |
| WHO GHO | NCDMORT3070 | cvd_mortality_30_70 | CVD mortality probability 30-70 |
| WHO GHO | (IHD mortality rate) | ihd_mort_rate | IHD-specific mortality |
| WHO GHO | (Stroke mortality rate) | stroke_mort_rate | Stroke-specific mortality |

Note: IHD and stroke indicator codes will be discovered via WHO catalog search at implementation time.

### 3.3 WHO Sex/Age Filtering

All WHO observations filtered to `Dim1 == "BTSX"` (both sexes) before harmonization. If `Dim1` column absent, deduplicate by country-year taking first row with warning.

### 3.4 Robustness

- `get_json()`: Catches `RequestException` (not just `HTTPError`). Retries on 502/503/429/Timeout/ConnectionError. Terminal raise after exhaustion. Response size cap 50MB.
- Indicator code validation: `re.fullmatch(r'[A-Za-z0-9_.\-]{1,80}', code)` before URL interpolation.
- `zscore_series()`: Returns zeros when `std < 1e-12`. Never produces NaN/inf.
- Column validation: `prepare_model_panel()` checks required columns exist before proceeding. Clear error message listing missing columns.
- Exception handling: Catch `(RequestException, ValueError, KeyError)` specifically. No bare `except Exception`.
- User-Agent header: `"HyperAtlas/4.0 (research)"` on all requests.
- Paths: `pathlib.Path` throughout. No string concatenation for file paths.

### 3.5 Fixture Fallback

30 countries spanning HIC/UMIC/LMIC/LIC, seeded `default_rng(42)`. Used when API unavailable. Clearly flagged in output metadata (`data_source: "fixture"` vs `"api"`).

---

## 4. Bayesian Engine

### 4.1 Model Specification

**Measurement model:**
```
y_prev_obs[c,t] = prev_true[c,t] + eps_prev,    eps_prev ~ N(0, sigma_obs_prev^2)
y_treat_obs[c,t] = treat_true[c,t] + eps_treat,  eps_treat ~ N(0, sigma_obs_treat^2)
```

**Structural model:**
```
prev_true[c,t]  = X[c,t] @ beta_prev  + u_prev[c]  + phi_prev  * prev_true[c,t-1] + eta_prev[c,t]
treat_true[c,t] = X[c,t] @ beta_treat + u_treat[c] + phi_treat * treat_true[c,t-1] + alpha * prev_true[c,t] + eta_treat[c,t]
```

**CVD outcome model:**
```
ihd_mort[c,t]    = gamma_0 + gamma_1 * prev_true[c,t] + gamma_2 * treat_true[c,t] + gamma_3 * gdp_z + v_ihd[c] + eta_ihd[c,t]
stroke_mort[c,t] = delta_0 + delta_1 * prev_true[c,t] + delta_2 * treat_true[c,t] + delta_3 * gdp_z + v_stroke[c] + eta_stroke[c,t]
```

**Design matrix X:** `[intercept, log_gdp_z, health_exp_z, urban_z, year_z]` (all z-scored with ddof=1, zero-guarded).

### 4.2 Priors

| Parameter | Prior | Rationale |
|-----------|-------|-----------|
| beta_prev | MVN([30, 0, -0.5, 1.0, -0.1], diag([10^2, 3^2, 3^2, 3^2, 2^2])) | GDP prior near 0 (non-monotonic relationship); urbanization prior positive |
| beta_treat | MVN([40, 2, 1, 0.5, 0.5], diag([12^2, 5^2, 5^2, 5^2, 2^2])) | GDP increases treatment (well-established) |
| alpha (prev->treat) | N(+0.3, 0.5^2) | Positive: detection drives treatment (NCD-RisC 2021) |
| gamma, delta (CVD) | MVN(0, diag(5^2)) | Weakly informative, let data speak |
| sigma^2 (all residual) | InvGamma(3, 1) | Weakly informative, proper |
| sigma_u^2 (all RE) | InvGamma(3, 1) | Weakly informative, proper |
| sigma_obs^2 | InvGamma(3, 1) | Measurement error variance, sampled |
| phi (AR coefficients) | Uniform(-0.95, 0.95) | Stationarity constraint |

### 4.3 Gibbs Sampler Details

- **Full conditionals** for all blocks: beta (MVN), u (normal), sigma^2 (inv-gamma), latent states (precision-weighted from measurement + structural), phi (Metropolis-Hastings within Gibbs using N(phi_current, 0.05^2) proposal, truncated to (-0.95, 0.95), since no conjugate form), alpha (normal).
- **Latent state update:** Draw from proper full conditional: `p(latent | y_obs, params) = N(weighted_mean, weighted_var)` where weights come from measurement precision `1/sigma_obs^2` and structural precision `1/sigma_eta^2`.
- **RNG:** `np.random.default_rng(seed)` passed as parameter. Default seed=2026. All draws use `rng.normal()`, `rng.multivariate_normal()`, `rng.gamma()`.
- **Iterations:** 3,000 total, 1,000 burn-in, thin by 2 = 1,000 post-burn-in draws. Uniform thinning across all parameter types.
- **ddof=1** for all z-scoring.

### 4.4 Convergence Diagnostics

Computed automatically after sampling:

| Diagnostic | Method | Threshold |
|------------|--------|-----------|
| Rhat | Split-chain (split single chain into 2 halves) | Warn if > 1.05, error if > 1.10 |
| ESS | Bulk ESS via autocorrelation | Warn if < 400 |
| Trace export | Raw arrays saved to bundle | For external plotting |

Diagnostics saved to `convergence.json` in the TruthCert bundle.

### 4.5 Counterfactual Engine

Three HEARTS-aligned scenarios, computed per MCMC draw:

1. **+25pp treatment** -- WHO benchmark absolute increase
2. **80% treatment target** -- WHO HEARTS global goal (country-specific delta)
3. **Custom shift** -- user-configurable pp change

For each scenario and each MCMC draw:
- Compute counterfactual treatment coverage (capped at 100%)
- Propagate through CVD outcome model to get delta IHD and delta stroke mortality
- Store per-draw results

Output: posterior mean + 95% CrI for each scenario's mortality reduction, per country.

### 4.6 TruthCert Bundle

```
bundle/
  parameter_summary.csv     -- all params with mean, median, sd, 95% CrI
  posterior_draws.parquet    -- full chains (1000 draws x all params)
  counterfactual.csv         -- 3 scenarios x countries, with CrIs
  convergence.json           -- Rhat, ESS per parameter, pass/warn/fail flags
  provenance.json            -- {data_hashes, code_version, seed, timestamp, indicator_codes, n_countries, n_years}
  panel_fitted.csv           -- posterior state estimates (latent prevalence, treatment, IHD, stroke)
  coverage_summary.csv       -- data completeness metrics
```

---

## 5. Dashboard

### 5.1 Overview

Single HTML file (`hyper_atlas.html`), targeting 3,000-5,000 lines. Three tabs. Data from JSON bundle (embedded base64-gzipped or adjacent file).

### 5.2 Tab 1: Global Map

- D3 v7 choropleth with TopoJSON world boundaries (~150KB embedded)
- **Metric selector dropdown:** HTN prevalence | treatment coverage | treatment gap | IHD mortality | stroke mortality
- **Time slider:** Snaps to available years (2010, 2015, 2019)
- **Color scales:** Sequential (YlOrRd for prevalence/mortality), diverging (RdYlGn for treatment gap)
- **Hover tooltip:** Country name, metric value, 95% CrI, data source (observed/modeled)
- **Click interaction:** Navigates to Tab 2 with clicked country pre-selected
- **Legend:** Continuous color bar with tick marks

### 5.3 Tab 2: Country Explorer

- **Country selector:** Dropdown with search (type-ahead)
- **Left panel -- Trajectory charts (4):**
  - HTN prevalence over time: observed points (circles) + posterior ribbon (95% CrI shaded)
  - Treatment coverage: same format
  - IHD mortality rate: same format
  - Stroke mortality rate: same format
- **Right panel -- Counterfactual overlay:**
  - Toggle buttons for each HEARTS scenario
  - Scenario ribbons overlaid on mortality charts (different color, with CrI bands)
  - Summary box: "Under +25pp treatment: IHD mortality reduces by X% (95% CrI: Y-Z%)"
- **Bottom strip:** Country random effects as forest plot (dot + CI for each RE component)
- **Export:** Download button generates CSV of displayed country's data

### 5.4 Tab 3: Evidence Dashboard

- **Parameter table:** All model parameters with mean, 95% CrI, Rhat, ESS. Color-coded rows (green/yellow/red by convergence).
- **Convergence traffic lights:** Summary badges -- "All Rhat < 1.05" (green) or "3 parameters > 1.05" (yellow/red)
- **Data coverage heatmap:** Countries (rows) x years (columns), cells colored by missingness (green = observed, gray = missing, blue = modeled)
- **Provenance footer:** Seed, timestamp, data hashes, code version. Copyable.

### 5.5 Technical Details

- D3.js v7 (CDN with integrity hash)
- TopoJSON client v3 (CDN)
- Vanilla JS, no framework
- CSS: dark/light mode toggle, responsive (mobile-friendly)
- Seeded PRNG: xoshiro128** for any client-side jitter/sampling
- `role="application"` only on map container
- Keyboard navigation: arrow keys on map, tab through controls
- All interactive elements have ARIA labels

---

## 6. Testing

### 6.1 Data Engine (15 tests)

1. Fixture determinism: same seed -> identical DataFrame
2. WHO sex filter: only BTSX rows survive
3. WHO age filter: correct age band selected
4. Indicator registry: register + fetch + list cycle
5. Retry logic: mock 502 -> 502 -> 200 succeeds
6. Retry exhaustion: mock 502 x 3 raises clearly
7. zscore guard: all-identical input returns zeros, no NaN
8. zscore ddof=1: verify std matches expected
9. Column validation: missing WHO columns raises with message
10. Indicator code sanitization: `../` rejected, valid codes pass
11. Response size limit: oversized payload triggers error
12. Panel merge: no duplicate country-year rows after WHO join
13. Year filtering: only requested range survives
14. Fixture country count: exactly 30 countries
15. Pathlib usage: output files created in correct directory

### 6.2 Bayesian Engine (25 tests)

1. Seed reproducibility: two runs with same seed -> identical output
2. Different seeds: two runs with different seeds -> different output
3. Beta posterior (synthetic): converges to known OLS estimate within tolerance
4. Variance sampling: inv-gamma mean converges to known sigma^2
5. Country RE shrinkage: with large sigma_u, REs near zero
6. Country RE freedom: with small sigma_u, REs match country means
7. AR(1) phi: stays within (-0.95, 0.95) across all draws
8. AR(1) recovery: phi posterior mean near true phi on synthetic AR data
9. Measurement error: latent != observed when sigma_obs large
10. Measurement error: latent ~ observed when sigma_obs ~ 0
11. Prev-treat link: positive posterior mean on synthetic positive-correlation data
12. Rhat computation: known good chain -> Rhat ~ 1.0
13. Rhat computation: known bad chain (two modes) -> Rhat > 1.1
14. ESS computation: iid draws -> ESS ~ n_draws
15. ESS computation: highly autocorrelated -> ESS << n_draws
16. Counterfactual +25pp: treatment capped at 100%
17. Counterfactual uncertainty: CrI width > 0
18. Counterfactual propagation: mortality reduction has correct sign
19. IHD vs stroke: different gamma/delta -> different mortality reductions
20. Edge case: single country (n_country=1)
21. Edge case: two time points (minimal panel)
22. Edge case: one covariate all-missing -> graceful handling
23. TruthCert bundle: all 7 files present
24. Provenance hash: matches SHA-256 of input panel
25. Parameter summary: 95% CrI (not 94%)

### 6.3 Dashboard (10 tests, Playwright)

1. HTML loads without console errors
2. Map renders correct number of country paths
3. Time slider updates map fill colors
4. Metric selector changes color scale
5. Country click navigates to explorer tab
6. Counterfactual toggle shows/hides scenario ribbons
7. Dark/light mode toggle works
8. Export CSV produces valid file with correct columns
9. Convergence traffic lights match test data
10. Keyboard: Tab through all interactive controls

**Total: 50 tests.**

---

## 7. Review Findings Coverage

Every P0 and P1 from the review is addressed:

| Finding | Fix Location | Section |
|---------|-------------|---------|
| P0-1 Latent refresh invalid | model/gibbs.py -- proper full conditional | 4.3 |
| P0-2 No random seed | model/gibbs.py -- default_rng(seed) | 4.3 |
| P0-3 Fixed variances | model/gibbs.py -- inv-gamma sampling | 4.2, 4.3 |
| P0-4 WHO sex filter | data/who.py -- Dim1==BTSX filter | 3.3 |
| P0-5 Link sign backward | model/gibbs.py -- alpha prior +0.3 | 4.2 |
| P0-6 get_json fragile | data/registry.py -- RequestException + terminal raise | 3.4 |
| P0-7 zscore div by zero | data/harmonize.py -- zero guard | 3.4 |
| P1-1 No counterfactual CrI | model/counterfactual.py -- per-draw propagation | 4.5 |
| P1-2 94% CrI | model/gibbs.py -- 95% (0.025/0.975) | 4.3 |
| P1-3 Asymmetric thinning | model/gibbs.py -- uniform thinning | 4.3 |
| P1-4 Sparse WHO data | data/harmonize.py -- document available years | 3.2 |
| P1-5 Missing confounders | data/indicators.py -- urbanization added | 3.2 |
| P1-6 GDP prior direction | model/gibbs.py -- prior near 0 | 4.2 |
| P1-7 URL injection | data/registry.py -- regex validation | 3.4 |
| P1-8 Broad exceptions | all data/ modules -- specific catches | 3.4 |
| P1-9 Global mutable CFG | Config passed as parameter | 3.4 |
| P1-10 Missing column guard | data/harmonize.py -- check before prepare | 3.4 |

---

## 8. Delivery Milestones

| Phase | Deliverable | Test Count |
|-------|------------|------------|
| Phase 1 | Data Engine (registry + 8 indicators + fixtures + validation) | 15 |
| Phase 2 | Bayesian Engine (Gibbs + diagnostics + counterfactual + TruthCert) | 25 |
| Phase 3 | Dashboard (map + explorer + evidence tabs) | 10 |
| Phase 4 | Integration test + full pipeline run + review-findings.md update | -- |

---

## 9. Non-Goals

- Real-time data streaming (batch pipeline only)
- GBD/IHME data integration (WHO GHO only)
- User authentication or multi-user access
- Server deployment (local-first, single HTML)
- Causal inference claims (ecological associations only, documented as such)
