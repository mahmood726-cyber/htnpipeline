## Multi-Persona Review: htn_pipeline.py
### Date: 2026-04-03
### Summary: 7 P0, 10 P1, 10 P2

---

#### P0 -- Critical

- **P0-1** [Statistical/Domain]: Latent state refresh (lines 441-442) is ad-hoc `0.7*observed + 0.3*model + noise(0,0.25)`, NOT a valid Gibbs full conditional. Destroys posterior validity -- CrIs are not interpretable as Bayesian credible intervals.
  - Suggested fix: Either derive proper full conditional from a measurement-error model `y_obs = y_true + N(0, sigma_obs^2)`, giving precision-weighted posterior; or relabel as regularized iterative estimator and drop CrI claims.

- **P0-2** [Statistical/SWE]: No random seed before Gibbs sampler (line 400+). All `np.random.normal()` / `np.random.multivariate_normal()` calls use unseeded legacy global RNG. Violates CLAUDE.md determinism requirement. Fixture function (line 161) correctly uses `default_rng(42)` but sampler does not.
  - Suggested fix: Add `rng = np.random.default_rng(seed)` parameter, use `rng.normal()` / `rng.multivariate_normal()` throughout.

- **P0-3** [Statistical]: Four variance parameters (`s_prev2`, `s_treat2`, `s_u_prev2`, `s_u_treat2`, lines 387-390) are fixed constants, never sampled. CrIs are conditional on arbitrary values. Model is effectively penalized regression, not Bayesian inference.
  - Suggested fix: Add inverse-gamma full conditionals for each variance parameter.

- **P0-4** [Domain]: `harmonize_who_observations` (lines 272-298) does NOT filter by sex dimension (`Dim1`). WHO indicators return rows for BTSX/MLE/FMLE. Merge produces multiple rows per country-year, mixing male/female/combined estimates.
  - Suggested fix: Filter `df[df["Dim1"] == "BTSX"]` before extracting values.

- **P0-5** [Domain]: `a_prev_to_treat` prior = -0.2 (lines 382, 434) says higher prevalence -> lower treatment. Epidemiologically backward. NCD-RisC Lancet 2021 shows strong *positive* ecological correlation (detection drives treatment).
  - Suggested fix: Change prior to +0.3 (positive), widen variance.

- **P0-6** [SWE]: `get_json()` (lines 55-68) only catches `HTTPError`. Timeout/ConnectionError are NOT retried -- they propagate immediately on first attempt, making the 3-retry mechanism ineffective for transient network failures. Also, if `r.json()` raises `ValueError` (malformed JSON), function implicitly returns `None`.
  - Suggested fix: Catch `requests.exceptions.RequestException`; add terminal `raise` after loop.

- **P0-7** [SWE]: `zscore_series` (line 337) divides by `std(ddof=0)` with no zero-guard. If all values identical (plausible for sparse WHO data), produces NaN/inf that silently corrupts design matrix X and all posteriors.
  - Suggested fix: Return zeros when std < epsilon; or raise with clear message.

#### P1 -- Important

- **P1-1** [Statistical]: Counterfactual (line 462) is naive `posterior_mean + 20` with no uncertainty propagation. Should use posterior draws (`treat_draws` exists) and propagate through model.
  - Suggested fix: Compute counterfactual per MCMC draw, report CrI.

- **P1-2** [Statistical]: 94% CrI (quantiles 0.03/0.97) is non-standard. Epidemiology uses 95% (0.025/0.975). No justification documented.
  - Suggested fix: Change to 0.025/0.975; or document rationale.

- **P1-3** [Statistical]: Thinning applied asymmetrically -- parameter draws saved every iteration (500), latent draws thinned by 2 (250). Different Monte Carlo error for parameters vs. latent state summaries.
  - Suggested fix: Thin consistently or not at all.

- **P1-4** [Domain]: WHO HTN data is NOT annual -- typically 2010, 2015, 2019 only. Panel requests 2015-2024 but most years are empty. After `dropna`, effective N per country is 1-2 observations. Model is severely underpowered for country-level RE + 4 fixed effects.
  - Suggested fix: Restructure as cross-sectional model on available years; or use interpolation with explicit documentation.

- **P1-5** [Domain]: Missing key confounders -- urbanization rate, BMI/obesity, sodium intake, age structure. GDP and health spending are distal; model lacks proximal risk factors.
  - Suggested fix: Add World Bank urbanization indicator (`SP.URB.TOTL.IN.ZS`) at minimum.

- **P1-6** [Domain]: GDP prior on prevalence (beta=-1.0, line 380) assumes richer = lower HTN. Epidemiologically ambiguous -- LMICs now have *higher* prevalence than HICs (NCD-RisC 2021). Relationship is non-monotonic.
  - Suggested fix: Set prior near zero with wide variance to let data speak; or add quadratic term.

- **P1-7** [Security]: Unsanitized WHO-derived indicator codes interpolated into URLs (lines 76, 253-256). If WHO API compromised, crafted codes could enable OData injection or SSRF.
  - Suggested fix: Validate codes with `re.fullmatch(r'[A-Za-z0-9_.\-]{1,80}', code)`.

- **P1-8** [Security/SWE]: Broad `except Exception` at lines 118, 266, 317 masks real errors. Schema changes, programming bugs, MemoryError all silently swallowed. Line 118 silently substitutes real data with synthetic fixtures.
  - Suggested fix: Catch `(requests.RequestException, ValueError, KeyError)` specifically.

- **P1-9** [SWE]: Global mutable `CFG` (line 47) creates hidden coupling. Functions read config implicitly. Not testable in isolation; not safe for concurrent use.
  - Suggested fix: Pass `Config` as explicit parameter.

- **P1-10** [SWE]: `prepare_model_panel` (lines 344-349) assumes `htn_prevalence_pct` and `htn_treatment_pct` columns exist. If both WHO indicators fail, `KeyError` with no clear message.
  - Suggested fix: Check required columns before calling; raise clear error if WHO data missing.

#### P2 -- Minor

- **P2-1** [Statistical]: `ddof=0` gives population SD; `ddof=1` more correct for sample standardization. ~0.8% difference at N=60.
- **P2-2** [Statistical]: 500 post-burn-in draws (250 thinned) is marginal. No convergence diagnostics (Rhat, ESS) computed.
- **P2-3** [Statistical]: Matrix inversion `np.linalg.inv(post_prec)` (lines 417, 426) without regularization. Use Cholesky or add jitter for numerical stability.
- **P2-4** [SWE]: Dead imports: `io` (line 3), `Iterable` (line 8). Dead function: `who_measure` (line 244).
- **P2-5** [SWE]: `import os` inside function body (line 51). Should be at module level.
- **P2-6** [SWE]: File paths use string concatenation (`f"{outdir}/file.csv"`) instead of `pathlib.Path`.
- **P2-7** [SWE]: Parameter summary (lines 464-510) is 47 lines of repetitive manual indexing. Should be a loop.
- **P2-8** [Security]: No `User-Agent` header on HTTP requests. APIs may rate-limit unknown agents.
- **P2-9** [Security]: No response size limit on `r.json()`. Large payload could cause OOM.
- **P2-10** [Domain]: Fixture countries missing Pacific Islands (highest global HTN prevalence) and Central Asia.

#### False Positive Watch
- DOR = exp(mu1 + mu2): not relevant (no DOR in this code)
- Clayton copula theta: not relevant
- This is a pragmatic prototype per docstring -- some P0s are by design but should be documented

---

### Status: REVIEW COMPLETE -- awaiting fix decisions
