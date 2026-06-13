# htnpipeline

A Bayesian hypertension-prevalence pipeline (HyperAtlas v4) that harmonizes
WHO Global Health Observatory and World Bank indicators into a country-year
panel, fits a state-space Gibbs sampler, and writes an auditable output bundle.

## What the code does

- **Data layer** (`data/`): fetches 8 registered indicators (3 World Bank,
  5 WHO GHO), filters WHO observations to both-sex (BTSX), and merges them into
  a wide country-year panel. World Bank indicators fall back to a deterministic
  fixture when the API is unavailable; WHO indicators fall back to a fixture or
  an empty frame so the pipeline can run offline.
- **Model layer** (`model/`): a state-space Gibbs sampler
  (`StateSpaceGibbs`) for prevalence and treatment coverage with country random
  effects, AR(1) terms, and an optional CVD-mortality sub-model. Includes
  split-chain R-hat / ESS diagnostics and a counterfactual coverage engine.
- **Bundle** (`model/truthcert.py`): writes a 7-file output bundle
  (parameter summary, posterior draws, counterfactuals, convergence,
  fitted panel, coverage summary, provenance with an input SHA-256 hash).
- **Dashboard** (`index.html`): an interactive D3 choropleth dashboard for the
  fitted estimates (uses external CDN map data; requires network access).

## Run

```bash
python run_pipeline.py --max-countries 30 --n-iter 3000 --burn 1000 --thin 2
```

## Tests

```bash
python -m pytest -q
```

See `E156-PROTOCOL.md` for the study protocol and primary estimand.
