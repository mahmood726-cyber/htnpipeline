#!/usr/bin/env python
"""HyperAtlas v4 -- Global Hypertension Pipeline."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from data.config import Config
from data.harmonize import build_panel, prepare_model_panel
from model.gibbs import StateSpaceGibbs
from model.truthcert import generate_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="HyperAtlas v4 pipeline")
    parser.add_argument("--max-countries", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-iter", type=int, default=3000)
    parser.add_argument("--burn", type=int, default=1000)
    parser.add_argument("--thin", type=int, default=2)
    parser.add_argument("--outdir", type=str, default="v4_output")
    args = parser.parse_args()

    config = Config(
        max_countries=args.max_countries,
        seed=args.seed,
        n_iter=args.n_iter,
        burn=args.burn,
        thin=args.thin,
        outdir=Path(args.outdir),
        bundle_dir=Path(args.outdir) / "bundle",
    )
    config.outdir.mkdir(parents=True, exist_ok=True)

    print("Phase 1: Fetching and harmonizing data...")
    panel, sources = build_panel(config)
    panel.to_csv(config.outdir / "panel_raw.csv", index=False)
    with open(config.outdir / "indicator_sources.json", "w") as f:
        json.dump(sources, f, indent=2)
    print(f"  Panel: {len(panel)} rows, {panel['iso3c'].nunique()} countries")

    print("Phase 2: Preparing model panel...")
    model_panel = prepare_model_panel(panel)
    model_panel.to_csv(config.outdir / "panel_prepared.csv", index=False)
    print(f"  Model-ready: {len(model_panel)} rows")

    print(f"Phase 3: Running Gibbs sampler ({config.n_iter} iterations, burn={config.burn})...")
    gibbs = StateSpaceGibbs(seed=config.seed, n_iter=config.n_iter, burn=config.burn, thin=config.thin)
    result = gibbs.fit(model_panel)
    print(f"  Stored {len(result.alpha_draws)} post-burn-in draws")

    print("Phase 4: Generating TruthCert bundle...")
    bundle_path = generate_bundle(result, panel, bundle_dir=config.bundle_dir, seed=config.seed)
    print(f"  Bundle saved to {bundle_path}")

    params = result.parameter_summary()
    print("\nParameter summary:")
    print(params.to_string(index=False))

    conv = json.loads((config.bundle_dir / "convergence.json").read_text())
    print(f"\nConvergence: {conv['overall_status']} ({conv['n_warn']} warn, {conv['n_fail']} fail)")
    print("\nDone.")


if __name__ == "__main__":
    main()
