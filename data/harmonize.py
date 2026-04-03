"""Panel builder, zscore with zero-guard, and column validation.

Task 4 — HyperAtlas v4 pipeline.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from data.config import Config
from data.indicators import registry

logger = logging.getLogger(__name__)

# Columns expected to come from World Bank indicators
_WB_COLS = ("gdp_pc_usd", "health_exp_pct_gdp", "urban_pct")


def zscore_series(s: pd.Series) -> pd.Series:
    """Z-score a numeric Series using ddof=1.

    Zero-guard: when the standard deviation is less than 1e-12 (constant or
    single-element series), returns a zero-filled Series rather than NaN/inf.

    Parameters
    ----------
    s:
        Numeric pandas Series.

    Returns
    -------
    pd.Series
        Z-scored Series (or zeros when std < 1e-12).  Never contains NaN or
        infinity due to a constant input.
    """
    std = s.std(ddof=1)
    if std < 1e-12 or np.isnan(std):
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    return (s - s.mean()) / std


def build_panel(config: Config) -> pd.DataFrame:
    """Fetch all registered indicators and merge into a single panel.

    Uses ``registry.fetch_all`` (which catches per-indicator failures) then
    performs a successive outer-join merge on ``iso3c`` + ``year``.  Rows are
    filtered to ``[config.start_year, config.end_year]`` and sorted.

    Parameters
    ----------
    config:
        Pipeline configuration dataclass.

    Returns
    -------
    pd.DataFrame
        Wide-format panel sorted by ``iso3c``, ``year``.
    """
    dfs = registry.fetch_all(config.start_year, config.end_year, config)

    panel: pd.DataFrame | None = None
    for name, df in dfs.items():
        if df is None or df.empty:
            logger.info("  Skipping empty indicator: %s", name)
            continue
        if "iso3c" not in df.columns or "year" not in df.columns:
            logger.warning("  Indicator %s missing iso3c/year columns; skipping.", name)
            continue
        if panel is None:
            panel = df.copy()
        else:
            panel = panel.merge(df, on=["iso3c", "year"], how="outer")

    if panel is None:
        return pd.DataFrame()

    # Filter to configured year range
    panel = panel[
        (panel["year"] >= config.start_year) & (panel["year"] <= config.end_year)
    ].copy()

    return panel.sort_values(["iso3c", "year"]).reset_index(drop=True)


def prepare_model_panel(
    panel: pd.DataFrame,
    required_outcomes: list[str] | None = None,
) -> pd.DataFrame:
    """Validate required columns, create model features, and z-score predictors.

    Derived columns added
    ---------------------
    - ``country_id`` — integer category code (0-based, alphabetical by iso3c)
    - ``year_c`` — year centred at mid-point of observed year range
    - ``log_gdp`` — natural log of ``gdp_pc_usd``
    - ``log_gdp_z`` — z-score of ``log_gdp``
    - ``health_exp_z`` — z-score of ``health_exp_pct_gdp``
    - ``urban_z`` — z-score of ``urban_pct``
    - ``year_z`` — z-score of ``year``

    Parameters
    ----------
    panel:
        Wide-format panel DataFrame (output of :func:`build_panel` or similar).
    required_outcomes:
        List of column names that must be present.  Raises ``ValueError``
        listing all missing columns if any are absent.

    Returns
    -------
    pd.DataFrame
        Panel with additional model-ready columns.

    Raises
    ------
    ValueError
        When one or more required columns are missing from *panel*.
    """
    # --- column validation ---------------------------------------------------
    required_outcomes = required_outcomes or []
    required = list(_WB_COLS) + required_outcomes
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}.  "
            f"Available columns: {list(panel.columns)}"
        )

    result = panel.copy()

    # --- country_id ----------------------------------------------------------
    countries_sorted = sorted(result["iso3c"].dropna().unique())
    id_map = {c: i for i, c in enumerate(countries_sorted)}
    result["country_id"] = result["iso3c"].map(id_map).astype("Int64")

    # --- year_c --------------------------------------------------------------
    year_mid = (result["year"].min() + result["year"].max()) / 2.0
    result["year_c"] = result["year"] - year_mid

    # --- log_gdp -------------------------------------------------------------
    result["log_gdp"] = np.log(result["gdp_pc_usd"].clip(lower=1e-9))

    # --- z-scored predictors -------------------------------------------------
    result["log_gdp_z"] = zscore_series(result["log_gdp"])
    result["health_exp_z"] = zscore_series(result["health_exp_pct_gdp"])
    result["urban_z"] = zscore_series(result["urban_pct"])
    result["year_z"] = zscore_series(result["year"].astype(float))

    return result
