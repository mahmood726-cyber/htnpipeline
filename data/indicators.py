"""8 indicator definitions for HyperAtlas v4 pipeline.

Registers all indicators into a global ``registry`` instance.  World Bank
indicators fall back to the deterministic fixture when the API call fails.
WHO indicators catch exceptions and return an empty DataFrame on failure so
the pipeline can continue with partial data.
"""
from __future__ import annotations

import logging

import pandas as pd

from data.registry import IndicatorRegistry
from data.worldbank import fetch_worldbank_indicator, worldbank_fixture
from data.who import fetch_who_indicator, harmonize_who_observations, who_fixture

logger = logging.getLogger(__name__)

registry = IndicatorRegistry()


# ---------------------------------------------------------------------------
# World Bank indicators — try API, fall back to fixture
# ---------------------------------------------------------------------------

@registry.register("gdp_pc_usd", source="worldbank", code="NY.GDP.PCAP.CD")
def _fetch_gdp_pc_usd(start_year: int, end_year: int, config) -> pd.DataFrame:
    """GDP per capita (current USD). Fallback: deterministic fixture."""
    try:
        df = fetch_worldbank_indicator("NY.GDP.PCAP.CD", start_year, end_year, config)
        if not df.empty:
            df = df.rename(columns={"NY.GDP.PCAP.CD": "gdp_pc_usd"})
            return df
    except Exception as exc:
        logger.warning("WB GDP fetch failed (%s); using fixture.", exc)
    fix = worldbank_fixture(start_year, end_year, config.seed)
    return fix[["iso3c", "year", "gdp_pc_usd"]].copy()


@registry.register("health_exp_pct_gdp", source="worldbank", code="SH.XPD.CHEX.GD.ZS")
def _fetch_health_exp(start_year: int, end_year: int, config) -> pd.DataFrame:
    """Current health expenditure (% of GDP). Fallback: deterministic fixture."""
    try:
        df = fetch_worldbank_indicator("SH.XPD.CHEX.GD.ZS", start_year, end_year, config)
        if not df.empty:
            df = df.rename(columns={"SH.XPD.CHEX.GD.ZS": "health_exp_pct_gdp"})
            return df
    except Exception as exc:
        logger.warning("WB health-exp fetch failed (%s); using fixture.", exc)
    fix = worldbank_fixture(start_year, end_year, config.seed)
    return fix[["iso3c", "year", "health_exp_pct_gdp"]].copy()


@registry.register("urban_pct", source="worldbank", code="SP.URB.TOTL.IN.ZS")
def _fetch_urban_pct(start_year: int, end_year: int, config) -> pd.DataFrame:
    """Urban population (% of total). Fallback: deterministic fixture."""
    try:
        df = fetch_worldbank_indicator("SP.URB.TOTL.IN.ZS", start_year, end_year, config)
        if not df.empty:
            df = df.rename(columns={"SP.URB.TOTL.IN.ZS": "urban_pct"})
            return df
    except Exception as exc:
        logger.warning("WB urban-pct fetch failed (%s); using fixture.", exc)
    fix = worldbank_fixture(start_year, end_year, config.seed)
    return fix[["iso3c", "year", "urban_pct"]].copy()


# ---------------------------------------------------------------------------
# WHO GHO indicators
# ---------------------------------------------------------------------------

@registry.register("htn_prevalence_pct", source="who", code="NCD_HYP_PREVALENCE_A")
def _fetch_htn_prevalence(start_year: int, end_year: int, config) -> pd.DataFrame:
    """Hypertension prevalence (age-standardised %). Filtered to BTSX. Falls back to fixture."""
    try:
        df = fetch_who_indicator("NCD_HYP_PREVALENCE_A", config)
        result = harmonize_who_observations(df, "htn_prevalence_pct")
        if "year" in result.columns:
            result = result[
                (result["year"] >= start_year) & (result["year"] <= end_year)
            ]
        if not result.empty and "htn_prevalence_pct" in result.columns:
            return result.reset_index(drop=True)
    except Exception as exc:
        logger.warning("WHO htn_prevalence fetch failed (%s); using fixture.", exc)
    fix = who_fixture(start_year, end_year, getattr(config, "seed", 42))
    return fix[["iso3c", "year", "htn_prevalence_pct"]].copy()


@registry.register("htn_treatment_pct", source="who", code="NCD_HYP_TREATMENT_A")
def _fetch_htn_treatment(start_year: int, end_year: int, config) -> pd.DataFrame:
    """Hypertension treatment coverage (%). Filtered to BTSX. Falls back to fixture."""
    try:
        df = fetch_who_indicator("NCD_HYP_TREATMENT_A", config)
        result = harmonize_who_observations(df, "htn_treatment_pct")
        if "year" in result.columns:
            result = result[
                (result["year"] >= start_year) & (result["year"] <= end_year)
            ]
        if not result.empty and "htn_treatment_pct" in result.columns:
            return result.reset_index(drop=True)
    except Exception as exc:
        logger.warning("WHO htn_treatment fetch failed (%s); using fixture.", exc)
    fix = who_fixture(start_year, end_year, getattr(config, "seed", 42))
    return fix[["iso3c", "year", "htn_treatment_pct"]].copy()


@registry.register("cvd_mortality_30_70", source="who", code="NCDMORT3070")
def _fetch_cvd_mortality(start_year: int, end_year: int, config) -> pd.DataFrame:
    """Probability of dying between 30-70 from CVD/cancer/diabetes/CRD (%). BTSX."""
    df = fetch_who_indicator("NCDMORT3070", config)
    result = harmonize_who_observations(df, "cvd_mortality_30_70")
    if "year" in result.columns:
        result = result[
            (result["year"] >= start_year) & (result["year"] <= end_year)
        ]
    return result.reset_index(drop=True)


@registry.register("ihd_mort_rate", source="who", code="MORT_100")
def _fetch_ihd_mort_rate(start_year: int, end_year: int, config) -> pd.DataFrame:
    """IHD mortality rate. Returns empty DataFrame on API failure."""
    try:
        df = fetch_who_indicator("MORT_100", config)
        result = harmonize_who_observations(df, "ihd_mort_rate")
        if "year" in result.columns:
            result = result[
                (result["year"] >= start_year) & (result["year"] <= end_year)
            ]
        return result.reset_index(drop=True)
    except Exception as exc:
        logger.warning("WHO IHD mort-rate fetch failed (%s); returning empty.", exc)
        return pd.DataFrame(columns=["iso3c", "year", "ihd_mort_rate"])


@registry.register("stroke_mort_rate", source="who", code="MORT_200")
def _fetch_stroke_mort_rate(start_year: int, end_year: int, config) -> pd.DataFrame:
    """Stroke mortality rate. Returns empty DataFrame on API failure."""
    try:
        df = fetch_who_indicator("MORT_200", config)
        result = harmonize_who_observations(df, "stroke_mort_rate")
        if "year" in result.columns:
            result = result[
                (result["year"] >= start_year) & (result["year"] <= end_year)
            ]
        return result.reset_index(drop=True)
    except Exception as exc:
        logger.warning("WHO stroke mort-rate fetch failed (%s); returning empty.", exc)
        return pd.DataFrame(columns=["iso3c", "year", "stroke_mort_rate"])
