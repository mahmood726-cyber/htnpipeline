"""WHO GHO fetcher with BTSX sex filter for HyperAtlas v4 pipeline."""
from __future__ import annotations

import logging
import warnings

import pandas as pd

from data.http_client import get_json
from data.registry import validate_indicator_code

logger = logging.getLogger(__name__)

_WHO_GHO_BASE = "https://ghoapi.azureedge.net/api"


def filter_who_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Filter WHO GHO data to BTSX (both-sex) rows only.

    If the ``Dim1`` column is present, keep only rows where ``Dim1 == 'BTSX'``.
    If ``Dim1`` is absent, deduplicate by SpatialDim+TimeDim keeping the first
    row (with a warning) so downstream code always gets at most one row per
    country-year.

    Parameters
    ----------
    df:
        Raw WHO GHO observation DataFrame.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only both-sex rows.
    """
    if df.empty:
        return df

    if "Dim1" in df.columns:
        return df[df["Dim1"] == "BTSX"].copy()

    # Dim1 absent — warn and deduplicate by SpatialDim+TimeDim
    warnings.warn(
        "Dim1 column absent from WHO response; deduplicating by SpatialDim+TimeDim "
        "keeping first row. Results may mix sexes.",
        UserWarning,
        stacklevel=2,
    )
    key_cols = [c for c in ["SpatialDim", "TimeDim"] if c in df.columns]
    if key_cols:
        return df.drop_duplicates(subset=key_cols, keep="first").copy()
    return df.copy()


def fetch_who_indicator(indicator_code: str, config) -> pd.DataFrame:
    """Fetch observations for *indicator_code* from the WHO GHO OData API.

    Parameters
    ----------
    indicator_code:
        WHO GHO indicator code (validated against allowlist).
    config:
        Pipeline ``Config`` dataclass providing ``user_agent`` and
        ``max_response_bytes``.

    Returns
    -------
    pd.DataFrame
        Raw observations DataFrame.  Columns reflect the GHO API response.
    """
    validate_indicator_code(indicator_code)
    url = f"{_WHO_GHO_BASE}/{indicator_code}"
    payload = get_json(
        url,
        user_agent=config.user_agent,
        max_bytes=config.max_response_bytes,
        sleep_base=0.5,
    )
    value_list = payload.get("value", [])
    return pd.DataFrame(value_list)


def harmonize_who_observations(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Apply sex filter and extract standardised iso3c/year/value columns.

    Steps
    -----
    1. Apply :func:`filter_who_sex` (keep BTSX / deduplicate).
    2. Map ``SpatialDim`` → ``iso3c``, ``TimeDim`` → ``year``,
       ``NumericValue`` → *col_name*.
    3. Drop rows with null iso3c, year, or value.
    4. Deduplicate by country-year keeping first row.

    Parameters
    ----------
    df:
        Raw WHO GHO observation DataFrame.
    col_name:
        Name to assign to the value column in the output.

    Returns
    -------
    pd.DataFrame
        Harmonised DataFrame with columns ``iso3c``, ``year``, *col_name*.
    """
    if df.empty:
        return pd.DataFrame(columns=["iso3c", "year", col_name])

    filtered = filter_who_sex(df)

    # Map column names
    rename: dict[str, str] = {}
    if "SpatialDim" in filtered.columns:
        rename["SpatialDim"] = "iso3c"
    if "TimeDim" in filtered.columns:
        rename["TimeDim"] = "year"
    if "NumericValue" in filtered.columns:
        rename["NumericValue"] = col_name

    result = filtered.rename(columns=rename)

    # Keep only required columns that exist
    keep = [c for c in ["iso3c", "year", col_name] if c in result.columns]
    result = result[keep].copy()

    # Coerce types
    if "year" in result.columns:
        result["year"] = pd.to_numeric(result["year"], errors="coerce").astype("Int64")
    if col_name in result.columns:
        result[col_name] = pd.to_numeric(result[col_name], errors="coerce")

    # Drop rows with nulls in key columns
    result = result.dropna(subset=[c for c in ["iso3c", "year", col_name] if c in result.columns])

    # Deduplicate country-year
    key_cols = [c for c in ["iso3c", "year"] if c in result.columns]
    if key_cols:
        result = result.drop_duplicates(subset=key_cols, keep="first")

    result = result.reset_index(drop=True)
    return result
