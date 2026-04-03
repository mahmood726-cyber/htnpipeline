"""World Bank fetcher with deterministic fixture for HyperAtlas v4 pipeline."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from data.http_client import get_json
from data.registry import validate_indicator_code

logger = logging.getLogger(__name__)

_WB_API_BASE = "https://api.worldbank.org/v2/country/all/indicator"

# 30-country reference table: iso3c -> (name, base_gdp, base_health_exp, base_urban)
_FIXTURE_COUNTRIES: list[tuple[str, str, float, float, float]] = [
    ("GBR", "United Kingdom",       46000.0, 10.2,  83.9),
    ("USA", "United States",        65000.0, 16.7,  82.7),
    ("DEU", "Germany",              48000.0, 11.7,  77.5),
    ("FRA", "France",               42000.0, 11.3,  81.2),
    ("JPN", "Japan",                40000.0, 10.9,  91.8),
    ("CAN", "Canada",               46000.0, 10.8,  81.6),
    ("AUS", "Australia",            55000.0,  9.4,  86.2),
    ("ITA", "Italy",                34000.0,  8.7,  71.0),
    ("ESP", "Spain",                30000.0,  9.1,  80.8),
    ("KOR", "Korea, Rep.",          32000.0,  8.4,  81.4),
    ("BRA", "Brazil",                8700.0,  9.6,  87.1),
    ("MEX", "Mexico",               10000.0,  5.4,  80.7),
    ("ARG", "Argentina",            10500.0,  9.5,  92.1),
    ("COL", "Colombia",              6100.0,  7.7,  81.4),
    ("CHL", "Chile",                15000.0,  9.3,  87.7),
    ("CHN", "China",                10500.0,  5.4,  61.4),
    ("IND", "India",                 2100.0,  3.3,  35.0),
    ("IDN", "Indonesia",             4300.0,  2.9,  56.6),
    ("THA", "Thailand",              7200.0,  3.8,  51.0),
    ("VNM", "Viet Nam",              3700.0,  5.5,  37.3),
    ("NGA", "Nigeria",               2100.0,  3.8,  52.0),
    ("ZAF", "South Africa",          6300.0,  8.3,  67.4),
    ("KEN", "Kenya",                 1800.0,  4.6,  28.0),
    ("ETH", "Ethiopia",               930.0,  3.5,  22.2),
    ("EGY", "Egypt, Arab Rep.",      3600.0,  4.7,  42.8),
    ("TUR", "Turkiye",               9600.0,  4.3,  76.1),
    ("SAU", "Saudi Arabia",         23000.0,  6.4,  84.1),
    ("RUS", "Russian Federation",   11500.0,  5.3,  74.8),
    ("POL", "Poland",               17000.0,  6.3,  60.1),
    ("BGD", "Bangladesh",            2500.0,  2.5,  38.2),
]


def fetch_worldbank_indicator(
    indicator: str,
    start_year: int,
    end_year: int,
    config,
) -> pd.DataFrame:
    """Fetch all-country data for *indicator* from World Bank API v2.

    Parameters
    ----------
    indicator:
        World Bank indicator code (e.g. ``NY.GDP.PCAP.CD``).
    start_year, end_year:
        Inclusive year range.
    config:
        Pipeline ``Config`` dataclass providing ``user_agent`` and
        ``max_response_bytes``.

    Returns
    -------
    pd.DataFrame
        Columns: ``iso3c``, ``year``, *indicator* (last segment used as name).
    """
    validate_indicator_code(indicator)
    url = f"{_WB_API_BASE}/{indicator}"
    params: dict = {
        "format": "json",
        "per_page": 10000,
        "date": f"{start_year}:{end_year}",
    }

    payload = get_json(
        url,
        params=params,
        user_agent=config.user_agent,
        max_bytes=config.max_response_bytes,
        sleep_base=0.5,
    )

    # WB API v2 returns [metadata_dict, data_list]
    if not isinstance(payload, list) or len(payload) < 2:
        return pd.DataFrame(columns=["iso3c", "year", indicator])

    records = payload[1] or []
    rows = []
    for rec in records:
        iso3c = (rec.get("country") or {}).get("id", None)
        year_raw = rec.get("date", None)
        value = rec.get("value", None)
        if iso3c and year_raw and value is not None:
            rows.append({"iso3c": iso3c, "year": int(year_raw), indicator: float(value)})

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["iso3c", "year", indicator])
    return df.drop_duplicates(subset=["iso3c", "year"]).reset_index(drop=True)


def worldbank_fixture(
    start_year: int,
    end_year: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a deterministic fixture DataFrame for 30 countries.

    Produces synthetic but plausible values for:
    - ``gdp_pc_usd`` — GDP per capita (USD)
    - ``health_exp_pct_gdp`` — Health expenditure (% of GDP)
    - ``urban_pct`` — Urban population (%)

    The growth model per year is:
    - ``gdp = base * (1 + 0.03*t + N(0, 0.02))``
    - ``health = base * (1 + 0.01*t + N(0, 0.01))``
    - ``urban = min(base * (1 + 0.005*t + N(0, 0.003)), 100)``

    where ``t = (year - start_year) / max(end_year - start_year, 1)``.

    Parameters
    ----------
    start_year, end_year:
        Inclusive year range.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: ``iso3c``, ``year``, ``gdp_pc_usd``, ``health_exp_pct_gdp``,
        ``urban_pct``.  Sorted by ``iso3c``, ``year``.
    """
    rng = np.random.default_rng(seed)
    years = list(range(start_year, end_year + 1))
    n_years = len(years)
    denom = max(end_year - start_year, 1)

    rows: list[dict] = []
    for iso3c, _name, base_gdp, base_health, base_urban in _FIXTURE_COUNTRIES:
        for year in years:
            t = (year - start_year) / denom
            gdp = base_gdp * (1.0 + 0.03 * t + rng.normal(0.0, 0.02))
            health = base_health * (1.0 + 0.01 * t + rng.normal(0.0, 0.01))
            urban = min(base_urban * (1.0 + 0.005 * t + rng.normal(0.0, 0.003)), 100.0)
            rows.append({
                "iso3c": iso3c,
                "year": year,
                "gdp_pc_usd": float(gdp),
                "health_exp_pct_gdp": float(health),
                "urban_pct": float(urban),
            })

    df = pd.DataFrame(rows)
    return df.sort_values(["iso3c", "year"]).reset_index(drop=True)
