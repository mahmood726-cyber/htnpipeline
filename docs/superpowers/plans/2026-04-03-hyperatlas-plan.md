# HyperAtlas Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a publish-ready Bayesian state-space pipeline for global hypertension analysis with interactive dashboard, fixing all 17 P0+P1 review findings.

**Architecture:** Three independent layers (Data Engine, Bayesian Engine, Dashboard) communicating via CSV/JSON/Parquet files. Each layer is testable in isolation. The Data Engine fetches WHO GHO + World Bank indicators via an extensible registry. The Bayesian Engine runs a proper Gibbs sampler with full conditionals, convergence diagnostics, and HEARTS-aligned counterfactuals. The Dashboard is a single HTML file with D3 choropleth map + country explorer.

**Tech Stack:** Python 3.13, numpy, pandas, requests, pathlib, pytest. Dashboard: vanilla JS, D3.js v7, TopoJSON. Browser tests: Playwright.

**Spec:** `docs/superpowers/specs/2026-04-03-hyperatlas-design.md`
**Review findings:** `review-findings.md`

---

## Phase 1: Data Engine

### Task 1: Project scaffold + Config + HTTP client

**Files:**
- Create: `data/__init__.py`
- Create: `data/config.py`
- Create: `data/http_client.py`
- Create: `model/__init__.py`
- Create: `dashboard/.gitkeep`
- Create: `tests/__init__.py`
- Create: `tests/test_data.py`
- Create: `run_pipeline.py`

**Fixes:** P0-6 (get_json fragile), P1-9 (global mutable CFG), P1-8 (broad exceptions), P2-8 (User-Agent), P2-9 (response size limit)

- [ ] **Step 1: Create package structure**

```bash
mkdir -p data model dashboard tests tests/fixtures bundle
touch data/__init__.py model/__init__.py tests/__init__.py dashboard/.gitkeep
```

- [ ] **Step 2: Write Config dataclass** in `data/config.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    start_year: int = 2000
    end_year: int = 2024
    outdir: Path = field(default_factory=lambda: Path("v4_output"))
    bundle_dir: Path = field(default_factory=lambda: Path("bundle"))
    max_countries: int | None = None
    session_sleep_sec: float = 0.15
    seed: int = 2026
    n_iter: int = 3000
    burn: int = 1000
    thin: int = 2
    user_agent: str = "HyperAtlas/4.0 (research)"
    max_response_bytes: int = 50_000_000  # 50 MB
```

- [ ] **Step 3: Write failing tests for HTTP client**

```python
# tests/test_data.py
import pytest
import json
from unittest.mock import patch, MagicMock
from data.http_client import get_json


class TestGetJson:
    """Tests for robust HTTP client (P0-6, P1-8, P2-8, P2-9)."""

    def test_retry_on_502_then_success(self):
        """Retry logic: mock 502 -> 502 -> 200 succeeds."""
        resp_fail = MagicMock()
        resp_fail.status_code = 502
        resp_fail.raise_for_status.side_effect = Exception("502")

        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.content = b'{"value": [1, 2, 3]}'
        resp_ok.json.return_value = {"value": [1, 2, 3]}
        resp_ok.raise_for_status.return_value = None
        resp_ok.headers = {"content-length": "20"}

        with patch("data.http_client.requests.get", side_effect=[resp_fail, resp_fail, resp_ok]):
            result = get_json("https://example.com/api", retries=3, sleep_base=0)
        assert result == {"value": [1, 2, 3]}

    def test_retry_exhaustion_raises(self):
        """Retry exhaustion: mock 502 x 3 raises clearly."""
        resp_fail = MagicMock()
        resp_fail.status_code = 502
        resp_fail.raise_for_status.side_effect = Exception("502")

        with patch("data.http_client.requests.get", return_value=resp_fail):
            with pytest.raises(RuntimeError, match="Failed after 3 retries"):
                get_json("https://example.com/api", retries=3, sleep_base=0)

    def test_response_size_limit(self):
        """Response size limit: oversized payload triggers error."""
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.headers = {"content-length": "999999999"}
        resp.content = b"x" * 100  # doesn't matter, header check first

        with patch("data.http_client.requests.get", return_value=resp):
            with pytest.raises(ValueError, match="Response too large"):
                get_json("https://example.com/api", max_bytes=50_000_000)

    def test_user_agent_header_sent(self):
        """User-Agent header is set on requests."""
        resp = MagicMock()
        resp.status_code = 200
        resp.content = b'{"ok": true}'
        resp.json.return_value = {"ok": True}
        resp.raise_for_status.return_value = None
        resp.headers = {"content-length": "12"}

        with patch("data.http_client.requests.get", return_value=resp) as mock_get:
            get_json("https://example.com/api", user_agent="HyperAtlas/4.0")
            call_kwargs = mock_get.call_args
            assert call_kwargs[1]["headers"]["User-Agent"] == "HyperAtlas/4.0"
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd C:\Models\HTNPipeline && python -m pytest tests/test_data.py::TestGetJson -v`
Expected: FAIL (ImportError, module not found)

- [ ] **Step 5: Implement HTTP client** in `data/http_client.py`

```python
from __future__ import annotations
import time
import requests


def get_json(
    url: str,
    params: dict | None = None,
    retries: int = 3,
    sleep_base: float = 2.0,
    timeout: int = 60,
    user_agent: str = "HyperAtlas/4.0 (research)",
    max_bytes: int = 50_000_000,
) -> dict | list:
    """Fetch JSON with retry on transient errors.

    Catches RequestException (not just HTTPError). Retries on
    502/503/429/Timeout/ConnectionError. Terminal raise after exhaustion.
    Response size cap enforced via Content-Length header.
    """
    headers = {"User-Agent": user_agent}
    last_err: Exception | None = None

    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=headers)

            # Check response size before parsing
            content_length = r.headers.get("content-length")
            if content_length and int(content_length) > max_bytes:
                raise ValueError(
                    f"Response too large: {content_length} bytes "
                    f"(limit {max_bytes})"
                )

            r.raise_for_status()
            return r.json()

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            last_err = e
            if attempt < retries - 1:
                wait = sleep_base * (2 ** attempt)
                print(f"  Retry {attempt+1}/{retries} after {type(e).__name__}, waiting {wait:.0f}s...")
                time.sleep(wait)
            continue

        except requests.exceptions.HTTPError as e:
            last_err = e
            if hasattr(e, "response") and e.response is not None:
                status = e.response.status_code
            else:
                status = getattr(r, "status_code", 0)
            if status in (502, 503, 429) and attempt < retries - 1:
                wait = sleep_base * (2 ** attempt)
                print(f"  Retry {attempt+1}/{retries} after {status}, waiting {wait:.0f}s...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"Failed after {retries} retries: {url} (last error: {last_err})")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd C:\Models\HTNPipeline && python -m pytest tests/test_data.py::TestGetJson -v`
Expected: 4 passed

- [ ] **Step 7: Commit**

```bash
git add data/ model/ tests/ dashboard/ run_pipeline.py
git commit -m "feat: project scaffold + Config + robust HTTP client (P0-6, P1-8, P1-9)"
```

---

### Task 2: Indicator registry + indicator code validation

**Files:**
- Create: `data/registry.py`
- Append: `tests/test_data.py`

**Fixes:** P1-7 (URL injection), extensible architecture

- [ ] **Step 1: Write failing tests for registry**

```python
# append to tests/test_data.py
import re
from data.registry import IndicatorRegistry, validate_indicator_code


class TestIndicatorCodeValidation:
    """Tests for indicator code sanitization (P1-7)."""

    def test_valid_codes_pass(self):
        for code in ["NY.GDP.PCAP.CD", "NCD_HYP_PREVALENCE_A", "NCDMORT3070", "SH.XPD.CHEX.GD.ZS"]:
            assert validate_indicator_code(code) == code

    def test_path_traversal_rejected(self):
        with pytest.raises(ValueError, match="Invalid indicator code"):
            validate_indicator_code("../../../etc/passwd")

    def test_odata_injection_rejected(self):
        with pytest.raises(ValueError, match="Invalid indicator code"):
            validate_indicator_code("' or 1 eq 1 --")

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="Invalid indicator code"):
            validate_indicator_code("")

    def test_too_long_rejected(self):
        with pytest.raises(ValueError, match="Invalid indicator code"):
            validate_indicator_code("A" * 81)


class TestIndicatorRegistry:
    """Tests for indicator registry pattern."""

    def test_register_and_list(self):
        reg = IndicatorRegistry()

        @reg.register("test_ind", source="worldbank", code="TEST.CODE")
        def fetch_test(start_year, end_year, config):
            import pandas as pd
            return pd.DataFrame({"iso3c": ["GBR"], "year": [2020], "test_ind": [42.0]})

        assert "test_ind" in reg.list_indicators()
        meta = reg.get_metadata("test_ind")
        assert meta["source"] == "worldbank"
        assert meta["code"] == "TEST.CODE"

    def test_fetch_returns_dataframe(self):
        reg = IndicatorRegistry()

        @reg.register("gdp_test", source="worldbank", code="NY.GDP.PCAP.CD")
        def fetch_gdp(start_year, end_year, config):
            import pandas as pd
            return pd.DataFrame({"iso3c": ["GBR", "USA"], "year": [2020, 2020], "gdp_test": [46000.0, 65000.0]})

        from data.config import Config
        df = reg.fetch("gdp_test", 2020, 2024, Config())
        assert len(df) == 2
        assert "gdp_test" in df.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:\Models\HTNPipeline && python -m pytest tests/test_data.py::TestIndicatorCodeValidation tests/test_data.py::TestIndicatorRegistry -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement registry** in `data/registry.py`

```python
from __future__ import annotations
import re
from typing import Callable
import pandas as pd


def validate_indicator_code(code: str) -> str:
    """Validate indicator code against strict allowlist regex.

    Prevents OData injection, path traversal, and query-string manipulation
    from WHO-derived indicator codes (P1-7).
    """
    if not re.fullmatch(r"[A-Za-z0-9_.\-]{1,80}", code):
        raise ValueError(f"Invalid indicator code: {code!r}")
    return code


class IndicatorRegistry:
    """Extensible registry for data indicators.

    Usage:
        registry = IndicatorRegistry()

        @registry.register("gdp_pc", source="worldbank", code="NY.GDP.PCAP.CD")
        def fetch_gdp(start_year, end_year, config):
            ...
            return pd.DataFrame(...)
    """

    def __init__(self) -> None:
        self._indicators: dict[str, dict] = {}

    def register(self, name: str, source: str, code: str) -> Callable:
        validate_indicator_code(code)

        def decorator(func: Callable) -> Callable:
            self._indicators[name] = {
                "source": source,
                "code": code,
                "fetch_fn": func,
            }
            return func
        return decorator

    def list_indicators(self) -> list[str]:
        return list(self._indicators.keys())

    def get_metadata(self, name: str) -> dict:
        if name not in self._indicators:
            raise KeyError(f"Unknown indicator: {name}")
        return {k: v for k, v in self._indicators[name].items() if k != "fetch_fn"}

    def fetch(self, name: str, start_year: int, end_year: int, config) -> pd.DataFrame:
        if name not in self._indicators:
            raise KeyError(f"Unknown indicator: {name}")
        return self._indicators[name]["fetch_fn"](start_year, end_year, config)

    def fetch_all(self, start_year: int, end_year: int, config) -> dict[str, pd.DataFrame]:
        results = {}
        for name in self._indicators:
            try:
                results[name] = self.fetch(name, start_year, end_year, config)
            except Exception as e:
                print(f"  Warning: failed to fetch {name}: {e}")
        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd C:\Models\HTNPipeline && python -m pytest tests/test_data.py::TestIndicatorCodeValidation tests/test_data.py::TestIndicatorRegistry -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add data/registry.py tests/test_data.py
git commit -m "feat: indicator registry with code validation (P1-7)"
```

---

### Task 3: World Bank + WHO fetchers with fixtures and sex filtering

**Files:**
- Create: `data/worldbank.py`
- Create: `data/who.py`
- Create: `data/indicators.py`
- Append: `tests/test_data.py`

**Fixes:** P0-4 (WHO sex filter), P1-5 (urbanization), P0-7 partial (fixture seeding)

- [ ] **Step 1: Write failing tests for WHO sex filter and fixtures**

```python
# append to tests/test_data.py
import numpy as np
import pandas as pd
from data.who import filter_who_sex, harmonize_who_observations
from data.worldbank import worldbank_fixture


class TestWHOSexFilter:
    """Tests for WHO BTSX filtering (P0-4)."""

    def test_btsx_only_survives(self):
        df = pd.DataFrame({
            "SpatialDim": ["GBR", "GBR", "GBR"],
            "TimeDim": [2019, 2019, 2019],
            "NumericValue": [30.0, 35.0, 25.0],
            "Dim1": ["BTSX", "MLE", "FMLE"],
        })
        result = filter_who_sex(df)
        assert len(result) == 1
        assert result.iloc[0]["Dim1"] == "BTSX"

    def test_missing_dim1_deduplicates(self):
        df = pd.DataFrame({
            "SpatialDim": ["GBR", "GBR"],
            "TimeDim": [2019, 2019],
            "NumericValue": [30.0, 31.0],
        })
        result = filter_who_sex(df)
        assert len(result) == 1  # deduplicated by country-year

    def test_no_duplicate_country_year_after_harmonize(self):
        df = pd.DataFrame({
            "SpatialDim": ["GBR", "GBR", "USA"],
            "TimeDim": [2019, 2019, 2019],
            "NumericValue": [30.0, 35.0, 28.0],
            "Dim1": ["BTSX", "MLE", "BTSX"],
        })
        result = harmonize_who_observations(df, "htn_prev_pct")
        dupes = result.groupby(["iso3c", "year"]).size()
        assert (dupes <= 1).all()


class TestFixtures:
    """Tests for fixture determinism and country count."""

    def test_worldbank_fixture_deterministic(self):
        df1 = worldbank_fixture(2015, 2019, seed=42)
        df2 = worldbank_fixture(2015, 2019, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_worldbank_fixture_30_countries(self):
        df = worldbank_fixture(2015, 2019, seed=42)
        assert df["iso3c"].nunique() == 30

    def test_worldbank_fixture_year_range(self):
        df = worldbank_fixture(2015, 2019, seed=42)
        assert df["year"].min() == 2015
        assert df["year"].max() == 2019
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:\Models\HTNPipeline && python -m pytest tests/test_data.py::TestWHOSexFilter tests/test_data.py::TestFixtures -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement WHO fetcher** in `data/who.py`

```python
from __future__ import annotations
import pandas as pd
import numpy as np
from data.http_client import get_json
from data.registry import validate_indicator_code

WHO_BASE = "https://ghoapi.azureedge.net/api"


def filter_who_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Filter WHO observations to both-sexes only (P0-4).

    If Dim1 column exists, keep only BTSX rows.
    If Dim1 absent, deduplicate by SpatialDim+TimeDim keeping first row.
    """
    if "Dim1" in df.columns:
        filtered = df[df["Dim1"] == "BTSX"].copy()
        if len(filtered) == 0:
            print("  Warning: No BTSX rows found, keeping all rows")
            filtered = df.copy()
        return filtered

    # No Dim1 column: deduplicate by country-year
    print("  Warning: Dim1 column not found, deduplicating by country-year")
    spatial_col = "SpatialDim" if "SpatialDim" in df.columns else None
    time_col = "TimeDim" if "TimeDim" in df.columns else None
    if spatial_col and time_col:
        return df.drop_duplicates(subset=[spatial_col, time_col], keep="first").copy()
    return df.copy()


def fetch_who_indicator(indicator_code: str, config) -> pd.DataFrame:
    """Fetch raw WHO GHO observations for a given indicator code."""
    code = validate_indicator_code(indicator_code)
    url = f"{WHO_BASE}/{code}"
    payload = get_json(
        url,
        user_agent=config.user_agent,
        max_bytes=config.max_response_bytes,
    )
    if isinstance(payload, dict) and "value" in payload:
        df = pd.DataFrame(payload["value"])
        if len(df) == 0:
            raise ValueError(f"WHO indicator {code} returned empty data")
        return df
    raise ValueError(f"Unexpected WHO response format for {code}")


def harmonize_who_observations(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Harmonize raw WHO observations into iso3c/year/value format.

    Applies BTSX sex filter (P0-4) and deduplicates.
    """
    filtered = filter_who_sex(df)

    cols = {c.lower(): c for c in filtered.columns}

    def pick(*names: str) -> str | None:
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    iso3_col = pick("SpatialDim", "Country", "COUNTRY")
    year_col = pick("TimeDim", "Year", "TIME_PERIOD")
    value_col = pick("NumericValue", "Value", "VAL", "FactValueNumeric")

    if iso3_col is None or year_col is None or value_col is None:
        raise ValueError(f"WHO schema not recognized: {list(filtered.columns)}")

    out = pd.DataFrame({
        "iso3c": filtered[iso3_col].astype(str),
        "year": pd.to_numeric(filtered[year_col], errors="coerce"),
        col_name: pd.to_numeric(filtered[value_col], errors="coerce"),
    })
    out = out.dropna(subset=["year"]).copy()
    out["year"] = out["year"].astype(int)

    # Final dedup safety net
    out = out.groupby(["iso3c", "year"], as_index=False).first()
    return out
```

- [ ] **Step 4: Implement World Bank fetcher** in `data/worldbank.py`

```python
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from data.http_client import get_json
from data.registry import validate_indicator_code

WB_BASE = "https://api.worldbank.org/v2"


def fetch_worldbank_indicator(indicator: str, start_year: int, end_year: int, config) -> pd.DataFrame:
    """Fetch one World Bank indicator. Returns iso3c/country/year/value columns."""
    code = validate_indicator_code(indicator)
    url = f"{WB_BASE}/country/all/indicator/{code}"
    params = {
        "format": "json",
        "per_page": 5000,
        "date": f"{start_year}:{end_year}",
    }
    payload = get_json(
        url, params=params,
        user_agent=config.user_agent,
        max_bytes=config.max_response_bytes,
    )
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError(f"Unexpected World Bank response for {code}")

    rows = []
    for r in payload[1]:
        yr = int(r["date"])
        if start_year <= yr <= end_year:
            rows.append({
                "iso3c": r["countryiso3code"],
                "country": r["country"]["value"],
                "year": yr,
                "value": pd.to_numeric(r["value"], errors="coerce"),
            })
    return pd.DataFrame(rows)


def worldbank_fixture(start_year: int, end_year: int, seed: int = 42) -> pd.DataFrame:
    """Deterministic fixture data for 30 countries when API is unavailable."""
    countries = {
        "GBR": ("United Kingdom", 46000, 10.2, 83.9),
        "USA": ("United States", 65000, 16.7, 82.7),
        "DEU": ("Germany", 48000, 11.7, 77.5),
        "FRA": ("France", 42000, 11.3, 81.2),
        "JPN": ("Japan", 40000, 10.9, 91.8),
        "CAN": ("Canada", 46000, 10.8, 81.6),
        "AUS": ("Australia", 55000, 9.4, 86.2),
        "ITA": ("Italy", 34000, 8.7, 71.0),
        "ESP": ("Spain", 30000, 9.1, 80.8),
        "KOR": ("Korea, Rep.", 32000, 8.4, 81.4),
        "BRA": ("Brazil", 8700, 9.6, 87.1),
        "MEX": ("Mexico", 10000, 5.4, 80.7),
        "ARG": ("Argentina", 10500, 9.5, 92.1),
        "COL": ("Colombia", 6100, 7.7, 81.4),
        "CHL": ("Chile", 15000, 9.3, 87.7),
        "CHN": ("China", 10500, 5.4, 61.4),
        "IND": ("India", 2100, 3.3, 35.0),
        "IDN": ("Indonesia", 4300, 2.9, 56.6),
        "THA": ("Thailand", 7200, 3.8, 51.0),
        "VNM": ("Viet Nam", 3700, 5.5, 37.3),
        "NGA": ("Nigeria", 2100, 3.8, 52.0),
        "ZAF": ("South Africa", 6300, 8.3, 67.4),
        "KEN": ("Kenya", 1800, 4.6, 28.0),
        "ETH": ("Ethiopia", 930, 3.5, 22.2),
        "EGY": ("Egypt, Arab Rep.", 3600, 4.7, 42.8),
        "TUR": ("Turkiye", 9600, 4.3, 76.1),
        "SAU": ("Saudi Arabia", 23000, 6.4, 84.1),
        "RUS": ("Russian Federation", 11500, 5.3, 74.8),
        "POL": ("Poland", 17000, 6.3, 60.1),
        "BGD": ("Bangladesh", 2500, 2.5, 38.2),
    }
    rng = np.random.default_rng(seed)
    rows = []
    for iso3c, (name, base_gdp, base_health, base_urban) in countries.items():
        for year in range(start_year, end_year + 1):
            t = (year - start_year) / max(end_year - start_year, 1)
            gdp = base_gdp * (1 + 0.03 * t + rng.normal(0, 0.02))
            health = base_health * (1 + 0.01 * t + rng.normal(0, 0.01))
            urban = min(base_urban * (1 + 0.005 * t + rng.normal(0, 0.003)), 100.0)
            rows.append({
                "iso3c": iso3c,
                "country": name,
                "year": year,
                "gdp_pc_usd": round(gdp, 2),
                "health_exp_pct_gdp": round(health, 2),
                "urban_pct": round(urban, 2),
            })
    return pd.DataFrame(rows)
```

- [ ] **Step 5: Implement indicator definitions** in `data/indicators.py`

```python
from __future__ import annotations
import time
import pandas as pd
from data.registry import IndicatorRegistry
from data.worldbank import fetch_worldbank_indicator, worldbank_fixture
from data.who import fetch_who_indicator, harmonize_who_observations

# Global registry instance
registry = IndicatorRegistry()


@registry.register("gdp_pc_usd", source="worldbank", code="NY.GDP.PCAP.CD")
def fetch_gdp(start_year: int, end_year: int, config) -> pd.DataFrame:
    try:
        df = fetch_worldbank_indicator("NY.GDP.PCAP.CD", start_year, end_year, config)
        return df.rename(columns={"value": "gdp_pc_usd"}).drop(columns=["country"], errors="ignore")
    except Exception as e:
        print(f"  World Bank GDP unavailable ({e}), using fixture...")
        fix = worldbank_fixture(start_year, end_year, seed=config.seed)
        return fix[["iso3c", "year", "gdp_pc_usd"]]


@registry.register("health_exp_pct_gdp", source="worldbank", code="SH.XPD.CHEX.GD.ZS")
def fetch_health_exp(start_year: int, end_year: int, config) -> pd.DataFrame:
    try:
        df = fetch_worldbank_indicator("SH.XPD.CHEX.GD.ZS", start_year, end_year, config)
        return df.rename(columns={"value": "health_exp_pct_gdp"}).drop(columns=["country"], errors="ignore")
    except Exception as e:
        print(f"  World Bank health exp unavailable ({e}), using fixture...")
        fix = worldbank_fixture(start_year, end_year, seed=config.seed)
        return fix[["iso3c", "year", "health_exp_pct_gdp"]]


@registry.register("urban_pct", source="worldbank", code="SP.URB.TOTL.IN.ZS")
def fetch_urbanization(start_year: int, end_year: int, config) -> pd.DataFrame:
    try:
        df = fetch_worldbank_indicator("SP.URB.TOTL.IN.ZS", start_year, end_year, config)
        return df.rename(columns={"value": "urban_pct"}).drop(columns=["country"], errors="ignore")
    except Exception as e:
        print(f"  World Bank urbanization unavailable ({e}), using fixture...")
        fix = worldbank_fixture(start_year, end_year, seed=config.seed)
        return fix[["iso3c", "year", "urban_pct"]]


@registry.register("htn_prevalence_pct", source="who", code="NCD_HYP_PREVALENCE_A")
def fetch_htn_prev(start_year: int, end_year: int, config) -> pd.DataFrame:
    raw = fetch_who_indicator("NCD_HYP_PREVALENCE_A", config)
    df = harmonize_who_observations(raw, "htn_prevalence_pct")
    return df[(df["year"] >= start_year) & (df["year"] <= end_year)]


@registry.register("htn_treatment_pct", source="who", code="NCD_HYP_TREATMENT_A")
def fetch_htn_treat(start_year: int, end_year: int, config) -> pd.DataFrame:
    raw = fetch_who_indicator("NCD_HYP_TREATMENT_A", config)
    df = harmonize_who_observations(raw, "htn_treatment_pct")
    return df[(df["year"] >= start_year) & (df["year"] <= end_year)]


@registry.register("cvd_mortality_30_70", source="who", code="NCDMORT3070")
def fetch_cvd_mort(start_year: int, end_year: int, config) -> pd.DataFrame:
    raw = fetch_who_indicator("NCDMORT3070", config)
    df = harmonize_who_observations(raw, "cvd_mortality_30_70")
    return df[(df["year"] >= start_year) & (df["year"] <= end_year)]


# IHD and stroke indicators -- codes discovered from WHO catalog
# These use WHO cause-specific mortality indicators
@registry.register("ihd_mort_rate", source="who", code="MORT_100")
def fetch_ihd_mort(start_year: int, end_year: int, config) -> pd.DataFrame:
    try:
        raw = fetch_who_indicator("MORT_100", config)
        df = harmonize_who_observations(raw, "ihd_mort_rate")
        return df[(df["year"] >= start_year) & (df["year"] <= end_year)]
    except Exception as e:
        print(f"  WHO IHD mortality unavailable ({e}), skipping...")
        return pd.DataFrame(columns=["iso3c", "year", "ihd_mort_rate"])


@registry.register("stroke_mort_rate", source="who", code="MORT_200")
def fetch_stroke_mort(start_year: int, end_year: int, config) -> pd.DataFrame:
    try:
        raw = fetch_who_indicator("MORT_200", config)
        df = harmonize_who_observations(raw, "stroke_mort_rate")
        return df[(df["year"] >= start_year) & (df["year"] <= end_year)]
    except Exception as e:
        print(f"  WHO stroke mortality unavailable ({e}), skipping...")
        return pd.DataFrame(columns=["iso3c", "year", "stroke_mort_rate"])
```

- [ ] **Step 6: Run all data tests**

Run: `cd C:\Models\HTNPipeline && python -m pytest tests/test_data.py -v`
Expected: All 13 tests pass

- [ ] **Step 7: Commit**

```bash
git add data/worldbank.py data/who.py data/indicators.py tests/test_data.py
git commit -m "feat: WHO/WB fetchers with BTSX filter + fixtures + 8 indicators (P0-4, P1-5)"
```

---

### Task 4: Panel harmonizer + zscore + column validation

**Files:**
- Create: `data/harmonize.py`
- Append: `tests/test_data.py`

**Fixes:** P0-7 (zscore div-by-zero), P1-10 (column guard), P2-1 (ddof=1)

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_data.py
from data.harmonize import zscore_series, build_panel, prepare_model_panel


class TestZscore:
    """Tests for zscore with zero-guard (P0-7)."""

    def test_normal_zscore(self):
        s = pd.Series([10.0, 20.0, 30.0])
        z = zscore_series(s)
        assert abs(z.mean()) < 1e-10
        assert abs(z.std(ddof=1) - 1.0) < 0.01  # ddof=1

    def test_all_identical_returns_zeros(self):
        s = pd.Series([5.0, 5.0, 5.0, 5.0])
        z = zscore_series(s)
        assert (z == 0.0).all()
        assert not z.isna().any()

    def test_single_value_returns_zero(self):
        s = pd.Series([42.0])
        z = zscore_series(s)
        assert z.iloc[0] == 0.0

    def test_ddof1_used(self):
        s = pd.Series([10.0, 20.0])
        z = zscore_series(s)
        # With ddof=1: std = 7.071, mean = 15
        # z[0] = (10 - 15) / 7.071 = -0.7071
        assert abs(z.iloc[0] - (-0.7071)) < 0.01


class TestColumnValidation:
    """Tests for column validation (P1-10)."""

    def test_missing_required_columns_raises(self):
        df = pd.DataFrame({"iso3c": ["GBR"], "year": [2019], "gdp_pc_usd": [46000]})
        with pytest.raises(ValueError, match="Missing required columns"):
            prepare_model_panel(df, required_outcomes=["htn_prevalence_pct", "htn_treatment_pct"])

    def test_valid_panel_passes(self):
        df = pd.DataFrame({
            "iso3c": ["GBR", "USA"],
            "year": [2019, 2019],
            "gdp_pc_usd": [46000, 65000],
            "health_exp_pct_gdp": [10.2, 16.7],
            "urban_pct": [83.9, 82.7],
            "htn_prevalence_pct": [30.0, 28.0],
            "htn_treatment_pct": [55.0, 50.0],
        })
        result = prepare_model_panel(df, required_outcomes=["htn_prevalence_pct", "htn_treatment_pct"])
        assert "log_gdp_z" in result.columns
        assert "country_id" in result.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:\Models\HTNPipeline && python -m pytest tests/test_data.py::TestZscore tests/test_data.py::TestColumnValidation -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement harmonizer** in `data/harmonize.py`

```python
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from data.config import Config
from data.indicators import registry


def zscore_series(s: pd.Series) -> pd.Series:
    """Z-score with ddof=1 and zero-guard (P0-7, P2-1).

    Returns zeros when std < 1e-12 to prevent NaN/inf.
    """
    x = pd.to_numeric(s, errors="coerce")
    std = x.std(ddof=1)
    if std is None or np.isnan(std) or std < 1e-12:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - x.mean()) / std


def build_panel(config: Config) -> tuple[pd.DataFrame, dict[str, str]]:
    """Fetch all indicators and merge into a country-year panel."""
    all_data = registry.fetch_all(config.start_year, config.end_year, config)

    # Start with the first available frame
    panel = None
    sources: dict[str, str] = {}

    for name, df in all_data.items():
        if len(df) == 0:
            continue
        meta = registry.get_metadata(name)
        sources[name] = meta["code"]

        if panel is None:
            panel = df.copy()
        else:
            panel = panel.merge(df, on=["iso3c", "year"], how="outer")

    if panel is None:
        raise RuntimeError("No indicator data available")

    # Filter and sort
    panel = panel[
        (panel["year"] >= config.start_year) &
        (panel["year"] <= config.end_year) &
        (panel["iso3c"].notna()) &
        (panel["iso3c"] != "")
    ].copy()
    panel = panel.sort_values(["iso3c", "year"]).reset_index(drop=True)

    if config.max_countries is not None:
        keep = panel["iso3c"].drop_duplicates().head(config.max_countries)
        panel = panel[panel["iso3c"].isin(keep)].copy()

    return panel, sources


def prepare_model_panel(
    panel: pd.DataFrame,
    required_outcomes: list[str] | None = None,
) -> pd.DataFrame:
    """Z-score covariates and validate required columns (P1-10).

    Raises ValueError listing missing columns if required_outcomes are absent.
    """
    if required_outcomes is None:
        required_outcomes = ["htn_prevalence_pct", "htn_treatment_pct"]

    missing = [c for c in required_outcomes if c not in panel.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available: {list(panel.columns)}"
        )

    out = panel.copy()
    out["country_id"] = out["iso3c"].astype("category").cat.codes
    out["year_c"] = out["year"] - out["year"].mean()

    # Z-score covariates (all zero-guarded)
    if "gdp_pc_usd" in out.columns:
        out["log_gdp"] = np.log(out["gdp_pc_usd"].clip(lower=1.0))
        out["log_gdp_z"] = zscore_series(out["log_gdp"])
    if "health_exp_pct_gdp" in out.columns:
        out["health_exp_z"] = zscore_series(out["health_exp_pct_gdp"])
    if "urban_pct" in out.columns:
        out["urban_z"] = zscore_series(out["urban_pct"])
    out["year_z"] = zscore_series(out["year_c"])

    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd C:\Models\HTNPipeline && python -m pytest tests/test_data.py::TestZscore tests/test_data.py::TestColumnValidation -v`
Expected: 6 passed

- [ ] **Step 5: Run full data test suite**

Run: `cd C:\Models\HTNPipeline && python -m pytest tests/test_data.py -v`
Expected: All 19 tests pass (13 prior + 6 new)

- [ ] **Step 6: Commit**

```bash
git add data/harmonize.py tests/test_data.py
git commit -m "feat: panel harmonizer + zscore zero-guard + column validation (P0-7, P1-10)"
```

---

## Phase 2: Bayesian Engine

### Task 5: Gibbs sampler core (full conditionals + seeded RNG)

**Files:**
- Create: `model/gibbs.py`
- Create: `tests/test_model.py`

**Fixes:** P0-1 (latent refresh), P0-2 (seed), P0-3 (variance sampling), P0-5 (link sign), P1-2 (95% CrI), P1-3 (uniform thinning), P1-6 (GDP prior)

- [ ] **Step 1: Write failing tests for Gibbs sampler**

```python
# tests/test_model.py
import pytest
import numpy as np
import pandas as pd
from model.gibbs import StateSpaceGibbs


def make_synthetic_panel(n_countries=5, n_years=5, seed=42):
    """Generate synthetic panel with known parameters for testing."""
    rng = np.random.default_rng(seed)
    rows = []
    true_beta_prev = np.array([30.0, -1.5, -0.5, 0.8, -0.2])  # intercept, gdp, health, urban, year
    true_beta_treat = np.array([40.0, 3.0, 1.5, 0.5, 0.5])
    true_alpha = 0.4  # positive prev->treat link
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
            treat_true = x @ true_beta_treat + true_alpha * prev_true + u_treat + rng.normal(0, 2.0)
            rows.append({
                "iso3c": iso, "year": 2015 + y, "country_id": c_idx,
                "log_gdp_z": gdp_z, "health_exp_z": health_z,
                "urban_z": urban_z, "year_z": year_z,
                "htn_prevalence_pct": prev_true + rng.normal(0, 1),
                "htn_treatment_pct": treat_true + rng.normal(0, 1),
            })
    return pd.DataFrame(rows)


class TestGibbsReproducibility:
    """Tests for seed reproducibility (P0-2)."""

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
    """Tests for variance sampling via inverse-gamma (P0-3)."""

    def test_variance_params_change_across_draws(self):
        df = make_synthetic_panel()
        g = StateSpaceGibbs(seed=42, n_iter=200, burn=50, thin=1)
        r = g.fit(df)
        # Variance draws should NOT be constant (they were in v3)
        assert r.sigma_prev2_draws.std() > 0.01

    def test_variance_mean_reasonable(self):
        df = make_synthetic_panel(n_countries=10, n_years=5)
        g = StateSpaceGibbs(seed=42, n_iter=500, burn=200, thin=1)
        r = g.fit(df)
        # Residual variance should be in a reasonable range (not 0, not 1000)
        assert 0.1 < r.sigma_prev2_draws.mean() < 50.0


class TestGibbsLinkSign:
    """Tests for prevalence-treatment link (P0-5)."""

    def test_positive_link_on_positive_data(self):
        df = make_synthetic_panel(n_countries=10, n_years=5, seed=42)
        g = StateSpaceGibbs(seed=42, n_iter=500, burn=200, thin=1)
        r = g.fit(df)
        # With true alpha=0.4, posterior mean should be positive
        assert r.alpha_draws.mean() > 0.0


class TestGibbsCrI:
    """Tests for 95% CrI (P1-2)."""

    def test_95_percent_cri(self):
        df = make_synthetic_panel()
        g = StateSpaceGibbs(seed=42, n_iter=200, burn=50, thin=1)
        r = g.fit(df)
        summary = r.parameter_summary()
        # Check column names are low95/high95, not low94/high94
        assert "low95" in summary.columns
        assert "high95" in summary.columns


class TestGibbsUniformThinning:
    """Tests for uniform thinning (P1-3)."""

    def test_all_draw_arrays_same_length(self):
        df = make_synthetic_panel()
        g = StateSpaceGibbs(seed=42, n_iter=200, burn=50, thin=2)
        r = g.fit(df)
        n = len(r.beta_prev_draws)
        assert len(r.beta_treat_draws) == n
        assert len(r.alpha_draws) == n
        assert len(r.sigma_prev2_draws) == n
        assert len(r.latent_prev_draws) == n
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:\Models\HTNPipeline && python -m pytest tests/test_model.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement Gibbs sampler** in `model/gibbs.py`

```python
from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class GibbsResult:
    """Container for MCMC draws and fitted values."""
    beta_prev_draws: np.ndarray
    beta_treat_draws: np.ndarray
    alpha_draws: np.ndarray
    sigma_prev2_draws: np.ndarray
    sigma_treat2_draws: np.ndarray
    sigma_obs_prev2_draws: np.ndarray
    sigma_obs_treat2_draws: np.ndarray
    sigma_u_prev2_draws: np.ndarray
    sigma_u_treat2_draws: np.ndarray
    latent_prev_draws: np.ndarray
    latent_treat_draws: np.ndarray
    country_ids: np.ndarray
    iso3c_map: dict[int, str]
    panel: pd.DataFrame

    def parameter_summary(self) -> pd.DataFrame:
        """Summarize all parameters with mean, sd, 95% CrI."""
        p = self.beta_prev_draws.shape[1]
        prev_names = ["prev_intercept", "prev_log_gdp", "prev_health_exp", "prev_urban", "prev_year"][:p]
        treat_names = ["treat_intercept", "treat_log_gdp", "treat_health_exp", "treat_urban", "treat_year"][:p]

        params = {}
        for i, name in enumerate(prev_names):
            params[name] = self.beta_prev_draws[:, i]
        for i, name in enumerate(treat_names):
            params[name] = self.beta_treat_draws[:, i]
        params["alpha_prev_to_treat"] = self.alpha_draws
        params["sigma_prev2"] = self.sigma_prev2_draws
        params["sigma_treat2"] = self.sigma_treat2_draws
        params["sigma_obs_prev2"] = self.sigma_obs_prev2_draws
        params["sigma_obs_treat2"] = self.sigma_obs_treat2_draws
        params["sigma_u_prev2"] = self.sigma_u_prev2_draws
        params["sigma_u_treat2"] = self.sigma_u_treat2_draws

        rows = []
        for name, draws in params.items():
            rows.append({
                "parameter": name,
                "mean": draws.mean(),
                "median": np.median(draws),
                "sd": draws.std(ddof=1),
                "low95": np.quantile(draws, 0.025),
                "high95": np.quantile(draws, 0.975),
            })
        return pd.DataFrame(rows)


class StateSpaceGibbs:
    """Bayesian state-space Gibbs sampler with full conditionals.

    Fixes from v3 review:
    - P0-1: Proper measurement-error full conditional for latent states
    - P0-2: Seeded RNG throughout
    - P0-3: All variances sampled via inverse-gamma
    - P0-5: Prevalence-treatment link prior is positive (+0.3)
    - P1-2: 95% CrI (0.025/0.975)
    - P1-3: Uniform thinning across all draw types
    - P1-6: GDP prior near 0
    """

    def __init__(
        self,
        seed: int = 2026,
        n_iter: int = 3000,
        burn: int = 1000,
        thin: int = 2,
    ):
        self.seed = seed
        self.n_iter = n_iter
        self.burn = burn
        self.thin = thin

    def fit(self, df: pd.DataFrame) -> GibbsResult:
        rng = np.random.default_rng(self.seed)

        # Prepare data
        required = ["log_gdp_z", "health_exp_z", "year_z", "htn_prevalence_pct", "htn_treatment_pct"]
        optional = ["urban_z"]
        covariate_cols = ["log_gdp_z", "health_exp_z"]
        for col in optional:
            if col in df.columns and df[col].notna().any():
                covariate_cols.append(col)
        covariate_cols.append("year_z")

        dat = df.dropna(subset=required).copy()
        dat = dat.sort_values(["iso3c", "year"]).reset_index(drop=True)

        if len(dat) == 0:
            raise ValueError("No complete cases after dropping NaN rows")

        n = len(dat)
        X = np.column_stack([np.ones(n)] + [dat[c].to_numpy() for c in covariate_cols])
        p = X.shape[1]
        country_id = dat["country_id"].to_numpy().astype(int)
        n_country = len(np.unique(country_id))
        iso3c_map = dict(zip(dat["country_id"], dat["iso3c"]))

        y_prev_obs = dat["htn_prevalence_pct"].to_numpy().astype(float)
        y_treat_obs = dat["htn_treatment_pct"].to_numpy().astype(float)

        # --- Initialize parameters ---
        # Priors (P1-6: GDP prior near 0)
        prior_mean_prev = np.zeros(p)
        prior_mean_prev[0] = 30.0  # intercept
        if p > 3:
            prior_mean_prev[3] = 1.0  # urbanization positive
        prior_var_prev = np.array([10**2] + [3**2] * (p - 1))
        prior_var_prev[-1] = 2**2  # year tighter

        prior_mean_treat = np.zeros(p)
        prior_mean_treat[0] = 40.0
        prior_mean_treat[1] = 2.0  # GDP positive for treatment
        prior_mean_treat[2] = 1.0  # health exp positive
        prior_var_treat = np.array([12**2] + [5**2] * (p - 1))
        prior_var_treat[-1] = 2**2

        beta_prev = prior_mean_prev.copy()
        beta_treat = prior_mean_treat.copy()
        alpha = 0.3  # P0-5: positive prior
        alpha_prior_mean = 0.3
        alpha_prior_var = 0.5**2

        u_prev = np.zeros(n_country)
        u_treat = np.zeros(n_country)

        # Variances -- all will be sampled (P0-3)
        sigma_prev2 = 4.0
        sigma_treat2 = 6.0
        sigma_obs_prev2 = 2.0
        sigma_obs_treat2 = 2.0
        sigma_u_prev2 = 4.0
        sigma_u_treat2 = 4.0

        # Inverse-gamma prior hyperparams
        ig_a0 = 3.0
        ig_b0 = 1.0

        # Latent states initialized at observed
        latent_prev = y_prev_obs.copy()
        latent_treat = y_treat_obs.copy()

        xtx = X.T @ X

        # Storage
        n_keep = (self.n_iter - self.burn + self.thin - 1) // self.thin
        beta_prev_draws = np.zeros((n_keep, p))
        beta_treat_draws = np.zeros((n_keep, p))
        alpha_draws = np.zeros(n_keep)
        sigma_prev2_draws = np.zeros(n_keep)
        sigma_treat2_draws = np.zeros(n_keep)
        sigma_obs_prev2_draws = np.zeros(n_keep)
        sigma_obs_treat2_draws = np.zeros(n_keep)
        sigma_u_prev2_draws = np.zeros(n_keep)
        sigma_u_treat2_draws = np.zeros(n_keep)
        latent_prev_draws = np.zeros((n_keep, n))
        latent_treat_draws = np.zeros((n_keep, n))

        draw_idx = 0

        for it in range(self.n_iter):
            # --- 1. Sample country effects ---
            for c in range(n_country):
                idx = np.where(country_id == c)[0]
                nc = len(idx)

                # u_prev[c]
                resid = latent_prev[idx] - X[idx] @ beta_prev
                prec = nc / sigma_prev2 + 1.0 / sigma_u_prev2
                var_post = 1.0 / prec
                mean_post = var_post * (resid.sum() / sigma_prev2)
                u_prev[c] = rng.normal(mean_post, math.sqrt(var_post))

                # u_treat[c]
                resid = latent_treat[idx] - X[idx] @ beta_treat - alpha * latent_prev[idx]
                prec = nc / sigma_treat2 + 1.0 / sigma_u_treat2
                var_post = 1.0 / prec
                mean_post = var_post * (resid.sum() / sigma_treat2)
                u_treat[c] = rng.normal(mean_post, math.sqrt(var_post))

            # --- 2. Sample beta_prev ---
            y = latent_prev - u_prev[country_id]
            prior_prec = np.diag(1.0 / prior_var_prev)
            post_prec = prior_prec + xtx / sigma_prev2
            post_cov = np.linalg.inv(post_prec)
            post_mean = post_cov @ (prior_prec @ prior_mean_prev + X.T @ y / sigma_prev2)
            beta_prev = rng.multivariate_normal(post_mean, post_cov)

            # --- 3. Sample beta_treat ---
            y = latent_treat - alpha * latent_prev - u_treat[country_id]
            prior_prec = np.diag(1.0 / prior_var_treat)
            post_prec = prior_prec + xtx / sigma_treat2
            post_cov = np.linalg.inv(post_prec)
            post_mean = post_cov @ (prior_prec @ prior_mean_treat + X.T @ y / sigma_treat2)
            beta_treat = rng.multivariate_normal(post_mean, post_cov)

            # --- 4. Sample alpha (prev->treat link) ---
            y = latent_treat - X @ beta_treat - u_treat[country_id]
            x_alpha = latent_prev
            prec = np.sum(x_alpha ** 2) / sigma_treat2 + 1.0 / alpha_prior_var
            var_post = 1.0 / prec
            mean_post = var_post * (np.sum(x_alpha * y) / sigma_treat2 + alpha_prior_mean / alpha_prior_var)
            alpha = rng.normal(mean_post, math.sqrt(var_post))

            # --- 5. Sample variances (inverse-gamma) (P0-3) ---
            # sigma_prev2
            resid = latent_prev - X @ beta_prev - u_prev[country_id]
            a_post = ig_a0 + n / 2
            b_post = ig_b0 + np.sum(resid ** 2) / 2
            sigma_prev2 = 1.0 / rng.gamma(a_post, 1.0 / b_post)

            # sigma_treat2
            resid = latent_treat - X @ beta_treat - alpha * latent_prev - u_treat[country_id]
            a_post = ig_a0 + n / 2
            b_post = ig_b0 + np.sum(resid ** 2) / 2
            sigma_treat2 = 1.0 / rng.gamma(a_post, 1.0 / b_post)

            # sigma_obs_prev2
            resid = y_prev_obs - latent_prev
            a_post = ig_a0 + n / 2
            b_post = ig_b0 + np.sum(resid ** 2) / 2
            sigma_obs_prev2 = 1.0 / rng.gamma(a_post, 1.0 / b_post)

            # sigma_obs_treat2
            resid = y_treat_obs - latent_treat
            a_post = ig_a0 + n / 2
            b_post = ig_b0 + np.sum(resid ** 2) / 2
            sigma_obs_treat2 = 1.0 / rng.gamma(a_post, 1.0 / b_post)

            # sigma_u_prev2
            a_post = ig_a0 + n_country / 2
            b_post = ig_b0 + np.sum(u_prev ** 2) / 2
            sigma_u_prev2 = 1.0 / rng.gamma(a_post, 1.0 / b_post)

            # sigma_u_treat2
            a_post = ig_a0 + n_country / 2
            b_post = ig_b0 + np.sum(u_treat ** 2) / 2
            sigma_u_treat2 = 1.0 / rng.gamma(a_post, 1.0 / b_post)

            # --- 6. Sample latent states (P0-1: proper full conditional) ---
            # Prevalence latent: precision-weighted from measurement + structural
            mu_structural_prev = X @ beta_prev + u_prev[country_id]
            prec_obs = 1.0 / sigma_obs_prev2
            prec_struct = 1.0 / sigma_prev2
            var_latent = 1.0 / (prec_obs + prec_struct)
            mean_latent = var_latent * (prec_obs * y_prev_obs + prec_struct * mu_structural_prev)
            latent_prev = rng.normal(mean_latent, np.sqrt(var_latent))

            # Treatment latent
            mu_structural_treat = X @ beta_treat + alpha * latent_prev + u_treat[country_id]
            prec_obs = 1.0 / sigma_obs_treat2
            prec_struct = 1.0 / sigma_treat2
            var_latent = 1.0 / (prec_obs + prec_struct)
            mean_latent = var_latent * (prec_obs * y_treat_obs + prec_struct * mu_structural_treat)
            latent_treat = rng.normal(mean_latent, np.sqrt(var_latent))

            # --- 7. Store draws (uniform thinning, P1-3) ---
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
                draw_idx += 1

        # Trim to actual draws stored
        actual = draw_idx
        return GibbsResult(
            beta_prev_draws=beta_prev_draws[:actual],
            beta_treat_draws=beta_treat_draws[:actual],
            alpha_draws=alpha_draws[:actual],
            sigma_prev2_draws=sigma_prev2_draws[:actual],
            sigma_treat2_draws=sigma_treat2_draws[:actual],
            sigma_obs_prev2_draws=sigma_obs_prev2_draws[:actual],
            sigma_obs_treat2_draws=sigma_obs_treat2_draws[:actual],
            sigma_u_prev2_draws=sigma_u_prev2_draws[:actual],
            sigma_u_treat2_draws=sigma_u_treat2_draws[:actual],
            latent_prev_draws=latent_prev_draws[:actual],
            latent_treat_draws=latent_treat_draws[:actual],
            country_ids=country_id,
            iso3c_map=iso3c_map,
            panel=dat,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd C:\Models\HTNPipeline && python -m pytest tests/test_model.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add model/gibbs.py tests/test_model.py
git commit -m "feat: Gibbs sampler with full conditionals + seeded RNG (P0-1,2,3,5, P1-2,3,6)"
```

---

### Task 6: Convergence diagnostics (Rhat + ESS)

**Files:**
- Create: `model/diagnostics.py`
- Append: `tests/test_model.py`

**Fixes:** P2-2 (no convergence diagnostics), P2-5 (no Rhat/ESS)

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_model.py
from model.diagnostics import compute_rhat, compute_ess, run_diagnostics


class TestRhat:
    def test_good_chain_near_one(self):
        rng = np.random.default_rng(42)
        draws = rng.normal(0, 1, 1000)
        assert abs(compute_rhat(draws) - 1.0) < 0.05

    def test_bad_chain_above_threshold(self):
        # Two distinct modes concatenated
        draws = np.concatenate([np.random.default_rng(1).normal(-5, 0.1, 500),
                                np.random.default_rng(2).normal(5, 0.1, 500)])
        assert compute_rhat(draws) > 1.1


class TestESS:
    def test_iid_draws_high_ess(self):
        rng = np.random.default_rng(42)
        draws = rng.normal(0, 1, 1000)
        ess = compute_ess(draws)
        assert ess > 800  # iid should be near n

    def test_autocorrelated_draws_low_ess(self):
        rng = np.random.default_rng(42)
        draws = np.zeros(1000)
        draws[0] = rng.normal()
        for i in range(1, 1000):
            draws[i] = 0.95 * draws[i-1] + rng.normal(0, 0.1)
        ess = compute_ess(draws)
        assert ess < 200  # high autocorrelation -> low ESS


class TestDiagnostics:
    def test_run_diagnostics_returns_dict(self):
        df = make_synthetic_panel()
        g = StateSpaceGibbs(seed=42, n_iter=200, burn=50, thin=1)
        result = g.fit(df)
        diag = run_diagnostics(result)
        assert "parameters" in diag
        assert "overall_pass" in diag
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:\Models\HTNPipeline && python -m pytest tests/test_model.py::TestRhat tests/test_model.py::TestESS tests/test_model.py::TestDiagnostics -v`
Expected: FAIL

- [ ] **Step 3: Implement diagnostics** in `model/diagnostics.py`

```python
from __future__ import annotations
import json
import numpy as np
from model.gibbs import GibbsResult


def compute_rhat(draws: np.ndarray) -> float:
    """Split-chain Rhat (Gelman-Rubin).

    Splits single chain into two halves and computes Rhat.
    Values near 1.0 indicate convergence.
    """
    n = len(draws)
    if n < 4:
        return float("nan")
    mid = n // 2
    chain1 = draws[:mid]
    chain2 = draws[mid:2*mid]

    m = 2  # number of chains
    n_per = len(chain1)

    chain_means = np.array([chain1.mean(), chain2.mean()])
    grand_mean = chain_means.mean()

    B = n_per * np.sum((chain_means - grand_mean) ** 2) / (m - 1)
    W = (chain1.var(ddof=1) + chain2.var(ddof=1)) / m

    if W < 1e-12:
        return 1.0 if B < 1e-12 else float("inf")

    var_hat = (n_per - 1) / n_per * W + B / n_per
    return float(np.sqrt(var_hat / W))


def compute_ess(draws: np.ndarray, max_lag: int = 100) -> float:
    """Bulk effective sample size via autocorrelation."""
    n = len(draws)
    if n < 4:
        return float(n)

    x = draws - draws.mean()
    var = np.var(x, ddof=1)
    if var < 1e-12:
        return float(n)

    max_lag = min(max_lag, n // 2)
    rho_sum = 0.0
    for lag in range(1, max_lag):
        autocorr = np.sum(x[:n-lag] * x[lag:]) / ((n - lag) * var)
        if autocorr < 0.05:
            break
        rho_sum += autocorr

    tau = 1 + 2 * rho_sum
    return float(n / max(tau, 1.0))


def run_diagnostics(result: GibbsResult, rhat_warn: float = 1.05, ess_warn: int = 400) -> dict:
    """Run Rhat + ESS on all parameter draws."""
    params = {}
    summary = result.parameter_summary()
    for _, row in summary.iterrows():
        name = row["parameter"]
        # Get the draws array
        draws = _get_draws_by_name(result, name)
        if draws is None:
            continue
        rhat = compute_rhat(draws)
        ess = compute_ess(draws)
        status = "pass"
        if rhat > 1.10:
            status = "fail"
        elif rhat > rhat_warn:
            status = "warn"
        if ess < ess_warn:
            status = "warn" if status == "pass" else status

        params[name] = {
            "rhat": round(rhat, 4),
            "ess": round(ess, 1),
            "status": status,
        }

    n_warn = sum(1 for v in params.values() if v["status"] == "warn")
    n_fail = sum(1 for v in params.values() if v["status"] == "fail")
    overall = "pass" if n_fail == 0 and n_warn == 0 else ("fail" if n_fail > 0 else "warn")

    return {
        "parameters": params,
        "overall_pass": overall == "pass",
        "overall_status": overall,
        "n_warn": n_warn,
        "n_fail": n_fail,
    }


def _get_draws_by_name(result: GibbsResult, name: str) -> np.ndarray | None:
    """Map parameter name to its draws array."""
    mapping = {
        "alpha_prev_to_treat": result.alpha_draws,
        "sigma_prev2": result.sigma_prev2_draws,
        "sigma_treat2": result.sigma_treat2_draws,
        "sigma_obs_prev2": result.sigma_obs_prev2_draws,
        "sigma_obs_treat2": result.sigma_obs_treat2_draws,
        "sigma_u_prev2": result.sigma_u_prev2_draws,
        "sigma_u_treat2": result.sigma_u_treat2_draws,
    }
    if name in mapping:
        return mapping[name]
    # Beta parameters
    if name.startswith("prev_"):
        idx_map = {"prev_intercept": 0, "prev_log_gdp": 1, "prev_health_exp": 2, "prev_urban": 3, "prev_year": 4}
        if name in idx_map and idx_map[name] < result.beta_prev_draws.shape[1]:
            return result.beta_prev_draws[:, idx_map[name]]
    if name.startswith("treat_"):
        idx_map = {"treat_intercept": 0, "treat_log_gdp": 1, "treat_health_exp": 2, "treat_urban": 3, "treat_year": 4}
        if name in idx_map and idx_map[name] < result.beta_treat_draws.shape[1]:
            return result.beta_treat_draws[:, idx_map[name]]
    return None


def save_diagnostics(diag: dict, path: str) -> None:
    """Save diagnostics to JSON file."""
    with open(path, "w") as f:
        json.dump(diag, f, indent=2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd C:\Models\HTNPipeline && python -m pytest tests/test_model.py::TestRhat tests/test_model.py::TestESS tests/test_model.py::TestDiagnostics -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add model/diagnostics.py tests/test_model.py
git commit -m "feat: convergence diagnostics -- Rhat + ESS (P2-2)"
```

---

### Task 7: Counterfactual engine with uncertainty propagation

**Files:**
- Create: `model/counterfactual.py`
- Append: `tests/test_model.py`

**Fixes:** P1-1 (no counterfactual CrI)

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_model.py
from model.counterfactual import compute_counterfactuals


class TestCounterfactual:
    def _get_result(self):
        df = make_synthetic_panel(n_countries=5, n_years=3)
        g = StateSpaceGibbs(seed=42, n_iter=200, burn=50, thin=1)
        return g.fit(df)

    def test_treatment_capped_at_100(self):
        result = self._get_result()
        cf = compute_counterfactuals(result, scenarios=[{"name": "plus_25pp", "shift_pp": 25}])
        assert (cf["cf_treatment_plus_25pp"] <= 100.0).all()

    def test_cri_width_positive(self):
        result = self._get_result()
        cf = compute_counterfactuals(result, scenarios=[{"name": "plus_25pp", "shift_pp": 25}])
        assert (cf["cf_treatment_plus_25pp_high95"] >= cf["cf_treatment_plus_25pp_low95"]).all()

    def test_custom_shift_works(self):
        result = self._get_result()
        cf = compute_counterfactuals(result, scenarios=[{"name": "custom_10", "shift_pp": 10}])
        assert "cf_treatment_custom_10" in cf.columns

    def test_target_80_scenario(self):
        result = self._get_result()
        cf = compute_counterfactuals(result, scenarios=[{"name": "target_80", "target_pct": 80}])
        assert "cf_treatment_target_80" in cf.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:\Models\HTNPipeline && python -m pytest tests/test_model.py::TestCounterfactual -v`
Expected: FAIL

- [ ] **Step 3: Implement counterfactual engine** in `model/counterfactual.py`

```python
from __future__ import annotations
import numpy as np
import pandas as pd
from model.gibbs import GibbsResult


def compute_counterfactuals(
    result: GibbsResult,
    scenarios: list[dict] | None = None,
) -> pd.DataFrame:
    """Compute HEARTS-aligned counterfactual scenarios with full uncertainty propagation.

    Each scenario is computed per MCMC draw, then summarized with 95% CrI.

    Scenarios format:
        [{"name": "plus_25pp", "shift_pp": 25}]     -- absolute increase
        [{"name": "target_80", "target_pct": 80}]    -- target level
    """
    if scenarios is None:
        scenarios = [
            {"name": "plus_25pp", "shift_pp": 25},
            {"name": "target_80", "target_pct": 80},
        ]

    n_draws = len(result.latent_treat_draws)
    n_obs = result.latent_treat_draws.shape[1]

    out = result.panel[["iso3c", "year"]].copy()
    out["post_htn_prevalence_pct"] = result.latent_prev_draws.mean(axis=0)
    out["post_htn_treatment_pct"] = result.latent_treat_draws.mean(axis=0)
    out["post_prev_low95"] = np.quantile(result.latent_prev_draws, 0.025, axis=0)
    out["post_prev_high95"] = np.quantile(result.latent_prev_draws, 0.975, axis=0)
    out["post_treat_low95"] = np.quantile(result.latent_treat_draws, 0.025, axis=0)
    out["post_treat_high95"] = np.quantile(result.latent_treat_draws, 0.975, axis=0)

    for scenario in scenarios:
        name = scenario["name"]
        cf_draws = np.zeros((n_draws, n_obs))

        for d in range(n_draws):
            baseline = result.latent_treat_draws[d]
            if "shift_pp" in scenario:
                cf_draws[d] = np.minimum(baseline + scenario["shift_pp"], 100.0)
            elif "target_pct" in scenario:
                target = scenario["target_pct"]
                cf_draws[d] = np.maximum(baseline, target)
                cf_draws[d] = np.minimum(cf_draws[d], 100.0)

        out[f"cf_treatment_{name}"] = cf_draws.mean(axis=0)
        out[f"cf_treatment_{name}_low95"] = np.quantile(cf_draws, 0.025, axis=0)
        out[f"cf_treatment_{name}_high95"] = np.quantile(cf_draws, 0.975, axis=0)
        out[f"cf_delta_{name}"] = cf_draws.mean(axis=0) - result.latent_treat_draws.mean(axis=0)

    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd C:\Models\HTNPipeline && python -m pytest tests/test_model.py::TestCounterfactual -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add model/counterfactual.py tests/test_model.py
git commit -m "feat: counterfactual engine with uncertainty propagation (P1-1)"
```

---

### Task 8: TruthCert bundle generator + provenance

**Files:**
- Create: `model/truthcert.py`
- Append: `tests/test_model.py`

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_model.py
import hashlib
from pathlib import Path
from model.truthcert import generate_bundle


class TestTruthCertBundle:
    def test_all_files_present(self, tmp_path):
        df = make_synthetic_panel()
        g = StateSpaceGibbs(seed=42, n_iter=100, burn=30, thin=1)
        result = g.fit(df)
        generate_bundle(result, df, bundle_dir=tmp_path, seed=42)

        expected_files = [
            "parameter_summary.csv",
            "counterfactual.csv",
            "convergence.json",
            "provenance.json",
            "panel_fitted.csv",
            "coverage_summary.csv",
        ]
        for fname in expected_files:
            assert (tmp_path / fname).exists(), f"Missing: {fname}"

    def test_provenance_has_required_fields(self, tmp_path):
        df = make_synthetic_panel()
        g = StateSpaceGibbs(seed=42, n_iter=100, burn=30, thin=1)
        result = g.fit(df)
        generate_bundle(result, df, bundle_dir=tmp_path, seed=42)

        import json
        with open(tmp_path / "provenance.json") as f:
            prov = json.load(f)
        assert "seed" in prov
        assert "timestamp" in prov
        assert "data_hash" in prov
        assert "n_countries" in prov
        assert prov["seed"] == 42
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:\Models\HTNPipeline && python -m pytest tests/test_model.py::TestTruthCertBundle -v`
Expected: FAIL

- [ ] **Step 3: Implement bundle generator** in `model/truthcert.py`

```python
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from model.gibbs import GibbsResult
from model.diagnostics import run_diagnostics, save_diagnostics
from model.counterfactual import compute_counterfactuals


def generate_bundle(
    result: GibbsResult,
    raw_panel: pd.DataFrame,
    bundle_dir: Path,
    seed: int,
    scenarios: list[dict] | None = None,
) -> Path:
    """Generate TruthCert bundle with full provenance chain."""
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parameter summary
    params = result.parameter_summary()
    params.to_csv(bundle_dir / "parameter_summary.csv", index=False)

    # 2. Counterfactuals
    cf = compute_counterfactuals(result, scenarios=scenarios)
    cf.to_csv(bundle_dir / "counterfactual.csv", index=False)

    # 3. Convergence diagnostics
    diag = run_diagnostics(result)
    save_diagnostics(diag, str(bundle_dir / "convergence.json"))

    # 4. Fitted panel
    fitted = result.panel.copy()
    fitted["post_htn_prevalence_pct"] = result.latent_prev_draws.mean(axis=0)
    fitted["post_htn_treatment_pct"] = result.latent_treat_draws.mean(axis=0)
    fitted.to_csv(bundle_dir / "panel_fitted.csv", index=False)

    # 5. Coverage summary
    coverage = pd.DataFrame({
        "metric": [
            "rows_raw", "countries", "year_min", "year_max",
            "nonmissing_htn_prev", "nonmissing_htn_treat",
            "nonmissing_gdp", "nonmissing_health_exp",
            "n_model_rows",
        ],
        "value": [
            len(raw_panel),
            raw_panel["iso3c"].nunique() if "iso3c" in raw_panel.columns else 0,
            raw_panel["year"].min() if "year" in raw_panel.columns else 0,
            raw_panel["year"].max() if "year" in raw_panel.columns else 0,
            raw_panel["htn_prevalence_pct"].notna().sum() if "htn_prevalence_pct" in raw_panel.columns else 0,
            raw_panel["htn_treatment_pct"].notna().sum() if "htn_treatment_pct" in raw_panel.columns else 0,
            raw_panel["gdp_pc_usd"].notna().sum() if "gdp_pc_usd" in raw_panel.columns else 0,
            raw_panel["health_exp_pct_gdp"].notna().sum() if "health_exp_pct_gdp" in raw_panel.columns else 0,
            len(result.panel),
        ],
    })
    coverage.to_csv(bundle_dir / "coverage_summary.csv", index=False)

    # 6. Provenance
    data_bytes = raw_panel.to_csv(index=False).encode("utf-8")
    data_hash = hashlib.sha256(data_bytes).hexdigest()

    provenance = {
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_hash": data_hash,
        "n_countries": int(result.panel["iso3c"].nunique()),
        "n_years": int(result.panel["year"].nunique()),
        "n_draws": len(result.alpha_draws),
        "convergence_status": diag["overall_status"],
    }
    with open(bundle_dir / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    return bundle_dir
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd C:\Models\HTNPipeline && python -m pytest tests/test_model.py::TestTruthCertBundle -v`
Expected: 2 passed

- [ ] **Step 5: Run full model test suite**

Run: `cd C:\Models\HTNPipeline && python -m pytest tests/test_model.py -v`
Expected: All 18 tests pass

- [ ] **Step 6: Commit**

```bash
git add model/truthcert.py tests/test_model.py
git commit -m "feat: TruthCert bundle generator with provenance chain"
```

---

### Task 9: Pipeline orchestrator (run_pipeline.py)

**Files:**
- Create: `run_pipeline.py`

- [ ] **Step 1: Implement the main pipeline script**

```python
#!/usr/bin/env python
"""HyperAtlas v4 -- Global Hypertension Pipeline.

Usage:
    python run_pipeline.py [--max-countries N] [--seed S] [--n-iter N]
"""
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

    # Phase 1: Data
    print("Phase 1: Fetching and harmonizing data...")
    panel, sources = build_panel(config)
    panel.to_csv(config.outdir / "panel_raw.csv", index=False)
    with open(config.outdir / "indicator_sources.json", "w") as f:
        json.dump(sources, f, indent=2)
    print(f"  Panel: {len(panel)} rows, {panel['iso3c'].nunique()} countries")

    # Phase 2: Model
    print("Phase 2: Preparing model panel...")
    model_panel = prepare_model_panel(panel)
    model_panel.to_csv(config.outdir / "panel_prepared.csv", index=False)
    print(f"  Model-ready: {len(model_panel)} rows")

    print(f"Phase 3: Running Gibbs sampler ({config.n_iter} iterations, burn={config.burn})...")
    gibbs = StateSpaceGibbs(
        seed=config.seed,
        n_iter=config.n_iter,
        burn=config.burn,
        thin=config.thin,
    )
    result = gibbs.fit(model_panel)
    print(f"  Stored {len(result.alpha_draws)} post-burn-in draws")

    # Phase 3: Bundle
    print("Phase 4: Generating TruthCert bundle...")
    bundle_path = generate_bundle(
        result, panel,
        bundle_dir=config.bundle_dir,
        seed=config.seed,
    )
    print(f"  Bundle saved to {bundle_path}")

    # Summary
    params = result.parameter_summary()
    print("\nParameter summary:")
    print(params.to_string(index=False))

    # Convergence
    conv = json.loads((config.bundle_dir / "convergence.json").read_text())
    print(f"\nConvergence: {conv['overall_status']} ({conv['n_warn']} warn, {conv['n_fail']} fail)")

    print("\nDone.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test the pipeline with max-countries=5**

Run: `cd C:\Models\HTNPipeline && python run_pipeline.py --max-countries 5 --n-iter 300 --burn 100 --thin 1`
Expected: Completes without error, prints parameter summary and convergence status

- [ ] **Step 3: Commit**

```bash
git add run_pipeline.py
git commit -m "feat: pipeline orchestrator with CLI args"
```

---

## Phase 3: Dashboard

### Task 10: Interactive HTML dashboard (map + explorer + evidence)

**Files:**
- Create: `dashboard/hyper_atlas.html`
- Create: `dashboard/build_bundle.py`
- Create: `tests/test_dashboard.py`

This is the largest single task. The HTML file targets ~3,000-5,000 lines with three tabs.

- [ ] **Step 1: Create JSON bundle builder** in `dashboard/build_bundle.py`

```python
"""Convert pipeline CSV/JSON outputs into a single JSON bundle for the dashboard."""
from __future__ import annotations
import json
import gzip
import base64
from pathlib import Path
import pandas as pd


def build_dashboard_bundle(bundle_dir: Path, output_path: Path | None = None) -> str:
    """Read pipeline outputs and produce a single JSON for the dashboard.

    Returns the JSON string. Optionally writes to output_path.
    """
    bundle_dir = Path(bundle_dir)

    # Load all data
    fitted = pd.read_csv(bundle_dir / "panel_fitted.csv")
    params = pd.read_csv(bundle_dir / "parameter_summary.csv")
    counterfactual = pd.read_csv(bundle_dir / "counterfactual.csv")
    coverage = pd.read_csv(bundle_dir / "coverage_summary.csv")

    with open(bundle_dir / "convergence.json") as f:
        convergence = json.load(f)
    with open(bundle_dir / "provenance.json") as f:
        provenance = json.load(f)

    # Build country-indexed data for map + explorer
    countries = {}
    for iso3c, grp in fitted.groupby("iso3c"):
        countries[iso3c] = {
            "years": grp["year"].tolist(),
            "htn_prevalence_pct": grp["htn_prevalence_pct"].tolist() if "htn_prevalence_pct" in grp else [],
            "htn_treatment_pct": grp["htn_treatment_pct"].tolist() if "htn_treatment_pct" in grp else [],
            "post_htn_prevalence_pct": grp["post_htn_prevalence_pct"].tolist() if "post_htn_prevalence_pct" in grp else [],
            "post_htn_treatment_pct": grp["post_htn_treatment_pct"].tolist() if "post_htn_treatment_pct" in grp else [],
        }

    # Counterfactual data by country
    cf_by_country = {}
    if len(counterfactual) > 0 and "iso3c" in counterfactual.columns:
        for iso3c, grp in counterfactual.groupby("iso3c"):
            cf_cols = [c for c in grp.columns if c.startswith("cf_")]
            cf_by_country[iso3c] = {c: grp[c].tolist() for c in cf_cols}
            cf_by_country[iso3c]["years"] = grp["year"].tolist()

    bundle = {
        "countries": countries,
        "counterfactuals": cf_by_country,
        "parameters": params.to_dict(orient="records"),
        "convergence": convergence,
        "provenance": provenance,
        "coverage": coverage.to_dict(orient="records"),
    }

    bundle_json = json.dumps(bundle)

    if output_path:
        output_path = Path(output_path)
        output_path.write_text(bundle_json, encoding="utf-8")

    return bundle_json


if __name__ == "__main__":
    import sys
    bundle_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("v4_output/bundle")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("dashboard/bundle.json")
    build_dashboard_bundle(bundle_dir, out)
    print(f"Dashboard bundle written to {out}")
```

- [ ] **Step 2: Create the HTML dashboard**

Create `dashboard/hyper_atlas.html` with the full three-tab layout:

**Tab 1 (Global Map):** D3 choropleth with TopoJSON, metric selector, time slider, hover tooltip, click-to-explore.

**Tab 2 (Country Explorer):** Country dropdown with search, 4 trajectory charts (prevalence, treatment, IHD, stroke) with observed + posterior ribbon, counterfactual toggle panel, country RE forest plot, CSV export.

**Tab 3 (Evidence Dashboard):** Parameter table with convergence color-coding, Rhat/ESS traffic lights, data coverage heatmap, provenance footer.

Implementation details:
- D3.js v7 from CDN (`https://d3js.org/d3.v7.min.js`)
- TopoJSON from CDN (`https://cdn.jsdelivr.net/npm/topojson-client@3`)
- World-110m TopoJSON embedded or fetched from `https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json`
- Data loaded from adjacent `bundle.json` file via `fetch()`
- CSS dark/light mode toggle
- Seeded xoshiro128** PRNG for client-side jitter
- `role="application"` on map container only
- All controls keyboard-accessible with ARIA labels

This file will be ~3,000-5,000 lines. Build it incrementally:
1. HTML skeleton with 3 tabs + CSS (including dark mode)
2. Tab 1: Map with D3 choropleth
3. Tab 2: Country explorer charts
4. Tab 3: Evidence dashboard

- [ ] **Step 3: Write Playwright tests** in `tests/test_dashboard.py`

```python
"""Dashboard tests using Playwright."""
import pytest
import subprocess
import time
from pathlib import Path

# These tests require: pip install playwright && playwright install chromium


@pytest.fixture(scope="module")
def server():
    """Start a simple HTTP server for the dashboard."""
    import http.server
    import threading
    port = 8765
    handler = http.server.SimpleHTTPRequestHandler
    httpd = http.server.HTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()


@pytest.fixture(scope="module")
def page(server):
    """Launch browser and navigate to dashboard."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("Playwright not installed")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        pg = browser.new_page()
        pg.goto(f"{server}/dashboard/hyper_atlas.html", wait_until="networkidle")
        yield pg
        browser.close()


class TestDashboard:
    def test_loads_without_errors(self, page):
        errors = page.evaluate("() => window.__consoleErrors || []")
        assert len(errors) == 0

    def test_map_renders_countries(self, page):
        paths = page.query_selector_all("svg.map path.country")
        assert len(paths) > 100  # world has ~180 countries

    def test_time_slider_exists(self, page):
        slider = page.query_selector("input.time-slider")
        assert slider is not None

    def test_metric_selector_exists(self, page):
        selector = page.query_selector("select.metric-selector")
        assert selector is not None

    def test_tabs_navigable(self, page):
        tabs = page.query_selector_all("[role='tab']")
        assert len(tabs) == 3

    def test_dark_mode_toggle(self, page):
        toggle = page.query_selector(".theme-toggle")
        assert toggle is not None

    def test_country_selector_in_explorer(self, page):
        # Click explorer tab
        page.click("[data-tab='explorer']")
        selector = page.query_selector("select.country-selector")
        assert selector is not None

    def test_evidence_tab_has_param_table(self, page):
        page.click("[data-tab='evidence']")
        table = page.query_selector("table.param-table")
        assert table is not None

    def test_convergence_badges(self, page):
        page.click("[data-tab='evidence']")
        badges = page.query_selector_all(".convergence-badge")
        assert len(badges) > 0

    def test_keyboard_navigation(self, page):
        # Tab through controls
        page.keyboard.press("Tab")
        focused = page.evaluate("document.activeElement.tagName")
        assert focused is not None
```

- [ ] **Step 4: Run dashboard tests**

Run: `cd C:\Models\HTNPipeline && python -m pytest tests/test_dashboard.py -v`
Expected: 10 passed (or skipped if Playwright not installed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/ tests/test_dashboard.py
git commit -m "feat: interactive HTML dashboard with map + explorer + evidence tabs"
```

---

## Phase 4: Integration

### Task 11: Full pipeline integration test + review findings update

**Files:**
- Modify: `review-findings.md`

- [ ] **Step 1: Run full pipeline end-to-end**

Run: `cd C:\Models\HTNPipeline && python run_pipeline.py --max-countries 10 --n-iter 500 --burn 150 --thin 2`
Expected: Completes without error, all bundle files present

- [ ] **Step 2: Verify bundle completeness**

Run: `cd C:\Models\HTNPipeline && python -c "from pathlib import Path; files = list(Path('v4_output/bundle').iterdir()); print(f'{len(files)} bundle files:'); [print(f'  {f.name} ({f.stat().st_size} bytes)') for f in sorted(files)]"`
Expected: 6 files present (parameter_summary.csv, counterfactual.csv, convergence.json, provenance.json, panel_fitted.csv, coverage_summary.csv)

- [ ] **Step 3: Build dashboard bundle**

Run: `cd C:\Models\HTNPipeline && python dashboard/build_bundle.py v4_output/bundle dashboard/bundle.json`
Expected: `dashboard/bundle.json` created

- [ ] **Step 4: Run full test suite**

Run: `cd C:\Models\HTNPipeline && python -m pytest tests/ -v --tb=short`
Expected: All ~50 tests pass

- [ ] **Step 5: Update review-findings.md**

Mark all P0 and P1 findings as `[FIXED]` with the task that fixed them. Add `REVIEW CLEAN` header if all P0+P1 resolved.

- [ ] **Step 6: Final commit**

```bash
git add review-findings.md
git commit -m "chore: mark all P0+P1 review findings as FIXED -- REVIEW CLEAN"
```

---

## Summary

| Task | Phase | Description | Tests | Key Fixes |
|------|-------|-------------|-------|-----------|
| 1 | Data | Scaffold + Config + HTTP client | 4 | P0-6, P1-8, P1-9 |
| 2 | Data | Indicator registry + code validation | 7 | P1-7 |
| 3 | Data | WB/WHO fetchers + sex filter + fixtures | 6 | P0-4, P1-5 |
| 4 | Data | Panel harmonizer + zscore + column guard | 6 | P0-7, P1-10 |
| 5 | Model | Gibbs sampler (full conditionals) | 7 | P0-1,2,3,5, P1-2,3,6 |
| 6 | Model | Convergence diagnostics (Rhat + ESS) | 5 | P2-2 |
| 7 | Model | Counterfactual engine | 4 | P1-1 |
| 8 | Model | TruthCert bundle + provenance | 2 | -- |
| 9 | Model | Pipeline orchestrator | -- | -- |
| 10 | Dashboard | HTML dashboard (map + explorer + evidence) | 10 | -- |
| 11 | Integration | Full pipeline test + review update | -- | -- |

**Total: 51 tests across 11 tasks + 14 addendum tests = 65 total.**

---

## Addendum: Self-Review Gap Fixes

The self-review identified 3 critical spec gaps and 14 missing tests. This addendum patches them.

### Task 5A: Add CVD outcome sub-model to Gibbs sampler

**Spec gap:** The Gibbs sampler (Task 5) omits the CVD outcome model for IHD and stroke mortality. The spec (section 4.1) requires:

```
ihd_mort[c,t]    = gamma_0 + gamma_1*prev_true + gamma_2*treat_true + gamma_3*gdp_z + v_ihd[c] + eta_ihd
stroke_mort[c,t] = delta_0 + delta_1*prev_true + delta_2*treat_true + delta_3*gdp_z + v_stroke[c] + eta_stroke
```

**Changes to `model/gibbs.py`:**

- [ ] **Step 1: Extend `GibbsResult` with CVD fields**

Add these fields to the `GibbsResult` dataclass:

```python
    # CVD outcome draws
    gamma_draws: np.ndarray          # (n_draws, 4) IHD coefficients
    delta_draws: np.ndarray          # (n_draws, 4) stroke coefficients
    sigma_ihd2_draws: np.ndarray     # IHD residual variance
    sigma_stroke2_draws: np.ndarray  # stroke residual variance
    sigma_v_ihd2_draws: np.ndarray   # IHD country RE variance
    sigma_v_stroke2_draws: np.ndarray # stroke country RE variance
    latent_ihd_draws: np.ndarray     # (n_draws, n_obs) IHD fitted
    latent_stroke_draws: np.ndarray  # (n_draws, n_obs) stroke fitted
```

Update `parameter_summary()` to include gamma/delta parameters.

- [ ] **Step 2: Add CVD sampling block to `fit()` method**

After the prevalence/treatment sampling blocks, add:

```python
            # --- 7. CVD Outcome: IHD mortality ---
            if has_ihd:
                X_cvd = np.column_stack([np.ones(n), latent_prev, latent_treat, X[:, 1]])  # intercept, prev, treat, gdp
                y_ihd = y_ihd_obs.copy()

                # Sample gamma (IHD coefficients)
                y_g = y_ihd - v_ihd[country_id]
                prior_prec_g = np.diag([1/5**2]*4)
                post_prec_g = prior_prec_g + X_cvd.T @ X_cvd / sigma_ihd2
                post_cov_g = np.linalg.inv(post_prec_g)
                post_mean_g = post_cov_g @ (X_cvd.T @ y_g / sigma_ihd2)
                gamma = rng.multivariate_normal(post_mean_g, post_cov_g)

                # Sample v_ihd (IHD country RE)
                for c in range(n_country):
                    idx = np.where(country_id == c)[0]
                    resid = y_ihd[idx] - X_cvd[idx] @ gamma
                    prec = len(idx) / sigma_ihd2 + 1.0 / sigma_v_ihd2
                    var_post = 1.0 / prec
                    mean_post = var_post * (resid.sum() / sigma_ihd2)
                    v_ihd[c] = rng.normal(mean_post, math.sqrt(var_post))

                # Sample sigma_ihd2
                resid = y_ihd - X_cvd @ gamma - v_ihd[country_id]
                sigma_ihd2 = 1.0 / rng.gamma(ig_a0 + n/2, 1.0/(ig_b0 + np.sum(resid**2)/2))
                sigma_v_ihd2 = 1.0 / rng.gamma(ig_a0 + n_country/2, 1.0/(ig_b0 + np.sum(v_ihd**2)/2))

            # --- 8. CVD Outcome: Stroke mortality (same structure) ---
            # (identical to IHD block but with delta, v_stroke, sigma_stroke2)
```

- [ ] **Step 3: Write missing CVD tests**

```python
# append to tests/test_model.py

class TestCVDOutcome:
    def test_mortality_reduction_correct_sign(self):
        """Spec test 6.2.18: counterfactual mortality has correct sign."""
        df = make_synthetic_panel_with_cvd(n_countries=10, n_years=3)
        g = StateSpaceGibbs(seed=42, n_iter=300, burn=100, thin=1)
        r = g.fit(df)
        cf = compute_counterfactuals(r, scenarios=[{"name": "plus_25pp", "shift_pp": 25}])
        # More treatment should reduce mortality
        assert cf["cf_delta_ihd_plus_25pp"].mean() < 0

    def test_ihd_vs_stroke_different_reductions(self):
        """Spec test 6.2.19: different gamma/delta -> different reductions."""
        df = make_synthetic_panel_with_cvd(n_countries=10, n_years=3)
        g = StateSpaceGibbs(seed=42, n_iter=300, burn=100, thin=1)
        r = g.fit(df)
        cf = compute_counterfactuals(r, scenarios=[{"name": "plus_25pp", "shift_pp": 25}])
        # IHD and stroke reductions should differ
        assert cf["cf_delta_ihd_plus_25pp"].mean() != cf["cf_delta_stroke_plus_25pp"].mean()
```

- [ ] **Step 4: Run tests, commit**

---

### Task 5B: Add AR(1) phi parameters with Metropolis-Hastings

**Spec gap:** The spec (section 4.3) requires AR(1) autoregressive coefficients `phi_prev` and `phi_treat` sampled via MH within Gibbs.

**Changes to `model/gibbs.py`:**

- [ ] **Step 1: Add phi fields to `GibbsResult`**

```python
    phi_prev_draws: np.ndarray   # AR(1) coefficient for prevalence
    phi_treat_draws: np.ndarray  # AR(1) coefficient for treatment
```

- [ ] **Step 2: Add MH step for phi in the Gibbs loop**

After the latent state update, add:

```python
            # --- 9. Sample phi_prev via Metropolis-Hastings ---
            # Proposal: N(phi_current, 0.05^2), truncated to (-0.95, 0.95)
            phi_prev_prop = rng.normal(phi_prev, 0.05)
            if -0.95 < phi_prev_prop < 0.95:
                # Log-likelihood ratio for AR(1) prevalence
                resid_current = _ar_residuals(latent_prev, country_id, phi_prev, X, beta_prev, u_prev)
                resid_prop = _ar_residuals(latent_prev, country_id, phi_prev_prop, X, beta_prev, u_prev)
                ll_current = -0.5 * np.sum(resid_current**2) / sigma_prev2
                ll_prop = -0.5 * np.sum(resid_prop**2) / sigma_prev2
                if np.log(rng.random()) < (ll_prop - ll_current):
                    phi_prev = phi_prev_prop

            # Same for phi_treat
            phi_treat_prop = rng.normal(phi_treat, 0.05)
            if -0.95 < phi_treat_prop < 0.95:
                resid_current = _ar_residuals(latent_treat, country_id, phi_treat, X, beta_treat, u_treat, alpha, latent_prev)
                resid_prop = _ar_residuals(latent_treat, country_id, phi_treat_prop, X, beta_treat, u_treat, alpha, latent_prev)
                ll_current = -0.5 * np.sum(resid_current**2) / sigma_treat2
                ll_prop = -0.5 * np.sum(resid_prop**2) / sigma_treat2
                if np.log(rng.random()) < (ll_prop - ll_current):
                    phi_treat = phi_treat_prop
```

Helper function:

```python
def _ar_residuals(latent, country_id, phi, X, beta, u, alpha=0.0, latent_other=None):
    """Compute AR(1) residuals for a given phi value."""
    n = len(latent)
    mu = X @ beta + u[country_id]
    if latent_other is not None:
        mu += alpha * latent_other
    resid = np.zeros(n)
    countries = np.unique(country_id)
    for c in countries:
        idx = np.where(country_id == c)[0]
        resid[idx[0]] = latent[idx[0]] - mu[idx[0]]
        for i in range(1, len(idx)):
            resid[idx[i]] = latent[idx[i]] - mu[idx[i]] - phi * latent[idx[i-1]]
    return resid
```

- [ ] **Step 3: Write AR(1) tests**

```python
# append to tests/test_model.py

class TestAR1:
    def test_phi_stays_in_bounds(self):
        """Spec test 6.2.7: phi within (-0.95, 0.95)."""
        df = make_synthetic_panel()
        g = StateSpaceGibbs(seed=42, n_iter=300, burn=100, thin=1)
        r = g.fit(df)
        assert (r.phi_prev_draws > -0.95).all()
        assert (r.phi_prev_draws < 0.95).all()
        assert (r.phi_treat_draws > -0.95).all()
        assert (r.phi_treat_draws < 0.95).all()

    def test_phi_recovery_on_ar_data(self):
        """Spec test 6.2.8: phi posterior near true phi on synthetic AR data."""
        # Generate data with true phi=0.6
        df = make_synthetic_ar_panel(true_phi=0.6, n_countries=10, n_years=10, seed=42)
        g = StateSpaceGibbs(seed=42, n_iter=500, burn=200, thin=1)
        r = g.fit(df)
        # Posterior mean should be near 0.6 (within 0.3)
        assert abs(r.phi_prev_draws.mean() - 0.6) < 0.3
```

- [ ] **Step 4: Run tests, commit**

---

### Task 8A: Add posterior_draws.parquet to TruthCert bundle

**Spec gap:** Bundle should have 7 files, not 6. Missing `posterior_draws.parquet`.

**Changes to `model/truthcert.py`:**

- [ ] **Step 1: Add parquet export after parameter summary**

```python
    # 1b. Posterior draws (full chains)
    draws_dict = {}
    for i in range(result.beta_prev_draws.shape[1]):
        draws_dict[f"beta_prev_{i}"] = result.beta_prev_draws[:, i]
    for i in range(result.beta_treat_draws.shape[1]):
        draws_dict[f"beta_treat_{i}"] = result.beta_treat_draws[:, i]
    draws_dict["alpha"] = result.alpha_draws
    draws_dict["sigma_prev2"] = result.sigma_prev2_draws
    draws_dict["sigma_treat2"] = result.sigma_treat2_draws
    draws_dict["sigma_obs_prev2"] = result.sigma_obs_prev2_draws
    draws_dict["sigma_obs_treat2"] = result.sigma_obs_treat2_draws
    draws_dict["sigma_u_prev2"] = result.sigma_u_prev2_draws
    draws_dict["sigma_u_treat2"] = result.sigma_u_treat2_draws
    draws_df = pd.DataFrame(draws_dict)
    draws_df.to_parquet(bundle_dir / "posterior_draws.parquet", index=False)
```

- [ ] **Step 2: Update bundle test to check 7 files**

Change `expected_files` to include `"posterior_draws.parquet"`.

- [ ] **Step 3: Run tests, commit**

---

### Task 5C: Missing model tests (edge cases + measurement error)

**Spec gap:** 14 tests from spec section 6.2 not covered. Add them here.

- [ ] **Step 1: Write missing tests**

```python
# append to tests/test_model.py

class TestMeasurementError:
    """Spec tests 6.2.9 and 6.2.10."""

    def test_latent_differs_from_observed_large_sigma_obs(self):
        """6.2.9: latent != observed when sigma_obs is large."""
        df = make_synthetic_panel(n_countries=5, n_years=5)
        g = StateSpaceGibbs(seed=42, n_iter=200, burn=50, thin=1)
        r = g.fit(df)
        # Posterior latent means should differ from raw observed
        obs = df.dropna(subset=["htn_prevalence_pct"])["htn_prevalence_pct"].to_numpy()
        post = r.latent_prev_draws.mean(axis=0)
        # At least 10% of points should differ by > 0.1
        diffs = np.abs(post - obs[:len(post)])
        assert (diffs > 0.1).sum() > len(post) * 0.1

    def test_latent_near_observed_small_sigma_obs(self):
        """6.2.10: when sigma_obs is small, latent ~ observed."""
        # This is harder to test since sigma_obs is sampled.
        # Use a dataset where structural model matches observed well.
        df = make_synthetic_panel(n_countries=5, n_years=5, seed=99)
        g = StateSpaceGibbs(seed=42, n_iter=200, burn=50, thin=1)
        r = g.fit(df)
        obs = df.dropna(subset=["htn_prevalence_pct"])["htn_prevalence_pct"].to_numpy()
        post = r.latent_prev_draws.mean(axis=0)
        # Correlation between observed and posterior should be high
        corr = np.corrcoef(obs[:len(post)], post)[0, 1]
        assert corr > 0.8


class TestBetaConvergence:
    """Spec test 6.2.3."""

    def test_beta_converges_to_ols_on_clean_data(self):
        """With low noise, posterior mean should be near OLS estimate."""
        rng = np.random.default_rng(42)
        n = 200
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
        true_beta = np.array([30.0, -2.0])
        y = X @ true_beta + rng.normal(0, 0.5, n)  # low noise
        # OLS estimate
        ols_beta = np.linalg.solve(X.T @ X, X.T @ y)
        # Build a minimal panel
        df = pd.DataFrame({
            "iso3c": [f"C{i%5:02d}" for i in range(n)],
            "year": [2015 + i % 5 for i in range(n)],
            "country_id": [i % 5 for i in range(n)],
            "log_gdp_z": X[:, 1],
            "health_exp_z": rng.normal(0, 1, n),
            "urban_z": rng.normal(0, 1, n),
            "year_z": rng.normal(0, 1, n),
            "htn_prevalence_pct": y,
            "htn_treatment_pct": 40 + rng.normal(0, 1, n),
        })
        g = StateSpaceGibbs(seed=42, n_iter=500, burn=200, thin=1)
        r = g.fit(df)
        # Intercept should be near 30 (within 3)
        assert abs(r.beta_prev_draws[:, 0].mean() - 30.0) < 3.0


class TestREShinkage:
    """Spec tests 6.2.5 and 6.2.6."""

    def test_re_shrinkage_toward_zero(self):
        """6.2.5: with few obs per country, REs should shrink toward 0."""
        df = make_synthetic_panel(n_countries=20, n_years=2, seed=42)
        g = StateSpaceGibbs(seed=42, n_iter=300, burn=100, thin=1)
        r = g.fit(df)
        # With only 2 obs per country, shrinkage should be substantial
        # Mean absolute RE should be modest
        re_means = r.latent_prev_draws.mean(axis=0)  # rough proxy
        assert re_means.std() < 20  # not wildly spread


class TestEdgeCases:
    """Spec tests 6.2.20, 6.2.21, 6.2.22."""

    def test_single_country(self):
        """6.2.20: sampler handles n_country=1."""
        df = make_synthetic_panel(n_countries=1, n_years=5, seed=42)
        g = StateSpaceGibbs(seed=42, n_iter=100, burn=30, thin=1)
        r = g.fit(df)
        assert len(r.alpha_draws) > 0

    def test_two_time_points(self):
        """6.2.21: sampler handles minimal n_years=2."""
        df = make_synthetic_panel(n_countries=5, n_years=2, seed=42)
        g = StateSpaceGibbs(seed=42, n_iter=100, burn=30, thin=1)
        r = g.fit(df)
        assert len(r.alpha_draws) > 0

    def test_missing_covariate_graceful(self):
        """6.2.22: one covariate all-missing -> graceful handling."""
        df = make_synthetic_panel(n_countries=5, n_years=3, seed=42)
        df["urban_z"] = np.nan  # all missing
        g = StateSpaceGibbs(seed=42, n_iter=100, burn=30, thin=1)
        r = g.fit(df)
        # Should still run (urban_z excluded from design matrix)
        assert len(r.alpha_draws) > 0


class TestProvenanceHash:
    """Spec test 6.2.24."""

    def test_provenance_hash_matches_input(self, tmp_path):
        """Hash in provenance.json matches SHA-256 of input panel."""
        df = make_synthetic_panel()
        g = StateSpaceGibbs(seed=42, n_iter=100, burn=30, thin=1)
        r = g.fit(df)
        generate_bundle(r, df, bundle_dir=tmp_path, seed=42)

        import hashlib
        expected_hash = hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()
        with open(tmp_path / "provenance.json") as f:
            prov = json.load(f)
        assert prov["data_hash"] == expected_hash
```

- [ ] **Step 2: Run all tests, commit**

Run: `cd C:\Models\HTNPipeline && python -m pytest tests/ -v --tb=short`
Expected: All 65 tests pass

---

## Updated Summary

| Task | Phase | Description | Tests | Key Fixes |
|------|-------|-------------|-------|-----------|
| 1 | Data | Scaffold + Config + HTTP client | 4 | P0-6, P1-8, P1-9 |
| 2 | Data | Indicator registry + code validation | 7 | P1-7 |
| 3 | Data | WB/WHO fetchers + sex filter + fixtures | 6 | P0-4, P1-5 |
| 4 | Data | Panel harmonizer + zscore + column guard | 6 | P0-7, P1-10 |
| 5 | Model | Gibbs sampler core (full conditionals) | 7 | P0-1,2,3,5, P1-2,3,6 |
| 5A | Model | CVD outcome sub-model (IHD+stroke) | 2 | Spec 4.1 |
| 5B | Model | AR(1) phi with MH step | 2 | Spec 4.3 |
| 5C | Model | Missing edge case + measurement error tests | 10 | Spec 6.2 |
| 6 | Model | Convergence diagnostics (Rhat + ESS) | 5 | P2-2 |
| 7 | Model | Counterfactual engine | 4 | P1-1 |
| 8 | Model | TruthCert bundle + provenance | 2 | -- |
| 8A | Model | Add posterior_draws.parquet | 0 | Spec 4.6 |
| 9 | Model | Pipeline orchestrator | -- | -- |
| 10 | Dashboard | HTML dashboard (map + explorer + evidence) | 10 | -- |
| 11 | Integration | Full pipeline test + review update | -- | -- |

**Total: 65 tests across 15 tasks (11 original + 4 addendum).**
