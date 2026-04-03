"""Tests for data.http_client — Task 1 (4 tests)."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch, call

from data.http_client import get_json


class TestRetryOn502ThenSuccess(unittest.TestCase):
    """Test that get_json retries on 502 and succeeds on 200."""

    def test_retry_on_502_then_success(self):
        bad_resp = MagicMock()
        bad_resp.status_code = 502
        bad_resp.headers = {}
        bad_resp.json.return_value = {}

        good_resp = MagicMock()
        good_resp.status_code = 200
        good_resp.headers = {"Content-Length": "10"}
        good_resp.json.return_value = {"ok": True}

        with patch("requests.get", side_effect=[bad_resp, bad_resp, good_resp]) as mock_get:
            result = get_json("http://example.com/api", retries=3, sleep_base=0)

        self.assertEqual(result, {"ok": True})
        self.assertEqual(mock_get.call_count, 3)


class TestRetryExhaustionRaises(unittest.TestCase):
    """Test that RuntimeError is raised after retry exhaustion."""

    def test_retry_exhaustion_raises(self):
        bad_resp = MagicMock()
        bad_resp.status_code = 502
        bad_resp.headers = {}

        with patch("requests.get", return_value=bad_resp):
            with self.assertRaises(RuntimeError):
                get_json("http://example.com/api", retries=3, sleep_base=0)


class TestResponseSizeLimit(unittest.TestCase):
    """Test that oversized Content-Length raises ValueError."""

    def test_response_size_limit(self):
        big_resp = MagicMock()
        big_resp.status_code = 200
        big_resp.headers = {"Content-Length": str(100_000_000)}

        with patch("requests.get", return_value=big_resp):
            with self.assertRaises(ValueError):
                get_json("http://example.com/api", max_bytes=50_000_000, sleep_base=0)


class TestUserAgentHeaderSent(unittest.TestCase):
    """Test that the User-Agent header is forwarded to requests.get."""

    def test_user_agent_header_sent(self):
        good_resp = MagicMock()
        good_resp.status_code = 200
        good_resp.headers = {"Content-Length": "5"}
        good_resp.json.return_value = {}

        with patch("requests.get", return_value=good_resp) as mock_get:
            get_json(
                "http://example.com/api",
                user_agent="TestAgent/1.0",
                sleep_base=0,
            )

        _, kwargs = mock_get.call_args
        sent_headers = kwargs.get("headers", {})
        self.assertEqual(sent_headers.get("User-Agent"), "TestAgent/1.0")


import pytest
import re
from data.registry import IndicatorRegistry, validate_indicator_code


class TestIndicatorCodeValidation:
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


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Tasks 3 + 4 tests (12 new)
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
from data.who import filter_who_sex, harmonize_who_observations
from data.worldbank import worldbank_fixture
from data.harmonize import zscore_series, prepare_model_panel


class TestWHOSexFilter:
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
        assert len(result) == 1

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


class TestZscore:
    def test_normal_zscore(self):
        s = pd.Series([10.0, 20.0, 30.0])
        z = zscore_series(s)
        assert abs(z.mean()) < 1e-10
        assert abs(z.std(ddof=1) - 1.0) < 0.01

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
        assert abs(z.iloc[0] - (-0.7071)) < 0.01


class TestColumnValidation:
    def test_missing_required_columns_raises(self):
        df = pd.DataFrame({"iso3c": ["GBR"], "year": [2019], "gdp_pc_usd": [46000]})
        with pytest.raises(ValueError, match="Missing required columns"):
            prepare_model_panel(df, required_outcomes=["htn_prevalence_pct", "htn_treatment_pct"])

    def test_valid_panel_passes(self):
        df = pd.DataFrame({
            "iso3c": ["GBR", "USA"],
            "year": [2019, 2019],
            "gdp_pc_usd": [46000.0, 65000.0],
            "health_exp_pct_gdp": [10.2, 16.7],
            "urban_pct": [83.9, 82.7],
            "htn_prevalence_pct": [30.0, 28.0],
            "htn_treatment_pct": [55.0, 50.0],
        })
        result = prepare_model_panel(df, required_outcomes=["htn_prevalence_pct", "htn_treatment_pct"])
        assert "log_gdp_z" in result.columns
        assert "country_id" in result.columns
