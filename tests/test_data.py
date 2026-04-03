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
