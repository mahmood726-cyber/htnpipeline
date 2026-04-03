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


if __name__ == "__main__":
    unittest.main()
