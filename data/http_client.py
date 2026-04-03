"""Robust HTTP client with retry logic for HyperAtlas v4 pipeline."""
from __future__ import annotations

import time
from typing import Any

import requests
from requests.exceptions import RequestException

# HTTP status codes that warrant a retry
_RETRY_STATUSES = {429, 502, 503}


def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    retries: int = 5,
    sleep_base: float = 1.0,
    timeout: float = 30.0,
    user_agent: str = "HyperAtlas/4.0 (research)",
    max_bytes: int = 50_000_000,
) -> Any:
    """Fetch *url* and return parsed JSON.

    Retries on transient HTTP errors (429/502/503) and network-level
    failures (Timeout, ConnectionError).  Raises ``RuntimeError`` after
    retry exhaustion — never returns ``None``.

    Parameters
    ----------
    url:
        Target URL.
    params:
        Optional query-string parameters.
    retries:
        Total number of attempts (including the first).
    sleep_base:
        Base sleep between retries (seconds); actual sleep = sleep_base * attempt.
        Set to 0 in tests to avoid slow-downs.
    timeout:
        Per-request socket timeout in seconds.
    user_agent:
        Value sent in the ``User-Agent`` request header.
    max_bytes:
        Maximum allowed response size (checked via Content-Length header).
        Raises ``ValueError`` when exceeded.

    Returns
    -------
    Any
        Parsed JSON payload.

    Raises
    ------
    ValueError
        When ``Content-Length`` exceeds *max_bytes*.
    RuntimeError
        When all retry attempts are exhausted.
    """
    headers = {"User-Agent": user_agent}
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout,
            )

            # Check response size before reading body
            content_length = resp.headers.get("Content-Length")
            if content_length is not None:
                if int(content_length) > max_bytes:
                    raise ValueError(
                        f"Response Content-Length {content_length} bytes exceeds "
                        f"limit of {max_bytes} bytes for {url}"
                    )

            if resp.status_code in _RETRY_STATUSES:
                last_exc = RuntimeError(
                    f"HTTP {resp.status_code} from {url} (attempt {attempt}/{retries})"
                )
                if attempt < retries:
                    time.sleep(sleep_base * attempt)
                continue

            resp.raise_for_status()
            return resp.json()

        except ValueError:
            # Re-raise size-limit errors immediately — do not retry
            raise
        except RequestException as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(sleep_base * attempt)
            continue

    raise RuntimeError(
        f"All {retries} attempts failed for {url}. Last error: {last_exc}"
    )
