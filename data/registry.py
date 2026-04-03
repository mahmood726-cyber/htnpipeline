from __future__ import annotations
import re
from typing import Callable
import pandas as pd


def validate_indicator_code(code: str) -> str:
    """Validate indicator code against strict allowlist regex.
    Prevents OData injection, path traversal, query-string manipulation."""
    if not re.fullmatch(r"[A-Za-z0-9_.\-]{1,80}", code):
        raise ValueError(f"Invalid indicator code: {code!r}")
    return code


class IndicatorRegistry:
    def __init__(self) -> None:
        self._indicators: dict[str, dict] = {}

    def register(self, name: str, source: str, code: str) -> Callable:
        validate_indicator_code(code)
        def decorator(func: Callable) -> Callable:
            self._indicators[name] = {"source": source, "code": code, "fetch_fn": func}
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
