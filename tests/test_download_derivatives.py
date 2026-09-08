from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.download_derivatives import download_derivatives


class _FakeResponse:
    def __init__(self, payload: list[dict[str, object]]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> list[dict[str, object]]:
        return self._payload


class _FakeSession:
    payloads = {
        "/futures/data/openInterestHist": [{"timestamp": 1_700_000_000_000, "sumOpenInterest": "100"}],
        "/futures/data/globalLongShortAccountRatio": [{"timestamp": 1_700_000_000_000, "longShortRatio": "1.2"}],
        "/fapi/v1/fundingRate": [{"fundingTime": 1_700_000_000_000, "fundingRate": "0.0001"}],
    }

    def __enter__(self) -> "_FakeSession":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def get(self, url: str, *, params: dict[str, object], timeout: float) -> _FakeResponse:
        path = "/" + url.split("/", 3)[-1]
        return _FakeResponse(self.payloads[path])


class DownloadDerivativesTests(unittest.TestCase):
    def test_download_writes_feeds_and_coverage_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch("scripts.download_derivatives.requests.Session", return_value=_FakeSession()):
                metadata = download_derivatives(Path(directory), base_url="https://example.test")

            self.assertEqual(set(metadata), {"open_interest_5m", "long_short_ratio_5m", "funding_events"})
            self.assertTrue((Path(directory) / "coverage.json").exists())
            self.assertEqual(metadata["funding_events"]["rows"], 1)
            coverage = json.loads((Path(directory) / "coverage.json").read_text(encoding="utf-8"))
            self.assertEqual(coverage["open_interest_5m"]["columns"], ["timestamp", "sumOpenInterest"])


if __name__ == "__main__":
    unittest.main()