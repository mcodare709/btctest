from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from btc_perp.expected_return import EXPECTED_RETURN_SCHEMA_VERSION
from btc_perp.expected_return_runtime import ExpectedReturnModels
from btc_perp.protocol import EXECUTION_PROTOCOL_VERSION


class RuntimeTests(unittest.TestCase):
    def test_runtime_rejects_timeframe_mismatch(self) -> None:
        class FakeRegressor:
            def load_model(self, path: str) -> None:
                pass

        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "model"
            prefix.with_suffix(".json").write_text(json.dumps({
                "schema_version": EXPECTED_RETURN_SCHEMA_VERSION,
                "execution_protocol": EXECUTION_PROTOCOL_VERSION,
                "timeframe": "30s",
                "horizon_bars": 10,
                "feature_names": ["return_1"],
            }), encoding="utf-8")
            with patch("btc_perp.expected_return_runtime._catboost_regressor", return_value=FakeRegressor):
                with self.assertRaisesRegex(ValueError, "timeframe"):
                    ExpectedReturnModels.load(prefix, expected_timeframe="1min", expected_horizon_bars=10)

    def test_runtime_policy_keeps_flat_below_edge(self) -> None:
        class FakeModel:
            def __init__(self, value: float) -> None:
                self.value = value

            def predict(self, frame: pd.DataFrame) -> np.ndarray:
                return np.array([self.value])

        from btc_perp.expected_return import ExpectedReturnBundle
        from btc_perp.config import BacktestConfig

        models = ExpectedReturnModels(
            ExpectedReturnBundle(FakeModel(0.0001), "long", ("return_1",), {}),
            ExpectedReturnBundle(FakeModel(0.0002), "short", ("return_1",), {}),
        )
        self.assertEqual(models.signal(pd.Series({"return_1": np.nan}), BacktestConfig())[1], 0)


if __name__ == "__main__":
    unittest.main()
