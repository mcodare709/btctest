"""Runtime loading and policy for paired expected-return artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .expected_return import EXPECTED_RETURN_SCHEMA_VERSION, ExpectedReturnBundle, _catboost_regressor, expected_direction
from .protocol import EXECUTION_PROTOCOL_VERSION


class ExpectedReturnModels:
    def __init__(self, long_model: ExpectedReturnBundle, short_model: ExpectedReturnBundle) -> None:
        self.long_model = long_model
        self.short_model = short_model

    @classmethod
    def load(cls, prefix: str | Path, *, expected_timeframe: str | None = None, expected_horizon_bars: int | None = None) -> "ExpectedReturnModels":
        regressor = _catboost_regressor()
        prefix = Path(prefix)
        metadata_path = prefix.with_suffix(".json")
        if not metadata_path.exists():
            raise ValueError("expected-return model metadata is required")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != EXPECTED_RETURN_SCHEMA_VERSION:
            raise ValueError("expected-return model schema does not match current version")
        if metadata.get("execution_protocol") != EXECUTION_PROTOCOL_VERSION:
            raise ValueError("expected-return model execution protocol does not match current semantics")
        if expected_timeframe is not None and metadata.get("timeframe") != expected_timeframe:
            raise ValueError("expected-return model timeframe does not match requested inference timeframe")
        if expected_horizon_bars is not None and metadata.get("horizon_bars") != expected_horizon_bars:
            raise ValueError("expected-return model horizon does not match requested inference horizon")
        names = tuple(metadata.get("feature_names", ()))
        if not names or len(set(names)) != len(names):
            raise ValueError("expected-return model feature manifest is invalid")
        bundles = []
        for side in ("long", "short"):
            model = regressor()
            model.load_model(str(prefix.with_name(f"{prefix.name}_{side}.cbm")))
            bundles.append(ExpectedReturnBundle(model, side, names, metadata))
        return cls(bundles[0], bundles[1])

    def predict_returns(self, row: pd.Series) -> tuple[float, float]:
        return self.long_model.predict(row), self.short_model.predict(row)

    def signal(self, row: pd.Series, config: Any) -> tuple[float, int]:
        long_expected, short_expected = self.predict_returns(row)
        direction = expected_direction(long_expected, short_expected, config.min_expected_edge_bps / 10_000.0)
        return long_expected, direction
