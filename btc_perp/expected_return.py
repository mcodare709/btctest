"""Cost-aware long/short expected-return models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .data import validate_market_data
from .evaluation import chronological_train_validation_indices
from .features import build_features
from .feature_schema import CORE_FEATURES
from .protocol import EXECUTION_PROTOCOL_VERSION, infer_timeframe, net_horizon_returns

EXPECTED_RETURN_SCHEMA_VERSION = "expected-net-return-v1"


def _catboost_regressor() -> Any:
    try:
        from catboost import CatBoostRegressor
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("CatBoost is required in the llm environment") from error
    return CatBoostRegressor


def make_expected_return_dataset(
    market_data: pd.DataFrame,
    *,
    horizon_bars: int = 10,
    feature_names: Sequence[str] = CORE_FEATURES,
    fee_rate: float = 0.0004,
    slippage_bps: float = 1.0,
    default_spread_bps: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return features and executable long/short net-return targets."""

    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")
    frame = validate_market_data(market_data.reset_index() if isinstance(market_data.index, pd.DatetimeIndex) else market_data)
    features = build_features(frame)
    names = tuple(feature_names)
    missing = sorted(set(names).difference(features.columns))
    if missing:
        raise ValueError(f"feature schema contains unavailable columns: {missing}")
    targets = net_horizon_returns(
        features,
        horizon_bars,
        fee_rate=fee_rate,
        slippage_bps=slippage_bps,
        default_spread_bps=default_spread_bps,
    )[["long_net_return", "short_net_return"]]
    valid = targets.notna().all(axis=1)
    x = features.loc[valid, list(names)].replace([np.inf, -np.inf], np.nan).astype(float)
    return x, targets.loc[valid]


def expected_direction(long_expected: float, short_expected: float, threshold: float) -> int:
    """Map expected net returns to LONG=1, FLAT=0, SHORT=-1."""

    if long_expected > threshold and long_expected > short_expected:
        return 1
    if short_expected > threshold and short_expected > long_expected:
        return -1
    return 0


@dataclass
class ExpectedReturnBundle:
    model: Any
    side: str
    feature_names: tuple[str, ...]
    metadata: dict[str, Any]

    def _row_frame(self, row: pd.Series) -> pd.DataFrame:
        values = {name: row.get(name, np.nan) for name in self.feature_names}
        return pd.DataFrame([values]).replace([np.inf, -np.inf], np.nan).astype(float)

    def predict(self, row: pd.Series) -> float:
        return float(np.asarray(self.model.predict(self._row_frame(row))).reshape(-1)[0])


def train_expected_return_models(
    market_data: pd.DataFrame,
    output_prefix: str | Path,
    *,
    horizon_bars: int = 10,
    timeframe: str | None = None,
    data_source: str = "unknown",
    feature_names: Sequence[str] = CORE_FEATURES,
    iterations: int = 400,
    depth: int = 6,
    learning_rate: float = 0.05,
) -> dict[str, Any]:
    """Train separate CatBoost regressors for long and short net return."""

    frame = validate_market_data(market_data.reset_index() if isinstance(market_data.index, pd.DatetimeIndex) else market_data)
    resolved_timeframe = infer_timeframe(frame)
    if timeframe is not None and timeframe != resolved_timeframe:
        raise ValueError(f"requested timeframe {timeframe!r} does not match input frequency {resolved_timeframe!r}")
    x, targets = make_expected_return_dataset(frame, horizon_bars=horizon_bars, feature_names=feature_names)
    if len(x) < 100:
        raise ValueError("training data must contain at least 100 rows")
    train_end, validation_start = chronological_train_validation_indices(
        len(x),
        horizon_bars=horizon_bars,
    )
    regressor = _catboost_regressor()
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for side in ("long", "short"):
        model = regressor(loss_function="RMSE", iterations=iterations, depth=depth, learning_rate=learning_rate, random_seed=42, verbose=False, allow_writing_files=False)
        model.fit(x.iloc[:train_end], targets[f"{side}_net_return"].iloc[:train_end], eval_set=(x.iloc[validation_start:], targets[f"{side}_net_return"].iloc[validation_start:]), verbose=False)
        model.save_model(str(prefix.with_name(f"{prefix.name}_{side}.cbm")))
    metadata = {
        "schema_version": EXPECTED_RETURN_SCHEMA_VERSION,
        "execution_protocol": EXECUTION_PROTOCOL_VERSION,
        "feature_names": list(x.columns),
        "feature_schema_version": "named-v1",
        "timeframe": resolved_timeframe,
        "source_timeframe": resolved_timeframe,
        "horizon_bars": horizon_bars,
        "horizon_seconds": horizon_bars * int(pd.Timedelta(resolved_timeframe).total_seconds()),
        "data_source": data_source,
        "training_start": x.index[0].isoformat(),
        "training_end": x.index[train_end - 1].isoformat(),
        "training_rows": train_end,
        "purged_rows": validation_start - train_end,
        "validation_rows": len(x) - validation_start,
        "cost_assumptions": {"fee_rate": 0.0004, "slippage_bps": 1.0, "default_spread_bps": 2.0},
    }
    prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
