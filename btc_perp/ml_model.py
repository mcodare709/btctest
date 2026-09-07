"""Optional CatBoost model with a cost-aware three-class target.

The dependency is intentionally optional. The rule-based cost-aware baseline
remains available when no trained model artifact exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .data import validate_market_data
from .features import build_features


MODEL_FEATURES = (
    "return_1",
    "return_2",
    "return_10",
    "ema_gap_pct",
    "atr_pct",
    "realized_vol",
    "volume_z",
    "order_book_imbalance",
    "trade_imbalance",
    "cvd_change",
    "spread_bps",
    "oi_change",
    "liquidation_ratio",
    "long_short_log_ratio",
    "funding_rate",
)
WARMUP_FEATURES = (
    "return_1",
    "return_2",
    "return_10",
    "ema_gap_pct",
    "atr_pct",
    "realized_vol",
    "volume_z",
)


def _catboost() -> Any:
    try:
        from catboost import CatBoostClassifier
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "CatBoost is optional; install it with `pip install catboost` in the llm environment"
        ) from error
    return CatBoostClassifier


def _feature_frame(market_data: pd.DataFrame) -> pd.DataFrame:
    frame = market_data.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame = validate_market_data(frame)
    return build_features(frame)


def make_supervised_dataset(
    market_data: pd.DataFrame,
    *,
    horizon_bars: int = 10,
    fee_rate: float = 0.0004,
    slippage_bps: float = 1.0,
    default_spread_bps: float = 2.0,
    min_edge_bps: float = 2.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build leakage-safe features and ``down/flat/up`` labels.

    The flat class covers future returns that do not clear estimated round-trip
    costs plus a small edge buffer. Labels use future close only for the target;
    all model features are current-row or backward-looking.
    """

    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")
    features = _feature_frame(market_data)
    future_return = features["close"].shift(-horizon_bars) / features["close"] - 1.0
    cost_band_bps = 2.0 * (fee_rate * 10_000.0 + slippage_bps) + default_spread_bps + min_edge_bps
    cost_band = cost_band_bps / 10_000.0

    valid = features[list(WARMUP_FEATURES)].notna().all(axis=1) & future_return.notna()
    x = features.loc[valid, list(MODEL_FEATURES)].copy()
    x["spread_bps"] = x["spread_bps"].fillna(default_spread_bps)
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    future = future_return.loc[valid]
    y = pd.Series(
        np.select([future < -cost_band, future > cost_band], [0, 2], default=1),
        index=x.index,
        name="target",
        dtype="int64",
    )
    return x, y


@dataclass
class CatBoostBundle:
    model: Any
    feature_names: tuple[str, ...] = MODEL_FEATURES

    @classmethod
    def load(cls, path: str | Path) -> "CatBoostBundle":
        classifier = _catboost()
        model = classifier()
        model.load_model(str(path))
        metadata_path = Path(path).with_suffix(".json")
        feature_names = MODEL_FEATURES
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            feature_names = tuple(metadata.get("feature_names", MODEL_FEATURES))
        if tuple(feature_names) != MODEL_FEATURES:
            raise ValueError("model feature schema does not match current feature engineering")
        return cls(model=model, feature_names=feature_names)

    def _row_frame(self, row: pd.Series) -> pd.DataFrame:
        values = {name: row.get(name, 0.0) for name in self.feature_names}
        frame = pd.DataFrame([values]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return frame.astype(float)

    def predict_probabilities(self, row: pd.Series) -> dict[int, float]:
        values = np.asarray(self.model.predict_proba(self._row_frame(row)))[0]
        classes = getattr(self.model, "classes_", np.arange(len(values)))
        return {int(label): float(probability) for label, probability in zip(classes, values)}

    def signal(self, row: pd.Series, config: BacktestConfig) -> tuple[float, int]:
        probabilities = self.predict_probabilities(row)
        probability_up = probabilities.get(2, 0.0)
        probability_down = probabilities.get(0, 0.0)
        raw_direction = 0
        if probability_up >= config.long_probability_threshold and probability_up > probability_down:
            raw_direction = 1
        elif probability_down >= (1.0 - config.short_probability_threshold) and probability_down > probability_up:
            raw_direction = -1
        if raw_direction == 0 or not config.cost_aware_filter:
            return probability_up, raw_direction

        spread_bps = row.get("spread_bps", config.default_spread_bps)
        spread_bps = config.default_spread_bps if spread_bps is None or pd.isna(spread_bps) else max(float(spread_bps), 0.0)
        volatility = max(
            float(row.get("realized_vol", 0.0) or 0.0),
            float(row.get("atr_pct", 0.0) or 0.0),
            config.minimum_stop_pct,
        )
        confidence = abs(probability_up - probability_down)
        gross_edge_bps = confidence * volatility * 10_000.0
        round_trip_cost_bps = 2.0 * (config.fee_rate * 10_000.0 + config.slippage_bps) + spread_bps
        expected_edge_bps = gross_edge_bps - round_trip_cost_bps
        return probability_up, raw_direction if expected_edge_bps >= config.min_expected_edge_bps else 0


def train_catboost(
    market_data: pd.DataFrame,
    output_path: str | Path,
    *,
    horizon_bars: int = 10,
    iterations: int = 400,
    depth: int = 6,
    learning_rate: float = 0.05,
) -> dict[str, Any]:
    """Train a chronological CatBoost classifier and save a model artifact."""

    classifier = _catboost()
    x, y = make_supervised_dataset(market_data, horizon_bars=horizon_bars)
    if len(x) < 100 or y.nunique() < 3:
        raise ValueError("training data must contain at least 100 rows and all three target classes")
    split = int(len(x) * 0.8)
    train_end = max(split - horizon_bars, 1)
    model = classifier(
        loss_function="MultiClass",
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(x.iloc[:train_end], y.iloc[:train_end], eval_set=(x.iloc[split:], y.iloc[split:]), verbose=False)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    metadata = {
        "feature_names": list(MODEL_FEATURES),
        "horizon_bars": horizon_bars,
        "target_classes": {"0": "down", "1": "flat", "2": "up"},
        "train_rows": train_end,
        "purged_rows": split - train_end,
        "validation_rows": len(x) - split,
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
