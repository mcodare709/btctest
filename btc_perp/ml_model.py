"""Optional CatBoost model with a cost-aware three-class target.

The dependency is intentionally optional. The rule-based cost-aware baseline
remains available when no trained model artifact exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .data import validate_market_data
from .evaluation import chronological_train_validation_indices
from .features import build_features
from .protocol import EXECUTION_PROTOCOL_VERSION, infer_timeframe, net_horizon_returns


MODEL_FEATURES = (
    "return_1",
    "return_2",
    "return_10",
    "ema_gap_pct",
    "atr_pct",
    "realized_vol",
    "volume_z",
    "quote_volume_z",
    "trade_count_z",
    "average_trade_size_quote",
    "order_book_imbalance",
    "trade_imbalance",
    "taker_buy_quote_ratio",
    "cvd_change",
    "spread_bps",
    "oi_change",
    "liquidation_ratio",
    "long_short_log_ratio",
    "funding_rate",
    "latest_known_funding_rate", "funding_change", "funding_zscore",
    "mark_index_basis_bps", "premium_index", "predicted_funding_rate",
    "oi_zscore", "oi_acceleration", "price_oi_interaction",
    "microprice", "weighted_mid_bps", "spread_change_bps", "order_book_imbalance_change",
    "depth_imbalance_5", "depth_imbalance_10", "trade_imbalance_mean_10",
    "trade_imbalance_acceleration", "cvd_rolling_change", "taker_buy_ratio", "taker_sell_ratio",
    "has_orderbook", "has_open_interest", "has_liquidation", "has_long_short_ratio", "has_funding",)

FEATURE_GROUPS = {
    "price_momentum": ("return_1", "return_2", "return_10", "ema_gap_pct"),
    "volatility": ("atr_pct", "realized_vol"),
    "volume": ("volume_z", "quote_volume_z", "trade_count_z", "average_trade_size_quote"),
    "order_flow": ("trade_imbalance", "taker_buy_quote_ratio", "cvd_change", "trade_imbalance_mean_10", "trade_imbalance_acceleration", "cvd_rolling_change", "taker_buy_ratio", "taker_sell_ratio"),
    "order_book": ("order_book_imbalance", "spread_bps", "microprice", "weighted_mid_bps", "spread_change_bps", "order_book_imbalance_change", "depth_imbalance_5", "depth_imbalance_10"),
    "derivatives": ("oi_change", "liquidation_ratio", "long_short_log_ratio", "funding_rate", "latest_known_funding_rate", "funding_change", "funding_zscore", "mark_index_basis_bps", "premium_index", "predicted_funding_rate", "oi_zscore", "oi_acceleration", "price_oi_interaction"),
}
FEATURE_SCHEMA_VERSION = "canonical-v2"
WARMUP_FEATURES = (
    "return_1",
    "return_2",
    "return_10",
    "ema_gap_pct",
    "atr_pct",
    "realized_vol",
    "volume_z",
)


def select_feature_manifest(
    features: pd.DataFrame,
    *,
    min_coverage: float = 0.95,
) -> dict[str, Any]:
    """Select non-constant candidate features actually supported by this dataset."""

    if not 0.0 < min_coverage <= 1.0:
        raise ValueError("min_coverage must be in (0, 1]")
    coverage: dict[str, float] = {}
    selected: list[str] = []
    for name in MODEL_FEATURES:
        if name not in features:
            coverage[name] = 0.0
            continue
        values = pd.to_numeric(features[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
        ratio = float(values.notna().mean())
        coverage[name] = ratio
        if ratio >= min_coverage and values.dropna().nunique() > 1:
            selected.append(name)
    if not selected:
        raise ValueError("no non-constant model features meet the coverage threshold")
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "min_coverage": min_coverage,
        "features": selected,
        "coverage": coverage,
    }

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
    feature_names: Sequence[str] | None = None,
    min_feature_coverage: float = 0.95,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build leakage-safe features and ``down/flat/up`` labels.

    The flat class covers future returns that do not clear estimated round-trip
    costs plus a small edge buffer. Labels use next-bar-open execution for the target;
    all model features are current-row or backward-looking.
    """

    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")
    features = _feature_frame(market_data)
    net_returns = net_horizon_returns(
        features,
        horizon_bars,
        fee_rate=fee_rate,
        slippage_bps=slippage_bps,
        default_spread_bps=default_spread_bps,
    )
    threshold = min_edge_bps / 10_000.0
    valid = features[list(WARMUP_FEATURES)].notna().all(axis=1) & net_returns[["long_net_return", "short_net_return"]].notna().all(axis=1)
    resolved_features = tuple(feature_names) if feature_names is not None else tuple(
        select_feature_manifest(features.loc[valid], min_coverage=min_feature_coverage)["features"]
    )
    unsupported = sorted(set(resolved_features).difference(features.columns))
    if unsupported:
        raise ValueError(f"feature manifest contains columns absent from dataset: {unsupported}")
    x = features.loc[valid, list(resolved_features)].copy().replace([np.inf, -np.inf], np.nan).astype(float)
    executable = net_returns.loc[valid]
    y = pd.Series(
        np.select(
            [executable["short_net_return"] > threshold, executable["long_net_return"] > threshold],
            [0, 2],
            default=1,
        ),
        index=x.index,
        name="target",
        dtype="int64",
    )
    return x, y


@dataclass
class CatBoostBundle:
    model: Any
    feature_names: tuple[str, ...] = MODEL_FEATURES
    metadata: dict[str, Any] | None = None

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_timeframe: str | None = None,
        expected_horizon_bars: int | None = None,
    ) -> "CatBoostBundle":
        classifier = _catboost()
        model = classifier()
        model.load_model(str(path))
        metadata_path = Path(path).with_suffix(".json")
        if not metadata_path.exists():
            raise ValueError("model metadata is required for timeframe and horizon validation")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        feature_names = tuple(metadata.get("feature_names", ()))
        if not feature_names or len(set(feature_names)) != len(feature_names) or not set(feature_names).issubset(MODEL_FEATURES):
            raise ValueError("model feature manifest is not compatible with current feature engineering")
        if metadata.get("execution_protocol") != EXECUTION_PROTOCOL_VERSION:
            raise ValueError("model execution protocol does not match current backtest semantics")
        if expected_timeframe is not None and metadata.get("timeframe") != expected_timeframe:
            raise ValueError("model timeframe does not match requested inference timeframe")
        if expected_horizon_bars is not None and metadata.get("horizon_bars") != expected_horizon_bars:
            raise ValueError("model horizon does not match requested inference horizon")
        return cls(model=model, feature_names=feature_names, metadata=metadata)

    def _row_frame(self, row: pd.Series) -> pd.DataFrame:
        values = {name: row.get(name, np.nan) for name in self.feature_names}
        frame = pd.DataFrame([values]).replace([np.inf, -np.inf], np.nan)
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
    timeframe: str | None = None,
    data_source: str = "unknown",
) -> dict[str, Any]:
    """Train a chronological CatBoost classifier and save a model artifact."""

    validated_data = validate_market_data(market_data.reset_index() if isinstance(market_data.index, pd.DatetimeIndex) else market_data)
    inferred_timeframe = infer_timeframe(validated_data)
    if timeframe is not None and timeframe != inferred_timeframe:
        raise ValueError(f"requested timeframe {timeframe!r} does not match input frequency {inferred_timeframe!r}")
    resolved_timeframe = inferred_timeframe
    classifier = _catboost()
    x, y = make_supervised_dataset(validated_data, horizon_bars=horizon_bars)
    if len(x) < 100 or y.nunique() < 3:
        raise ValueError("training data must contain at least 100 rows and all three target classes")
    train_end, validation_start = chronological_train_validation_indices(
        len(x),
        horizon_bars=horizon_bars,
    )
    model = classifier(
        loss_function="MultiClass",
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(x.iloc[:train_end], y.iloc[:train_end], eval_set=(x.iloc[validation_start:], y.iloc[validation_start:]), verbose=False)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    metadata = {
        "execution_protocol": EXECUTION_PROTOCOL_VERSION,
        "source_timeframe": resolved_timeframe,
        "timeframe": resolved_timeframe,
        "horizon_seconds": horizon_bars * int(pd.Timedelta(resolved_timeframe).total_seconds()),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "data_source": data_source,
        "train_start": x.index[0].isoformat(),
        "train_end": x.index[train_end - 1].isoformat(),
        "cost_assumptions": {
            "fee_rate": 0.0004,
            "slippage_bps": 1.0,
            "default_spread_bps": 2.0,
            "cost_model": "next-open-v1",
            "min_edge_bps": 2.0,
        },
        "feature_names": list(x.columns),
        "feature_manifest": select_feature_manifest(_feature_frame(validated_data)),
        "horizon_bars": horizon_bars,
        "target_classes": {"0": "down", "1": "flat", "2": "up"},
        "train_rows": train_end,
        "purged_rows": validation_start - train_end,
        "validation_rows": len(x) - validation_start,
        "catboost_parameters": {"iterations": iterations, "depth": depth, "learning_rate": learning_rate, "random_seed": 42},
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
