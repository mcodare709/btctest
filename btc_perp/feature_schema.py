"""Named feature contracts for long-history and recent-data models."""

CORE_FEATURES = (
    "return_1", "return_2", "return_10", "ema_gap_pct", "atr_pct",
    "realized_vol", "volume_z", "trade_imbalance", "cvd_change",
    "trade_imbalance_mean_10", "trade_imbalance_acceleration",
    "cvd_rolling_change", "taker_buy_ratio", "taker_sell_ratio",
    "mark_index_basis_bps", "premium_index",
)
DERIVATIVES_FEATURES = CORE_FEATURES + (
    "oi_change", "oi_zscore", "oi_acceleration", "funding_rate",
    "funding_change", "funding_zscore", "liquidation_ratio",
    "long_short_log_ratio",
)
MICROSTRUCTURE_FEATURES = CORE_FEATURES + (
    "order_book_imbalance", "spread_bps", "microprice", "weighted_mid_bps",
    "depth_imbalance_5", "depth_imbalance_10",
)
FEATURE_SCHEMAS = {
    "core": CORE_FEATURES,
    "derivatives": DERIVATIVES_FEATURES,
    "microstructure": MICROSTRUCTURE_FEATURES,
}


def feature_schema(name: str) -> tuple[str, ...]:
    try:
        return tuple(FEATURE_SCHEMAS[name])
    except KeyError as error:
        raise ValueError(f"unknown feature schema: {name!r}") from error
