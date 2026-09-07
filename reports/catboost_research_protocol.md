# BTCUSDT Perpetual CatBoost Research Protocol

## Scope

This document defines the canonical research protocol.  Synthetic data is
permitted only for smoke tests and is not evidence of profitability or alpha.

## Dataset columns

Required market data: `timestamp`, `open`, `high`, `low`, `close`, `volume`.
Optional feeds: `funding_rate` (event-only), `open_interest`, `bid_price`,
`ask_price`, `bid_size`, `ask_size`, taker buy/sell volume, liquidation volume,
and long/short ratio.  Additional optional feeds accepted by the feature
pipeline are `mark_price`, `index_price`, `premium_index`,
`predicted_funding_rate`, `bid_depth_5`, `ask_depth_5`, `bid_depth_10`, and
`ask_depth_10`.

## Feature pipeline

`btc_perp.features.build_features` is the only canonical pipeline for training
and backtest.  Every feature at row `t` uses row `t` or earlier data only.

| Group | Features / formulas |
|---|---|
| Price / momentum | `return_1`, `return_2`, `return_10`, `ema_gap_pct` |
| Volatility | ATR / close, rolling return volatility |
| Volume | rolling volume z-score |
| Order flow | taker imbalance, CVD changes, rolling imbalance and acceleration, buy/sell ratios |
| Order book | top-of-book imbalance, spread, microprice, weighted-mid deviation, spread/imbalance changes, depth-5/depth-10 imbalance when supplied |
| Derivatives | OI change/z-score/acceleration, price×OI, liquidation ratio, long-short log ratio, funding state/change/z-score, mark-index basis, premium and predicted funding when supplied |

Availability indicators (`has_orderbook`, `has_open_interest`,
`has_liquidation`, `has_long_short_ratio`, `has_funding`) distinguish an absent
feed from a measured zero. Missing optional values stay `NaN`; CatBoost handles
them natively.

## Label, entry, exit, and costs

A signal observed after close `t` enters at `open[t+1]`. A horizon of `N` bars
exits at `open[t+1+N]`, unless stop loss, liquidation, or explicit signal exit
occurs first. `max_holding_bars=N` therefore forces a `time_exit` after exactly
N elapsed bars.

The supervised three-class label uses executable net returns from
`btc_perp.protocol.net_horizon_returns`:

`net long = open[t+1+N] / open[t+1] - 1 - round-trip cost - funding`

`net short = -(open[t+1+N] / open[t+1] - 1) - round-trip cost + funding`

Round-trip cost contains two fees, two slippage terms, and entry/exit half
spreads. Funding is charged only on known event rows after entry through exit,
as in the bar engine. `Up` means long net return clears the safety threshold;
`Down` means short net return clears it; otherwise `Flat`. This is a fixed-time
label. Stop and liquidation path labels require trade-level/event-level replay
and are an explicit remaining limitation.

## Splits and evaluation

Use chronological purged walk-forward splits. A purge must be at least the
label horizon. Each CatBoost fold trains only on its own historical training
segment and reports only its future OOS prediction rows. Do not report pooled
in-sample metrics as performance.

Required real-data outputs: class and prediction distributions; confusion
matrix; precision, recall, F1 by class; macro F1; balanced accuracy; reliability
table and Brier/ECE; expected-return MAE/RMSE; confidence-bucket net returns;
feature and SHAP importance; ablations for price, volume, order-flow,
order-book, derivatives, and full model; and cost-aware OOS trading metrics.

## Artifact metadata

Artifacts record execution protocol, source timeframe, horizon bars/seconds,
feature schema version, source, training range, cost model/assumptions, and
CatBoost parameters. Loading rejects incompatible execution protocol,
timeframe, or horizon.

## Known limitations and next work

There is no real Binance trade/order-book/mark/index history in this repository
yet. Consequently no real-data class distribution, feature importance, SHAP,
calibration curve, ablation result, or OOS trading result is claimed here.
Next: ingest timestamp-aligned Binance data, add long/short liquidation fields
and scheduled funding timestamps, replay stop/liquidation at event resolution,
fit calibration only on prior validation OOS, then generate the required report.
