# BTCUSDT 永續合約研究工作流

最後更新：2026-09-08  
分支政策：只使用 `main`，不建立分支。

## 1. 研究目標與成功條件

目標不是預測下一根 K 棒，而是在 BTCUSDT 永續合約中驗證：價格、成交方向、Order Book 與衍生品特徵，是否能在扣除 Fee、Spread、Slippage 與 Funding 後，對短期可成交淨報酬提供穩定、可重現的 OOS 預測能力。

策略有效的最低標準：

- 只以 purged walk-forward 的 Out-of-Sample 結果判斷。
- 與 buy-and-hold、random-sign、簡單 momentum benchmark 比較。
- 結果需包含成本、Funding、停損、持倉上限與清算事件。
- 不使用 synthetic data 的獲利結果作為 alpha 證據。

## 2. 資料工作流

```text
Binance 官方月度 archive
  -> UTC 正規化、排序、去重與 OHLC 驗證
  -> canonical raw 1m bars
  -> build_features()（僅當前與過去資料）
  -> feature availability / constant-feature 檢查
  -> 依 timeframe 重採樣或建立實驗資料集
  -> purged walk-forward OOS
```

### 已完成資料集

本機忽略資料，不能提交到 Git：

- `data/processed/btc_usdt_perp_1m_raw.csv.gz`
- `data/processed/btc_usdt_perp_1m_features.csv.gz`
- `data/processed/dataset_metadata.json`
- `data/processed/model_feature_availability.json`

目前 1m canonical dataset：

| 項目 | 狀態 |
| --- | --- |
| 期間 | 2020-01-01 至 2026-08-31 UTC |
| 列數 | 3,506,400 |
| 欄位數 | 53 |
| 重複 timestamp | 0 |
| 缺少分鐘 | 0 |
| 非法 OHLC | 0 |

重建與檢查：

```powershell
conda run -n llm python scripts\build_binance_dataset.py --archive-root data\binance --output-dir data\processed
conda run -n llm python scripts\inspect_dataset_availability.py
```

## 3. 特徵與缺失值政策

唯一特徵管線是 `btc_perp/features.py:build_features()`；訓練、backtest 與 future live/paper 必須共用它。

- 所有 rolling 特徵只讀當前列及過去列；warm-up 維持 `NaN`。
- 缺少的外部 feed 保持 `NaN`，絕不以 `0` 偽造市場值。
- `has_orderbook`、`has_open_interest`、`has_liquidation`、`has_long_short_ratio`、`has_funding` 是逐列 availability indicator。
- 進入本資料集模型的欄位必須有至少 95% 覆蓋率且不是常數。

目前可用的非固定模型特徵（16）：

```text
return_1, return_2, return_10, ema_gap_pct, atr_pct, realized_vol,
volume_z, trade_imbalance, cvd_change, mark_index_basis_bps,
premium_index, trade_imbalance_mean_10, trade_imbalance_acceleration,
cvd_rolling_change, taker_buy_ratio, taker_sell_ratio
```

目前不納入本輪訓練：歷史 L2 order book/depth、bid-ask spread、OI、liquidations、long/short ratio、Funding。原因是完整 2020–2026 覆蓋不足；保留欄位與 `NaN`/indicator，等取得可信歷史來源後再開啟。

## 4. 執行、Label 與風險語意

每根 bar `t` 收盤後才產生訊號；可成交 entry 是 `open[t+1]`。horizon `h` 的固定時間 label 使用：

```text
gross_long = open[t+1+h] / open[t+1] - 1
net_long   = gross_long - round_trip_fee - slippage - spread - funding
net_short  = -gross_long - round_trip_fee - slippage - spread + funding
```

三分類 target：

```text
Down: short_net_return > threshold
Up:   long_net_return  > threshold
Flat: 其他情況
```

backtest 的 `max_holding_bars` 會實際強制 time exit；只有 Stop Loss、Liquidation 或明確 exit 能提前離場。資金、名目倉位、手續費、滑價、spread、Funding 與 maintenance-margin/liquidation 都由 `BacktestConfig` 管理。

注意：目前固定時間 supervised label 尚未重播 intrabar stop/liquidation path；這是未來 label 與 engine 需要進一步完全一致的工作。

## 5. 模型與評估工作流

CatBoost 維持三分類 baseline。artifact metadata 綁定：

- `source_timeframe` / `timeframe`
- `horizon_bars` / `horizon_seconds`
- `feature_schema_version` 與 feature names
- 訓練期間、資料來源、成本模型、費率、滑價與 CatBoost parameters

模型載入遇到 timeframe 或 horizon mismatch 必須拒絕，不能 silent mismatch。

評估順序：

1. 使用 purged_walk_forward_splits()；對 next-open、horizon h 的 target，train/validation/test 邊界至少保留 h + 1 個 purge rows。
2. 每個 fold 只用過去資料訓練，僅保留未見未來的 OOS 預測。
3. calibration 只使用先前 OOS fold，不使用當前或未來 test。
4. replay OOS signal 到同一個 backtest engine。
5. 輸出 class metrics、confidence buckets、calibration、feature importance、交易成本與 OOS trading metrics。

執行入口請見 `btc_perp/cli.py`；OOS 研究實作位於 `btc_perp/oos.py`。

### 5.1 Expected-return 與 RL 實驗擴充

- expected_return.py 與 return_policy.py 以單一 gross forecast 推導 long/short net return，保留成本與 missing-value 語意。
- 所有模型訓練先切 chronological fit/validation；fit 與 validation 之間保留 horizon_bars + 1 purge rows。
- walk_forward_return_models.py、walk_forward_accuracy.py 與 compare_return_models.py 的 OOS test 不參與 CatBoost eval_set 或 best-iteration 選擇。
- rl_env.py 是 dependency-free execution environment；gym_env.py 只提供 Gymnasium adapter。smoke_rl_env.py 可做 deterministic smoke test。
- RL/expected-return 結果仍屬研究實驗；需在多 fold OOS、成本、benchmark 與 regime 分析完成後才可解讀。

## 6. 已完成工程項目

- 下一根 open 的 execution / label 對齊。
- `max_holding_bars` 實際 time exit 與單元測試。
- event-only Funding 資料驗證；禁止 forward-fill 成每根 bar 的計費資料。
- canonical feature pipeline、模型 metadata 驗證與成本後三分類 label。
- purged walk-forward、OOS-only baseline/backtest、benchmarks、parameter sensitivity。
- prior-OOS isotonic probability calibration、分類報告、confidence bucket 與 replay backtest。
- Binance monthly archive downloader、清洗、特徵建置與 availability report。

## 7. 尚未完成與優先順序

1. 取得可稽核、長時間覆蓋的 L2 bookTicker/depth、OI、Funding、liquidation、long/short ratio 資料；不可任意 forward-fill。
2. 將資料 availability manifest 接到訓練介面，讓模型 schema 只使用這次資料真正可用的欄位。
3. 讓 fixed-horizon label 用相同 execution replay 計入 stop、liquidation、time exit，完全對齊 engine。
4. 使用真實資料跑多個 timeframe 的 purged walk-forward OOS，產生版本化報告。
5. 執行 feature-group ablation：Price、Volume、Order Flow、Order Book、Derivatives、Full。
6. 加 SHAP 與 market-regime feature analysis。
7. Paper engine 加入 Mark Price 的即時風控來源，並與 offline strategy interface 完全共用。
8. 比較 predicted expected net return 與 realized net return；未達標前不宣稱有 alpha。

## 8. 驗證與提交規則

每次資料或模型邏輯修改後至少執行：

```powershell
conda run -n llm python -m unittest discover -s tests -v
git -c safe.directory=D:/BTC diff --check
```
目前驗證基線：2026-09-09，77/77 tests passed。資料檔、模型產物與 outputs 不提交 Git；程式、測試、設定與可重跑腳本提交至 main。

## 9. 2026-09-08 Protocol Corrections

- `resample_market_data()` only down-samples regular bars; 1m -> 30s is rejected. Buckets are left-closed and start-labeled.
- `train-model` infers source frequency from timestamp deltas. A declared timeframe mismatch fails; model metadata uses inferred frequency.
- `backtest --model` requires explicit `--timeframe` and `--horizon-bars` and validates both against artifact metadata.
- Scheduled `time_exit` executes at the scheduled bar open before that bar intrabar path. Liquidation uses aligned Mark Price fields when provided, otherwise an explicit contract-price fallback.
- CatBoost feature artifacts store a selected manifest. Inference preserves missing values as `NaN`.
- OOS Flat is no-trade with zero realized/expected net return. Benchmarks are replayed through the cost-aware engine.
- Free Kline columns `quote_volume`, `trade_count`, `taker_buy_quote` are retained. For true 30s studies, download `aggTrades` and run `scripts\build_aggtrade_30s_dataset.py`; never upsample 1m bars.