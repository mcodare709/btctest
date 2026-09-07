# BTCUSDT Perpetual Research Baseline

這是一個離線、成本感知的 BTCUSDT 永續合約回測 MVP。它不是交易所下單程式，也不會把 synthetic data 的結果當成策略證據。

目前包含：

- CSV 資料驗證與 OHLCV/衍生品/微結構欄位 resample。
- ATR、EMA、報酬、成交量 z-score、realized volatility、Order Book Imbalance、Trade Imbalance、OI change 等 feature。
- 可解釋的 rule-based baseline，輸出 `prob_up` 與 Long/Short/Flat。
- 成本感知 baseline：只有預估毛邊際扣除雙邊 fee、slippage、spread 後仍超過安全緩衝才交易。
- 可選 CatBoost 三分類模型（down/flat/up），標籤先扣除估計交易成本；模型檔不存在時不會假裝使用 ML。
- 下一根 K 棒開盤執行，避免用收盤訊號偷看未來。
- 以單筆風險反推名目倉位，並受 leverage cap 限制。
- 手續費、滑價、spread、funding、停損、持有時間、maintenance margin / liquidation event。
- Total/Annualized Return、Sharpe、Sortino、Max Drawdown、Win Rate、Profit Factor、Long/Short PnL、Trading/Funding Cost、Trade Count。
- 30 秒→5 分鐘、1 分鐘→15 分鐘、5 分鐘→1 小時三組比較設定。

## 安裝與驗證

專案規定使用 `llm` environment：

```powershell
conda run -n llm python -m unittest discover -s tests -v
```

不需要額外安裝 sklearn。CatBoost 是可選依賴：`conda run -n llm python -m pip install -e ".[ml]"`。必須先在時間序列 walk-forward split 上比較成本後表現，不能直接把訓練集結果當成策略證據。

## 資料格式

必需欄位：`timestamp,open,high,low,close,volume`。`timestamp` 可為 ISO-8601 或 Unix milliseconds。

可選欄位：

`funding_rate,funding_event,open_interest,bid_price,ask_price,bid_size,ask_size,taker_buy_volume,taker_sell_volume,liquidation_volume,long_short_ratio`

Funding 的安全約定：只有在真正 funding event 的資料列填入該事件 rate，其餘列留空；不要把 8 小時 funding rate forward-fill 到每個 30 秒 bar。若提供 `funding_event`，它必須在該列為 true；驗證器會拒絕連續 non-null funding rows。

執行成本 snapshot（`bid_price/ask_price/bid_size/ask_size`）被視為該 bar 開盤可取得的資料；若你的 vendor 欄位其實是收盤 snapshot，請先 shift 到下一個 bar 或改用明確的 open snapshot，否則會造成執行成本 look-ahead。

固定風險 sizing 會把停損距離加上預估雙邊 fee、slippage、spread 成本後反推名目倉位；gap 穿越停損或 liquidation 仍可能超過風險預算。

## 執行

單組回測：

```powershell
conda run -n llm python -m btc_perp.cli backtest `
  --data .\data\BTCUSDT.csv `
  --timeframe 30s `
  --horizon-bars 10 `
  --output-dir .\outputs\30s_5m
```

訓練可選 CatBoost 模型（需要真實歷史資料，不要使用 synthetic data 作為策略證據）：

```powershell
conda run -n llm python -m btc_perp.cli train-model `
  --data .\data\BTCUSDT.csv `
  --horizon-bars 10 `
  --output .\outputs\models\btc_catboost.cbm
```

載入模型回測：

```powershell
conda run -n llm python -m btc_perp.cli backtest `
  --data .\data\BTCUSDT.csv `
  --timeframe 30s `
  --horizon-bars 10 `
  --model .\outputs\models\btc_catboost.cbm `
  --output-dir .\outputs\catboost_30s_5m
```

三組比較：

```powershell
conda run -n llm python -m btc_perp.cli compare `
  --data .\data\BTCUSDT.csv `
  --output-dir .\outputs\comparison
```

輸出包含 `summary.json`、`equity_curve.csv`、`trades.csv`。沒有真實歷史資料時，可先產生只用於 smoke test 的 synthetic CSV：

```powershell
conda run -n llm python -m btc_perp.synthetic --output .\examples\synthetic_30s.csv
```

## 研究限制

這個 baseline 不代表有正期望值。要得到可信結論，仍需：交易所原始 trades/order book/funding/liquidation 資料、嚴格時間切分、out-of-sample/walk-forward、參數敏感度、不同市場 regime、資料缺口與延遲處理，以及 paper trading 觀察。即時 30 秒更新的 connector 目前刻意未實作，避免把 backtest engine 和未驗證的交易所 API 混在一起。
