# BTCUSDT Live Paper Monitor

```powershell
conda run -n llm python dashboard\server.py
```

Open `http://127.0.0.1:8765` locally, or open the Tailscale Serve URL from another device. The server runs the shared HFT paper engine from 100 USDT: it consumes Binance USDⓈ-M Futures public WebSocket `trade` and `bookTicker` streams, polls public funding data more slowly, and makes a microstructure decision about once per second. The page reads the server account about once per second and uses REST only for slower display context. It never calls an order endpoint and does not require an API key.

The page shows a full closed-trade ledger with entry/exit time, side, quantity, prices, notional, gross PnL, trading costs, Funding, and net PnL. The HFT baseline uses 5-second return/volatility, taker-flow imbalance, and order-book imbalance; it now estimates gross edge and rejects signals that do not clear round-trip fee, slippage, spread, and a 2 bps safety margin. Signals must also pass a 72%/28% entry threshold and three consecutive one-second confirmations; positions have a 10-second minimum hold and a 5-second re-entry cooldown to reduce fee-driven churn. The PnL summary reconciles account equity against realized PnL and the current open position. The simulation's maximum notional leverage cap is 20x; position sizing can still choose less based on the 1% risk budget and stop distance. The server persists the shared state in `D:\BTC\.btc_paper_state.json` with a backup and revision number; browsers only keep a last-known display cache. The reset button atomically clears the shared session.

If the browser cannot reach the shared server, simulation stops and the page shows `共享 API 錯誤`; it will not continue a separate local timeline. If the WebSocket disconnects, the engine reconnects and shows `高頻／斷線`; the paper account does not advance until market data returns. The badge `高頻／伺服器` means the server is advancing the account; all browsers are views of the same account.
