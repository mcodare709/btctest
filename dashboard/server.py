"""Shared BTCUSDT paper-monitor server.

The browser is only a view and signal calculator. This process owns the
canonical paper account, grants one short-lived simulation lease, and rejects
stale writes with a revision check. It never calls a private Binance API or
an order endpoint.
"""

from __future__ import annotations

import json
import math
import os
import secrets
import threading
import time
from collections import deque
from copy import deepcopy
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.request import urlopen
from urllib.parse import urlparse

from cpp_core import MarketFeatures, StrategyCore

try:
    import websocket
except ImportError:  # pragma: no cover - reported in the engine status
    websocket = None


ROOT = Path(__file__).resolve().parent
# Keep the account outside the directory served by SimpleHTTPRequestHandler.
STATE_PATH = ROOT.parent / ".btc_paper_state.json"
STATE_PATH = Path(os.environ.get("PAPER_STATE_PATH", str(STATE_PATH)))
BACKUP_PATH = STATE_PATH.with_suffix(".bak")
HOST = os.environ.get("PAPER_HOST", "127.0.0.1")
PORT = int(os.environ.get("PAPER_PORT", "8765"))
LEASE_SECONDS = 120
MAX_BODY_BYTES = 2 * 1024 * 1024
SCHEMA_VERSION = 1
HFT_MODE = os.environ.get("PAPER_HFT_MODE", "1").lower() not in {"0", "false", "off"}
HFT_WS_URL = "wss://fstream.binance.com/ws"
HFT_STREAMS = ["btcusdt@trade", "btcusdt@bookTicker"]
HFT_DECISION_MS = 1000
HFT_CHECKPOINT_SECONDS = 5
HFT_MAX_HOLD_MS = 60_000
HFT_ENTRY_PROB = 0.72
HFT_CONFIRMATIONS = 3
HFT_MIN_HOLD_MS = 10_000
HFT_REENTRY_COOLDOWN_MS = 5_000
HFT_MIN_EDGE_BPS = 2.0
HFT_FEE_BPS = 4.0
HFT_SLIPPAGE_BPS = 1.0
HFT_MIN_MOVE_BPS = 10.0
HFT_RISK_PER_TRADE = 0.02
HFT_TAKE_PROFIT_PCT = 0.01
HFT_ENGINE = None


class StateStoreError(RuntimeError):
    """The canonical state cannot be safely read or written."""


def initial_state() -> dict[str, Any]:
    return {
        "cash": 100.0,
        "equity": 100.0,
        "position": None,
        "trades": [],
        "logs": [],
        "lastBar": None,
        "lastFundingAt": 0,
        "nextFundingAt": 0,
    }


def finite_number(value: Any, field: str, *, minimum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field} must be a number") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return number


def _side(value: Any, field: str) -> int:
    number = finite_number(value, field)
    if number not in (-1, 1):
        raise ValueError(f"{field} must be -1 or 1")
    return int(number)


def _validate_position(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("position must be an object or null")
    position = dict(value)
    position["side"] = _side(position.get("side"), "position.side")
    position["qty"] = finite_number(position.get("qty"), "position.qty", minimum=0.0)
    position["entry"] = finite_number(position.get("entry"), "position.entry", minimum=0.0)
    position["entryNotional"] = finite_number(position.get("entryNotional", position["qty"] * position["entry"]), "position.entryNotional", minimum=0.0)
    position["entryCost"] = finite_number(position.get("entryCost"), "position.entryCost", minimum=0.0)
    position["funding"] = finite_number(position.get("funding", 0), "position.funding")
    position["entrySpreadBps"] = finite_number(position.get("entrySpreadBps", 0), "position.entrySpreadBps", minimum=0.0)
    position["stop"] = finite_number(position.get("stop"), "position.stop", minimum=0.0)
    position["opened"] = finite_number(position.get("opened"), "position.opened", minimum=0.0)
    return position


def _validate_trade(value: Any, index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"trades[{index}] must be an object")
    trade = dict(value)
    trade["side"] = _side(trade.get("side"), f"trades[{index}].side")
    for field in ("qty", "entry", "exit", "entryNotional", "exitNotional"):
        trade[field] = finite_number(trade.get(field), f"trades[{index}].{field}", minimum=0.0)
    for field in ("gross", "entryCost", "exitCost", "net", "cost", "funding", "entrySpreadBps", "exitSpreadBps", "cashBeforeClose", "rawCashAfterClose", "cashAfterClose", "liquidationAdjustment"):
        trade[field] = finite_number(trade.get(field, 0), f"trades[{index}].{field}")
    for field in ("opened", "time"):
        trade[field] = finite_number(trade.get(field), f"trades[{index}].{field}", minimum=0.0)
    reason = trade.get("reason")
    if not isinstance(reason, str) or len(reason) > 100:
        raise ValueError(f"trades[{index}].reason must be a short string")
    return trade


def _validate_log(value: Any, index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"logs[{index}] must be an object")
    message = value.get("message")
    if not isinstance(message, str) or len(message) > 500:
        raise ValueError(f"logs[{index}].message must be a short string")
    timestamp = value.get("time")
    if not isinstance(timestamp, (str, int, float)):
        raise ValueError(f"logs[{index}].time must be a timestamp")
    if isinstance(timestamp, (int, float)):
        finite_number(timestamp, f"logs[{index}].time", minimum=0.0)
    return {"time": timestamp, "message": message}


def normalize_state(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("state must be an object")

    cash = finite_number(payload.get("cash"), "cash", minimum=0.0)
    equity = finite_number(payload.get("equity"), "equity", minimum=0.0)
    trades_payload = payload.get("trades")
    logs_payload = payload.get("logs")
    if not isinstance(trades_payload, list) or not isinstance(logs_payload, list):
        raise ValueError("trades and logs must be arrays")
    if len(trades_payload) > 1000 or len(logs_payload) > 100:
        raise ValueError("state history is too large")

    last_bar = payload.get("lastBar")
    if last_bar is not None:
        last_bar = finite_number(last_bar, "lastBar", minimum=0.0)

    return {
        "cash": cash,
        "equity": equity,
        "position": _validate_position(payload.get("position")),
        "trades": [_validate_trade(item, index) for index, item in enumerate(trades_payload)],
        "logs": [_validate_log(item, index) for index, item in enumerate(logs_payload)],
        "lastBar": last_bar,
        "lastFundingAt": finite_number(payload.get("lastFundingAt"), "lastFundingAt", minimum=0.0),
        "nextFundingAt": finite_number(payload.get("nextFundingAt"), "nextFundingAt", minimum=0.0),
    }


def _record(state: dict[str, Any], revision: int) -> dict[str, Any]:
    return {"schemaVersion": SCHEMA_VERSION, "revision": revision, "state": state}


class SharedPaperState:
    lock = threading.RLock()
    live_record: dict[str, Any] | None = None
    owner: str | None = None
    lease_token: str | None = None
    lease_generation = 0
    lease_until_mono = 0.0

    @classmethod
    def _write_record(cls, record: dict[str, Any]) -> None:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = STATE_PATH.with_name(f".{STATE_PATH.name}.{os.getpid()}.tmp")
        encoded = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        try:
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            if STATE_PATH.exists():
                backup_temporary = BACKUP_PATH.with_name(f".{BACKUP_PATH.name}.{os.getpid()}.tmp")
                with STATE_PATH.open("rb") as source, backup_temporary.open("wb") as backup:
                    backup.write(source.read())
                    backup.flush()
                    os.fsync(backup.fileno())
                os.replace(backup_temporary, BACKUP_PATH)
            os.replace(temporary, STATE_PATH)
        except OSError as error:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise StateStoreError(f"cannot persist canonical state: {error}") from error

    @classmethod
    def _read_record(cls) -> dict[str, Any]:
        if not STATE_PATH.exists():
            record = _record(normalize_state(initial_state()), 0)
            cls._write_record(record)
            return record
        try:
            payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("schemaVersion") != SCHEMA_VERSION:
                raise ValueError("unsupported state schema")
            revision = payload.get("revision")
            if not isinstance(revision, int) or revision < 0:
                raise ValueError("invalid state revision")
            state = normalize_state(payload.get("state"))
            return _record(state, revision)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            # Do not silently reset a possibly valuable paper-trading ledger.
            raise StateStoreError(f"canonical state is unreadable: {error}") from error

    @classmethod
    def _current_record(cls) -> dict[str, Any]:
        if cls.live_record is None:
            cls.live_record = cls._read_record()
        return cls.live_record

    @classmethod
    def current_state_copy(cls) -> dict[str, Any]:
        with cls.lock:
            return deepcopy(cls._current_record()["state"])

    @classmethod
    def publish_live(cls, state: dict[str, Any], *, persist: bool = False) -> dict[str, Any]:
        with cls.lock:
            current = cls._current_record()
            new_record = _record(normalize_state(state), current["revision"] + 1)
            cls.live_record = new_record
            if persist:
                cls._write_record(new_record)
            return new_record

    @classmethod
    def checkpoint_live(cls) -> None:
        with cls.lock:
            if cls.live_record is not None:
                cls._write_record(cls.live_record)

    @classmethod
    def reset_from_engine(cls) -> dict[str, Any]:
        with cls.lock:
            current = cls._current_record()
            new_record = _record(normalize_state(initial_state()), current["revision"] + 1)
            cls.live_record = new_record
            cls._write_record(new_record)
            return deepcopy(new_record["state"])

    @classmethod
    def _lease_active(cls) -> bool:
        return cls.owner is not None and cls.lease_token is not None and cls.lease_until_mono > time.monotonic()

    @classmethod
    def _lease_until_wall_ms(cls) -> int:
        if not cls._lease_active():
            return 0
        remaining = max(0.0, cls.lease_until_mono - time.monotonic())
        return int((time.time() + remaining) * 1000)

    @classmethod
    def _lease_matches(cls, client_id: str, token: Any, generation: Any) -> bool:
        return (
            cls._lease_active()
            and cls.owner == client_id
            and isinstance(token, str)
            and secrets.compare_digest(cls.lease_token or "", token)
            and isinstance(generation, int)
            and generation == cls.lease_generation
        )

    @classmethod
    def _response(cls, record: dict[str, Any], *, leader: bool, include_token: bool = False) -> dict[str, Any]:
        return {
            "mode": "hft" if HFT_MODE else "legacy",
            "leader": leader,
            "ownerActive": cls._lease_active(),
            "leaseUntil": cls._lease_until_wall_ms(),
            "leaseGeneration": cls.lease_generation,
            "leaseToken": cls.lease_token if include_token and leader else None,
            "revision": record["revision"],
            "state": record["state"],
            "engine": HFT_ENGINE.snapshot() if HFT_ENGINE is not None else None,
        }

    @classmethod
    def acquire(cls, client_id: str) -> dict[str, Any]:
        with cls.lock:
            if HFT_MODE:
                return cls._response(cls._current_record(), leader=False)
            if not cls._lease_active() or cls.owner == client_id:
                cls.owner = client_id
                cls.lease_token = secrets.token_urlsafe(32)
                cls.lease_generation += 1
                cls.lease_until_mono = time.monotonic() + LEASE_SECONDS
                leader = True
            else:
                leader = False
            return cls._response(cls._read_record(), leader=leader, include_token=True)

    @classmethod
    def save_from_leader(cls, client_id: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        with cls.lock:
            record = cls._current_record()
            if HFT_MODE:
                return 409, {"error": "server HFT engine owns the paper state", **cls._response(record, leader=False)}
            if not cls._lease_matches(client_id, payload.get("leaseToken"), payload.get("leaseGeneration")):
                return 409, {"error": "leader lease required", **cls._response(record, leader=False)}
            revision = payload.get("revision")
            if not isinstance(revision, int) or revision != record["revision"]:
                return 409, {"error": "state revision is stale", **cls._response(record, leader=False)}
            state = normalize_state(payload.get("state"))
            new_record = _record(state, record["revision"] + 1)
            cls._write_record(new_record)
            cls.live_record = new_record
            cls.lease_until_mono = time.monotonic() + LEASE_SECONDS
            return 200, cls._response(new_record, leader=True, include_token=True)

    @classmethod
    def reset_from_leader(cls, client_id: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        with cls.lock:
            record = cls._current_record()
            if HFT_MODE:
                state = cls.reset_from_engine()
                record = cls._current_record()
                if HFT_ENGINE is not None:
                    HFT_ENGINE.load_state(state)
                return 200, cls._response(record, leader=False)
            if not cls._lease_matches(client_id, payload.get("leaseToken"), payload.get("leaseGeneration")):
                return 409, {"error": "leader lease required", **cls._response(record, leader=False)}
            revision = payload.get("revision")
            if not isinstance(revision, int) or revision != record["revision"]:
                return 409, {"error": "state revision is stale", **cls._response(record, leader=False)}
            new_record = _record(normalize_state(initial_state()), record["revision"] + 1)
            cls._write_record(new_record)
            cls.live_record = new_record
            cls.lease_until_mono = time.monotonic() + LEASE_SECONDS
            return 200, cls._response(new_record, leader=True, include_token=True)

    @classmethod
    def release(cls, client_id: str, payload: dict[str, Any]) -> bool:
        with cls.lock:
            if not cls._lease_matches(client_id, payload.get("leaseToken"), payload.get("leaseGeneration")):
                return False
            cls.owner = None
            cls.lease_token = None
            cls.lease_until_mono = 0.0
            return True

    @classmethod
    def public_state(cls) -> dict[str, Any]:
        with cls.lock:
            record = cls._current_record()
            return {
                "mode": "hft" if HFT_MODE else "legacy",
                "state": record["state"],
                "revision": record["revision"],
                "ownerActive": cls._lease_active(),
                "engine": HFT_ENGINE.snapshot() if HFT_ENGINE is not None else None,
            }


class HFTPaperEngine:
    """Server-side one-second paper trader fed by public Binance streams."""

    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.state_lock = threading.Lock()
        self.state = SharedPaperState.current_state_copy()
        self.prices: deque[tuple[int, float]] = deque(maxlen=20_000)
        self.flow: deque[tuple[int, float]] = deque(maxlen=20_000)
        self.bid = 0.0
        self.ask = 0.0
        self.bid_qty = 0.0
        self.ask_qty = 0.0
        self.last_price = 0.0
        self.funding_rate = 0.0
        self.next_funding_at = int(self.state.get("nextFundingAt", 0))
        self.last_event_at = 0
        self.last_decision_at = 0
        self.last_close_at = 0
        self.last_checkpoint = time.monotonic()
        self.last_funding_poll = 0.0
        self.strategy = StrategyCore()
        self.decision_count = 0
        self.connected = False
        self.error = "尚未連線"
        self.score = 0.0
        self.probability = 0.5
        self.direction = 0
        self.raw_direction = 0
        self.gross_edge_bps = 0.0
        self.expected_edge_bps = 0.0
        self.round_trip_cost_bps = 0.0
        self.candidate_direction = 0
        self.candidate_count = 0
        self.spread_bps = 0.0
        self.flow_imbalance = 0.0
        self.book_imbalance = 0.0
        self.ret5s = 0.0
        self.vol_bps = 5.0
        self.event_counts: dict[str, int] = {}
        self.thread = threading.Thread(target=self._run, name="btc-hft-paper", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def snapshot(self) -> dict[str, Any]:
        # These are scalar snapshots. The engine owns mutation on its worker
        # thread; returning primitives avoids lock inversion with the state API.
        return {
            "mode": "hft",
            "decisionBackend": self.strategy.backend,
            "connected": self.connected,
            "error": self.error,
            "lastPrice": self.last_price,
            "lastEventAt": self.last_event_at,
            "lastDecisionAt": self.last_decision_at,
            "decisionCount": self.decision_count,
            "score": self.score,
            "probability": self.probability,
            "direction": self.direction,
            "rawDirection": self.raw_direction,
            "grossEdgeBps": self.gross_edge_bps,
            "expectedEdgeBps": self.expected_edge_bps,
            "roundTripCostBps": self.round_trip_cost_bps,
            "spreadBps": self.spread_bps,
            "flowImbalance": self.flow_imbalance,
            "bookImbalance": self.book_imbalance,
            "ret5s": self.ret5s,
            "volBps": self.vol_bps,
            "fundingRate": self.funding_rate,
            "nextFundingAt": self.next_funding_at,
            "eventCounts": dict(self.event_counts),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        with self.state_lock:
            self.state = deepcopy(state)

    def _trim(self, now_ms: int) -> None:
        cutoff = now_ms - 60_000
        while self.prices and self.prices[0][0] < cutoff:
            self.prices.popleft()
        while self.flow and self.flow[0][0] < cutoff:
            self.flow.popleft()

    def _metrics(self, now_ms: int) -> tuple[float, float, float, float, float]:
        price = self.last_price
        if not price or len(self.prices) < 2:
            return 0.0, 0.0, 0.0, 5.0, 0.0
        old = next((value for timestamp, value in reversed(self.prices) if timestamp <= now_ms - 5_000), None)
        ret5 = price / old - 1.0 if old and now_ms - next(timestamp for timestamp, value in reversed(self.prices) if timestamp <= now_ms - 5_000) <= 7_500 else 0.0
        recent = [
            value
            for timestamp, value in self.prices
            if timestamp >= now_ms - 5_000 and price * 0.95 <= value <= price * 1.05
        ]
        if len(recent) < 2:
            recent = [price]
        returns = [recent[index] / recent[index - 1] - 1.0 for index in range(1, len(recent)) if recent[index - 1]]
        if returns:
            mean = sum(returns) / len(returns)
            variance = sum((value - mean) ** 2 for value in returns) / len(returns)
            vol_pct = max(math.sqrt(variance), (max(recent) - min(recent)) / price, 0.0005)
        else:
            vol_pct = 0.0005
        vol_bps = min(vol_pct * 10_000.0, 200.0)
        signed = sum(value for timestamp, value in self.flow if timestamp >= now_ms - 5_000)
        absolute = sum(abs(value) for timestamp, value in self.flow if timestamp >= now_ms - 5_000)
        flow_imbalance = signed / absolute if absolute else 0.0
        bid = self.bid or price
        ask = self.ask or price
        spread_bps = max(0.0, (ask - bid) / price * 10_000)
        book_total = self.bid_qty + self.ask_qty
        book_imbalance = (self.bid_qty - self.ask_qty) / book_total if book_total else 0.0
        return ret5, flow_imbalance, book_imbalance, vol_bps, spread_bps

    @staticmethod
    def _sigmoid(score: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, score))))

    def _mark_equity(self, price: float) -> float:
        position = self.state.get("position")
        if not position:
            return self.state["cash"]
        return self.state["cash"] + position["qty"] * (price - position["entry"]) * position["side"]

    def _log(self, message: str, now_ms: int) -> None:
        self.state["logs"].insert(0, {"time": now_ms, "message": message})
        self.state["logs"] = self.state["logs"][:30]

    def _close(self, price: float, reason: str, now_ms: int) -> None:
        position = self.state.get("position")
        if not position:
            return
        bid = self.bid or price
        ask = self.ask or price
        exit_price = bid if position["side"] == 1 else ask
        spread_bps = max(0.0, (ask - bid) / price * 10_000) if price else 0.0
        exit_cost = position["qty"] * exit_price * (0.0004 + 0.0001)
        cash_before = self.state["cash"]
        gross = position["qty"] * (exit_price - position["entry"]) * position["side"]
        raw_cash = cash_before + gross - exit_cost
        self.state["cash"] = max(0.0, raw_cash) if reason == "liquidation" else raw_cash
        liquidation_adjustment = self.state["cash"] - raw_cash
        net = gross - position["entryCost"] - exit_cost - position.get("funding", 0.0) + liquidation_adjustment
        self.state["trades"].insert(0, {
            "side": position["side"],
            "qty": position["qty"],
            "entry": position["entry"],
            "exit": exit_price,
            "entryNotional": position["entryNotional"],
            "exitNotional": position["qty"] * exit_price,
            "gross": gross,
            "entryCost": position["entryCost"],
            "exitCost": exit_cost,
            "net": net,
            "cost": position["entryCost"] + exit_cost,
            "funding": position.get("funding", 0.0),
            "entrySpreadBps": position.get("entrySpreadBps", 0.0),
            "exitSpreadBps": spread_bps,
            "cashBeforeClose": cash_before,
            "rawCashAfterClose": raw_cash,
            "cashAfterClose": self.state["cash"],
            "liquidationAdjustment": liquidation_adjustment,
            "reason": reason,
            "opened": position["opened"],
            "time": now_ms,
        })
        self.state["trades"] = self.state["trades"][:1000]
        self._log(f"高頻平倉 {('做多' if position['side'] == 1 else '做空')} · 淨 PnL {net:+.2f} USDT", now_ms)
        self.state["position"] = None
        self.last_close_at = now_ms

    def _open(self, side: int, price: float, stop_pct: float, spread_bps: float, now_ms: int) -> None:
        if self.state.get("position") or self.state["cash"] <= 0:
            return
        stop_pct = min(0.02, max(0.001, stop_pct))
        bid = self.bid or price
        ask = self.ask or price
        entry = ask if side == 1 else bid
        cost_rate = 2 * 0.0004 + 2 * 0.0001 + max(0.0, spread_bps) / 10_000
        notional = min((self.state["cash"] * HFT_RISK_PER_TRADE) / max(stop_pct + cost_rate, 0.0001), self.state["cash"] * 20.0)
        if notional <= 0:
            return
        qty = notional / entry
        entry_cost = qty * entry * (0.0004 + 0.0001)
        self.state["cash"] -= entry_cost
        self.state["position"] = {
            "side": side,
            "qty": qty,
            "entry": entry,
            "entryNotional": notional,
            "entryCost": entry_cost,
            "funding": 0.0,
            "entrySpreadBps": spread_bps,
            "stop": entry * (1.0 - side * stop_pct),
            "takeProfit": entry * (1.0 + side * HFT_TAKE_PROFIT_PCT),
            "opened": now_ms,
        }
        self._log(f"高頻開倉 {('做多' if side == 1 else '做空')} · 名目 {notional:.2f} USDT", now_ms)

    def _event_risk_check(self, now_ms: int) -> None:
        """Check stop and liquidation on every market event, never on a timer."""
        position = self.state.get("position")
        if position is None or not self.last_price:
            return
        liquidation = self._mark_equity(self.last_price) <= abs(position["qty"] * self.last_price) * 0.005
        stop_hit = self.last_price <= position["stop"] if position["side"] == 1 else self.last_price >= position["stop"]
        take_profit_hit = self.last_price >= position.get("takeProfit", float("inf")) if position["side"] == 1 else self.last_price <= position.get("takeProfit", float("-inf"))
        if liquidation:
            self._close(self.last_price, "liquidation", now_ms)
        elif stop_hit:
            self._close(self.last_price, "stop", now_ms)
        elif take_profit_hit:
            self._close(self.last_price, "take profit", now_ms)

    def _decide(self, now_ms: int) -> None:
        if now_ms - self.last_decision_at < HFT_DECISION_MS or not self.last_price:
            return
        self.last_decision_at = now_ms
        self.decision_count += 1
        self._trim(now_ms)
        self._poll_funding()
        ret5, flow, book, vol_bps, spread_bps = self._metrics(now_ms)
        decision = self.strategy.evaluate(
            MarketFeatures(ret5, flow, book, vol_bps, spread_bps),
            self.candidate_direction,
            self.candidate_count,
        )
        volatility = max(vol_bps / 10_000, 0.0005)
        confirmed_direction = decision.confirmed_direction
        self.candidate_direction = decision.candidate_direction
        self.candidate_count = decision.candidate_count
        self.score = decision.score
        self.probability = decision.probability_up
        self.direction = decision.direction
        self.raw_direction = decision.raw_direction
        self.gross_edge_bps = decision.gross_edge_bps
        self.expected_edge_bps = decision.expected_edge_bps
        self.round_trip_cost_bps = decision.round_trip_cost_bps
        self.spread_bps = spread_bps
        self.flow_imbalance = flow
        self.book_imbalance = book
        self.ret5s = ret5
        self.vol_bps = vol_bps

        position = self.state.get("position")
        if position and self.next_funding_at and now_ms >= self.next_funding_at:
            payment = position["qty"] * self.last_price * self.funding_rate * position["side"]
            self.state["cash"] -= payment
            position["funding"] = position.get("funding", 0.0) + payment
            self.state["lastFundingAt"] = self.next_funding_at
            self.next_funding_at += 8 * 60 * 60 * 1000
            self.state["nextFundingAt"] = self.next_funding_at

        if position:
            stop_hit = self.last_price <= position["stop"] if position["side"] == 1 else self.last_price >= position["stop"]
            take_profit_hit = self.last_price >= position.get("takeProfit", float("inf")) if position["side"] == 1 else self.last_price <= position.get("takeProfit", float("-inf"))
            liquidation = self._mark_equity(self.last_price) <= abs(position["qty"] * self.last_price) * 0.005
            held_ms = now_ms - position["opened"]
            timed_out = held_ms >= HFT_MAX_HOLD_MS
            if liquidation:
                self._close(self.last_price, "liquidation", now_ms)
            elif stop_hit:
                self._close(self.last_price, "stop", now_ms)
            elif timed_out or (confirmed_direction and held_ms >= HFT_MIN_HOLD_MS and confirmed_direction != position["side"]):
                self._close(self.last_price, "signal exit", now_ms)
        if (
            not self.state.get("position")
            and confirmed_direction
            and now_ms - self.last_close_at >= HFT_REENTRY_COOLDOWN_MS
        ):
            self._open(confirmed_direction, self.last_price, max(0.001, 2 * volatility), spread_bps, now_ms)
        self.state["lastBar"] = now_ms
        self.state["nextFundingAt"] = self.next_funding_at
        self.state["equity"] = self._mark_equity(self.last_price)
        persist = time.monotonic() - self.last_checkpoint >= HFT_CHECKPOINT_SECONDS
        SharedPaperState.publish_live(self.state, persist=persist)
        if persist:
            self.last_checkpoint = time.monotonic()

    def _handle(self, message: dict[str, Any]) -> None:
        data = message.get("data", message)
        event = data.get("e")
        self.event_counts[event or "unknown"] = self.event_counts.get(event or "unknown", 0) + 1
        event_time = int(data.get("E") or data.get("T") or int(time.time() * 1000))
        self.last_event_at = event_time
        if event in {"aggTrade", "trade"}:
            price = float(data["p"])
            quantity = float(data["q"])
            if not math.isfinite(price) or price <= 0 or not math.isfinite(quantity) or quantity <= 0:
                return
            signed_quantity = -quantity if data.get("m") else quantity
            self.last_price = price
            self.prices.append((event_time, price))
            self.flow.append((event_time, signed_quantity))
        elif event == "bookTicker":
            self.bid = float(data.get("b", self.bid))
            self.ask = float(data.get("a", self.ask))
            self.bid_qty = float(data.get("B", self.bid_qty))
            self.ask_qty = float(data.get("A", self.ask_qty))
            if not self.last_price:
                self.last_price = (self.bid + self.ask) / 2
        elif event == "markPriceUpdate":
            self.funding_rate = float(data.get("r", self.funding_rate))
            self.next_funding_at = int(data.get("T") or self.next_funding_at or 0)
            self.state["nextFundingAt"] = self.next_funding_at
        self._event_risk_check(event_time)
        self._decide(event_time)

    def _poll_funding(self) -> None:
        if time.monotonic() - self.last_funding_poll < 30:
            return
        self.last_funding_poll = time.monotonic()
        try:
            with urlopen("https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT", timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.funding_rate = float(payload.get("lastFundingRate", self.funding_rate))
            self.next_funding_at = int(payload.get("nextFundingTime", self.next_funding_at or 0))
            self.state["nextFundingAt"] = self.next_funding_at
        except Exception:
            return

    def _run(self) -> None:
        if websocket is None:
            self.error = "缺少 websocket-client；高頻引擎未啟動"
            return
        while not self.stop_event.is_set():
            connection = None
            try:
                connection = websocket.create_connection(HFT_WS_URL, timeout=30, enable_multithread=True)
                connection.settimeout(30)
                connection.send(json.dumps({"method": "SUBSCRIBE", "params": HFT_STREAMS, "id": 1}))
                self.connected = True
                self.error = ""
                while not self.stop_event.is_set():
                    raw = connection.recv()
                    if not raw:
                        raise ConnectionError("Binance WebSocket closed")
                    self._handle(json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw))
            except Exception as error:  # reconnect with a visible status
                self.connected = False
                self.error = f"{type(error).__name__}: {error}"[:240]
                try:
                    SharedPaperState.checkpoint_live()
                except StateStoreError:
                    pass
                self.stop_event.wait(3)
            finally:
                if connection is not None:
                    try:
                        connection.close()
                    except Exception:
                        pass


class Handler(SimpleHTTPRequestHandler):
    server_version = "BTCSharedPaper/2.0"

    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("invalid request body size") from error
        if content_length <= 0 or content_length > MAX_BODY_BYTES:
            raise ValueError("invalid request body size")
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be an object")
        return payload

    @staticmethod
    def _client_id(payload: dict[str, Any]) -> str:
        client_id = payload.get("clientId")
        if not isinstance(client_id, str) or not 8 <= len(client_id) <= 128:
            raise ValueError("clientId is required")
        return client_id

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/paper-state":
            try:
                self._json(SharedPaperState.public_state())
            except StateStoreError as error:
                self._json({"error": str(error)}, status=503)
            return
        if path.startswith("/api/") or any(part.startswith(".") for part in path.split("/") if part):
            self._json({"error": "not found"}, status=404)
            return
        try:
            super().do_GET()
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self._body()
            client_id = self._client_id(payload)
            if path == "/api/lease/acquire":
                self._json(SharedPaperState.acquire(client_id))
                return
            if path == "/api/paper-state":
                status, response = SharedPaperState.save_from_leader(client_id, payload)
                self._json(response, status=status)
                return
            if path == "/api/paper-reset":
                status, response = SharedPaperState.reset_from_leader(client_id, payload)
                self._json(response, status=status)
                return
            if path == "/api/lease/release":
                self._json({"released": SharedPaperState.release(client_id, payload)})
                return
            self._json({"error": "not found"}, status=404)
        except StateStoreError as error:
            self._json({"error": str(error)}, status=503)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._json({"error": str(error)}, status=400)

    def log_message(self, format_string: str, *args: Any) -> None:
        return


def main() -> None:
    global HFT_ENGINE
    handler = partial(Handler, directory=str(ROOT))
    server = ThreadingHTTPServer((HOST, PORT), handler)
    if HFT_MODE:
        HFT_ENGINE = HFTPaperEngine()
        HFT_ENGINE.start()
    print(f"BTC shared paper monitor: http://{HOST}:{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if HFT_ENGINE is not None:
            HFT_ENGINE.stop()
            try:
                SharedPaperState.checkpoint_live()
            except StateStoreError:
                pass
        server.server_close()


if __name__ == "__main__":
    main()
