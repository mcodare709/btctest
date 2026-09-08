"""Dependency-free portfolio environment for risk-aware RL experiments.

An action observed at close[t] executes at open[t+1]. Same-direction actions
open distinct lots; opposite-direction actions first close all existing lots.
The environment is intentionally independent of Gymnasium so execution
semantics can be unit-tested before an RL framework is added.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import ceil, log
from typing import Sequence

import numpy as np
import pandas as pd


class PortfolioAction(IntEnum):
    HOLD = 0
    CLOSE_ALL = 1
    LONG_1X = 2
    LONG_3X = 3
    LONG_5X = 4
    SHORT_1X = 5
    SHORT_3X = 6
    SHORT_5X = 7


_ACTION_TARGETS = {
    PortfolioAction.LONG_1X: (1, 1.0),
    PortfolioAction.LONG_3X: (1, 3.0),
    PortfolioAction.LONG_5X: (1, 5.0),
    PortfolioAction.SHORT_1X: (-1, 1.0),
    PortfolioAction.SHORT_3X: (-1, 3.0),
    PortfolioAction.SHORT_5X: (-1, 5.0),
}


@dataclass(frozen=True)
class RLPortfolioConfig:
    initial_equity: float = 100.0
    episode_leverage_choices: tuple[float, ...] = (3.0, 4.0, 5.0)
    base_allocation_fraction: float = 0.10
    maintenance_margin_rate: float = 0.005
    fee_rate: float = 0.0004
    slippage_bps: float = 1.0
    default_spread_bps: float = 2.0
    liquidation_fee_rate: float = 0.001
    max_holding_bars: int = 10
    recovery_step: float = 1.25
    max_recovery_multiplier: float = 1.50
    max_recovery_layers: int = 2
    max_drawdown: float = 0.20
    drawdown_penalty: float = 1.0
    liquidation_penalty: float = 0.25
    max_entries_per_window: int = 5
    entry_rate_window_minutes: float = 5.0
    expected_return_gate_enabled: bool = False
    min_expected_edge_bps: float = 5.0

    def __post_init__(self) -> None:
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        if not self.episode_leverage_choices or min(self.episode_leverage_choices) <= 0:
            raise ValueError("episode leverage choices must be positive")
        if not 0 < self.base_allocation_fraction <= 1:
            raise ValueError("base_allocation_fraction must be in (0, 1]")
        if not 0 <= self.maintenance_margin_rate < 1:
            raise ValueError("maintenance_margin_rate must be in [0, 1)")
        if min(self.fee_rate, self.slippage_bps, self.default_spread_bps, self.liquidation_fee_rate) < 0:
            raise ValueError("cost assumptions cannot be negative")
        if self.max_holding_bars <= 0:
            raise ValueError("max_holding_bars must be positive")
        if self.recovery_step < 1 or self.max_recovery_multiplier < 1 or self.max_recovery_layers < 0:
            raise ValueError("recovery constraints are invalid")
        if not 0 < self.max_drawdown < 1:
            raise ValueError("max_drawdown must be in (0, 1)")
        if self.drawdown_penalty < 0 or self.liquidation_penalty < 0:
            raise ValueError("reward penalties cannot be negative")
        if self.max_entries_per_window <= 0 or self.entry_rate_window_minutes <= 0:
            raise ValueError("entry rate limits must be positive")
        if self.min_expected_edge_bps < 0:
            raise ValueError("min_expected_edge_bps cannot be negative")


@dataclass
class PortfolioLot:
    side: int
    quantity: float
    entry_price: float
    entry_index: int
    entry_cost: float
    funding_cost: float = 0.0


class RLPortfolioEnv:
    """Account-level next-open simulator with overlapping same-side lots."""

    ACCOUNT_OBSERVATIONS = (
        "account_net_leverage",
        "account_gross_leverage",
        "account_unrealized_return",
        "account_drawdown",
        "account_lot_count",
        "account_loss_streak",
        "account_recovery_layer",
        "episode_max_leverage",
    )

    def __init__(
        self,
        market_data: pd.DataFrame,
        *,
        feature_names: Sequence[str] = (),
        config: RLPortfolioConfig | None = None,
        seed: int = 42,
    ) -> None:
        required = {"open", "high", "low", "close"}
        missing = sorted(required.difference(market_data.columns))
        if missing:
            raise ValueError(f"market_data missing required columns: {missing}")
        if len(market_data) < 2:
            raise ValueError("market_data requires at least two rows")
        unavailable = sorted(set(feature_names).difference(market_data.columns))
        if unavailable:
            raise ValueError(f"feature columns unavailable: {unavailable}")
        prices = market_data[list(required)].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(prices.to_numpy(dtype=float)).all() or (prices <= 0).any().any():
            raise ValueError("OHLC prices must be finite and positive")
        self.frame = market_data.copy()
        self.feature_names = tuple(feature_names)
        self.config = config or RLPortfolioConfig()
        self.rng = np.random.default_rng(seed)
        self.cash = self.config.initial_equity
        self.lots: list[PortfolioLot] = []
        self.trades: list[dict[str, float | int | str]] = []
        self.cursor = 0
        self.stop_index = len(self.frame) - 1
        self.episode_max_leverage = min(self.config.episode_leverage_choices)
        self.peak_equity = self.config.initial_equity
        self.drawdown = 0.0
        self.loss_streak = 0
        self.recovery_layer = 0
        self.liquidated = False
        self.mark_price_fallback_used = False
        self.entry_execution_indices: list[int] = []
        self.entry_attempts_blocked = 0

    @property
    def observation_names(self) -> tuple[str, ...]:
        masks = tuple(f"{name}__missing" for name in self.feature_names)
        return self.feature_names + masks + self.ACCOUNT_OBSERVATIONS

    def reset(
        self,
        *,
        start_index: int = 0,
        max_steps: int | None = None,
        episode_max_leverage: float | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        if not 0 <= start_index < len(self.frame) - 1:
            raise ValueError("start_index must leave at least one next bar")
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be positive")
        selected = float(
            self.rng.choice(self.config.episode_leverage_choices)
            if episode_max_leverage is None
            else episode_max_leverage
        )
        if selected not in self.config.episode_leverage_choices:
            raise ValueError("episode_max_leverage must be one of episode_leverage_choices")
        self.cash = self.config.initial_equity
        self.lots = []
        self.trades = []
        self.cursor = start_index
        natural_stop = len(self.frame) - 1
        self.stop_index = natural_stop if max_steps is None else min(natural_stop, start_index + max_steps)
        self.episode_max_leverage = selected
        self.peak_equity = self.config.initial_equity
        self.drawdown = 0.0
        self.loss_streak = 0
        self.recovery_layer = 0
        self.liquidated = False
        self.mark_price_fallback_used = False
        self.entry_execution_indices = []
        self.entry_attempts_blocked = 0
        return self._observation(), self._info()

    def _entries_in_rate_window(self, execution_index: int) -> int:
        recent_indices = self.entry_execution_indices[-self.config.max_entries_per_window:]
        label = self.frame.index[execution_index]
        if isinstance(label, pd.Timestamp):
            cutoff = label - pd.Timedelta(minutes=self.config.entry_rate_window_minutes)
            return sum(self.frame.index[index] > cutoff for index in recent_indices)
        fallback_bars = ceil(self.config.entry_rate_window_minutes)
        cutoff_index = execution_index - fallback_bars
        return sum(index > cutoff_index for index in recent_indices)

    def _entry_rate_available(self, execution_index: int) -> bool:
        return self._entries_in_rate_window(execution_index) < self.config.max_entries_per_window

    def _expected_return_gate(self) -> tuple[float, float, int]:
        row = self.frame.iloc[self.cursor]
        long_expected = row.get("long_expected_net_return", np.nan)
        short_expected = row.get("short_expected_net_return", np.nan)
        long_value = float(long_expected) if long_expected is not None and not pd.isna(long_expected) else np.nan
        short_value = float(short_expected) if short_expected is not None and not pd.isna(short_expected) else np.nan
        if not np.isfinite(long_value) or not np.isfinite(short_value):
            return long_value, short_value, 0
        threshold = self.config.min_expected_edge_bps / 10_000.0
        if long_value > threshold and long_value > short_value:
            return long_value, short_value, 1
        if short_value > threshold and short_value > long_value:
            return long_value, short_value, -1
        return long_value, short_value, 0

    def _numeric(self, row: pd.Series, name: str, default: float = 0.0) -> float:
        value = row.get(name, default)
        return default if value is None or pd.isna(value) else float(value)

    def _mark(self, row: pd.Series, stage: str) -> float:
        aliases = (f"mark_{stage}", f"mark_price_{stage}")
        if stage == "close":
            aliases += ("mark_price",)
        for name in aliases:
            value = row.get(name, np.nan)
            if value is not None and not pd.isna(value):
                return float(value)
        self.mark_price_fallback_used = True
        return self._numeric(row, stage)

    def _unrealized(self, price: float) -> float:
        return float(sum(lot.quantity * (price - lot.entry_price) * lot.side for lot in self.lots))

    def _equity(self, price: float) -> float:
        return self.cash + self._unrealized(price)

    def _gross_notional(self, price: float) -> float:
        return float(sum(lot.quantity * price for lot in self.lots))

    def _net_notional(self, price: float) -> float:
        return float(sum(lot.quantity * price * lot.side for lot in self.lots))

    def _execution(self, reference_price: float, quantity: float, side: int, row: pd.Series, *, is_entry: bool) -> tuple[float, float]:
        half_spread = max(self._numeric(row, "spread_bps", self.config.default_spread_bps), 0.0) / 20_000.0
        slippage = self.config.slippage_bps / 10_000.0
        adverse = side if is_entry else -side
        fill = reference_price * (1.0 + adverse * (half_spread + slippage))
        fee = quantity * fill * self.config.fee_rate
        friction = quantity * reference_price * (half_spread + slippage)
        return fill, fee + friction

    def _recovery_multiplier(self) -> float:
        return min(self.config.recovery_step**self.recovery_layer, self.config.max_recovery_multiplier)

    def _open_lot(self, side: int, action_leverage: float, price: float, row: pd.Series, index: int) -> bool:
        equity = self._equity(price)
        if equity <= 0:
            return False
        desired = equity * self.config.base_allocation_fraction * action_leverage * self._recovery_multiplier()
        existing = self._gross_notional(price)
        _, unit_entry_cost = self._execution(price, 1.0, side, row, is_entry=True)
        entry_cost_rate = unit_entry_cost / price
        capacity = max(
            (self.episode_max_leverage * equity - existing)
            / (1.0 + self.episode_max_leverage * entry_cost_rate),
            0.0,
        )
        notional = min(desired, capacity)
        if notional <= 1e-12:
            return False
        quantity = notional / price
        _, entry_cost = self._execution(price, quantity, side, row, is_entry=True)
        if entry_cost >= self.cash:
            return False
        self.cash -= entry_cost
        self.lots.append(PortfolioLot(side, quantity, price, index, entry_cost))
        return True

    def _close_lots(self, lots: Sequence[PortfolioLot], price: float, row: pd.Series, index: int, reason: str) -> float:
        total_net = 0.0
        for lot in list(lots):
            _, exit_cost = self._execution(price, lot.quantity, lot.side, row, is_entry=False)
            gross = lot.quantity * (price - lot.entry_price) * lot.side
            liquidation_fee = lot.quantity * price * self.config.liquidation_fee_rate if reason == "liquidation" else 0.0
            self.cash += gross - exit_cost - liquidation_fee
            net = gross - lot.entry_cost - exit_cost - liquidation_fee - lot.funding_cost
            total_net += net
            self.trades.append(
                {
                    "side": lot.side,
                    "entry_index": lot.entry_index,
                    "exit_index": index,
                    "quantity": lot.quantity,
                    "entry_price": lot.entry_price,
                    "exit_price": price,
                    "net_pnl": net,
                    "funding_cost": lot.funding_cost,
                    "exit_reason": reason,
                }
            )
            self.lots.remove(lot)
        if lots:
            if total_net < 0:
                self.loss_streak += 1
                self.recovery_layer = min(self.recovery_layer + 1, self.config.max_recovery_layers)
            elif total_net > 0:
                self.loss_streak = 0
                self.recovery_layer = 0
        return total_net

    def _maintenance_breached(self, price: float) -> bool:
        if not self.lots:
            return False
        equity = self._equity(price)
        maintenance = self._gross_notional(price) * self.config.maintenance_margin_rate
        return equity <= maintenance or equity <= 0

    def _liquidate(self, price: float, row: pd.Series, index: int) -> None:
        self._close_lots(tuple(self.lots), price, row, index, "liquidation")
        self.cash = max(self.cash, 0.0)
        self.liquidated = True

    def _apply_funding(self, row: pd.Series, open_mark: float) -> None:
        value = row.get("funding_rate", np.nan)
        if value is None or pd.isna(value):
            return
        rate = float(value)
        for lot in self.lots:
            payment = lot.quantity * open_mark * rate * lot.side
            self.cash -= payment
            lot.funding_cost += payment

    def _observation(self) -> np.ndarray:
        row = self.frame.iloc[self.cursor]
        values: list[float] = []
        masks: list[float] = []
        for name in self.feature_names:
            value = row.get(name, np.nan)
            missing = value is None or pd.isna(value) or not np.isfinite(float(value))
            values.append(0.0 if missing else float(value))
            masks.append(float(missing))
        mark = self._mark(row, "close")
        equity = max(self._equity(mark), 1e-12)
        account = [
            self._net_notional(mark) / equity,
            self._gross_notional(mark) / equity,
            self._unrealized(mark) / equity,
            self.drawdown,
            float(len(self.lots)),
            float(self.loss_streak),
            float(self.recovery_layer),
            self.episode_max_leverage,
        ]
        return np.asarray(values + masks + account, dtype=np.float32)

    def valid_action_mask(self) -> np.ndarray:
        mask = np.ones(len(PortfolioAction), dtype=bool)
        mask[PortfolioAction.CLOSE_ALL] = bool(self.lots)
        if self.lots:
            side = self.lots[0].side
            if side == 1:
                mask[[PortfolioAction.SHORT_1X, PortfolioAction.SHORT_3X, PortfolioAction.SHORT_5X]] = True
            else:
                mask[[PortfolioAction.LONG_1X, PortfolioAction.LONG_3X, PortfolioAction.LONG_5X]] = True
        if self.config.expected_return_gate_enabled:
            _, _, gate_direction = self._expected_return_gate()
            if gate_direction != 1:
                mask[[PortfolioAction.LONG_1X, PortfolioAction.LONG_3X, PortfolioAction.LONG_5X]] = False
            if gate_direction != -1:
                mask[[PortfolioAction.SHORT_1X, PortfolioAction.SHORT_3X, PortfolioAction.SHORT_5X]] = False
        execution_index = min(self.cursor + 1, self.stop_index)
        if not self._entry_rate_available(execution_index):
            mask[list(_ACTION_TARGETS)] = False
        return mask

    def _info(self) -> dict[str, object]:
        row = self.frame.iloc[self.cursor]
        mark = self._mark(row, "close")
        equity = self._equity(mark)
        long_expected, short_expected, gate_direction = self._expected_return_gate()
        return {
            "cursor": self.cursor,
            "equity": equity,
            "cash": self.cash,
            "lot_count": len(self.lots),
            "gross_leverage": self._gross_notional(mark) / max(equity, 1e-12),
            "episode_max_leverage": self.episode_max_leverage,
            "recovery_multiplier": self._recovery_multiplier(),
            "loss_streak": self.loss_streak,
            "drawdown": self.drawdown,
            "liquidated": self.liquidated,
            "mark_price_fallback_used": self.mark_price_fallback_used,
            "successful_entry_count": len(self.entry_execution_indices),
            "entry_attempts_blocked": self.entry_attempts_blocked,
            "entries_in_rate_window": self._entries_in_rate_window(self.cursor),
            "max_entries_per_window": self.config.max_entries_per_window,
            "entry_rate_window_minutes": self.config.entry_rate_window_minutes,
            "expected_return_gate_enabled": self.config.expected_return_gate_enabled,
            "min_expected_edge_bps": self.config.min_expected_edge_bps,
            "long_expected_net_return": long_expected,
            "short_expected_net_return": short_expected,
            "expected_return_gate_direction": gate_direction,
            "action_mask": self.valid_action_mask(),
        }

    def step(self, action: int | PortfolioAction) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        try:
            selected = PortfolioAction(int(action))
        except ValueError as error:
            raise ValueError(f"invalid action: {action}") from error
        if self.cursor >= self.stop_index or self.liquidated:
            raise RuntimeError("episode is finished; call reset")

        current_mark = self._mark(self.frame.iloc[self.cursor], "close")
        previous_equity = max(self._equity(current_mark), 1e-12)
        previous_drawdown = self.drawdown
        next_index = self.cursor + 1
        row = self.frame.iloc[next_index]
        trade_open = self._numeric(row, "open")
        open_mark = self._mark(row, "open")

        # Canonical ordering: funding -> open liquidation -> pending action ->
        # scheduled exit -> intrabar liquidation -> close mark-to-market.
        self._apply_funding(row, open_mark)
        if self._maintenance_breached(open_mark):
            self._liquidate(open_mark, row, next_index)

        opened = False
        entry_rate_limited = False
        action_block_reason: str | None = None
        effective_action = selected if not self.liquidated else PortfolioAction.HOLD
        if not self.liquidated:
            if selected == PortfolioAction.CLOSE_ALL:
                self._close_lots(tuple(self.lots), trade_open, row, next_index, "agent_close")
            elif selected in _ACTION_TARGETS:
                side, action_leverage = _ACTION_TARGETS[selected]
                _, _, gate_direction = self._expected_return_gate()
                if self.config.expected_return_gate_enabled and side != gate_direction:
                    entry_rate_limited = True
                    effective_action = PortfolioAction.HOLD
                    self.entry_attempts_blocked += 1
                    action_block_reason = "expected_return_gate"
                elif not self._entry_rate_available(next_index):
                    entry_rate_limited = True
                    effective_action = PortfolioAction.HOLD
                    self.entry_attempts_blocked += 1
                    action_block_reason = "open_rate_limit"
                else:
                    if self.lots and self.lots[0].side != side:
                        self._close_lots(tuple(self.lots), trade_open, row, next_index, "agent_reversal")
                    opened = self._open_lot(side, action_leverage, trade_open, row, next_index)
                    if opened:
                        self.entry_execution_indices.append(next_index)

            expired = [
                lot for lot in self.lots
                if next_index >= lot.entry_index + self.config.max_holding_bars
            ]
            if expired:
                self._close_lots(expired, trade_open, row, next_index, "time_exit")

            if self.lots:
                side = self.lots[0].side
                adverse_mark = self._mark(row, "low" if side == 1 else "high")
                if self._maintenance_breached(adverse_mark):
                    self._liquidate(adverse_mark, row, next_index)

        self.cursor = next_index
        close_mark = self._mark(row, "close")
        if not self.liquidated and self._maintenance_breached(close_mark):
            self._liquidate(close_mark, row, next_index)
        equity = max(self._equity(close_mark), 0.0)
        self.peak_equity = max(self.peak_equity, equity)
        self.drawdown = 1.0 - equity / max(self.peak_equity, 1e-12)
        terminated = self.liquidated or equity <= 0 or self.drawdown >= self.config.max_drawdown
        truncated = self.cursor >= self.stop_index
        if (terminated or truncated) and self.lots and not self.liquidated:
            reason = "risk_termination" if terminated else "episode_end"
            self._close_lots(tuple(self.lots), self._numeric(row, "close"), row, next_index, reason)
            equity = max(self.cash, 0.0)
            self.peak_equity = max(self.peak_equity, equity)
            self.drawdown = 1.0 - equity / max(self.peak_equity, 1e-12)
        reward = log(max(equity, 1e-12) / previous_equity)
        reward -= self.config.drawdown_penalty * max(self.drawdown - previous_drawdown, 0.0)
        if self.liquidated:
            reward -= self.config.liquidation_penalty
        info = self._info()
        info["opened_lot"] = opened
        info["entry_rate_limited"] = entry_rate_limited
        info["action_blocked"] = entry_rate_limited
        info["action_block_reason"] = action_block_reason if entry_rate_limited else None
        info["blocked_action_count"] = self.entry_attempts_blocked
        info["effective_action"] = int(effective_action)
        return self._observation(), float(reward), bool(terminated), bool(truncated), info
