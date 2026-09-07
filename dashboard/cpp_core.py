"""Typed bridge to the optional C++ HFT strategy library.

The Python server owns transport, persistence, and the public API. The C++ DLL
owns deterministic cost-aware signal scoring, signal confirmation, and sizing.
If the DLL has not been built, the reference implementation keeps paper mode
available and reports ``python-reference`` to the dashboard.
"""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LIBRARY = PROJECT_ROOT / "build" / "btc_strategy.dll"


@dataclass(frozen=True)
class MarketFeatures:
    return_5s: float
    flow_imbalance: float
    book_imbalance: float
    volatility_bps: float
    spread_bps: float


@dataclass(frozen=True)
class StrategyDecision:
    score: float
    probability_up: float
    gross_edge_bps: float
    expected_edge_bps: float
    round_trip_cost_bps: float
    raw_direction: int
    direction: int
    confirmed_direction: int
    candidate_direction: int
    candidate_count: int


class _StrategyInput(ctypes.Structure):
    _fields_ = [
        ("return_5s", ctypes.c_double),
        ("flow_imbalance", ctypes.c_double),
        ("book_imbalance", ctypes.c_double),
        ("volatility_bps", ctypes.c_double),
        ("spread_bps", ctypes.c_double),
        ("candidate_direction", ctypes.c_int),
        ("candidate_count", ctypes.c_int),
    ]


class _StrategyOutput(ctypes.Structure):
    _fields_ = [
        ("score", ctypes.c_double),
        ("probability_up", ctypes.c_double),
        ("gross_edge_bps", ctypes.c_double),
        ("expected_edge_bps", ctypes.c_double),
        ("round_trip_cost_bps", ctypes.c_double),
        ("raw_direction", ctypes.c_int),
        ("direction", ctypes.c_int),
        ("confirmed_direction", ctypes.c_int),
        ("next_candidate_direction", ctypes.c_int),
        ("next_candidate_count", ctypes.c_int),
    ]


class StrategyCore:
    """Cost-aware decision API with an optional C++ implementation."""

    def __init__(self, library_path: Path = DEFAULT_LIBRARY) -> None:
        self.backend = "python-reference"
        self._evaluate = None
        if library_path.exists():
            try:
                library = ctypes.CDLL(str(library_path))
                evaluate = library.btc_strategy_evaluate
                evaluate.argtypes = [ctypes.POINTER(_StrategyInput), ctypes.POINTER(_StrategyOutput)]
                evaluate.restype = None
                self._evaluate = evaluate
                self.backend = "cpp"
            except OSError as error:
                self.load_error = f"C++ DLL unavailable: {error}"

    def evaluate(
        self,
        features: MarketFeatures,
        candidate_direction: int,
        candidate_count: int,
    ) -> StrategyDecision:
        if self._evaluate is None:
            return self._reference(features, candidate_direction, candidate_count)
        request = _StrategyInput(
            features.return_5s,
            features.flow_imbalance,
            features.book_imbalance,
            features.volatility_bps,
            features.spread_bps,
            candidate_direction,
            candidate_count,
        )
        response = _StrategyOutput()
        self._evaluate(ctypes.byref(request), ctypes.byref(response))
        return StrategyDecision(
            response.score,
            response.probability_up,
            response.gross_edge_bps,
            response.expected_edge_bps,
            response.round_trip_cost_bps,
            response.raw_direction,
            response.direction,
            response.confirmed_direction,
            response.next_candidate_direction,
            response.next_candidate_count,
        )

    @staticmethod
    def _reference(
        features: MarketFeatures,
        candidate_direction: int,
        candidate_count: int,
    ) -> StrategyDecision:
        volatility_bps = max(features.volatility_bps, 5.0)
        score = (
            2.5 * (features.return_5s / (volatility_bps / 10_000.0))
            + 2.0 * features.flow_imbalance
            + 1.5 * features.book_imbalance
        )
        probability = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, score))))
        raw_direction = 1 if probability > 0.72 else -1 if probability < 0.28 else 0
        round_trip_cost = 10.0 + max(features.spread_bps, 0.0)
        gross_edge = abs(2.0 * probability - 1.0) * max(2.0 * volatility_bps, 10.0)
        expected_edge = gross_edge - round_trip_cost
        direction = raw_direction if expected_edge >= 2.0 else 0
        if direction and direction == candidate_direction:
            next_direction, next_count = direction, candidate_count + 1
        elif direction:
            next_direction, next_count = direction, 1
        else:
            next_direction, next_count = 0, 0
        return StrategyDecision(
            score,
            probability,
            gross_edge,
            expected_edge,
            round_trip_cost,
            raw_direction,
            direction,
            direction if next_count >= 3 else 0,
            next_direction,
            next_count,
        )
