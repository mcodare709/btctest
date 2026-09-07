#include "btc_core/strategy.hpp"

#include <algorithm>
#include <cmath>

namespace {

constexpr double kMinimumVolatilityBps = 5.0;

double clamp(double value, double lower, double upper) {
    return std::max(lower, std::min(value, upper));
}

double sigmoid(double score) {
    return 1.0 / (1.0 + std::exp(-clamp(score, -20.0, 20.0)));
}

bool is_finite(double value) {
    return std::isfinite(value);
}

}  // namespace

namespace btc_core {

Decision evaluate(const MarketFeatures& features, const StrategyConfig& config,
                  const ConfirmationState& confirmation) {
    const double volatility_bps = std::max(features.volatility_bps, kMinimumVolatilityBps);
    const double volatility = volatility_bps / 10'000.0;
    const double score = 2.5 * (features.return_5s / volatility)
        + 2.0 * features.flow_imbalance + 1.5 * features.book_imbalance;
    const double probability = sigmoid(score);
    const int raw_direction = probability > config.entry_probability ? 1
        : probability < (1.0 - config.entry_probability) ? -1 : 0;
    const double round_trip_cost = 2.0 * (config.fee_bps + config.slippage_bps)
        + std::max(features.spread_bps, 0.0);
    const double confidence = std::abs(2.0 * probability - 1.0);
    const double gross_edge = confidence * std::max(2.0 * volatility_bps, config.min_move_bps);
    const double expected_edge = gross_edge - round_trip_cost;
    const int direction = expected_edge >= config.min_edge_bps ? raw_direction : 0;

    ConfirmationState next{};
    if (direction != 0 && direction == confirmation.direction) {
        next = {direction, confirmation.count + 1};
    } else if (direction != 0) {
        next = {direction, 1};
    }

    return {
        score, probability, gross_edge, expected_edge, round_trip_cost,
        raw_direction, direction,
        next.count >= config.confirmations ? direction : 0, next,
    };
}

PositionSize size_position(double equity, double entry_price, double stop_distance_pct,
                           double risk_per_trade, double leverage,
                           double fee_bps, double slippage_bps, double spread_bps,
                           double minimum_stop_pct) {
    if (!is_finite(equity) || !is_finite(entry_price) || equity <= 0.0 || entry_price <= 0.0) {
        return {};
    }
    const double stop = std::max(stop_distance_pct, minimum_stop_pct);
    const double cost_rate = 2.0 * (fee_bps + slippage_bps) / 10'000.0
        + std::max(spread_bps, 0.0) / 10'000.0;
    const double risk_rate = stop + cost_rate;
    if (risk_rate <= 0.0) {
        return {};
    }
    const double risk_amount = equity * risk_per_trade;
    const double notional = std::min(risk_amount / risk_rate, equity * leverage);
    return {notional, notional / entry_price, notional * risk_rate, stop};
}

}  // namespace btc_core

extern "C" void btc_strategy_evaluate(const BtcStrategyInput* input, BtcStrategyOutput* output) {
    if (input == nullptr || output == nullptr) {
        return;
    }
    const btc_core::MarketFeatures features{
        input->return_5s, input->flow_imbalance, input->book_imbalance,
        input->volatility_bps, input->spread_bps,
    };
    const btc_core::ConfirmationState confirmation{
        input->candidate_direction, input->candidate_count,
    };
    const btc_core::Decision result = btc_core::evaluate(features, {}, confirmation);
    *output = {
        result.score, result.probability_up, result.gross_edge_bps,
        result.expected_edge_bps, result.round_trip_cost_bps, result.raw_direction,
        result.direction, result.confirmed_direction, result.next_confirmation.direction,
        result.next_confirmation.count,
    };
}

extern "C" void btc_position_size(const BtcPositionSizeInput* input, BtcPositionSizeOutput* output) {
    if (input == nullptr || output == nullptr) {
        return;
    }
    const btc_core::PositionSize result = btc_core::size_position(
        input->equity, input->entry_price, input->stop_distance_pct,
        input->risk_per_trade, input->leverage, input->fee_bps,
        input->slippage_bps, input->spread_bps, input->minimum_stop_pct);
    *output = {result.notional, result.quantity, result.risk_amount, result.stop_distance_pct};
}
