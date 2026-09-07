#pragma once

namespace btc_core {

struct StrategyConfig {
    double entry_probability = 0.72;
    double min_edge_bps = 2.0;
    double fee_bps = 4.0;
    double slippage_bps = 1.0;
    double min_move_bps = 10.0;
    int confirmations = 3;
};

struct MarketFeatures {
    double return_5s = 0.0;
    double flow_imbalance = 0.0;
    double book_imbalance = 0.0;
    double volatility_bps = 5.0;
    double spread_bps = 0.0;
};

struct ConfirmationState {
    int direction = 0;
    int count = 0;
};

struct Decision {
    double score = 0.0;
    double probability_up = 0.5;
    double gross_edge_bps = 0.0;
    double expected_edge_bps = 0.0;
    double round_trip_cost_bps = 0.0;
    int raw_direction = 0;
    int direction = 0;
    int confirmed_direction = 0;
    ConfirmationState next_confirmation{};
};

struct PositionSize {
    double notional = 0.0;
    double quantity = 0.0;
    double risk_amount = 0.0;
    double stop_distance_pct = 0.0;
};

Decision evaluate(const MarketFeatures& features, const StrategyConfig& config,
                  const ConfirmationState& confirmation);
PositionSize size_position(double equity, double entry_price, double stop_distance_pct,
                           double risk_per_trade, double leverage,
                           double fee_bps, double slippage_bps, double spread_bps,
                           double minimum_stop_pct);

}  // namespace btc_core

extern "C" {

struct BtcStrategyInput {
    double return_5s;
    double flow_imbalance;
    double book_imbalance;
    double volatility_bps;
    double spread_bps;
    int candidate_direction;
    int candidate_count;
};

struct BtcStrategyOutput {
    double score;
    double probability_up;
    double gross_edge_bps;
    double expected_edge_bps;
    double round_trip_cost_bps;
    int raw_direction;
    int direction;
    int confirmed_direction;
    int next_candidate_direction;
    int next_candidate_count;
};

struct BtcPositionSizeInput {
    double equity;
    double entry_price;
    double stop_distance_pct;
    double risk_per_trade;
    double leverage;
    double fee_bps;
    double slippage_bps;
    double spread_bps;
    double minimum_stop_pct;
};

struct BtcPositionSizeOutput {
    double notional;
    double quantity;
    double risk_amount;
    double stop_distance_pct;
};

void btc_strategy_evaluate(const BtcStrategyInput* input, BtcStrategyOutput* output);
void btc_position_size(const BtcPositionSizeInput* input, BtcPositionSizeOutput* output);

}
