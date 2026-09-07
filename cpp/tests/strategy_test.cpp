#include "btc_core/strategy.hpp"

#include <cassert>
#include <cmath>

int main() {
    const btc_core::StrategyConfig config{};
    const auto blocked = btc_core::evaluate({}, config, {});
    assert(blocked.direction == 0);
    assert(blocked.expected_edge_bps < 0.0);

    const btc_core::MarketFeatures strong{0.002, 0.8, 0.8, 20.0, 1.0};
    const auto first = btc_core::evaluate(strong, config, {});
    assert(first.direction == 1);
    assert(first.confirmed_direction == 0);
    const auto second = btc_core::evaluate(strong, config, first.next_confirmation);
    const auto third = btc_core::evaluate(strong, config, second.next_confirmation);
    assert(third.confirmed_direction == 1);

    const auto size = btc_core::size_position(100.0, 100.0, 0.02, 0.01, 5.0,
                                               0.0, 0.0, 0.0, 0.002);
    assert(std::abs(size.notional - 50.0) < 1e-9);
    assert(std::abs(size.quantity - 0.5) < 1e-9);
}
