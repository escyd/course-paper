import random
import numpy as np
import matplotlib.pyplot as plt

from AgentBasedModel.agents.multi_market import AssetMarket, FXMarket
from AgentBasedModel.agents.factory import build_multi_traders
from AgentBasedModel.events.multi_events import MultiMarketPriceShock
from AgentBasedModel.simulator.multi_simulator import MultiMarketSimulator


random.seed(42)
np.random.seed(42)


def build_system(tau=0.0, beta=0.0):
    market_a = AssetMarket(
        name="A",
        asset_name="ASSET_X",
        base_currency="CUR_A",
        price=100.0,
        std=10.0,
        volume=1000,
        transaction_cost=0.001,
        position=(2, 10),
    )

    market_b = AssetMarket(
        name="B",
        asset_name="ASSET_X",
        base_currency="CUR_B",
        price=102.0,
        std=10.0,
        volume=1000,
        transaction_cost=0.001,
        position=(18, 10),
    )

    fx_market = FXMarket(
        name="FX",
        base_currency="CUR_A",
        quote_currency="CUR_B",
        price=1.0,
        std=0.03,
        volume=500,
        transaction_cost=0.001,
        position=(10, 10),
    )

    markets = {
        "A": market_a,
        "B": market_b,
    }

    traders = build_multi_traders(
        markets=markets,
        fx_market=fx_market,
        grid_size=(20, 20),
        n_a_only_random=15,
        n_b_only_random=15,
        n_cross_random=10,
        n_a_only_fundamental=8,
        n_b_only_fundamental=8,
        n_cross_fundamental=10,
        n_fx_traders=8,
    )

    events = [
        MultiMarketPriceShock(
            market_name="A",
            iteration=200,
            delta_price=-10.0,
            spillover_beta=beta,
        )
    ]

    simulator = MultiMarketSimulator(
        markets=markets,
        fx_market=fx_market,
        traders=traders,
        events=events,
        tau=tau,
        beta_link=beta,
        link_strength=0.05,
    )
    return simulator


def run_and_plot(tau=0.0, beta=0.0, steps=500):
    simulator = build_system(tau=tau, beta=beta)
    simulator.simulate(steps)

    plt.figure(figsize=(10, 5))
    plt.plot(simulator.info.prices_a, label="Market A")
    plt.plot(simulator.info.prices_b, label="Market B")
    plt.axvline(200, linestyle="--")
    plt.title(f"Prices, tau={tau}, beta={beta}")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(simulator.info.price_gap, label="|P_A - P_B/FX|")
    plt.axvline(200, linestyle="--")
    plt.title(f"Price gap, tau={tau}, beta={beta}")
    plt.legend()
    plt.show()

    print("Recovery time:", simulator.recovery_time(shock_iteration=200))
    print("Volatility A:", simulator.volatility_before_after(shock_iteration=200, window=50))


if __name__ == "__main__":
    run_and_plot(tau=0.0, beta=0.0)
    run_and_plot(tau=0.5, beta=0.0)
    run_and_plot(tau=0.5, beta=0.4)