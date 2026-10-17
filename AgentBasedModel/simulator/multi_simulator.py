from AgentBasedModel.utils.liquidity import quote_state
from dataclasses import dataclass, field
import random
import statistics


@dataclass
class MultiSimulatorInfo:
    prices_a: list = field(default_factory=list)
    prices_b: list = field(default_factory=list)
    fx_rates: list = field(default_factory=list)
    price_gap: list = field(default_factory=list)
    traded_volume_a: list = field(default_factory=list)
    traded_volume_b: list = field(default_factory=list)
    fx_volume: list = field(default_factory=list)
    empty_books: list = field(default_factory=list)

    liquid_a: list = field(default_factory=list)
    liquid_b: list = field(default_factory=list)
    bid_empty_a: list = field(default_factory=list)
    ask_empty_a: list = field(default_factory=list)
    bid_empty_b: list = field(default_factory=list)
    ask_empty_b: list = field(default_factory=list)
    liquid_fx: list = field(default_factory=list)
    fallback_a: list = field(default_factory=list)
    fallback_b: list = field(default_factory=list)
    fallback_fx: list = field(default_factory=list)

    def record(self, simulator):
        for name, market in [('a', simulator.markets['A']), ('b', simulator.markets['B']),
                             ('fx', simulator.fx_market)]:
            state = quote_state(market) if market is not None else dict(two_sided=True)
            getattr(self, 'liquid_'+name).append(state['two_sided'])
            getattr(self, 'fallback_'+name).append(market is not None and not state['two_sided']
                                                  and market.shocked_price is None)
            if name != 'fx':
                getattr(self, 'bid_empty_'+name).append(state['bid_empty'])
                getattr(self, 'ask_empty_'+name).append(state['ask_empty'])

        p_a = simulator.markets["A"].price()
        p_b = simulator.markets["B"].price()
        fx = simulator.fx_market.price() if simulator.fx_market is not None else 1.0

        self.prices_a.append(p_a)
        self.prices_b.append(p_b)
        self.fx_rates.append(fx)

        p_b_in_a = p_b / max(fx, 1e-9)
        self.price_gap.append(abs(p_a - p_b_in_a))

        self.traded_volume_a.append(simulator.step_volume["A"])
        self.traded_volume_b.append(simulator.step_volume["B"])
        self.fx_volume.append(simulator.step_volume["FX"])
        # Observation only: retain the existing fallback-price behavior but flag it.
        self.empty_books.append(any(not book for market in
            [*simulator.markets.values(), simulator.fx_market] if market is not None
            for book in market.order_book.values()))


class MultiMarketSimulator:
    def __init__(
        self,
        markets: dict,
        fx_market,
        traders: list,
        events: list | None = None,
        tau: float = 0.0,
        market_noise_std: float = 0.03,
        fx_noise_std: float = 0.005,
        price_memory: float = 0.85,
        mean_reversion: float = 0.03,
        beta_link: float = 0.0,
        link_strength: float = 0.05,
    ):
        self.markets = markets
        self.fx_market = fx_market
        self.traders = traders
        self.events = events or []
        self.tau = tau
        self.info = MultiSimulatorInfo()
        self.current_iteration = 0
        self.step_volume = {"A": 0, "B": 0, "FX": 0}

        self.market_noise_std = market_noise_std
        self.fx_noise_std = fx_noise_std
        self.price_memory = price_memory
        self.mean_reversion = mean_reversion


        self.beta_link = beta_link
        self.link_strength = link_strength

        self.reference_price_a = markets["A"].price()
        self.reference_price_b = markets["B"].price()
        self.reference_fx = fx_market.price() if fx_market is not None else 1.0

    def reset_step_volume(self):
        self.step_volume = {"A": 0, "B": 0, "FX": 0}

    def apply_events(self):
        for event in self.events:
            if event.iteration == self.current_iteration:
                event.apply(self)

    def activate_traders(self):
        shuffled = self.traders[:]
        random.shuffle(shuffled)

        for trader in shuffled:
            before_assets = trader.assets
            before_cash_a = trader.cash["CUR_A"]
            before_cash_b = trader.cash["CUR_B"]

            trader.call(tau=self.tau)

            if trader.cash["CUR_A"] != before_cash_a or trader.assets != before_assets:
                self.step_volume["A"] += 1
            if trader.cash["CUR_B"] != before_cash_b or trader.assets != before_assets:
                self.step_volume["B"] += 1

            if trader.type == "FXTrader":
                self.step_volume["FX"] += 1

    def _smooth_price(self, old_price: float, target_price: float, reference_price: float, noise_std: float):
        noise = random.normalvariate(0, noise_std)

        new_price = (
            self.price_memory * old_price
            + (1 - self.price_memory) * target_price
            + self.mean_reversion * (reference_price - old_price)
            + noise
        )
        return max(0.01, round(new_price, 2))

    def apply_cross_market_link(self):

        if self.beta_link <= 0 or self.fx_market is None:
            return

        market_a = self.markets["A"]
        market_b = self.markets["B"]
        fx = self.fx_market.price()

        p_a = market_a.price()
        p_b_in_a = market_b.price() / max(fx, 1e-9)

        diff = p_a - p_b_in_a


        adjustment = self.link_strength * self.beta_link * diff

        new_a = max(0.01, round(market_a.last_trade_price - adjustment, 2))
        new_b_in_a = max(0.01, round(p_b_in_a + adjustment, 2))


        new_b = max(0.01, round(new_b_in_a * fx, 2))

        market_a.last_trade_price = new_a
        market_b.last_trade_price = new_b

        if market_a.shocked_price is not None:
            market_a.shocked_price = new_a
        if market_b.shocked_price is not None:
            market_b.shocked_price = new_b

    def apply_micro_dynamics(self):
        market_a = self.markets["A"]
        market_b = self.markets["B"]

        old_a = market_a.last_trade_price
        old_b = market_b.last_trade_price

        target_a = market_a.price()
        target_b = market_b.price()

        new_a = self._smooth_price(
            old_price=old_a,
            target_price=target_a,
            reference_price=self.reference_price_a,
            noise_std=self.market_noise_std,
        )
        new_b = self._smooth_price(
            old_price=old_b,
            target_price=target_b,
            reference_price=self.reference_price_b,
            noise_std=self.market_noise_std,
        )

        market_a.last_trade_price = new_a
        market_b.last_trade_price = new_b

        if market_a.shocked_price is not None:
            market_a.shocked_price = new_a
        if market_b.shocked_price is not None:
            market_b.shocked_price = new_b

        if self.fx_market is not None:
            old_fx = self.fx_market.last_trade_price
            target_fx = self.fx_market.price()

            new_fx = self._smooth_price(
                old_price=old_fx,
                target_price=target_fx,
                reference_price=self.reference_fx,
                noise_std=self.fx_noise_std,
            )
            self.fx_market.last_trade_price = round(new_fx, 4)

            if self.fx_market.shocked_price is not None:
                self.fx_market.shocked_price = round(new_fx, 4)

    def simulate(self, n_steps: int, silent: bool = False):
        for t in range(n_steps):
            self.current_iteration = t
            self.reset_step_volume()
            self.apply_events()
            self.activate_traders()
            self.apply_micro_dynamics()
            self.apply_cross_market_link()   # НОВОЕ
            self.info.record(self)

    def recovery_time(self, shock_iteration: int, tolerance: float = 1.0):
        for t in range(shock_iteration, len(self.info.price_gap)):
            if self.info.price_gap[t] <= tolerance:
                return t - shock_iteration
        return None

    def volatility_before_after(self, shock_iteration: int, window: int = 50):
        left = self.info.prices_a[max(1, shock_iteration - window):shock_iteration]
        right = self.info.prices_a[shock_iteration:shock_iteration + window]

        def returns(series):
            return [
                (series[i] - series[i - 1]) / max(series[i - 1], 1e-9)
                for i in range(1, len(series))
            ]

        left_r = returns(left) if len(left) > 2 else [0.0]
        right_r = returns(right) if len(right) > 2 else [0.0]

        return {
            "before": statistics.pstdev(left_r) if len(left_r) > 1 else 0.0,
            "after": statistics.pstdev(right_r) if len(right_r) > 1 else 0.0,
        }