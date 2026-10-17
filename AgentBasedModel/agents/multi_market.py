import random

from AgentBasedModel.utils import Order, OrderList


class AssetMarket:
    """
    Новый рынок актива для multi-market режима.
    Старый ExchangeAgent НЕ трогаем.
    """

    def __init__(
        self,
        name: str,
        asset_name: str,
        base_currency: str,
        price: float = 100.0,
        std: float = 25.0,
        volume: int = 1000,
        rf: float = 5e-4,
        transaction_cost: float = 0.0,
        position: tuple[int, int] = (0, 0),
    ):
        self.name = name
        self.asset_name = asset_name
        self.base_currency = base_currency
        self.position = position
        self.risk_free = rf
        self.transaction_cost = transaction_cost

        self.order_book = {
            "bid": OrderList("bid"),
            "ask": OrderList("ask"),
        }

        self.dividend_book = []
        self.last_trade_price = price

        # Новое поле: если есть активный шок, цена берётся отсюда
        self.shocked_price = None

        self._fill_book(price=price, std=std, volume=volume, div=rf * price)

    @staticmethod
    def _next_dividend(std: float = 5e-3) -> float:
        import math
        return math.exp(random.normalvariate(0, std))

    def _fill_book(self, price: float, std: float, volume: int, div: float = 0.05):
        prices1 = [round(random.normalvariate(price - std, std), 1) for _ in range(volume // 2)]
        prices2 = [round(random.normalvariate(price + std, std), 1) for _ in range(volume // 2)]
        quantities = [random.randint(1, 5) for _ in range(volume)]

        for p, q in zip(sorted(prices1 + prices2), quantities):
            if p > price:
                order = Order(round(p, 1), q, "ask", None)
                self.order_book["ask"].append(order)
            else:
                order = Order(round(p, 1), q, "bid", None)
                self.order_book["bid"].push(order)

        for _ in range(100):
            self.dividend_book.append(max(div, 0))
            div *= self._next_dividend()

    def spread(self):
        if self.order_book["bid"] and self.order_book["ask"]:
            return {
                "bid": self.order_book["bid"].first.price,
                "ask": self.order_book["ask"].first.price,
            }
        return None

    def spread_volume(self):
        if self.order_book["bid"] and self.order_book["ask"]:
            return {
                "bid": self.order_book["bid"].first.qty,
                "ask": self.order_book["ask"].first.qty,
            }
        return None

    def price(self) -> float:
        # Если шок активен, возвращаем шоковую цену
        if self.shocked_price is not None:
            return self.shocked_price

        spread = self.spread()
        if spread is not None:
            self.last_trade_price = round((spread["bid"] + spread["ask"]) / 2, 1)

        return self.last_trade_price

    def dividend(self, access: int = None):
        if access is None:
            return self.dividend_book[0]
        return self.dividend_book[:access]

    def market_order(self, order: Order):
        t_cost = self.transaction_cost
        if order.order_type == "bid":
            result = self.order_book["ask"].fulfill(order, t_cost)
        elif order.order_type == "ask":
            result = self.order_book["bid"].fulfill(order, t_cost)
        else:
            result = order

        # После реальной сделки убираем "жёсткую" шоковую фиксацию,
        # чтобы рынок снова мог жить своей ценой
        self.shocked_price = None
        spread = self.spread()
        if spread is not None:
            self.last_trade_price = round((spread["bid"] + spread["ask"]) / 2, 1)

        return result

    def limit_order(self, order: Order):
        spread = self.spread()
        if spread is None:
            return

        bid = spread["bid"]
        ask = spread["ask"]
        t_cost = self.transaction_cost

        if order.order_type == "bid":
            if order.price >= ask:
                order = self.order_book["ask"].fulfill(order, t_cost)
                if order.qty > 0:
                    self.order_book["bid"].insert(order)
            else:
                self.order_book["bid"].insert(order)

        elif order.order_type == "ask":
            if order.price <= bid:
                order = self.order_book["bid"].fulfill(order, t_cost)
                if order.qty > 0:
                    self.order_book["ask"].insert(order)
            else:
                self.order_book["ask"].insert(order)

        # Лимитная заявка тоже возвращает рынок под контроль стакана
        self.shocked_price = None
        spread = self.spread()
        if spread is not None:
            self.last_trade_price = round((spread["bid"] + spread["ask"]) / 2, 1)

    def cancel_order(self, order: Order):
        if order.order_type == "bid":
            self.order_book["bid"].remove(order)
        elif order.order_type == "ask":
            self.order_book["ask"].remove(order)

    def apply_price_shock(self, delta_price: float):
        """
        Быстрый патч:
        фиксируем новую цену явно, чтобы шок был виден на графике.
        """
        shocked = max(0.01, round(self.price() + delta_price, 1))
        self.shocked_price = shocked
        self.last_trade_price = shocked

    def __repr__(self):
        return (
            f"AssetMarket(name={self.name}, asset={self.asset_name}, "
            f"currency={self.base_currency}, price={self.price()}, pos={self.position})"
        )


class FXMarket(AssetMarket):
    """
    Отдельный маленький рынок обмена валют.
    """

    def __init__(
        self,
        name: str = "FX",
        base_currency: str = "CUR_A",
        quote_currency: str = "CUR_B",
        price: float = 1.0,
        std: float = 0.05,
        volume: int = 500,
        transaction_cost: float = 0.0,
        position: tuple[int, int] = (5, 5),
    ):
        self.quote_currency = quote_currency
        super().__init__(
            name=name,
            asset_name=f"{base_currency}/{quote_currency}",
            base_currency=base_currency,
            price=price,
            std=std,
            volume=volume,
            rf=0.0,
            transaction_cost=transaction_cost,
            position=position,
        )

    def convert(self, amount: float, from_currency: str, to_currency: str) -> float:
        """Rate is quote currency per base currency (CUR_B per CUR_A)."""
        rate = self.price()
        fee = 1.0 - self.transaction_cost

        if from_currency == self.base_currency and to_currency == self.quote_currency:
            return amount * rate * fee

        if from_currency == self.quote_currency and to_currency == self.base_currency:
            return amount / max(rate, 1e-9) * fee

        raise ValueError(f"Unsupported conversion: {from_currency} -> {to_currency}")

    def __repr__(self):
        return (
            f"FXMarket(name={self.name}, pair={self.asset_name}, "
            f"rate={self.price()}, pos={self.position})"
        )