import unittest
from unittest.mock import patch
from types import SimpleNamespace

from AgentBasedModel.agents.agents import ExchangeAgent, MarketMaker
from AgentBasedModel.utils.orders import Order, OrderList, EmptyOrderBookError


def single_market(fee=0.):
    exchange=ExchangeAgent(volume=0,transaction_cost=fee)
    exchange.order_book['bid'].append(Order(99.,500,'bid'))
    exchange.order_book['ask'].append(Order(101.,500,'ask'))
    return exchange


class MarketMakerFixTests(unittest.TestCase):
    def test_zero_bid_volume_sells_towards_target(self):
        market=single_market(.01);mm=MarketMaker(market,20000.,assets=99)
        mm.call()
        self.assertTrue(mm.panic)
        self.assertEqual(mm.assets,0)
        self.assertAlmostEqual(mm.cash,20000+99*99*.99)
        mm.call()
        self.assertFalse(mm.panic)
        self.assertEqual(len(mm.orders),2)

    def test_zero_ask_volume_buys_towards_target(self):
        market=single_market(.01);mm=MarketMaker(market,20000.,assets=-99)
        mm.call()
        self.assertTrue(mm.panic)
        self.assertEqual(mm.assets,0)
        self.assertAlmostEqual(mm.cash,20000-99*101*1.01)

    def test_partial_panic_fill_uses_actual_inventory(self):
        market=single_market();market.order_book['ask'].first.qty=2
        mm=MarketMaker(market,20000.,assets=-99)
        mm.call()
        self.assertEqual(mm.assets,-97)
        self.assertAlmostEqual(mm.cash,20000-202)
        self.assertEqual(market.price(),100.)

    def test_one_sided_market_uses_only_previous_midpoint(self):
        market=single_market();self.assertEqual(market.price(),100.)
        market.order_book['ask']=OrderList('ask')
        self.assertEqual(market.price(),100.)
        market.limit_order(Order(100,1,'ask'))
        MarketMaker(market,1000.).call()
        self.assertFalse(market.order_book['ask'])



from AgentBasedModel.agents.multi_market import AssetMarket, FXMarket
from AgentBasedModel.agents.multi_traders import RandomMultiTrader, FundamentalistMultiTrader


def asset_market(fee=.01):
    market=AssetMarket('A','ASSET','CUR_A',volume=0,transaction_cost=fee)
    return market


class ExecutionFixTests(unittest.TestCase):
    def test_partial_buy_conserves_assets_and_cash_less_fees(self):
        market=asset_market();seller=SimpleNamespace(cash=1000.,assets=5,orders=[])
        market.order_book['bid'].append(Order(90,100,'bid'))
        market.order_book['ask'].append(Order(100,1,'ask',seller))
        market.order_book['ask'].append(Order(110,10,'ask'))
        trader=RandomMultiTrader('buyer',{'A':market},cash_a=1000,assets=5)
        with patch('random.random',return_value=1.),patch('random.randint',return_value=3):trader.call()
        self.assertEqual(trader.assets,6)
        self.assertEqual(seller.assets,4)
        self.assertAlmostEqual(trader.cash['CUR_A'],899.)
        self.assertAlmostEqual(seller.cash,1099.)
        self.assertAlmostEqual(trader.cash['CUR_A']+seller.cash,2000-2.)
        self.assertEqual(market.order_book['ask'].first.price,110)

    def test_partial_sell_conserves_assets_and_cash_less_fees(self):
        market=asset_market();buyer=SimpleNamespace(cash=1000.,assets=5,orders=[])
        market.order_book['bid'].append(Order(100,1,'bid',buyer))
        market.order_book['bid'].append(Order(90,10,'bid'))
        market.order_book['ask'].append(Order(110,100,'ask'))
        trader=RandomMultiTrader('seller',{'A':market},cash_a=1000,assets=5)
        with patch('random.random',return_value=0.),patch('random.randint',return_value=3):trader.call()
        self.assertEqual(trader.assets,4);self.assertEqual(buyer.assets,6)
        self.assertAlmostEqual(trader.cash['CUR_A'],1099.)
        self.assertAlmostEqual(buyer.cash,899.)
        self.assertAlmostEqual(trader.cash['CUR_A']+buyer.cash,1998.)

    def test_fundamentalist_buy_and_sell_fees(self):
        for fundamental,expected_assets,expected_cash in [(120,6,899.),(80,4,1099.)]:
            with self.subTest(fundamental=fundamental):
                market=asset_market();market.order_book['bid'].append(Order(100,10,'bid'))
                market.order_book['ask'].append(Order(100,10,'ask'))
                trader=FundamentalistMultiTrader('fund',{'A':market},cash_a=1000,assets=5,fundamental_price=fundamental)
                trader.call()
                self.assertEqual(trader.assets,expected_assets)
                self.assertAlmostEqual(trader.cash['CUR_A'],expected_cash)

    def test_unfilled_order_does_not_change_balances(self):
        market=asset_market();market.order_book['bid'].append(Order(90,10,'bid'))
        market.order_book['ask'].append(Order(110,10,'ask'))
        trader=RandomMultiTrader('buyer',{'A':market},cash_a=1000,assets=5)
        result=trader.execute_market_order(market,100,3,'bid')
        self.assertEqual(result.qty,3);self.assertEqual(trader.assets,5)
        self.assertEqual(trader.cash['CUR_A'],1000)

    def test_affordability_includes_market_fee(self):
        market=asset_market();market.order_book['bid'].append(Order(90,10,'bid'))
        market.order_book['ask'].append(Order(100,10,'ask'))
        trader=RandomMultiTrader('buyer',{'A':market},cash_a=100,assets=5)
        with patch('random.random',return_value=1.),patch('random.randint',return_value=1):trader.call()
        self.assertEqual(trader.assets,5);self.assertEqual(trader.cash['CUR_A'],100)


from AgentBasedModel.agents.multi_traders import MultiTrader
from AgentBasedModel.simulator.multi_simulator import MultiMarketSimulator


def fx_market(rate=2., fee=.1):
    fx=FXMarket(price=rate,volume=0,transaction_cost=fee)
    fx.order_book['bid'].append(Order(rate,1000,'bid'))
    fx.order_book['ask'].append(Order(rate,1000,'ask'))
    return fx


def quoted_market(name,currency,bid,ask,fee=.01):
    m=AssetMarket(name,'ASSET',currency,price=(bid+ask)/2,volume=0,transaction_cost=fee)
    m.order_book['bid'].append(Order(bid,1000,'bid'))
    m.order_book['ask'].append(Order(ask,1000,'ask'))
    return m


class FXFixTests(unittest.TestCase):
    def test_quote_direction_and_roundtrip_fees(self):
        fx=fx_market()
        self.assertAlmostEqual(fx.convert(100,'CUR_A','CUR_B'),180)
        self.assertAlmostEqual(fx.convert(180,'CUR_B','CUR_A'),81)
        self.assertAlmostEqual(fx.convert(200,'CUR_B','CUR_A'),90)

    def test_funding_both_directions_includes_fx_fee(self):
        fx=fx_market()
        a=MultiTrader('A',{},fx,cash_a=100,cash_b=0)
        self.assertTrue(a.ensure_currency('CUR_B',90))
        self.assertAlmostEqual(a.cash['CUR_A'],50);self.assertAlmostEqual(a.cash['CUR_B'],90)
        b=MultiTrader('B',{},fx,cash_a=0,cash_b=200)
        self.assertTrue(b.ensure_currency('CUR_A',45))
        self.assertAlmostEqual(b.cash['CUR_B'],100);self.assertAlmostEqual(b.cash['CUR_A'],45)
        poor=MultiTrader('poor',{},fx,cash_a=10,cash_b=0)
        self.assertFalse(poor.ensure_currency('CUR_B',100))
        self.assertAlmostEqual(poor.cash['CUR_A'],0);self.assertAlmostEqual(poor.cash['CUR_B'],18)

    def test_effective_prices_have_consistent_units_and_fees(self):
        fx=fx_market()
        a=quoted_market('A','CUR_A',90,100)
        b=quoted_market('B','CUR_B',180,200)
        trader=MultiTrader('homeA',{'A':a,'B':b},fx,home_currency='CUR_A')
        self.assertAlmostEqual(trader.effective_buy_price('B',0),202/(2*.9))
        self.assertAlmostEqual(trader.effective_sell_price('B',0),178.2/2*.9)
        trader.home_currency='CUR_B'
        self.assertAlmostEqual(trader.effective_buy_price('A',0),101*2/.9)
        self.assertAlmostEqual(trader.effective_sell_price('A',0),89.1*2*.9)

    def test_gap_and_link_use_same_currency_conversion(self):
        fx=fx_market(fee=0)
        a=quoted_market('A','CUR_A',100,100)
        b=quoted_market('B','CUR_B',200,200)
        sim=MultiMarketSimulator({'A':a,'B':b},fx,[],beta_link=.4)
        sim.info.record(sim)
        self.assertEqual(sim.info.price_gap[-1],0)
        a.shocked_price=a.last_trade_price=120
        b.shocked_price=b.last_trade_price=200
        sim.apply_cross_market_link()
        self.assertAlmostEqual(a.price(),119.6)
        self.assertAlmostEqual(b.price(),200.8)
        self.assertAlmostEqual(a.price()+b.price()/2,220.)
        sim.info.record(sim)
        self.assertAlmostEqual(sim.info.price_gap[-1],19.2)

    def test_foreign_purchase_settles_fx_and_market_fees_once(self):
        market=quoted_market('B','CUR_B',90,100)
        seller=SimpleNamespace(cash=1000.,assets=5,orders=[])
        market.order_book['ask'].first.trader=seller
        fx=fx_market()
        trader=RandomMultiTrader('buyer',{'B':market},fx,cash_a=1000,cash_b=0,
                                 assets=5,access_mode='B_ONLY',home_currency='CUR_A')
        with patch('random.random',return_value=1.),patch('random.randint',return_value=1):trader.call()
        funding_a=101/(2*.9)
        self.assertAlmostEqual(trader.cash['CUR_A'],1000-funding_a)
        self.assertAlmostEqual(trader.cash['CUR_B'],0)
        self.assertEqual(trader.assets,6);self.assertEqual(seller.assets,4)
        self.assertAlmostEqual(seller.cash,1099)
        total_after=trader.cash['CUR_A']+trader.cash['CUR_B']/2+seller.cash/2
        self.assertAlmostEqual(1500-total_after,2/2+funding_a*.1)

    def test_empty_multi_book_is_flagged_without_inventing_liquidity(self):
        a=quoted_market('A','CUR_A',100,100);b=quoted_market('B','CUR_B',200,200)
        sim=MultiMarketSimulator({'A':a,'B':b},fx_market(),[])
        a.order_book['ask']=OrderList('ask')
        sim.info.record(sim)
        self.assertTrue(sim.info.empty_books[-1])
        self.assertFalse(a.order_book['ask'])


if __name__=='__main__':unittest.main()
