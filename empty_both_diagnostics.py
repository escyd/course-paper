"""Short observation-only accounting and reproducibility checks for empty books."""
from pathlib import Path
import json,random,time,math
from unittest.mock import patch
import numpy as np
import pandas as pd
from stylized_facts import build_baseline
from AgentBasedModel.utils.orders import OrderList
from AgentBasedModel.utils.liquidity import quote_history_summary
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'tables/empty_both_validation/20261017T163358Z'
PREVIOUS=ROOT/'tables/liquidity_validation/20261017T154909Z/after_final'

def run(seed,observed):
    sim=build_baseline(seed);counts=dict(fills=0,executed_qty=0,empty_both_agent_activations=0,agent_calls=0,payment_checks=0)
    old_fulfill=OrderList.fulfill
    def fulfill(book,order,fee):
        candidates=[];remaining=order.qty
        for node in book:
            if remaining<=0 or node>order:break
            candidates.append((node,node.qty));remaining-=min(node.qty,remaining)
        people={t for t in [order.trader,*[n.trader for n,q in candidates]] if t is not None}
        before={t:(t.cash,t.assets) for t in people};delta={t:[0.,0] for t in people};qty=order.qty
        result=old_fulfill(book,order,fee);actual=0
        for node,before_qty in candidates:
            q=before_qty-node.qty
            if q<=0:continue
            counts['fills']+=1;counts['executed_qty']+=q;actual+=q
            sign=1 if order.order_type=='bid' else -1
            for t,direction in [(order.trader,sign),(node.trader,-sign)]:
                if t is not None:
                    delta[t][0]+=-direction*node.price*q*(1+direction*fee)
                    delta[t][1]+=direction*q
        assert actual==qty-order.qty
        for t,(cash,assets) in before.items():
            assert math.isclose(t.cash,cash+delta[t][0],rel_tol=1e-12,abs_tol=1e-7)
            assert t.assets==assets+delta[t][1]
        return result
    if observed:
        for t in sim.traders:
            original=t.call
            def call(t=t,original=original):
                counts['agent_calls']+=1;before=(t.cash,t.assets);fills=counts['fills']
                both=not sim.exchange.order_book['bid'] and not sim.exchange.order_book['ask']
                if both:counts['empty_both_agent_activations']+=1
                result=original()
                if counts['fills']==fills:assert (t.cash,t.assets)==before
                return result
            t.call=call
        payments=sim._payments
        def checked_payments():
            div=sim.exchange.dividend();rf=sim.exchange.risk_free
            before={t:(t.cash,t.assets) for t in sim.traders};result=payments()
            for t,(cash,assets) in before.items():
                expected=cash+assets*div;expected+=expected*rf
                assert t.cash==expected and t.assets==assets
            counts['payment_checks']+=1
            return result
        sim._payments=checked_payments
        with patch.object(OrderList,'fulfill',fulfill):sim.simulate(10000,silent=True)
    else:sim.simulate(10000,silent=True)
    return sim,counts,random.getstate(),np.random.get_state()

def main():
    out=OUT/'pilot';out.mkdir(exist_ok=False);rows=[]
    for seed in [43,47,85]:
        start=time.perf_counter();a=run(seed,True);b=run(seed,False);sim=a[0]
        assert a[2]==b[2] and all(np.array_equal(x,y) for x,y in zip(a[3],b[3]))
        assert sim.info.prices==b[0].info.prices
        assert [(t.cash,t.assets,t.type) for t in sim.traders]==[(t.cash,t.assets,t.type) for t in b[0].traders]
        old=pd.read_csv(PREVIOUS/f'single_seed_{seed}/timeseries.csv',float_precision='round_trip')
        assert np.array_equal(sim.info.prices[:len(old)],old.price.to_numpy())
        pd.DataFrame(dict(price=sim.info.prices,liquidity_state=sim.info.liquidity_state)).to_csv(out/f'seed_{seed}.csv',index=False)
        rows.append(dict(seed=seed,completed=len(sim.info.prices),old_captured=len(old),old_prefix_exact=True,reproducible=True,observer_equivalent=True,**a[1],**quote_history_summary(sim.info.bid_empty,sim.info.ask_empty),elapsed_seconds=time.perf_counter()-start))
        print('PILOT',seed,rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(out/'verification.csv',index=False)
if __name__=='__main__':main()
