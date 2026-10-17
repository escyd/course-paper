"""Repeat the 50 seeds with accounting observers; compare every saved price."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing,json,hashlib,argparse
from pathlib import Path
import numpy as np
import pandas as pd
from empty_both_diagnostics import run
from AgentBasedModel.utils.liquidity import quote_history_summary
ROOT=Path(__file__).resolve().parent
OLD=ROOT/'tables/stylized_facts_v2/20261017T153832Z_86aa3981'

def job(args):
    seed,study=args;sim,counts,rng,nprng=run(seed,True)
    first=pd.read_csv(Path(study)/f'steps_10000/seed_{seed}/prices.csv',float_precision='round_trip').price.to_numpy()
    assert np.array_equal(first,sim.info.prices)
    old=pd.read_csv(OLD/f'steps_10000/seed_{seed}/prices.csv',float_precision='round_trip').price.to_numpy()
    assert np.array_equal(first[:len(old)],old)
    balances=[dict(cash=t.cash,assets=t.assets,type=t.type) for t in sim.traders]
    return dict(seed=seed,completed=len(first),prices_reproduced=True,old_price_prefix_exact=True,old_n_prices=len(old),
                balances_sha256=hashlib.sha256(json.dumps(balances,sort_keys=True).encode()).hexdigest(),
                **counts,**quote_history_summary(sim.info.bid_empty,sim.info.ask_empty))

def main(study):
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        rows=[]
        for r in pool.map(job,[(seed,str(study)) for seed in range(42,92)]):
            rows.append(r)
            if len(rows)%10==0:print('Reproduced and checked accounting',len(rows),'/50',flush=True)
    pd.DataFrame(rows).to_csv(Path(study)/'accounting_reproducibility.csv',index=False)
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('study');main(parser.parse_args().study)
