"""Observation-only diagnostics for the approved v2 model; unique output directories."""
import json, random, hashlib, importlib.util, time
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch
from collections import Counter
import numpy as np
import pandas as pd
from experiments import build_system, set_all_seeds, aggregate_results
from experiment_runner import _run_job
from AgentBasedModel.agents.multi_market import AssetMarket
from AgentBasedModel.simulator.multi_simulator import MultiSimulatorInfo

ROOT=Path(__file__).resolve().parent
AUDIT=ROOT/'tables/methodology_audit/20261017T122418Z'
def load(name):
    spec=importlib.util.spec_from_file_location(name,AUDIT/f'{name}.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def spells(flags):
    result=[];start=None
    for i,v in enumerate([*flags,False]):
        if v and start is None:start=i
        if not v and start is not None:result.append((start,i-start,i==len(flags)));start=None
    return result

def main():
    out=ROOT/'tables/v2_validation'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');out.mkdir(exist_ok=False)
    old={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('tables/**/*') if p.is_file() and 'stylized_facts_v2' not in str(p) and out not in p.parents}
    model={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('AgentBasedModel/**/*.py')}
    start=time.perf_counter(); trace=load('trace_market'); observed=trace.run(85,10000); reference=trace.run(85,10000,False)
    sim,quotes,trades,empty,mm,error,rng,nprng=observed
    assert sim.info.prices==reference[0].info.prices and rng==reference[6]
    assert all(np.array_equal(a,b) for a,b in zip(nprng,reference[7]))
    assert [(t.cash,t.assets,t.type) for t in sim.traders]==[(t.cash,t.assets,t.type) for t in reference[0].traders]
    pd.DataFrame(quotes).to_csv(out/'seed85_quotes.csv',index=False);pd.DataFrame(trades).to_csv(out/'seed85_trades.csv',index=False);pd.DataFrame(mm).to_csv(out/'seed85_mm.csv',index=False)
    (out/'seed85_failure.json').write_text(json.dumps(dict(seed=85,requested=10000,captured=len(sim.info.prices),error=error,empty_events=empty,observer_equivalent=True,invariants=trace.invariants(sim.exchange)),indent=2))
    print('Seed85 trace saved',len(sim.info.prices),flush=True)
    allrows=[];marketrows=[];spellrows=[];verification=[]
    scenarios=[(0.,0.,0.),(0.,0.,.001),(.5,.4,.005),(1.,.6,.01)]
    for scenario,(tau,beta,fee) in enumerate(scenarios):
      for seed in range(42,62):
        job=dict(run_id=seed-42,seed=seed,tau=tau,beta=beta,fx_cost=fee,shock_iteration=200,steps=500,save_full_timeseries=True)
        records=[];calls=Counter();held={};phase={'record':False}
        price_original=AssetMarket.price;record_original=MultiSimulatorInfo.record;build_original=build_system
        def build(**kwargs):held['sim']=build_original(**kwargs);return held['sim']
        def price(m):
            source='shock' if m.shocked_price is not None else ('midpoint' if m.order_book['bid'] and m.order_book['ask'] else 'fallback')
            calls[(m.name,source)]+=1
            return price_original(m)
        def record(info,s):
            before={m.name:dict(empty=not(bool(m.order_book['bid']) and bool(m.order_book['ask'])),fallback=m.shocked_price is None and not(bool(m.order_book['bid']) and bool(m.order_book['ask']))) for m in [*s.markets.values(),s.fx_market]}
            result=record_original(info,s)
            for name,p in zip(['A','B','FX'],[info.prices_a[-1],info.prices_b[-1],info.fx_rates[-1]]):
                records.append(dict(step=s.current_iteration,market=name,price=p,**before[name]))
            return result
        with patch('experiments.build_system',build),patch.object(AssetMarket,'price',price),patch.object(MultiSimulatorInfo,'record',record):row,series=_run_job(job)
        random_state=random.getstate();numpy_state=np.random.get_state()
        frame=pd.DataFrame(series); rec=pd.DataFrame(records)
        assert np.allclose(frame.price_gap,abs(frame.prices_a-frame.prices_b/frame.fx_rates),rtol=0,atol=1e-12)
        g=frame.price_gap.to_numpy(); hits=np.flatnonzero(g[200:]<=g[150:200].mean()+.5);expected=float(hits[0]) if len(hits) else np.nan
        assert (pd.isna(row['recovery_time']) and pd.isna(expected)) or row['recovery_time']==expected
        if seed==42:
            other,other_series=_run_job(job)
            assert frame.equals(pd.DataFrame(other_series)) and random_state==random.getstate()
            assert all(np.array_equal(a,b) for a,b in zip(numpy_state,np.random.get_state()))
            verification.append(dict(scenario=scenario,seed=seed,observer_equivalent=True))
        folder=out/f'scenario_{scenario}'/f'seed_{seed}';folder.mkdir(parents=True)
        frame.to_csv(folder/'timeseries.csv',index=False);rec.to_csv(folder/'price_sources.csv',index=False)
        row['scenario']=scenario;allrows.append(row)
        for name,part in rec.groupby('market'):
            flags=part.fallback.tolist(); emptyflags=part['empty'].tolist(); r=np.diff(np.log(part.price.to_numpy()));mask=np.array(flags[1:])&np.array(flags[:-1])
            ss=spells(emptyflags)
            marketrows.append(dict(scenario=scenario,seed=seed,market=name,empty_steps=sum(emptyflags),fallback_record_steps=sum(flags),max_one_sided_spell=max([x[1] for x in ss],default=0),first_empty=next((i for i,v in enumerate(emptyflags) if v),None),fallback_calls=calls[(name,'fallback')],price_calls=sum(v for (n,k),v in calls.items() if n==name),fallback_return_n=int(mask.sum()),fallback_nonzero_returns=int(np.count_nonzero(r[mask])),fallback_return_sd=float(np.std(r[mask])) if mask.any() else np.nan))
            for onset,duration,censored in ss:spellrows.append(dict(scenario=scenario,seed=seed,market=name,onset=onset,duration=duration,right_censored=censored))
      print('Multi scenario finished',scenario,flush=True)
    raw=pd.DataFrame(allrows);raw.to_csv(out/'multi_raw.csv',index=False);aggregate_results(raw).to_csv(out/'multi_summary.csv',index=False)
    pd.DataFrame(marketrows).to_csv(out/'multi_fallback.csv',index=False);pd.DataFrame(spellrows).to_csv(out/'multi_empty_spells.csv',index=False)
    unchanged=all(p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==h for name,h in old.items() for p in [ROOT/name])
    assert unchanged and all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h for name,h in model.items())
    (out/'verification.json').write_text(json.dumps(dict(model_version='paper-revision-v2',branch='paper-revision',scenarios=scenarios,seeds=list(range(42,62)),steps=500,shock=200,old_files_unchanged=len(old),model_unchanged=True,observer_checks=verification,elapsed_seconds=time.perf_counter()-start,notes='Fallback return statistics are descriptive, conditional on consecutive fallback samples; no counterfactual trading rule was introduced.'),indent=2))
    print('OUTPUT',out,flush=True)
if __name__=='__main__':main()
