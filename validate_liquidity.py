"""Short, paired diagnostics; no full-grid execution and no historical overwrites."""
import json,random,hashlib,time
from pathlib import Path
import numpy as np
import pandas as pd
from stylized_facts import build_baseline,diagnose
from AgentBasedModel.utils.liquidity import liquidity_summary,empty_spells
from experiment_runner import execute_jobs
from experiments import aggregate_results
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'tables/liquidity_validation/20261017T154909Z'
OLD_SINGLE=ROOT/'tables/stylized_facts_v2/20261017T153832Z_86aa3981'
OLD_MULTI=ROOT/'tables/v2_validation/20261017T153944Z'

def single(seed):
    sim=build_baseline(seed);error='';start=time.perf_counter()
    try:sim.simulate(10000,silent=True)
    except Exception as exc:error=repr(exc)
    return sim,error,random.getstate(),np.random.get_state(),time.perf_counter()-start


def main():
    out=OUT/'after_final';out.mkdir(exist_ok=False);rows=[];stats=[];comparisons=[]
    import pickle
    from AgentBasedModel.utils.liquidity import quote_state
    with (OUT/'before_probe.pkl').open('rb') as f:before_probe=pickle.load(f)
    for seed in [42,43,44,45,46,47,48,58,82,85]:
        sim,error,rng,nprng,elapsed=single(seed);reference=single(seed)
        assert sim.info.prices==reference[0].info.prices and error==reference[1] and rng==reference[2]
        assert all(np.array_equal(a,b) for a,b in zip(nprng,reference[3]))
        assert [(t.cash,t.assets,t.type) for t in sim.traders]==[(t.cash,t.assets,t.type) for t in reference[0].traders]
        prices=np.array(sim.info.prices);old=pd.read_csv(OLD_SINGLE/f'steps_10000/seed_{seed}/prices.csv',float_precision='round_trip').price.to_numpy()
        assert np.array_equal(prices[:len(old)],old)
        before = before_probe.get(seed)
        if before is not None and not before['error']:
            assert sim.info.prices==before['prices'] and rng==before['rng']
            assert [(t.cash,t.assets,t.type) for t in sim.traders]==before['agents']
            assert all(np.array_equal(a,b) for a,b in zip(nprng,before['nprng']))
        folder=out/f'single_seed_{seed}';folder.mkdir()
        pd.DataFrame(dict(step=range(len(prices)),price=prices,bid_empty=sim.info.bid_empty,ask_empty=sim.info.ask_empty,fallback=sim.info.fallback_used)).to_csv(folder/'timeseries.csv',index=False)
        (folder/'final_state.json').write_text(json.dumps(dict(quote_state(sim.exchange),error=error,last_midpoint=sim.exchange.last_midpoint),indent=2))
        pd.DataFrame(empty_spells(sim.info.empty_books),columns=['start','duration','right_censored']).to_csv(folder/'empty_spells.csv',index=False)
        row=dict(seed=seed,steps=10000,captured=len(prices),completed=len(prices)==10000 and not error,error=error,simulation_seconds=elapsed,old_captured=len(old),old_price_prefix_exact=True,reproducible_prices_balances_rng=True,fallback_price_calls=sim.exchange.fallback_calls,**liquidity_summary(sim.info.empty_books));rows.append(row)
        if not error:
            stat,acf=diagnose(prices,100,(10,20),40);stat.update(seed=seed);stats.append(stat);acf.to_csv(folder/'acf.csv',index=False)
        print('single',seed,'completed',len(prices),'empty',sum(sim.info.empty_books),flush=True)
    pd.DataFrame(rows).to_csv(out/'single_runs.csv',index=False);pd.DataFrame(stats).to_csv(out/'single_statistics.csv',index=False)
    jobs=[]
    for scenario,(tau,beta,fee) in enumerate([(0.,0.,0.),(0.,0.,.001),(.5,.4,.005),(1.,.6,.01)]):
        for seed in range(42,48):jobs.append(dict(run_id=seed-42,seed=seed,tau=tau,beta=beta,fx_cost=fee,steps=500,shock_iteration=200,extra={'scenario':scenario}))
    raw,summary=execute_jobs(jobs,aggregate_results,workers=4,output_root=out/'multi_parallel',save_full_timeseries=True)
    sr,ss=execute_jobs(jobs,aggregate_results,workers=1,output_root=out/'multi_serial',save_full_timeseries=True)
    pd.testing.assert_frame_equal(raw,sr)
    for i,r in enumerate(raw.itertuples()):
        new=pd.read_csv(Path(raw.attrs['output_dir'])/'runs'/f'{i:07d}_timeseries.csv',float_precision='round_trip')
        serial=pd.read_csv(Path(sr.attrs['output_dir'])/'runs'/f'{i:07d}_timeseries.csv',float_precision='round_trip');pd.testing.assert_frame_equal(new,serial)
        old=pd.read_csv(OLD_MULTI/f'scenario_{r.scenario}/seed_{r.seed}/timeseries.csv',float_precision='round_trip')
        for col in old.columns:assert np.array_equal(new[col],old[col]),(i,col)
        if pd.notna(r.liquidity_recovery_time):assert r.liquidity_recovery_time>=r.recovery_time
        comparisons.append(dict(scenario=r.scenario,seed=r.seed,all_old_series_exact=True,parallel_serial_exact=True,price_recovery=r.recovery_time,liquidity_recovery=r.liquidity_recovery_time))
    pd.DataFrame(comparisons).to_csv(out/'multi_comparison.csv',index=False)
    before=json.loads((OUT/'before_manifest.json').read_text());assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h for name,h in before.items())
    (out/'verification.json').write_text(json.dumps(dict(single_seeds=[r['seed'] for r in rows],single_completed=sum(r['completed'] for r in rows),single_steps=10000,multi_runs=24,multi_failed=int(raw.status.eq('failed').sum()),old_files_unchanged=len(before),single_prefixes_exact=True,multi_trajectories_unchanged=True,reproducible=True,multi_output=raw.attrs['output_dir'],sampling='single: before trading at original price capture; multi: after trading/microdynamics/link; durations in recorded steps'),indent=2))
    print('OUTPUT',out,flush=True)
if __name__=='__main__':main()
