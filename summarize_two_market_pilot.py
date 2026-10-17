"""Additional descriptive pilot summary, resampling whole shared-seed clusters."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd


def main(base):
    base=Path(base);protocol=json.loads((base/'pilot_protocol.json').read_text());verification=json.loads((base/'verification.json').read_text());root=Path(__file__).resolve().parent
    assert hashlib.sha256((root/'two_market_pilot.py').read_bytes()).hexdigest()==protocol['analysis_sha256']
    directory=Path(verification['parallel_results']);raw=pd.read_csv(directory/'raw_results.csv',float_precision='round_trip');liq=pd.read_csv(base/'liquidity_per_run.csv');sources=pd.read_csv(base/'recovery_price_sources.csv');spells=pd.read_csv(base/'empty_spells.csv');system=liq[liq.market=='A+B']
    data=system.merge(sources,on=['scenario','seed'],validate='one_to_one');rng=np.random.default_rng(20261017);rows=[]
    for metric in ['two_sided_fraction','one_sided_fraction','empty_both_fraction','fallback_fraction','empty_shock_fraction','ever_lost','ever_recovered','price_recovered','liquidity_recovered','price_recovery_without_two_sided']:
        matrix=data.pivot(index='seed',columns='scenario',values=metric).sort_index().astype(float).to_numpy();assert matrix.shape==(20,10)
        values=matrix.mean(axis=1);draw=values[rng.integers(0,20,(10000,20))].mean(axis=1);lo,hi=np.quantile(draw,[.025,.975]);rows.append(dict(metric=metric,mean=values.mean(),ci_low=lo,ci_high=hi,n_independent_seed_clusters=20,n_scenarios=10,method='whole-seed-cluster percentile bootstrap 95%, B=10000'))
    pd.DataFrame(rows).to_csv(base/'overall_seed_cluster_ci.csv',index=False)
    beta_pairs=0
    for low,high in [(0,2),(1,3),(4,6),(5,7)]:
        for offset in range(20):
            a=pd.read_csv(directory/'runs'/f'{low*20+offset:07d}_timeseries.csv',float_precision='round_trip');b=pd.read_csv(directory/'runs'/f'{high*20+offset:07d}_timeseries.csv',float_precision='round_trip')
            for col in ['liquid_a','liquid_b','liquid_fx','bid_empty_a','ask_empty_a','bid_empty_b','ask_empty_b','fallback_a','fallback_b','fallback_fx']:assert np.array_equal(a[col],b[col])
            beta_pairs+=1
    lost=spells[spells.market=='A+B'];assert lost.start.lt(200).all() and lost.right_censored.all()
    findings=dict(interpretation='exploratory pilot only; 20 independent seeds shared across 10 scenarios',pre_shock_liquidity_losses=len(lost),loss_onset_min=int(lost.start.min()),loss_onset_max=int(lost.start.max()),all_liquidity_losses_persist_to_horizon=bool(lost.right_censored.all()),beta_matched_pairs_with_identical_saved_liquidity_flags=beta_pairs,
        main_methodological_issue='59% of research runs lose two-sided asset liquidity before the shock; subsequent price-gap recovery with an active shock override is not market-liquidity recovery',
        caution='Do not drop illiquid paths: that selects on endogenous model outcomes. Retain all attempts and both recovery endpoints.',
        readiness='Technical execution and saving ready. Agree primary endpoint and interpretation of pre-shock liquidity collapse before a confirmatory full grid.',
        limitations=['20 seeds: no definitive parameter inference','eight corners and two intermediate scenarios do not identify full-grid responses','same seed across scenarios gives paired initialization, not identical later random-call sequences','FX market remained two-sided in all recorded samples','time fractions and spell durations use recorded samples, not intrastep events','price recovery conditional means and liquidity-aware conditional means can involve different sets of seeds'],
        supplementary_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (base/'findings.json').write_text(json.dumps(findings,indent=2));print(pd.DataFrame(rows).round(5).to_string(index=False));print(json.dumps(findings,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory');main(parser.parse_args().directory)
