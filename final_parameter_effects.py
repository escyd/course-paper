"""Postprocessing only: seed-paired effects and censoring-aware restricted times.

No simulations, model edits or modifications to historical metric definitions.
"""
import argparse,json,uuid
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from experiment_methodology import _bootstrap
from experiment_runner import _digest

MAIN=['avg_gap_after','final_gap','price_recovery_rate','liquidity_aware_recovery_rate','pre_shock_liquidity_failure_rate','price_restricted_wait_299','liquidity_restricted_wait_299']

def interval(values,family):
    x=np.asarray(values,dtype=float);x=x[np.isfinite(x)]
    if len(x)<2:return dict(simultaneous_ci_low=np.nan,simultaneous_ci_high=np.nan,n_seeds=len(x),constant_sample=False)
    mean=x.mean();se=x.std(ddof=1)/np.sqrt(len(x));q=t.ppf(1-.05/(2*family),len(x)-1)
    return dict(simultaneous_ci_low=mean-q*se,simultaneous_ci_high=mean+q*se,n_seeds=len(x),constant_sample=bool(np.ptp(x)==0))

def endpoint_data(raw):
    d=raw.copy()
    assert d.status.eq('completed').all() and d.data_valid.eq(True).all()
    for source,target,restricted in [('recovery_time','price_recovery_rate','price_restricted_wait_299'),('liquidity_recovery_time','liquidity_aware_recovery_rate','liquidity_restricted_wait_299')]:
        recovered=d[source+'_status'].eq('recovered');censored=d[source+'_status'].eq('censored')
        assert (recovered|censored).all() and d[source+'_followup'].eq(299).all()
        assert d.loc[censored,source].isna().all() and d.loc[recovered,source].between(0,299).all()
        d[target]=recovered.astype(float)
        # Integral of empirical survival over [0,299]. This is a NEW endpoint,
        # not an imputed event time: right-censored recovery_time stays NaN.
        d[restricted]=np.where(recovered,d[source],299.)
    assert d.pre_shock_history_complete.eq(True).all()
    d['pre_shock_liquidity_failure_rate']=d.pre_shock_liquidity_failure.astype(float)
    return d

def main(directory,verified):
    directory=Path(directory);verified=Path(verified)
    assert json.loads((verified/'verification.json').read_text())['scenario_statistics_recomputed_exact']
    out=directory/('parameter_effects_'+uuid.uuid4().hex[:8]);out.mkdir(exist_ok=False)
    raw=pd.read_csv(directory/'raw_results.csv',float_precision='round_trip');d=endpoint_data(raw)
    old=pd.read_csv(verified/'paired_comparisons.csv',float_precision='round_trip');effects=[];extra=[];contexts=[]
    scenario=d[['scenario_id','tau','beta','fx_cost']].drop_duplicates()
    group={k:v.set_index('seed').sort_index() for k,v in d.groupby('scenario_id')}
    # Every adjacent contrast, including the two new restricted waiting endpoints.
    contrast=old[['parameter','low_level','high_level','low_scenario','high_scenario']].drop_duplicates()
    for r in contrast.itertuples():
        a,b=group[r.low_scenario],group[r.high_scenario];assert a.index.equals(b.index)
        for metric in MAIN[-2:]:
            delta=(b[metric]-a[metric]).to_numpy();bs=_bootstrap(delta,('restricted-paired-v1',r.low_scenario,r.high_scenario,metric),1999)
            extra.append(dict(parameter=r.parameter,low_level=r.low_level,high_level=r.high_level,low_scenario=r.low_scenario,high_scenario=r.high_scenario,metric=metric,n_shared_seeds=500,n_eligible_pairs=500,low_paired_mean=a[metric].mean(),high_paired_mean=b[metric].mean(),mean_difference_high_minus_low=bs['mean'],median_difference_high_minus_low=bs['median'],mean_difference_ci_low=bs['mean_ci_low'],mean_difference_ci_high=bs['mean_ci_high'],median_difference_ci_low=bs['median_ci_low'],median_difference_ci_high=bs['median_ci_high'],estimand='restricted mean waiting time through 299; all 500 seeds, including administratively censored paths',unit='steps'))
    extended=pd.concat([old,pd.DataFrame(extra)],ignore_index=True)
    # Equal weight over grid contexts, then average inside each seed. The 100
    # scenario rows sharing a seed are NEVER treated as independent replicates.
    for parameter in ['tau','beta','fx_cost']:
        levels=sorted(d[parameter].unique());other=[p for p in ['tau','beta','fx_cost'] if p!=parameter]
        for low,high in zip(levels[:-1],levels[1:]):
            left=d[d[parameter]==low].set_index(['seed']+other)[MAIN].sort_index();right=d[d[parameter]==high].set_index(['seed']+other)[MAIN].sort_index();assert left.index.equals(right.index)
            deltas=(right-left).groupby(level='seed').mean();a=left.groupby(level='seed').mean();b=right.groupby(level='seed').mean();assert len(deltas)==500
            for metric in MAIN:
                x=deltas[metric].to_numpy();bs=_bootstrap(x,('grid-marginal-paired-v1',parameter,low,high,metric),1999)
                effects.append(dict(parameter=parameter,low_level=low,high_level=high,metric=metric,n_contexts=len(left)//500,n_seeds=500,low_mean=a[metric].mean(),high_mean=b[metric].mean(),mean_difference_high_minus_low=bs['mean'],pointwise_ci_low=bs['mean_ci_low'],pointwise_ci_high=bs['mean_ci_high'],**{k:v for k,v in interval(x,1).items() if k not in ['simultaneous_ci_low','simultaneous_ci_high','n_seeds']}))
                contexts.append(x)
    marginal=pd.DataFrame(effects);family=len(extended)+len(marginal);assert family==2192
    for i,r in extended.iterrows():
        a,b=group[r.low_scenario],group[r.high_scenario];metric=r.metric
        if metric in ['recovery_time','liquidity_recovery_time']:
            eligible=a[metric+'_status'].eq('recovered')&b[metric+'_status'].eq('recovered');x=(b.loc[eligible,metric]-a.loc[eligible,metric]).to_numpy()
        else:x=(b[metric]-a[metric]).to_numpy()
        ci=interval(x,family);assert len(x)==r.n_eligible_pairs
        extended.loc[i,'simultaneous_ci_low']=ci['simultaneous_ci_low'];extended.loc[i,'simultaneous_ci_high']=ci['simultaneous_ci_high'];extended.loc[i,'constant_sample']=ci['constant_sample']
    for i,x in enumerate(contexts):
        ci=interval(x,family);marginal.loc[i,'simultaneous_ci_low']=ci['simultaneous_ci_low'];marginal.loc[i,'simultaneous_ci_high']=ci['simultaneous_ci_high']
    extended.to_csv(out/'paired_effects_with_multiplicity.csv',index=False);marginal.to_csv(out/'parameter_effects_equal_grid_weights.csv',index=False)
    # Scenario restricted means are a supplement; original recovery times and
    # original reports remain unchanged.
    restricted=[]
    for key,part in d.groupby(['scenario_id','tau','beta','fx_cost']):
        for metric in MAIN[-2:]:
            x=part[metric].to_numpy();event='recovery_time' if metric.startswith('price') else 'liquidity_recovery_time'
            # Discrete survival-integral identity, no extrapolation past horizon.
            surv=sum(np.mean(part[event].isna()|(part[event]>k)) for k in range(299))
            assert np.isclose(surv,x.mean(),rtol=1e-13,atol=1e-12)
            restricted.append(dict(zip(['scenario_id','tau','beta','fx_cost'],key))|dict(metric=metric,**_bootstrap(x,('restricted-scenario-v1',key,metric),1999)))
    pd.DataFrame(restricted).to_csv(out/'scenario_restricted_waiting_times.csv',index=False)
    # Descriptive grid-average CIs preserve the common-seed cluster.
    overall=[]
    for metric in MAIN:
        x=d.groupby('seed')[metric].mean().to_numpy();overall.append(dict(metric=metric,n_seed_clusters=500,scenarios_per_seed=100,**_bootstrap(x,('overall-grid-v1',metric),1999)))
    pd.DataFrame(overall).to_csv(out/'grid_average_metrics.csv',index=False)
    protocol=dict(analysis='exploratory postprocessing, not preregistered confirmatory tests',sampling_unit='independent seed; all contexts sharing that seed kept together',marginal_estimand='adjacent high-minus-low effect averaged equally over existing other-parameter grid levels; not population/economy weights',restricted_time='area under empirical recovery survival over [0,299]; at common administrative censoring only, mean min(T,299). Original event times unchanged; no extrapolation.',conditional_times='original paired time comparisons use jointly recovered paths only; selected descriptive estimand, never unconditional recovery time',pointwise_intervals='deterministic paired percentile bootstrap B=1999, 95%; marginal bootstrap resamples whole seed clusters',multiplicity=dict(family_size=family,contents='2115 scenario-endpoint means + 77 marginal-endpoint means',method='Bonferroni two-sided Student-t intervals, nominal simultaneous 95%, approximate under sampling distribution assumptions',caveat='not exact distribution-free coverage; sparse/constant contrasts need caution. Bootstrap median intervals remain pointwise, not family-adjusted. No p-values or automated hypothesis decisions.'),script_sha256=_digest(Path(__file__)))
    (out/'analysis_protocol.json').write_text(json.dumps(protocol,indent=2))
    (out/'complete.json').write_text(json.dumps(dict(files={p.name:_digest(p) for p in out.iterdir() if p.is_file()}),indent=2));print('PARAMETER EFFECTS',out,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');p.add_argument('verified');a=p.parse_args();main(a.directory,a.verified)
