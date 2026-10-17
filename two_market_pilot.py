"""Ten-scenario pilot; immutable model, seed-level inference and unique output folders."""
import json,hashlib,time,itertools
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
from experiment_runner import execute_jobs,_source_hashes,_digest
from experiments import aggregate_results
from AgentBasedModel.utils.liquidity import quote_history_summary,empty_spells
ROOT=Path(__file__).resolve().parent
SCENARIOS=list(itertools.product([0.,1.],[0.,.6],[0.,.01]))+[(.3,.2,.003),(.5,.4,.005)]
SEEDS=list(range(42,62))

def mean_ci(values,rng):
    x=np.asarray(values,dtype=float);x=x[np.isfinite(x)]
    if not len(x):return np.nan,np.nan,np.nan,0
    draws=x[rng.integers(0,len(x),size=(10000,len(x)))].mean(axis=1)
    lo,hi=np.quantile(draws,[.025,.975]);return float(x.mean()),float(lo),float(hi),len(x)


def summarize_pilot(raw,directory,out):
    records=[];prices=[];spell_rows=[];fingerprints=[]
    for index,r in enumerate(raw.itertuples()):
        f=directory/'runs'/f'{index:07d}_timeseries.csv';d=pd.read_csv(f,float_precision='round_trip')
        assert len(d)==500 and np.isfinite(d[['prices_a','prices_b','fx_rates','price_gap']]).all().all()
        assert (d[['prices_a','prices_b','fx_rates']]>0).all().all()
        assert np.array_equal(d.price_gap.to_numpy(),abs(d.prices_a.to_numpy()-d.prices_b.to_numpy()/d.fx_rates.to_numpy()))
        a=d.liquid_a.to_numpy();b=d.liquid_b.to_numpy();both=a&b
        threshold=sum(d.price_gap.to_numpy()[150:200])/50+.5
        hits=np.flatnonzero(d.price_gap.to_numpy()[200:]<=threshold)
        liquid_hits=np.flatnonzero((d.price_gap.to_numpy()[200:]<=threshold)&both[200:])
        for name,events in [('recovery_time',hits),('liquidity_recovery_time',liquid_hits)]:
            value=getattr(r,name);assert (pd.isna(value) and not len(events)) or (len(events) and value==events[0])
        recovered=pd.notna(r.recovery_time);t=200+int(r.recovery_time) if recovered else None
        fallback=d.fallback_a.to_numpy()|d.fallback_b.to_numpy()
        empty_shock=(~a&~d.fallback_a.to_numpy())|(~b&~d.fallback_b.to_numpy())
        prices.append(dict(scenario=r.scenario,seed=r.seed,price_recovered=recovered,liquidity_recovered=pd.notna(r.liquidity_recovery_time),
                           price_recovery_without_two_sided=bool(not both[t]) if recovered else False,
                           price_recovery_with_fallback=bool(fallback[t]) if recovered else False,
                           price_recovery_with_empty_shock=bool(empty_shock[t]) if recovered else False))
        common=dict(scenario=r.scenario,seed=r.seed,tau=r.tau,beta=r.beta,fx_cost=r.fx_cost)
        for name in ['a','b']:
            bid=d['bid_empty_'+name].to_numpy();ask=d['ask_empty_'+name].to_numpy();q=quote_history_summary(bid.tolist(),ask.tolist());missing=bid|ask
            records.append(dict(common,market=name.upper(),two_sided_fraction=q['two_sided_steps']/500,one_sided_fraction=q['one_sided_steps']/500,empty_both_fraction=q['empty_both_steps']/500,
                                fallback_fraction=d['fallback_'+name].mean(),empty_shock_fraction=np.mean(missing&~d['fallback_'+name].to_numpy()),
                                max_empty_duration=q['longest_empty_period'],n_recoveries=q['two_sided_recoveries'],ever_lost=bool(q['empty_steps']),ever_recovered=bool(q['two_sided_recoveries'])))
            for s in empty_spells(missing.tolist()):spell_rows.append(dict(common,market=name.upper(),**s))
        # System fractions are disjoint: both markets two-sided / at least one fully empty / remaining one-sided.
        completely_empty=(d.bid_empty_a&d.ask_empty_a)|(d.bid_empty_b&d.ask_empty_b)
        miss=~both;spells=empty_spells(miss.tolist());rec=sum(x and not y for x,y in zip(miss,miss[1:]))
        records.append(dict(common,market='A+B',two_sided_fraction=both.mean(),one_sided_fraction=np.mean(miss&~completely_empty),empty_both_fraction=completely_empty.mean(),fallback_fraction=fallback.mean(),empty_shock_fraction=empty_shock.mean(),
                            max_empty_duration=max([s['duration'] for s in spells],default=0),n_recoveries=rec,ever_lost=bool(miss.any()),ever_recovered=bool(rec)))
        for s in spells:spell_rows.append(dict(common,market='A+B',**s))
        fingerprints.append(dict(common,prices_sha256=hashlib.sha256(d[['prices_a','prices_b','fx_rates']].to_numpy().tobytes()).hexdigest(),fx_two_sided_fraction=d.liquid_fx.mean(),fx_fallback_fraction=d.fallback_fx.mean()))
    liq=pd.DataFrame(records);recovery=pd.DataFrame(prices);fp=pd.DataFrame(fingerprints)
    assert np.allclose(liq.two_sided_fraction+liq.one_sided_fraction+liq.empty_both_fraction,1)
    assert fp.groupby('scenario').prices_sha256.nunique().eq(20).all()
    liq.to_csv(out/'liquidity_per_run.csv',index=False);recovery.to_csv(out/'recovery_price_sources.csv',index=False);pd.DataFrame(spell_rows).to_csv(out/'empty_spells.csv',index=False);fp.to_csv(out/'trajectory_fingerprints.csv',index=False)
    rng=np.random.default_rng(20261017);summary=[]
    for (scenario,market),part in liq.groupby(['scenario','market']):
        row=dict(scenario=scenario,market=market,n_runs=len(part),tau=part.tau.iloc[0],beta=part.beta.iloc[0],fx_cost=part.fx_cost.iloc[0])
        for metric in ['two_sided_fraction','one_sided_fraction','empty_both_fraction','fallback_fraction','empty_shock_fraction','max_empty_duration','n_recoveries']:
            mean,lo,hi,n=mean_ci(part[metric],rng);row.update({metric:mean,metric+'_ci_low':lo,metric+'_ci_high':hi})
        row['max_empty_duration_observed']=part.max_empty_duration.max()
        for metric in ['ever_lost','ever_recovered']:
            count=int(part[metric].sum());lo,hi=proportion_confint(count,len(part),method='wilson');row.update({metric+'_count':count,metric+'_fraction':count/len(part),metric+'_ci_low':lo,metric+'_ci_high':hi})
        summary.append(row)
    ls=pd.DataFrame(summary);ls.to_csv(out/'liquidity_by_scenario.csv',index=False)
    # Existing reporting retains both first-passage definitions, failure/censoring and Wilson/bootstrap CIs.
    financial=aggregate_results(raw)
    financial=financial.merge(pd.DataFrame([dict(scenario=i,tau=t,beta=b,fx_cost=f) for i,(t,b,f) in enumerate(SCENARIOS)]),on=['tau','beta','fx_cost'],validate='one_to_one')
    for metric in ['recovery_time','liquidity_recovery_time']:
        financial[metric+'_censored_ci_low']=1-financial[metric+'_fraction_completed_valid_ci_high']
        financial[metric+'_censored_ci_high']=1-financial[metric+'_fraction_completed_valid_ci_low']
    gaps=[]
    for scenario,part in raw.groupby('scenario'):
        for metric in ['avg_gap_before','avg_gap_after','max_gap_after','final_gap']:
            mean,lo,hi,n=mean_ci(part[metric],rng);gaps.append(dict(scenario=scenario,metric=metric,mean=mean,ci_low=lo,ci_high=hi,n_valid=n,method='seed bootstrap mean percentile 95%, B=10000'))
    pd.DataFrame(gaps).to_csv(out/'gap_confidence_intervals.csv',index=False)
    sources=recovery.groupby('scenario').agg(price_recovered=('price_recovered','sum'),liquidity_recovered=('liquidity_recovered','sum'),price_recovery_without_two_sided=('price_recovery_without_two_sided','sum'),price_recovery_with_fallback=('price_recovery_with_fallback','sum'),price_recovery_with_empty_shock=('price_recovery_with_empty_shock','sum')).reset_index()
    financial=financial.merge(sources,on='scenario');financial.to_csv(out/'financial_by_scenario.csv',index=False)
    # Exploratory matched high-minus-low contrasts on the eight corners only; no p-values.
    comparisons=[];system=liq[liq.market=='A+B'];corners=SCENARIOS[:8]
    for parameter,position in [('tau',0),('beta',1),('fx_cost',2)]:
        for low in corners:
            if low[position]!=0:continue
            high=tuple([1.,.6,.01][position] if i==position else value for i,value in enumerate(low));a=corners.index(low);b=corners.index(high)
            pair=system[system.scenario==a].merge(system[system.scenario==b],on='seed',suffixes=('_low','_high'));assert len(pair)==20
            for metric in ['two_sided_fraction','fallback_fraction','empty_shock_fraction','ever_lost','max_empty_duration']:
                delta=pair[metric+'_high'].astype(float)-pair[metric+'_low'].astype(float);mean,lo,hi,n=mean_ci(delta,rng)
                comparisons.append(dict(parameter=parameter,low_scenario=a,high_scenario=b,metric=metric,n_pairs=n,mean_high_minus_low=mean,ci_low=lo,ci_high=hi,interpretation='exploratory paired seed contrast; not confirmatory'))
    pd.DataFrame(comparisons).to_csv(out/'parameter_contrasts.csv',index=False)
    overview=ls[ls.market=='A+B'].merge(financial[['scenario','recovery_time_fraction_completed_valid','liquidity_recovery_time_fraction_completed_valid','recovery_time_conditional_mean','liquidity_recovery_time_conditional_mean','price_recovery_without_two_sided']],on='scenario')
    overview.to_csv(out/'pilot_overview.csv',index=False)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    x=np.arange(10);v=overview.sort_values('scenario');labels=[f'{int(r.scenario)}: {r.tau:g}, {r.beta:g}, {r.fx_cost:g}' for r in v.itertuples()]
    with plt.rc_context({'font.size':8}):
        fig,axes=plt.subplots(1,3,figsize=(13,5),layout='constrained')
        for ax,metric,title in [(axes[0],'two_sided_fraction','Time both A and B are two-sided'),(axes[1],'fallback_fraction','Time at least one asset price uses fallback')]:
            ax.barh(x,v[metric]);ax.errorbar(v[metric],x,xerr=[v[metric]-v[metric+'_ci_low'],v[metric+'_ci_high']-v[metric]],fmt='none',c='black',capsize=2);ax.set(yticks=x,yticklabels=labels if ax is axes[0] else [],title=title,xlim=(0,1));ax.xaxis.set_major_formatter(PercentFormatter(1))
        axes[0].set_ylabel('Scenario: tau, beta, FX cost')
        axes[2].barh(x-.15,v.recovery_time_fraction_completed_valid,height=.3,label='Price recovery');axes[2].barh(x+.15,v.liquidity_recovery_time_fraction_completed_valid,height=.3,label='Liquidity-aware recovery');axes[2].set(yticks=x,yticklabels=[],title='Recovery fraction (20 seeds)',xlim=(0,1));axes[2].xaxis.set_major_formatter(PercentFormatter(1));axes[2].legend(loc='lower right')
        fig.suptitle('Exploratory pilot; 500 steps, shock at 200; liquidity sampled at recorded prices')
        fig.savefig(out/'pilot_overview.png',dpi=200);fig.savefig(out/'pilot_overview.pdf');plt.close(fig)
    return liq,recovery,financial


def main():
    started=time.perf_counter();base=ROOT/'tables/two_market_pilot'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');base.mkdir(parents=True,exist_ok=False)
    old={str(p.relative_to(ROOT)):_digest(p) for p in ROOT.glob('tables/**/*') if p.is_file() and base not in p.parents};model=_source_hashes()
    full=set(itertools.product([0.,.1,.3,.5,1.],[0.,.2,.4,.6],[0.,.001,.003,.005,.01]));assert len(full)==100 and all(s in full for s in SCENARIOS)
    jobs=[dict(run_id=seed-42,seed=seed,tau=t,beta=b,fx_cost=f,steps=500,shock_iteration=200,extra=dict(scenario=i)) for i,(t,b,f) in enumerate(SCENARIOS) for seed in SEEDS]
    protocol=dict(model_version='paper-revision-v2',branch='paper-revision',scenarios=[dict(scenario=i,tau=t,beta=b,fx_cost=f) for i,(t,b,f) in enumerate(SCENARIOS)],seeds=SEEDS,steps=500,shock_iteration=200,n_research_runs=200,n_control_repeats=10,selection='8 corners + 2 intermediate scenarios from existing 100; before results',independence='unique seeds within scenarios; same seeds paired across scenarios, not independent observations across scenarios',liquidity='A and B separately; system both two-sided / at least one fully empty / remaining one-sided; FX separately',fallback='actual recorded fallback flags, separate from empty-book shock override',uncertainty='resample independent seeds, never time samples; percentile bootstrap 95% B=10000; recovery rates Wilson and conditional times original B=1999',parameter_comparison='matched high-minus-low corner contrasts, exploratory, no p-values or final conclusions',analysis_sha256=_digest(Path(__file__)),source_sha256=model)
    (base/'pilot_protocol.json').write_text(json.dumps(protocol,indent=2));pd.DataFrame(protocol['scenarios']).to_csv(base/'selected_scenarios.csv',index=False)
    raw,summary=execute_jobs(jobs,aggregate_results,workers=4,output_root=base/'parallel',save_full_timeseries=True);directory=Path(raw.attrs['output_dir']);assert len(raw)==200 and raw.groupby('scenario').seed.nunique().eq(20).all()
    controls=[dict(j) for j in reversed(jobs) if j['seed']==42];sr,ss=execute_jobs(controls,aggregate_results,workers=1,output_root=base/'serial_controls',save_full_timeseries=True);control_dir=Path(sr.attrs['output_dir'])
    pd.testing.assert_frame_equal(raw[raw.seed==42].sort_values('scenario').reset_index(drop=True),sr.sort_values('scenario').reset_index(drop=True))
    for index,c in enumerate(controls):
        primary=c['extra']['scenario']*20
        a=pd.read_csv(directory/'runs'/f'{primary:07d}_timeseries.csv',float_precision='round_trip');b=pd.read_csv(control_dir/'runs'/f'{index:07d}_timeseries.csv',float_precision='round_trip');pd.testing.assert_frame_equal(a,b)
    resumed,_=execute_jobs(jobs,aggregate_results,workers=1,resume=directory,save_full_timeseries=True);pd.testing.assert_frame_equal(raw,resumed)
    liq,recovery,financial=summarize_pilot(raw,directory,base)
    for folder in [directory,control_dir]:
        for marker in (folder/'runs').glob('*.json'):
            data=json.loads(marker.read_text());stem=marker.stem
            for suffix,key in [('.csv','csv_sha256'),('_timeseries.csv','timeseries_sha256'),('_empty_spells.csv','empty_spells_sha256')]:assert _digest(marker.with_name(stem+suffix))==data[key]
    assert _source_hashes()==model and all(_digest(ROOT/name)==h for name,h in old.items())
    verification=dict(research_runs=200,control_repeats=10,completed=int(raw.status.eq('completed').sum()),failed=int(raw.status.eq('failed').sum()),invalid=int((~raw.data_valid).sum()),serial_parallel_rows_and_all_series_exact=10,resume_recomputed=0,resume_rows_exact=True,unique_seeds_per_scenario=20,independent_initialization='each job resets Python and NumPy RNG; process scheduling and reversed control order do not change checked results',gap_and_recovery_checked_from_all_saved_series=True,checkpoint_checksums_verified=210,model_source_unchanged=True,old_result_files_unchanged=len(old),price_recovered=int(recovery.price_recovered.sum()),liquidity_recovered=int(recovery.liquidity_recovered.sum()),price_recovery_without_two_sided=int(recovery.price_recovery_without_two_sided.sum()),price_recovery_with_fallback=int(recovery.price_recovery_with_fallback.sum()),price_recovery_with_empty_shock=int(recovery.price_recovery_with_empty_shock.sum()),two_sided_recoveries=int(liq[liq.market=='A+B'].n_recoveries.sum()),parallel_results=str(directory),serial_controls=str(control_dir),elapsed_seconds=time.perf_counter()-started)
    (base/'verification.json').write_text(json.dumps(verification,indent=2));print('OUTPUT',base,flush=True);print(json.dumps(verification,indent=2),flush=True)
if __name__=='__main__':main()
