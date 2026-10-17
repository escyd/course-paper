"""Per-seed liquidity sensitivity; never join ACF/LB/ARCH across a time gap."""
import argparse,json,uuid,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from stylized_facts import diagnose
from AgentBasedModel.utils.liquidity import empty_spells,quote_history_summary


def liquid_segments(liquid,burn=100):
    valid=np.asarray(liquid,dtype=bool).copy();valid[:burn]=False
    return [(s['start'],s['start']+s['duration']) for s in empty_spells(valid.tolist())]


def marginal_statistics(r):
    r=np.asarray(r);row=dict(n_returns=len(r),zero_return_fraction=float(np.mean(r==0)) if len(r) else np.nan)
    sd=float(np.std(r,ddof=1)) if len(r)>1 else np.nan
    row['std_log_return']=sd
    row['status']='ok' if len(r)>=4 and np.isfinite(sd) and sd>0 else 'undefined_constant_or_insufficient'
    row['excess_kurtosis']=float(stats.kurtosis(r,fisher=True,bias=False)) if row['status']=='ok' else np.nan
    for k in [2,3,4]:row[f'tail_{k}sigma_frequency']=float(np.mean(abs(r-r.mean())>k*sd)) if row['status']=='ok' else np.nan
    return row


def analyze(study):
    study=Path(study);out=study/('liquidity_analysis_'+uuid.uuid4().hex[:8]);out.mkdir()
    protocol=dict(analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),burn_in=100,acf_lags=40,test_lags=[10,20],confidence=.95,bootstrap_seed=20261017,bootstrap_reps=10000,
        subsets='full; all adjacent two-sided pairs for marginal distributions; all contiguous two-sided segments for time-dependence tests',
        primary_liquid_acf='longest contiguous two-sided segment after global burn-in; earliest if tied; all other segments retained',
        gap_policy='never concatenate discontinuous returns for ACF, Ljung-Box or ARCH-LM',
        returns='log(P[t])-log(P[t-1]); no dividends; prices sampled at original capture',
        fallback='both endpoints lack two-sided quotes; transition returns reported separately',
        interpretation='descriptive paired subset sensitivity, not a causal counterfactual; conditional on sufficient nonconstant observations')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    liquidity=[];marginals=[];segments=[];acfrows=[];primary=[];source=pd.read_csv(study/'per_run_statistics.csv',float_precision='round_trip');full=source[(source.burn_in==100)&(source.status=='ok')]
    for f in sorted(study.glob('steps_10000/seed_*/prices.csv')):
        seed=int(f.parent.name.split('_')[1]);d=pd.read_csv(f,float_precision='round_trip');price=d.price.to_numpy();liquid=~(d.bid_empty.to_numpy()|d.ask_empty.to_numpy());r=np.diff(np.log(price));eligible=np.arange(1,len(price))>=101
        liquidity.append(dict(seed=seed,n_prices=len(d),**quote_history_summary(d.bid_empty.tolist(),d.ask_empty.tolist())))
        masks=dict(full=eligible,two_sided_pairs=eligible&liquid[1:]&liquid[:-1],fallback_internal=eligible&~liquid[1:]&~liquid[:-1],boundary=eligible&(liquid[1:]!=liquid[:-1]))
        for name,mask in masks.items():marginals.append(dict(seed=seed,subset=name,**marginal_statistics(r[mask])))
        blocks=liquid_segments(liquid,100);longest=max(blocks,key=lambda z:z[1]-z[0]) if blocks else None
        for start,end in blocks:
            base=dict(seed=seed,start=start,end_exclusive=end,n_prices=end-start,primary=(start,end)==longest,status='ok')
            try:
                row,table=diagnose(price[start:end],0,(10,20),40);base.update(row)
                table['seed']=seed;table['subset']='longest_two_sided' if (start,end)==longest else 'other_two_sided';table['start']=start;acfrows.append(table)
                if (start,end)==longest:primary.append(dict(row,seed=seed,subset='longest_two_sided'))
            except ValueError as exc:base.update(status='unavailable',error=str(exc))
            segments.append(base)
    raw=pd.concat([full.assign(subset='full'),pd.DataFrame(primary)],ignore_index=True)
    # Separate BH families per subset and test/transform/lag, across seeds.
    for name in [c for c in raw.columns if c.endswith('_p') and not c.endswith('_f_p')]:
        raw[name+'_bh_q']=np.nan
        for subset,part in raw.groupby('subset'):
            good=part[name].dropna();raw.loc[good.index,name+'_bh_q']=multipletests(good.to_numpy(),method='fdr_bh')[1]
    metrics=['excess_kurtosis','tail_2sigma_frequency','tail_3sigma_frequency','tail_4sigma_frequency','zero_return_fraction','acf_returns_1','acf_abs_returns_1','acf_squared_returns_1']
    ci=[];rng=np.random.default_rng(20261017)
    for subset,part in raw.groupby('subset'):
        for metric in metrics:
            x=part[metric].dropna().to_numpy();draw=np.median(x[rng.integers(0,len(x),(10000,len(x)))],axis=1);lo,hi=np.quantile(draw,[.025,.975]);ci.append(dict(subset=subset,metric=metric,n_valid=len(x),median=np.median(x),ci_low=lo,ci_high=hi))
    paired=raw[raw.subset=='full'].merge(raw[raw.subset=='longest_two_sided'],on='seed',suffixes=('_full','_liquid'))
    delta=[]
    affected={row['seed'] for row in liquidity if row['empty_steps']>0}
    for cohort,part in [('all_seeds',paired),('seeds_with_fallback',paired[paired.seed.isin(affected)])]:
        for metric in metrics:
            x=(part[metric+'_full']-part[metric+'_liquid']).to_numpy()
            if not len(x):continue
            draws=x[rng.integers(0,len(x),(10000,len(x)))]
            for estimand,point,draw in [('median',np.median(x),np.median(draws,axis=1)),('mean',np.mean(x),np.mean(draws,axis=1))]:
                lo,hi=np.quantile(draw,[.025,.975]);delta.append(dict(cohort=cohort,metric=metric,estimand=estimand,n_pairs=len(x),full_minus_liquid=point,ci_low=lo,ci_high=hi))
    pd.DataFrame(liquidity).to_csv(out/'liquidity_per_seed.csv',index=False);pd.DataFrame(marginals).to_csv(out/'return_regimes.csv',index=False);pd.DataFrame(segments).to_csv(out/'two_sided_segments.csv',index=False)
    raw.to_csv(out/'comparison_per_seed.csv',index=False);pd.DataFrame(ci).to_csv(out/'median_ci.csv',index=False);pd.DataFrame(delta).to_csv(out/'paired_sensitivity_ci.csv',index=False)
    tests=[]
    for subset,part in raw.groupby('subset'):
        for name in [c for c in raw if c.endswith('_bh_q')]:tests.append(dict(subset=subset,test=name,n_valid=part[name].notna().sum(),n_reject=(part[name]<.05).sum()))
    pd.DataFrame(tests).to_csv(out/'test_rejections.csv',index=False)
    fa=pd.read_csv(study/'acf_per_run.csv');fa=fa[(fa.burn_in==100)].assign(subset='full',start=100);allacf=pd.concat([fa,*acfrows],ignore_index=True);allacf.to_csv(out/'acf_by_segment.csv',index=False)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.size':9}):
        fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
        for ax,metric,title in zip(axes[0],['excess_kurtosis','tail_3sigma_frequency','zero_return_fraction'],['Excess kurtosis','3-sigma exceedance frequency','Zero-return fraction']):
            groups=[raw[raw.subset==s][metric].to_numpy() for s in ['full','longest_two_sided']];ax.boxplot(groups,tick_labels=['Full','Two-sided segment']);ax.set_title(title)
        for ax,transform in zip(axes[1],['returns','abs_returns','squared_returns']):
            for subset,label in [('full','Full'),('longest_two_sided','Two-sided segment')]:
                part=allacf[(allacf.subset==subset)&(allacf['transform']==transform)&(allacf.lag>0)];group=part.groupby('lag').acf;med=group.median();lo=group.quantile(.25);hi=group.quantile(.75);ax.plot(med.index,med,label=label);ax.fill_between(med.index,lo,hi,alpha=.15)
            ax.axhline(0,c='black',lw=.5);ax.set(title=transform+' ACF',xlabel='Lag',ylabel='Median and IQR across seeds');ax.legend()
        fig.suptitle('Same seeds; 10,000 steps; burn-in 100; no ACF across liquidity gaps')
        fig.savefig(out/'liquidity_sensitivity.png',dpi=200);fig.savefig(out/'liquidity_sensitivity.pdf');plt.close(fig)
    print('ANALYSIS',out,flush=True)
    return out
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('study');analyze(parser.parse_args().study)
