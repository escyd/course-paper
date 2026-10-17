"""Read-only liquidity events and prespecified scenario/cohort reporting."""
import hashlib
import json
from pathlib import Path
import uuid
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
from AgentBasedModel.utils.liquidity import empty_spells

VERSION='v2-methodology-1'
KEYS=['tau','beta','fx_cost']
DESCRIPTIVE=['avg_gap_before','avg_gap_after','max_gap_after','final_gap',
             'recovery_time','liquidity_recovery_time']


def liquidity_fields(info,shock,requested,status='completed'):
    """A/B quotes at t=shock-1 are immediately before the shock at t=shock."""
    def get(name):
        return list(info.get(name,[])) if isinstance(info,dict) else list(getattr(info,name,[])) if info is not None else []
    a,b=get('liquid_a'),get('liquid_b');n=len(get('prices_a'))
    available=n>0 and len(a)==len(b)==n and all(isinstance(x,(bool,np.bool_)) for x in a+b)
    fields=dict(liquidity_history_available=available,pre_shock_history_complete=False,
        pre_shock_two_sided=None,pre_shock_liquidity_a=None,pre_shock_liquidity_b=None,
        pre_shock_liquidity_failure=None,pre_shock_failure_observed=False,
        first_liquidity_loss_step=None,first_liquidity_loss_time_from_shock=None,
        post_shock_liquidity_restored=None,post_shock_restoration_observed=False,
        first_post_shock_restoration_step=None,system_empty_spells_json='[]',
        system_liquidity_recoveries=0,system_max_empty_duration=0)
    if not available:return fields
    two=np.asarray(a)&np.asarray(b);missing=~two;pre=missing[:min(shock,n)]
    full_pre=n>=shock and shock>0
    event=bool(pre.any());loss=np.flatnonzero(missing)
    restored=np.flatnonzero(missing[:-1]&two[1:])+1
    post=restored[restored>=shock]
    fields.update(pre_shock_history_complete=full_pre,
        pre_shock_two_sided=bool(two[shock-1]) if full_pre else None,
        pre_shock_liquidity_failure=event if event or full_pre else None,
        pre_shock_failure_observed=event,
        first_liquidity_loss_step=int(loss[0]) if len(loss) else None,
        first_liquidity_loss_time_from_shock=int(loss[0])-shock if len(loss) else None,
        post_shock_liquidity_restored=True if len(post) else False if status=='completed' and n==requested else None,
        post_shock_restoration_observed=bool(len(post)),
        first_post_shock_restoration_step=int(post[0]) if len(post) else None,
        system_empty_spells_json=json.dumps(empty_spells(missing.tolist())),
        system_liquidity_recoveries=len(restored),
        system_max_empty_duration=max([s['duration'] for s in empty_spells(missing.tolist())],default=0))
    if full_pre:
        for name,liquid in [('a',a),('b',b)]:
            bids,asks=get('bid_empty_'+name),get('ask_empty_'+name)
            if len(bids)==len(asks)==n:
                bid,ask=bids[shock-1],asks[shock-1]
                label='empty_both' if bid and ask else 'ask_only' if bid else 'bid_only' if ask else 'two_sided'
            else:label='two_sided' if liquid[shock-1] else 'not_two_sided'
            fields['pre_shock_liquidity_'+name]=label
    return fields


def _bootstrap(values,token,reps):
    values=np.asarray(values,dtype=float);values=values[np.isfinite(values)]
    if not len(values):return dict(n=0,mean=np.nan,median=np.nan,mean_ci_low=np.nan,mean_ci_high=np.nan,median_ci_low=np.nan,median_ci_high=np.nan)
    result=dict(n=len(values),mean=float(values.mean()),median=float(np.median(values)),mean_ci_low=np.nan,mean_ci_high=np.nan,median_ci_low=np.nan,median_ci_high=np.nan)
    if len(values)<2:return result
    seed=int.from_bytes(hashlib.sha256(repr(token).encode()).digest()[:8],'big');rng=np.random.default_rng(seed)
    means=[];medians=[]
    for offset in range(0,reps,100):
        sample=values[rng.integers(0,len(values),(min(100,reps-offset),len(values)))]
        means.extend(sample.mean(axis=1));medians.extend(np.median(sample,axis=1))
    result['mean_ci_low'],result['mean_ci_high']=map(float,np.quantile(means,[.025,.975]))
    result['median_ci_low'],result['median_ci_high']=map(float,np.quantile(medians,[.025,.975]))
    return result


def _rate(key,cohort,metric,numerator,denominator,estimand):
    lo,hi=proportion_confint(numerator,denominator,method='wilson') if denominator else (np.nan,np.nan)
    return dict(**key,cohort=cohort,metric=metric,numerator=int(numerator),denominator=int(denominator),
                estimate=numerator/denominator if denominator else np.nan,ci_low=lo,ci_high=hi,
                estimand=estimand,ci_method='Wilson pointwise 95%')


def methodology_tables(raw,reps=1999):
    """Every attempted row stays in 'all_attempts'; selected cohort retains failures."""
    rates=[];descriptive=[];counts=[]
    for key,all_rows in raw.groupby(KEYS,dropna=False,sort=True):
        keys=dict(zip(KEYS,key))
        if 'scenario_id' in all_rows:keys['scenario_id']=all_rows.scenario_id.iloc[0]
        pre_selected=all_rows.pre_shock_history_complete.eq(True)&all_rows.pre_shock_two_sided.eq(True)
        for cohort,part in [('all_attempts',all_rows),('two_sided_before_shock',all_rows.loc[pre_selected])]:
            valid=part.status.eq('completed')&part.data_valid.eq(True)
            pre_known=part.pre_shock_history_complete.eq(True)
            c=dict(**keys,cohort=cohort,n_attempted=len(part),n_completed=int(part.status.eq('completed').sum()),
                n_failed=int(part.status.eq('failed').sum()),n_invalid=int((~part.data_valid.eq(True)).sum()),
                n_status_unknown=int((~part.status.isin(['completed','failed'])).sum()),n_completed_valid=int(valid.sum()),
                n_pre_shock_window_observed=int(pre_known.sum()),n_pre_shock_window_unknown=int((~pre_known).sum()),
                n_pre_shock_failure_observed_in_incomplete_windows=int(((~pre_known)&part.pre_shock_failure_observed.eq(True)).sum()),
                n_parent_attempts=len(all_rows),n_conditional_selected=int(pre_selected.sum()),
                n_conditional_excluded_not_two_sided=int((all_rows.pre_shock_history_complete.eq(True)&all_rows.pre_shock_two_sided.eq(False)).sum()),
                n_conditional_excluded_unknown=int((~all_rows.pre_shock_history_complete.eq(True)).sum()))
            rates.append(_rate(keys,cohort,'completion_rate',int(part.status.eq('completed').sum()),len(part),'completed / all attempted in cohort'))
            rates.append(_rate(keys,cohort,'failure_rate',int(part.status.eq('failed').sum()),len(part),'technical failures / all attempted in cohort'))
            failures=int(part.loc[pre_known,'pre_shock_liquidity_failure'].eq(True).sum())
            rates.append(_rate(keys,cohort,'pre_shock_liquidity_failure_rate',failures,int(pre_known.sum()),'any sampled loss at t < shock / runs with entire pre-shock window observed, including later failures'))
            for metric,prefix in [('recovery_time','price'),('liquidity_recovery_time','liquidity_aware')]:
                labels=part[metric+'_status'];rec=labels.eq('recovered');cen=labels.eq('censored');eligible=rec|cen
                c[metric+'_n_recovered']=int(rec.sum());c[metric+'_n_censored']=int(cen.sum());c[metric+'_n_unavailable']=int((~eligible).sum());c[metric+'_n_event_before_failure']=int(part[metric+'_event_before_failure'].sum())
                assert int(rec.sum()+cen.sum()+(~eligible).sum())==len(part)
                rates.append(_rate(keys,cohort,prefix+'_recovery_rate',int(rec.sum()),int(eligible.sum()),'recovered completed valid / completed valid endpoint observations'))
                rates.append(_rate(keys,cohort,prefix+'_censoring_rate',int(cen.sum()),int(eligible.sum()),'unrecovered completed valid / completed valid endpoint observations'))
                rates.append(_rate(keys,cohort,prefix+'_completed_valid_and_recovered_rate_all_attempts',int(rec.sum()),len(part),'composite completion + validity + recovery / all attempts; not failure-as-censoring'))
            counts.append(c)
            for metric in DESCRIPTIVE:
                mask=valid.copy();estimand='completed-valid full-horizon outcomes within cohort'
                if metric in ['recovery_time','liquidity_recovery_time']:
                    mask&=part[metric+'_status'].eq('recovered');estimand='time conditional on recovered completed-valid paths; censored times not imputed'
                values=pd.to_numeric(part.loc[mask,metric],errors='coerce').to_numpy()
                values=values[np.isfinite(values)]
                descriptive.append(dict(**keys,cohort=cohort,metric=metric,**_bootstrap(values,(VERSION,key,cohort,metric),reps),
                    n_cohort_attempted=len(part),estimand=estimand,ci_method=f'seed bootstrap percentile pointwise 95%, B={reps}; undefined for n<2'))
    return pd.DataFrame(counts),pd.DataFrame(descriptive),pd.DataFrame(rates)


def write_methodology_outputs(raw,parent,reps=1999):
    """Publish analysis to a new folder; never edit historical run artifacts."""
    out=Path(parent)/('methodology_'+VERSION+'_'+uuid.uuid4().hex[:8]);out.mkdir(parents=True,exist_ok=False)
    from experiment_runner import _atomic_json, _digest
    counts,descriptive,rates=methodology_tables(raw,reps)
    raw.to_csv(out/'main_metrics.csv',index=False)
    counts.to_csv(out/'cohort_counts.csv',index=False);descriptive.to_csv(out/'descriptive_statistics.csv',index=False);rates.to_csv(out/'rate_statistics.csv',index=False)
    raw[raw.pre_shock_history_complete.eq(True)&raw.pre_shock_two_sided.eq(True)].to_csv(out/'conditional_main_metrics.csv',index=False)
    events=[]
    for r in raw.itertuples():
        for market,column in [('A','a_empty_periods_json'),('B','b_empty_periods_json'),('FX','fx_empty_periods_json'),('A+B','system_empty_spells_json')]:
            value=getattr(r,column,'[]')
            for spell in json.loads(value if isinstance(value,str) else '[]'):
                events.append(dict(tau=r.tau,beta=r.beta,fx_cost=r.fx_cost,seed=r.seed,run_id=r.run_id,market=market,status=r.status,**spell))
    pd.DataFrame(events,columns=['tau','beta','fx_cost','seed','run_id','market','status','start','duration','right_censored']).to_csv(out/'liquidity_periods.csv',index=False)
    (out/'methodology.json').write_text(json.dumps(dict(version=VERSION,bootstrap_reps=reps,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cohorts=['all_attempts','two_sided_before_shock'],conditional_caution='Descriptive endogenous selection, not an unbiased shock-effect estimate.',time_index='zero-based; pre-shock = recorded t=shock-1; absence sampled after each multi-market step',censoring='administrative at final recorded t; failed runs unavailable, event-before-failure flagged separately',time_statistics='conditional on recovery; NaN for no recoveries; no horizon substitution',rates='Wilson pointwise 95%; denominators explicit in every row; no complete-window ascertainment silently inferred for failed early paths',multiplicity='pointwise intervals, not simultaneous; no confirmatory multiple-testing decisions',comparison='same seeds paired across scenarios; no pooling scenario rows as independent seeds; no automatic effect inference'),indent=2))
    _atomic_json(dict(files={p.name:_digest(p) for p in out.iterdir() if p.is_file()},methodology_version=VERSION),out/'complete.json')
    return out
