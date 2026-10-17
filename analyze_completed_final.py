"""Verify every final artifact and compute prespecified adjacent paired comparisons."""
import argparse,hashlib,json,time,uuid
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as np
import pandas as pd
from experiment_runner import _digest,_source_hashes
from experiment_methodology import methodology_tables,_bootstrap

G_DIR=None;G_RAW=None;G_JOBS=None

def init_verifier(directory):
    global G_DIR,G_RAW,G_JOBS
    G_DIR=Path(directory)
    G_RAW=pd.read_csv(G_DIR/'raw_results.csv',float_precision='round_trip')
    G_JOBS=json.loads((G_DIR/'parameters.json').read_text())['config']['jobs']


def equal_cell(expected,actual):
    if expected is None or (isinstance(expected,float) and np.isnan(expected)):
        return pd.isna(actual)
    if isinstance(expected,str) and expected=='':return pd.isna(actual) or actual==''
    return expected==actual


def verify_one(index):
    marker=G_DIR/'runs'/f'{index:07d}.json';saved=json.loads(marker.read_text());row=saved['row'];raw=G_RAW.iloc[index];job=G_JOBS[index]
    for suffix,key in [('.csv','csv_sha256'),('_timeseries.csv','timeseries_sha256'),('_empty_spells.csv','empty_spells_sha256')]:
        assert _digest(marker.with_name(marker.stem+suffix))==saved[key],(index,key)
    for name,value in row.items():assert equal_cell(value,raw[name]),(index,name)
    for name in ['seed','run_id','tau','beta','fx_cost','steps','shock_iteration']:assert row[name]==job[name],(index,name)
    assert row['scenario_id']==job['extra']['scenario_id'] and row['design_sha256']==job['extra']['design_sha256']
    assert row['status']=='completed' and row['data_valid'] and row['steps_completed']==500
    d=pd.read_csv(marker.with_name(marker.stem+'_timeseries.csv'),float_precision='round_trip');assert len(d)==500
    p=d[['prices_a','prices_b','fx_rates']].to_numpy();assert np.isfinite(p).all() and (p>0).all()
    gap=np.abs(p[:,0]-p[:,1]/np.maximum(p[:,2],1e-9));assert np.array_equal(gap,d.price_gap.to_numpy())
    for name,value in [('avg_gap_before',gap[150:200].mean()),('avg_gap_after',gap[200:250].mean()),('max_gap_after',gap[200:250].max()),('final_gap',gap[-1])]:
        assert np.isclose(row[name],value,rtol=1e-12,atol=1e-12),(index,name)
    # The model stores Python floats in a list. Python 3.12 sum(float-list)
    # uses different arithmetic from sum(np.float64-array). Preserve the
    # original scalar types, slice, and operation order exactly.
    gap_list=gap.tolist()
    both=d.liquid_a.to_numpy()&d.liquid_b.to_numpy();threshold=sum(gap_list[150:200])/len(gap_list[150:200])+.5
    for metric,condition in [('recovery_time',gap<=threshold),('liquidity_recovery_time',(gap<=threshold)&both)]:
        hits=np.flatnonzero(condition[200:]);expected=int(hits[0]) if len(hits) else np.nan
        assert equal_cell(expected,row[metric]),(index,metric)
        assert row[metric+'_status']==('recovered' if len(hits) else 'censored')
        assert row[metric+'_censored']==(not len(hits))
        assert row[metric+'_followup']==299 and not row[metric+'_event_before_failure']
    assert row['pre_shock_two_sided']==bool(both[199])
    assert row['pre_shock_liquidity_failure']==bool((~both[:200]).any())
    first=np.flatnonzero(~both);assert equal_cell(int(first[0]) if len(first) else None,row['first_liquidity_loss_step'])
    restored=np.flatnonzero((~both[:-1])&both[1:])+1
    assert row['post_shock_restoration_observed']==bool((restored>=200).any())
    hits=np.flatnonzero((gap<=threshold)[200:]);t=200+int(hits[0]) if len(hits) else None
    return dict(seed=row['seed'],scenario_id=row['scenario_id'],marker_sha256=_digest(marker),pre_failure=bool(row['pre_shock_liquidity_failure']),
                price_recovered=bool(len(hits)),liquidity_recovered=row['liquidity_recovery_time_status']=='recovered',
                price_recovery_without_liquidity=bool(not both[t]) if t is not None else False,
                price_recovery_empty_shock=bool((not d.liquid_a.iloc[t] and not d.fallback_a.iloc[t]) or (not d.liquid_b.iloc[t] and not d.fallback_b.iloc[t])) if t is not None else False,
                price_recovery_fallback=bool(d.fallback_a.iloc[t] or d.fallback_b.iloc[t]) if t is not None else False,
                post_shock_restored=bool((restored>=200).any()))


def comparisons(raw,config,reps):
    levels={name:config[name+'_values'] for name in ['tau','beta','fx_cost']}
    scenario=raw[['scenario_id','tau','beta','fx_cost']].drop_duplicates();lookup={(r.tau,r.beta,r.fx_cost):r.scenario_id for r in scenario.itertuples()}
    groups={key:part.set_index('seed').sort_index() for key,part in raw.groupby('scenario_id')};rows=[]
    endpoints=['avg_gap_after','final_gap','price_recovery_rate','liquidity_aware_recovery_rate','pre_shock_liquidity_failure_rate','recovery_time','liquidity_recovery_time']
    for parameter in ['tau','beta','fx_cost']:
      for low in scenario.itertuples():
        value=getattr(low,parameter);where=levels[parameter].index(value)
        if where==len(levels[parameter])-1:continue
        high_value=levels[parameter][where+1];high_key=tuple(high_value if name==parameter else getattr(low,name) for name in ['tau','beta','fx_cost']);high_id=lookup[high_key]
        a,b=groups[low.scenario_id],groups[high_id];assert a.index.equals(b.index) and len(a)==500
        valid=a.status.eq('completed')&a.data_valid.eq(True)&b.status.eq('completed')&b.data_valid.eq(True)
        for metric in endpoints:
            eligible=valid.copy();selection='all shared completed-valid seeds; pre-shock losses retained'
            if metric in ['recovery_time','liquidity_recovery_time']:
                eligible&=a[metric+'_status'].eq('recovered')&b[metric+'_status'].eq('recovered')
                left=a[metric];right=b[metric];selection='jointly recovered completed-valid pairs; selected conditional estimand, not full-sample shock effect'
            elif metric=='pre_shock_liquidity_failure_rate':
                eligible&=a.pre_shock_history_complete.eq(True)&b.pre_shock_history_complete.eq(True)
                left=a.pre_shock_liquidity_failure.astype(float);right=b.pre_shock_liquidity_failure.astype(float)
            elif metric in ['price_recovery_rate','liquidity_aware_recovery_rate']:
                endpoint='recovery_time' if metric=='price_recovery_rate' else 'liquidity_recovery_time'
                eligible&=a[endpoint+'_status'].isin(['recovered','censored'])&b[endpoint+'_status'].isin(['recovered','censored'])
                left=a[endpoint+'_status'].eq('recovered').astype(float);right=b[endpoint+'_status'].eq('recovered').astype(float)
            else:left=a[metric];right=b[metric]
            values=(right[eligible]-left[eligible]).to_numpy(dtype=float)
            report=_bootstrap(values,('adjacent-paired-v1',low.scenario_id,high_id,metric),reps)
            rows.append(dict(parameter=parameter,low_level=value,high_level=high_value,low_scenario=low.scenario_id,high_scenario=high_id,
                metric=metric,n_shared_seeds=500,n_eligible_pairs=len(values),low_paired_mean=float(left[eligible].mean()),high_paired_mean=float(right[eligible].mean()),
                mean_difference_high_minus_low=report['mean'],median_difference_high_minus_low=report['median'],
                mean_difference_ci_low=report['mean_ci_low'],mean_difference_ci_high=report['mean_ci_high'],median_difference_ci_low=report['median_ci_low'],median_difference_ci_high=report['median_ci_high'],
                constant_observed_differences=bool(len(values)>0 and np.ptp(values)==0),estimand=selection,ci_method=f'pointwise paired seed percentile bootstrap 95%, B={reps}; not simultaneous',
                unit='rate fraction' if metric.endswith('_rate') else 'steps' if metric.endswith('_time') else 'CUR_A'))
    result=pd.DataFrame(rows);assert len(result)==235*7;return result


def main(directory,audit):
    directory=Path(directory);audit=Path(audit);started=time.perf_counter();out=directory/('verified_analysis_'+uuid.uuid4().hex[:8]);out.mkdir(exist_ok=False)
    plan=json.loads((audit/'preflight_and_pairing_plan.json').read_text());meta=json.loads((directory/'parameters.json').read_text())
    assert _source_hashes()==meta['config']['source_sha256']==plan['source_sha256']
    root=Path(__file__).resolve().parent
    assert _digest(root/'configs/final_experiment_v2.json')==plan['config_sha256']
    assert _digest(root/'docs/final_experiment_protocol.md')==plan['protocol_sha256']
    raw=pd.read_csv(directory/'raw_results.csv',float_precision='round_trip');assert len(raw)==50000
    assert raw.groupby('scenario_id').size().eq(500).all() and raw.scenario_id.nunique()==100
    assert raw.groupby('scenario_id').seed.apply(lambda x:sorted(x.tolist())==list(range(42,542))).all()
    assert not raw.duplicated(['scenario_id','seed']).any()
    all_reports=list(directory.glob('methodology_*/complete.json'));assert all_reports
    latest=max(all_reports,key=lambda p:p.stat().st_mtime_ns);report=latest.parent
    reference=json.loads(latest.read_text())['files']
    for marker in all_reports:
        manifest=json.loads(marker.read_text())['files'];assert manifest==reference
        for name,h in manifest.items():assert _digest(marker.parent/name)==h
    checked=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),initializer=init_verifier,initargs=(str(directory),)) as pool:
        for result in pool.map(verify_one,range(50000),chunksize=50):
            checked.append(result)
            if len(checked)%5000==0:print('Integrity and metric checks',len(checked),'/50000',flush=True)
    check=pd.DataFrame(checked);check.to_csv(out/'per_run_integrity_and_sources.csv',index=False)
    expected=methodology_tables(raw,plan['design']['bootstrap_reps'])
    for frame,name in zip(expected,['cohort_counts.csv','descriptive_statistics.csv','rate_statistics.csv']):
        actual=pd.read_csv(report/name,float_precision='round_trip');pd.testing.assert_frame_equal(frame,actual,check_dtype=False,check_exact=True)
    paired=comparisons(raw,plan['design'],plan['design']['bootstrap_reps']);paired.to_csv(out/'paired_comparisons.csv',index=False)
    (out/'paired_analysis.json').write_text(json.dumps(dict(plan=plan['paired_contrasts'],endpoints=plan['paired_endpoints'],contrast_count=235,endpoint_comparison_count=1645,seed_policy='same 500 seeds matched before subtraction',failure_policy='no silent dropping; shared endpoint eligibility reported explicitly',conditional_times=plan['paired_times'],uncertainty=plan['paired_uncertainty'],constant_difference_caution='Degenerate resampling intervals do not establish zero population uncertainty; no automated significance claims.',analysis_sha256=_digest(Path(__file__))),indent=2))
    check.groupby('scenario_id').sum(numeric_only=True).drop(columns=['seed']).to_csv(out/'recovery_sources_by_scenario.csv')
    old=json.loads((audit/'prior_results_sha256.json').read_text());assert all(_digest(root/name)==h for name,h in old.items())
    result=dict(simulations=50000,scenarios=100,seeds_per_scenario=500,completed=int(raw.status.eq('completed').sum()),failed=int(raw.status.eq('failed').sum()),invalid=int((~raw.data_valid.eq(True)).sum()),
        checkpoints_verified=50000,payload_hashes_verified=150000,checkpoint_markers_validated=50000,all_saved_metrics_checked=True,scenario_statistics_recomputed_exact=True,
        price_recovered=int(check.price_recovered.sum()),liquidity_recovered=int(check.liquidity_recovered.sum()),pre_shock_failure=int(check.pre_failure.sum()),
        price_recovery_without_liquidity=int(check.price_recovery_without_liquidity.sum()),price_recovery_empty_shock=int(check.price_recovery_empty_shock.sum()),price_recovery_fallback=int(check.price_recovery_fallback.sum()),post_shock_liquidity_restorations=int(check.post_shock_restored.sum()),
        old_files_unchanged=len(old),source_hashes_unchanged=True,paired_contrasts=235,paired_endpoint_comparisons=1645,main_report=str(report),paired_csv=str(out/'paired_comparisons.csv'),verification_elapsed_seconds=time.perf_counter()-started)
    (out/'verification.json').write_text(json.dumps(result,indent=2))
    (out/'complete.json').write_text(json.dumps(dict(files={p.name:_digest(p) for p in out.iterdir() if p.is_file()}),indent=2))
    print('VERIFIED',out,flush=True);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory');parser.add_argument('audit');args=parser.parse_args();main(args.directory,args.audit)
