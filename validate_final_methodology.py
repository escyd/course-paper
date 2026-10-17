"""Reanalyse immutable pilot, run only six-job smoke fixtures, estimate final resources."""
import json,time,shutil,hashlib
from pathlib import Path
from datetime import datetime
from unittest.mock import patch
import numpy as np
import pandas as pd
from experiment_methodology import liquidity_fields,methodology_tables,write_methodology_outputs
from experiment_runner import execute_jobs,_run_job,_digest
from experiments import aggregate_results
from final_experiment import prepare,DEFAULT_CONFIG,main as final_main
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'tables/methodology_validation/20261017T165633Z'
PILOT=ROOT/'tables/two_market_pilot/20261017T164518Z'


def main():
    design=json.loads(DEFAULT_CONFIG.read_text());scenarios,jobs=prepare(design)
    pd.DataFrame(scenarios).to_csv(OUT/'final_scenarios.csv',index=False)
    assert len(jobs)==50000 and all(len({j['seed'] for j in jobs if j['extra']['scenario_id']==s['scenario_id']})==500 for s in scenarios)
    plan=final_main(['--dry-run']);(OUT/'final_plan.json').write_text(json.dumps(plan,indent=2));del jobs
    lookup={(s['tau'],s['beta'],s['fx_cost']):s['scenario_id'] for s in scenarios}
    pilot_ver=json.loads((PILOT/'verification.json').read_text());directory=Path(pilot_ver['parallel_results']);old=pd.read_csv(directory/'raw_results.csv',float_precision='round_trip');fields=[]
    for i,r in enumerate(old.itertuples()):
        frame=pd.read_csv(directory/'runs'/f'{i:07d}_timeseries.csv',float_precision='round_trip');f=liquidity_fields({c:frame[c].tolist() for c in frame},r.shock_iteration,r.steps,r.status)
        pre=bool(frame.liquid_a.iloc[199] and frame.liquid_b.iloc[199]);failed=bool((~(frame.liquid_a.iloc[:200]&frame.liquid_b.iloc[:200])).any())
        assert f['pre_shock_two_sided']==pre and f['pre_shock_liquidity_failure']==failed
        if failed:assert f['first_liquidity_loss_step']==int(np.flatnonzero(~(frame.liquid_a&frame.liquid_b).to_numpy())[0])
        fields.append(f)
    enriched=old.copy()
    for c in fields[0]:enriched[c]=[x[c] for x in fields]
    enriched['scenario_id']=[lookup[(r.tau,r.beta,r.fx_cost)] for r in old.itertuples()]
    pd.testing.assert_frame_equal(enriched[old.columns],old,check_exact=True)
    start=time.perf_counter();analysis=write_methodology_outputs(enriched,OUT);pilot_analysis_seconds=time.perf_counter()-start
    counts=pd.read_csv(analysis/'cohort_counts.csv');rates=pd.read_csv(analysis/'rate_statistics.csv');desc=pd.read_csv(analysis/'descriptive_statistics.csv')
    allcounts=counts[counts.cohort=='all_attempts'];conditional=counts[counts.cohort=='two_sided_before_shock']
    assert allcounts.n_attempted.sum()==200 and conditional.n_attempted.sum()==82
    assert enriched.pre_shock_liquidity_failure.sum()==118
    assert enriched.post_shock_restoration_observed.sum()==0
    assert (allcounts.n_conditional_selected+allcounts.n_conditional_excluded_not_two_sided+allcounts.n_conditional_excluded_unknown).eq(allcounts.n_attempted).all()
    oldsummary=pd.read_csv(PILOT/'financial_by_scenario.csv',float_precision='round_trip')
    for r in oldsummary.itertuples():
        scenario=lookup[(r.tau,r.beta,r.fx_cost)]
        for metric,col in [('price_recovery_rate','recovery_time_fraction_completed_valid'),('liquidity_aware_recovery_rate','liquidity_recovery_time_fraction_completed_valid')]:
            x=rates[(rates.scenario_id==scenario)&(rates.cohort=='all_attempts')&(rates.metric==metric)].iloc[0]
            assert x.estimate==getattr(r,col) and x.denominator==20
        for metric,col in [('avg_gap_after','avg_gap_after_mean'),('recovery_time','recovery_time_conditional_mean'),('liquidity_recovery_time','liquidity_recovery_time_conditional_mean')]:
            x=desc[(desc.scenario_id==scenario)&(desc.cohort=='all_attempts')&(desc.metric==metric)].iloc[0]
            assert np.isclose(x['mean'],getattr(r,col),rtol=1e-12,atol=1e-12)
    # Six unique simulations, repeated for scheduling checks; never call the final launcher with --run.
    small=[]
    for t,b,f in [(.3,.2,.003),(1.,.6,.01)]:
        for seed in [42,43,44]:small.append(dict(tau=t,beta=b,fx_cost=f,run_id=seed-42,seed=seed,steps=500,shock_iteration=200,extra=dict(scenario_id=lookup[(t,b,f)])))
    raw,_=execute_jobs(small,aggregate_results,workers=4,output_root=OUT/'smoke_parallel',save_full_timeseries=True)
    serial,_=execute_jobs(small,aggregate_results,workers=1,output_root=OUT/'smoke_serial',save_full_timeseries=True)
    pd.testing.assert_frame_equal(raw,serial,check_exact=True)
    parallel=Path(raw.attrs['output_dir']);sd=Path(serial.attrs['output_dir'])
    for index,j in enumerate(small):
        path=parallel/'runs'/f'{index:07d}_timeseries.csv';assert path.read_bytes()==(sd/'runs'/path.name).read_bytes()
        original_index=int(old[(old.tau==j['tau'])&(old.beta==j['beta'])&(old.fx_cost==j['fx_cost'])&(old.seed==j['seed'])].index[0])
        assert path.read_bytes()==(directory/'runs'/f'{original_index:07d}_timeseries.csv').read_bytes()
    resume=OUT/'partial_resume';(resume/'runs').mkdir(parents=True)
    shutil.copy2(parallel/'parameters.json',resume/'parameters.json')
    for i in range(2):
        for suffix in ['.csv','.json','_timeseries.csv','_empty_spells.csv']:
            name=f'{i:07d}'+suffix;shutil.copy2(parallel/'runs'/name,resume/'runs'/name)
    before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in (resume/'runs').iterdir()}
    with patch('experiment_runner._run_job',wraps=_run_job) as worker:
        continued,_=execute_jobs(small,aggregate_results,workers=1,resume=resume,save_full_timeseries=True)
        assert len(worker.call_args_list)==4
    pd.testing.assert_frame_equal(raw,continued,check_exact=True)
    for name,(data,mtime) in before.items():assert (resume/'runs'/name).read_bytes()==data and (resume/'runs'/name).stat().st_mtime_ns==mtime
    report1=write_methodology_outputs(raw,parallel);report2=write_methodology_outputs(continued,resume)
    for name in ['main_metrics.csv','cohort_counts.csv','descriptive_statistics.csv','rate_statistics.csv','conditional_main_metrics.csv','liquidity_periods.csv']:
        assert (report1/name).read_bytes()==(report2/name).read_bytes()
    # Runtime-only benchmark at final per-scenario size, never published as scientific observations.
    timing=enriched[enriched.scenario==0].iloc[np.arange(500)%20].copy()
    durations=[]
    for _ in range(3):
        start=time.perf_counter();methodology_tables(timing,1999);durations.append(time.perf_counter()-start)
    summary_estimate=100*max(durations)
    meta=json.loads((directory/'parameters.json').read_text());created=datetime.fromisoformat(meta['created_utc']).timestamp()
    pilot_execution_seconds=max(p.stat().st_mtime for p in (directory/'runs').glob('*.json'))-created
    base_sim_seconds=pilot_execution_seconds*50000/200
    checkpoint_files=list((parallel/'runs').iterdir());logical=sum(p.stat().st_size for p in checkpoint_files)/6
    allocated=sum(p.stat().st_blocks*512 for p in checkpoint_files)/6
    report_bytes=sum(p.stat().st_size for p in report1.iterdir() if p.is_file())/6
    extra_metadata_bytes=(parallel/'parameters.json').stat().st_size/6*50000
    estimated_bytes=50000*(max(logical,allocated)+report_bytes)+extra_metadata_bytes
    free=shutil.disk_usage(ROOT).free
    resources=dict(reference_pilot_n=200,reference_pilot_parallel_seconds=pilot_execution_seconds,reference_parallel_runs_per_second=200/pilot_execution_seconds,
                   baseline_50000_simulation_minutes=base_sim_seconds/60,new_extraction_and_io_uncertainty='5-25% allowance; untested directory scaling at 200000 checkpoint files',
                   benchmark_synthetic_timing_only=True,benchmark_per_scenario_n=500,benchmark_methodology_seconds=durations,
                   estimated_100_scenario_analysis_seconds_upper=summary_estimate,
                   estimated_total_minutes_low=(base_sim_seconds*1.05+summary_estimate)/60,
                   estimated_total_minutes_high=(base_sim_seconds*1.25+summary_estimate)/60,
                   measured_checkpoint_logical_bytes_per_run=logical,measured_checkpoint_allocated_bytes_per_run=allocated,
                   estimated_output_GiB=estimated_bytes/2**30,recommended_free_GiB=estimated_bytes*1.3/2**30,
                   available_GiB=free/2**30,output_file_count_approx=200000,
                   limitations='Extrapolation on this Mac from 10 pilot scenarios; full-grid cache, contention, sleep and thermal throttling unmeasured; no full experiment executed.')
    (OUT/'resource_estimate.json').write_text(json.dumps(resources,indent=2))
    (OUT/'methodology_validation.json').write_text(json.dumps(dict(pilot_rows=200,pilot_primary_metrics_unchanged=True,full_cohort=200,conditional_cohort=82,pre_shock_failure_count=118,
        denominators_checked=True,short_unique_simulations=6,serial_repeats=6,resume_recomputed=4,resume_skipped_unchanged=2,
        serial_parallel_timeseries_exact=True,short_series_match_saved_pilot=True,methodology_reports_match_after_resume=True,
        final_grid_prepared=50000,final_grid_launched=0,pilot_report=str(analysis),smoke_report=str(report1),pilot_analysis_seconds=pilot_analysis_seconds),indent=2))
    old_files=json.loads((OUT/'before_manifest.json').read_text());assert all(_digest(ROOT/name)==value for name,value in old_files.items())
    models=json.loads((OUT/'model_hashes.json').read_text());assert all(_digest(ROOT/name)==value for name,value in models.items())
    print('VALIDATION',OUT,flush=True);print(json.dumps(resources,indent=2),flush=True)
if __name__=='__main__':main()
