"""Approved run supervision only: save a technical failure, then stop scheduling."""
import argparse,json,sys,time,traceback,shutil
from pathlib import Path
from datetime import datetime,timezone
import experiment_runner
from final_experiment import main as launch,DEFAULT_CONFIG
ROOT=Path(__file__).resolve().parent
class TechnicalRunError(RuntimeError):pass

class Tee:
    def __init__(self,terminal,log):self.terminal,self.log=terminal,log
    def write(self,text):self.terminal.write(text);self.log.write(text);self.log.flush()
    def flush(self):self.terminal.flush();self.log.flush()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--resume',type=Path);args=parser.parse_args()
    if shutil.disk_usage(ROOT).free<5*2**30:raise RuntimeError('Less than 5 GiB free before launch')
    audit=ROOT/'tables/final_run_audit'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');audit.mkdir(parents=True,exist_ok=False)
    log=(audit/'execution.log').open('w');terminal=sys.stdout;sys.stdout=Tee(terminal,log)
    original=experiment_runner._atomic_json;digest=experiment_runner._digest
    config=json.loads(DEFAULT_CONFIG.read_text());started=time.perf_counter()
    old={str(p.relative_to(ROOT)):digest(p) for p in ROOT.glob('tables/**/*') if p.is_file() and audit not in p.parents and 'final_experiments_v2' not in p.parts}
    original(old,audit/'prior_results_sha256.json')
    plan=dict(design=config,config_sha256=digest(DEFAULT_CONFIG),protocol_sha256=digest(ROOT/'docs/final_experiment_protocol.md'),
              source_sha256=experiment_runner._source_hashes(),supervisor_sha256=digest(Path(__file__)),resume=str(args.resume) if args.resume else None,
              paired_contrasts='235 adjacent one-factor level changes, other factors fixed; matched seeds; no exclusion for pre-shock liquidity loss',
              paired_endpoints=['avg_gap_after','final_gap','price_recovery_rate','liquidity_aware_recovery_rate','pre_shock_liquidity_failure_rate','recovery_time','liquidity_recovery_time'],
              paired_times='conditional on recovery in both members of pair; report joint-eligibility and discordant outcomes',
              paired_uncertainty='pointwise paired seed bootstrap 95%, B=1999; exploratory, not simultaneous; no automated significance claims',
              failure_policy='checkpoint the first failed/invalid row then raise; already submitted bounded jobs may finish during shutdown; no model edits')
    original(plan,audit/'preflight_and_pairing_plan.json')
    state=dict(directory=str(args.resume) if args.resume else None,completed=len(list((args.resume/'runs').glob('*.json'))) if args.resume else 0,total=50000)
    def guarded(value,path):
        original(value,path)
        if path.name=='parameters.json':state['directory']=str(path.parent)
        if path.parent.name=='runs' and isinstance(value,dict) and 'row' in value:
            row=value['row'];state['completed']+=1
            if row.get('status')!='completed' or not row.get('data_valid',False):
                error=dict(checkpoint=str(path),row=row,progress=state,elapsed_seconds=time.perf_counter()-started)
                original(error,audit/'technical_error.json')
                raise TechnicalRunError(f'Technical failure saved at {path}; stopping without model changes')
            if state['completed']%500==0 or state['completed']==50000:
                progress=dict(state,elapsed_seconds=time.perf_counter()-started)
                original(progress,audit/'progress.json')
                print(f"Progress {state['completed']}/50000; elapsed {(time.perf_counter()-started)/60:.1f} min",flush=True)
    experiment_runner._atomic_json=guarded
    try:
        print('AUDIT',audit,flush=True)
        directory=launch(['--run']+(['--resume',str(args.resume)] if args.resume else []))
        original(dict(status='simulation_and_scenario_reporting_complete',directory=str(directory),elapsed_seconds=time.perf_counter()-started),audit/'execution_complete.json')
    except BaseException:
        original(dict(status='stopped',traceback=traceback.format_exc(),progress=state),audit/'stopped.json');raise
    finally:
        experiment_runner._atomic_json=original;sys.stdout=terminal;log.close()
if __name__=='__main__':main()
