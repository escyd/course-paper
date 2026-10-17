"""Prepared experiment launcher. Default is a simulation-free dry run; --run is explicit."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
from AgentBasedModel.version import MODEL_VERSION
from experiment_methodology import VERSION,write_methodology_outputs

ROOT=Path(__file__).resolve().parent
DEFAULT_CONFIG=ROOT/'configs/final_experiment_v2.json'


def prepare(config):
    if config['model_version']!=MODEL_VERSION or config['methodology_version']!=VERSION:
        raise ValueError('Model or methodology version differs')
    if not 0<config['shock_iteration']<config['steps']:raise ValueError('Require 0 < shock < steps')
    if config['n_seeds']<1 or config['workers']<1 or config['bootstrap_reps']<100:raise ValueError('Invalid positive counts')
    if not 0<=config['base_seed']<2**32 or config['base_seed']+config['n_seeds']>2**32:raise ValueError('Invalid seed range')
    values=[config['fx_cost_values'],config['tau_values'],config['beta_values']]
    if any(not v or len(v)!=len(set(v)) for v in values):raise ValueError('Grid values must be nonempty and unique')
    grid=list(itertools.product(*values))
    if len(grid)!=config['expected_scenarios']:raise ValueError('Scenario count differs')
    digest=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    scenarios=[dict(scenario_id=f'S{i:03d}',tau=t,beta=b,fx_cost=f) for i,(f,t,b) in enumerate(grid)]
    jobs=[dict(tau=s['tau'],beta=s['beta'],fx_cost=s['fx_cost'],run_id=i,seed=config['base_seed']+i,
               steps=config['steps'],shock_iteration=config['shock_iteration'],
               extra=dict(scenario_id=s['scenario_id'],design_sha256=digest)) for s in scenarios for i in range(config['n_seeds'])]
    return scenarios,jobs


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--run',action='store_true');mode.add_argument('--dry-run',action='store_true')
    parser.add_argument('--resume',type=Path);args=parser.parse_args(argv)
    config=json.loads(args.config.read_text());scenarios,jobs=prepare(config)
    branch=subprocess.run(['git','branch','--show-current'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    if branch!=config['branch']:raise ValueError(f'Require branch {config["branch"]}; current {branch}')
    if args.resume and not args.run:raise ValueError('--resume requires explicit --run')
    if not args.run:
        plan=dict(mode='dry-run; no simulations executed',model_version=MODEL_VERSION,methodology_version=VERSION,
                  n_scenarios=len(scenarios),seeds_per_scenario=config['n_seeds'],total_simulations=len(jobs),
                  first_seed=config['base_seed'],last_seed=config['base_seed']+config['n_seeds']-1,
                  steps=config['steps'],shock_iteration=config['shock_iteration'],workers=config['workers'],
                  output_root=str(ROOT/config['output_root']),save_full_timeseries=config['save_full_timeseries'])
        print(json.dumps(plan,indent=2));return plan
    from experiment_runner import execute_jobs,_atomic_json
    from experiments import aggregate_results
    raw,_=execute_jobs(jobs,aggregate_results,workers=config['workers'],output_root=ROOT/config['output_root'],
                       resume=args.resume,save_full_timeseries=config['save_full_timeseries'])
    directory=Path(raw.attrs['output_dir'])
    _atomic_json(config,directory/'final_design.json')
    analysis=write_methodology_outputs(raw,directory,config['bootstrap_reps'])
    print('Execution:',directory);print('Methodology outputs:',analysis)
    return directory
if __name__=='__main__':main()
