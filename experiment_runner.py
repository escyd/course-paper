"""Execution and checkpoint storage only; financial calculations stay in experiments."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import uuid
import traceback

import numpy as np
import pandas as pd


def _atomic_csv(frame, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(value, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes():
    root = Path(__file__).resolve().parent
    paths = [root / 'experiments.py', root / 'experiment_runner.py', root / 'recovery_reporting.py',
             root / 'experiment_methodology.py', root / 'final_experiment.py']
    paths += sorted((root / 'AgentBasedModel').rglob('*.py'))
    return {str(p.relative_to(root)): _digest(p) for p in paths}


def _run_job(job):
    from experiments import build_system, set_all_seeds, summarize_run
    from recovery_reporting import METRICS, observation_fields
    from AgentBasedModel.version import MODEL_VERSION
    set_all_seeds(job['seed'])
    sim, error, failure_type = None, '', ''
    status = 'completed'
    try:
        sim = build_system(tau=job['tau'], beta=job['beta'], fx_cost=job['fx_cost'],
                           shock_iteration=job['shock_iteration'], **job.get('agent_kwargs', {}))
        sim.simulate(job['steps'])
    except Exception as exc:
        status, error, failure_type = 'failed', traceback.format_exc(), type(exc).__name__
        if sim is not None and any(not b for m in [*sim.markets.values(), sim.fx_market]
                                   if m is not None for b in m.order_book.values()):
            failure_type = 'empty_order_book'
    row = dict(run_id=job['run_id'], tau=job['tau'], beta=job['beta'], fx_cost=job['fx_cost'],
               shock_iteration=job['shock_iteration'], **{name: np.nan for name in METRICS})
    if sim is not None:
        try:
            row = summarize_run(sim, job['tau'], job['beta'], job['run_id'],
                                job['fx_cost'], shock_iteration=job['shock_iteration'], window=50)
        except Exception:
            status = 'failed'
            failure_type = failure_type or 'summary_error'
            error += traceback.format_exc()
    completed = len(sim.info.prices_a) if sim is not None else 0
    empty_steps = sum(bool(x) for x in getattr(sim.info, 'empty_books', [])) if sim is not None else 0
    data_valid = sim is not None and completed > 0
    if data_valid:
        arrays = [sim.info.prices_a, sim.info.prices_b, sim.info.fx_rates, sim.info.price_gap]
        data_valid = all(len(x) == completed and np.isfinite(x).all() for x in arrays)
        data_valid = data_valid and all(np.all(np.asarray(x) > 0) for x in arrays[:3])
    if status == 'completed' and completed != job['steps']:
        status, failure_type, error = 'failed', 'incomplete_simulation', 'Simulation returned before requested steps'
    row.update(observation_fields(row, status=status, requested=job['steps'], completed=completed,
                                  data_valid=data_valid))
    row.update(seed=job['seed'], steps=job['steps'], model_version=MODEL_VERSION,
               failure_type=failure_type, error=error, empty_book_steps=empty_steps,
               quotes_valid=(sim is not None and empty_steps == 0))
    from AgentBasedModel.utils.liquidity import liquidity_summary, empty_spells
    if sim is not None:
        for name in ['a', 'b', 'fx']:
            liquid = getattr(sim.info, 'liquid_'+name, [])
            for key, value in liquidity_summary([not x for x in liquid]).items():
                row[f'{name}_{key}'] = value
            row[f'{name}_empty_periods_json'] = json.dumps(empty_spells([not x for x in liquid]))
            row[f'{name}_fallback_steps'] = sum(getattr(sim.info, 'fallback_'+name, []))
    from experiment_methodology import liquidity_fields
    row.update(liquidity_fields(sim.info if sim is not None else None,job['shock_iteration'],job['steps'],status))
    row.update(job.get('extra', {}))
    series = None
    if job.get('save_full_timeseries'):
        series = vars(sim.info) if sim is not None else {key: [] for key in
            ['prices_a', 'prices_b', 'fx_rates', 'price_gap', 'traded_volume_a', 'traded_volume_b', 'fx_volume', 'empty_books']}
    return row, series


def execute_jobs(jobs, aggregate, *, workers=4, output_root='tables', resume=None,
                 save_full_timeseries=False, prefix='results'):
    if workers < 1 or not jobs:
        raise ValueError('Positive workers and at least one run are required')
    jobs = [dict(j, save_full_timeseries=save_full_timeseries) for j in jobs]
    for job in jobs:
        if job['steps'] <= job['shock_iteration'] or job['shock_iteration'] < 1:
            raise ValueError('Require 0 < shock_iteration < steps')
        if not 0 <= job['seed'] < 2**32:
            raise ValueError('Seed must be in [0, 2**32)')
    from AgentBasedModel.version import MODEL_VERSION
    config = dict(format_version=3, model_version=MODEL_VERSION, jobs=jobs, prefix=prefix,
                  source_sha256=_source_hashes(), python=platform.python_version(),
                  packages={name: importlib.metadata.version(name)
                            for name in ['numpy', 'pandas', 'scipy', 'matplotlib', 'statsmodels']})
    if resume is None:
        name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
        directory = Path(output_root) / name
        directory.mkdir(parents=True, exist_ok=False)
    else:
        directory = Path(resume)
        if not directory.is_dir():
            raise ValueError('Resume directory does not exist')
    lock = directory / '.running'
    # Prevent concurrent writers, including simultaneous resume attempts.
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        metadata_path = directory / 'parameters.json'
        if resume is not None:
            metadata = json.loads(metadata_path.read_text())
            if metadata['config'] != config:
                raise ValueError('Resume parameters, source code or dependency versions differ')
        else:
            git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
            defaults = __import__('inspect').signature(__import__('experiments').build_system)
            metadata = dict(config=config, created_utc=datetime.now(timezone.utc).isoformat(),
                            workers_at_creation=workers, platform=platform.platform(),
                            git_commit=git.stdout.strip(),
                            build_system_defaults={k: v.default for k, v in defaults.parameters.items()},
                            seed_policy='seed = base_seed + run_id; common seeds across scenarios',
                            reporting=dict(recovery_definition='unchanged first passage',
                                           rate_ci='Wilson 95%', conditional_mean_ci='seed bootstrap percentile 95%, B=1999; recovered completed runs only',
                                           failure_policy='terminal failures separate from administrative censoring',
                                           all_attempts_rate_estimand='completed, valid and recovered / all attempts; earlier events before failure counted separately',
                                           empty_book_policy='single: last valid midpoint for one or both empty sides, no history fails explicitly; multi: original fallback and shock override retained; no synthetic liquidity',
                                           liquidity_recovery='same gap band plus two-sided A and B; separate from price recovery',
                                           liquidity_sampling='single at price capture; multi at record; durations count sampled steps',
                                           fx_quote='CUR_B per CUR_A', gap_definition='abs(P_A - P_B / FX), CUR_A; excludes transport/trading costs'))
            _atomic_json(metadata, metadata_path)
        checkpoints = directory / 'runs'
        checkpoints.mkdir(exist_ok=True)
        rows, pending = {}, []
        for index, job in enumerate(jobs):
            result_path = checkpoints / f'{index:07d}.csv'
            marker_path = checkpoints / f'{index:07d}.json'
            if marker_path.exists():
                marker = json.loads(marker_path.read_text())
                if _digest(result_path) != marker['csv_sha256']:
                    raise ValueError(f'Checkpoint checksum mismatch: {result_path}')
                if save_full_timeseries and _digest(checkpoints / f'{index:07d}_timeseries.csv') != marker['timeseries_sha256']:
                    raise ValueError(f'Time series checksum mismatch: {index}')
                if 'empty_spells_sha256' in marker and _digest(checkpoints / f'{index:07d}_empty_spells.csv') != marker['empty_spells_sha256']:
                    raise ValueError(f'Liquidity checksum mismatch: {index}')
                rows[index] = {key: marker['row'][key] for key in marker['columns']}
            else:
                pending.append((index, job))
        print(f'Results: {directory.resolve()} | completed={len(rows)}, pending={len(pending)}, workers={workers}', flush=True)

        def store_result(index, result):
            row, series = result
            path = checkpoints / f'{index:07d}.csv'
            _atomic_csv(pd.DataFrame([row]), path)
            marker = dict(row=row, columns=list(row), csv_sha256=_digest(path))
            if series is not None:
                series_path = checkpoints / f'{index:07d}_timeseries.csv'
                _atomic_csv(pd.DataFrame({key: pd.Series(values) for key, values in series.items()}), series_path)
                marker['timeseries_sha256'] = _digest(series_path)
                from AgentBasedModel.utils.liquidity import empty_spells
                episodes = []
                for name in ['a', 'b', 'fx']:
                    for spell in empty_spells([not x for x in series.get('liquid_'+name, [])]):
                        episodes.append(dict(market=name, **spell))
                episodes_path = checkpoints / f'{index:07d}_empty_spells.csv'
                _atomic_csv(pd.DataFrame(episodes, columns=['market','start','duration','right_censored']), episodes_path)
                marker['empty_spells_sha256'] = _digest(episodes_path)
            # Publish completion only after all artifact files have been closed.
            _atomic_json(marker, checkpoints / f'{index:07d}.json')
            rows[index] = row

        if workers == 1:
            for index, job in pending:
                store_result(index, _run_job(job))
        elif pending:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                # Bound submissions so interruption does not leave a full experiment queued.
                iterator = iter(pending)
                futures = {}
                for _ in range(min(len(pending), workers * 2)):
                    index, job = next(iterator)
                    futures[pool.submit(_run_job, job)] = index
                while futures:
                    future = next(as_completed(futures))
                    index = futures.pop(future)
                    store_result(index, future.result())
                    item = next(iterator, None)
                    if item is not None:
                        index, job = item
                        futures[pool.submit(_run_job, job)] = index
        raw = pd.DataFrame([rows[i] for i in range(len(jobs))])
        summary = aggregate(raw)
        raw_name, summary_name = ('raw_results.csv', 'summary_results.csv') if prefix == 'results' else (f'{prefix}_raw.csv', f'{prefix}_summary.csv')
        _atomic_csv(raw, directory / raw_name)
        _atomic_csv(summary, directory / summary_name)
        raw.attrs['output_dir'] = summary.attrs['output_dir'] = str(directory.resolve())
        return raw, summary
    finally:
        lock.unlink()
