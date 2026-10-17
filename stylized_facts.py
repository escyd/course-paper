"""Diagnostics of the unchanged single-market baseline, one series per seed."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.metadata
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import platform
import random
import time
import traceback
import uuid

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import acf

from experiment_runner import _atomic_csv, _atomic_json, _digest


BASELINE = {'Random': 5, 'Fundamentalist': 5, 'Chartist': 5,
            'Universalist': 5, 'MarketMaker': 1}


def build_baseline(seed):
    # Match construction order and defaults in main.py; omit its external shock.
    from AgentBasedModel import (ExchangeAgent, Simulator, Random, Fundamentalist,
                                 Chartist, Universalist, MarketMaker)
    random.seed(seed)
    np.random.seed(seed)
    exchange = ExchangeAgent(volume=1000)
    traders = [*[Random(exchange, 1000) for _ in range(5)],
               *[Fundamentalist(exchange, 1000) for _ in range(5)],
               *[Chartist(exchange, 1000) for _ in range(5)],
               *[Universalist(exchange, 1000) for _ in range(5)],
               MarketMaker(exchange, 1000)]
    return Simulator(exchange=exchange, traders=traders, events=None)


def log_returns(prices, burn_in=100):
    """Discard price observations first; never bridge missing/invalid prices."""
    prices = np.asarray(prices, dtype=float)
    if burn_in < 0 or len(prices) - burn_in < 50:
        raise ValueError('Need at least 50 price observations after burn-in')
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError('Nonpositive/nonfinite prices: log returns undefined; no clipping')
    return np.diff(np.log(prices[burn_in:]))


def diagnose(prices, burn_in=100, lags=(10, 20), acf_lags=40):
    r = log_returns(prices, burn_in)
    if max(lags) >= len(r) // 5 or acf_lags >= len(r):
        raise ValueError('Insufficient observations for specified lags')
    sd = np.std(r, ddof=1)
    if not np.isfinite(sd) or sd == 0:
        raise ValueError('Constant returns: standardized diagnostics undefined')
    z = (r - r.mean()) / sd
    row = dict(burn_in=burn_in, n_returns=len(r), mean_log_return=float(r.mean()),
               std_log_return=float(sd), excess_kurtosis=float(stats.kurtosis(r, fisher=True, bias=False)),
               skewness=float(stats.skew(r, bias=False)), zero_return_fraction=float(np.mean(r == 0)))
    for threshold in (2, 3, 4):
        frequency = float(np.mean(np.abs(z) > threshold))
        gaussian = float(2 * stats.norm.sf(threshold))
        row.update({f'tail_{threshold}sigma_frequency': frequency,
                    f'tail_{threshold}sigma_gaussian': gaussian,
                    f'tail_{threshold}sigma_ratio': frequency / gaussian})
    tables = []
    for name, series in [('returns', r), ('abs_returns', np.abs(r)), ('squared_returns', r**2)]:
        values = acf(series, nlags=acf_lags, fft=True, adjusted=False)
        tables.extend(dict(burn_in=burn_in, transform=name, lag=i, acf=float(v))
                      for i, v in enumerate(values))
        lb = acorr_ljungbox(series, lags=list(lags), model_df=0, return_df=True)
        for lag in lags:
            row[f'lb_{name}_{lag}_stat'] = float(lb.loc[lag, 'lb_stat'])
            row[f'lb_{name}_{lag}_p'] = float(lb.loc[lag, 'lb_pvalue'])
        row[f'acf_{name}_1'] = float(values[1])
    # Fixed AR(1), chosen before results. Raw-return ARCH is retained as requested.
    x = np.column_stack([np.ones(len(r) - 1), r[:-1]])
    coefficients = np.linalg.lstsq(x, r[1:], rcond=None)[0]
    residuals = r[1:] - x @ coefficients
    row['ar1_intercept'], row['ar1_coefficient'] = map(float, coefficients)
    for lag in lags:
        for name, series, ddof in [('raw', r-r.mean(), 0), ('ar1', residuals, 2)]:
            lm, p, f, fp = het_arch(series, nlags=lag, ddof=ddof)
            row.update({f'arch_{name}_{lag}_stat': float(lm), f'arch_{name}_{lag}_p': float(p),
                        f'arch_{name}_{lag}_f': float(f), f'arch_{name}_{lag}_f_p': float(fp)})
    return row, pd.DataFrame(tables)


def plot_run(prices, row, acf_table, destination, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r = log_returns(prices, int(row['burn_in']))
    z = (r-r.mean()) / np.std(r, ddof=1)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    ax = axes.flat
    ax[0].plot(prices, lw=.7)
    ax[0].axvline(row['burn_in'], color='black', ls='--', lw=.7)
    ax[0].set(xlabel='Simulation step', ylabel='Price', title='Prices (dashed: burn-in)')
    ax[1].plot(np.arange(len(r)) + row['burn_in'] + 1, r, lw=.5)
    ax[1].set(xlabel='Simulation step', ylabel='Log return', title='Post burn-in returns')
    theoretical, ordered = stats.probplot(z, dist='norm', fit=False)
    ax[2].scatter(theoretical, ordered, s=3, rasterized=True)
    limits = [min(theoretical), max(theoretical)]
    ax[2].plot(limits, limits, color='black', lw=.8)
    ax[2].set(xlabel='Normal quantile', ylabel='Standardized return quantile', title='Normal Q–Q')
    absolute = np.sort(np.abs(z)); survival = np.arange(len(z), 0, -1) / len(z)
    ax[3].step(absolute, survival, where='post', label='Empirical')
    grid = np.linspace(0, max(4, absolute[-1]), 400)
    ax[3].plot(grid, 2*stats.norm.sf(grid), ls='--', label='Gaussian')
    ax[3].set(yscale='log', ylim=(max(1e-6, .5/len(z)), 1), xlabel='|Standardized return|',
              ylabel='Exceedance probability', title=f"Tails; excess kurtosis={row['excess_kurtosis']:.2f}")
    ax[3].legend(fontsize=8)
    for index, transforms in [(4, ['returns']), (5, ['abs_returns', 'squared_returns'])]:
        for transform in transforms:
            # Use explicit column access: DataFrame.transform is a method.
            part = acf_table[(acf_table['transform'] == transform) & (acf_table['lag'] > 0)]
            ax[index].plot(part.lag, part.acf, marker='.', ms=3, lw=.8, label=transform)
        bound = 1.96 / np.sqrt(len(r))
        ax[index].axhspan(-bound, bound, color='grey', alpha=.15)
        ax[index].axhline(0, color='black', lw=.5)
        ax[index].set(xlabel='Lag', ylabel='ACF', title='ACF (band: iid reference only)')
        ax[index].legend(fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.savefig(destination.with_suffix('.png'), dpi=220)
    fig.savefig(destination.with_suffix('.pdf'))
    plt.close(fig)


def _job(job):
    steps, seed, directory, config = job
    directory = Path(directory)
    start = time.perf_counter()
    sim = build_baseline(seed)
    error = None
    try:
        sim.simulate(steps, silent=True)
    except Exception:
        error = traceback.format_exc()
    simulation_seconds = time.perf_counter() - start
    from AgentBasedModel.utils.liquidity import quote_state
    if hasattr(sim, 'exchange'):
        _atomic_json(dict(quote_state(sim.exchange), fallback_price_calls=sim.exchange.fallback_calls,
                          error=error, completed=error is None), directory / 'final_liquidity.json')
    prices = np.asarray(sim.info.prices, dtype=float)
    frame = pd.DataFrame({'step': np.arange(len(prices)), 'price': prices})
    from AgentBasedModel.utils.liquidity import liquidity_summary
    liquidity = liquidity_summary(getattr(sim.info, 'empty_books', []))
    frame['bid_empty'] = getattr(sim.info, 'bid_empty', [None]*len(prices))
    frame['ask_empty'] = getattr(sim.info, 'ask_empty', [None]*len(prices))
    frame['fallback_used'] = getattr(sim.info, 'fallback_used', [None]*len(prices))
    frame['empty_both'] = getattr(sim.info, 'empty_both', [None]*len(prices))
    frame['liquidity_state'] = getattr(sim.info, 'liquidity_state', [None]*len(prices))
    if hasattr(sim.info, 'bid_empty'):
        from AgentBasedModel.utils.liquidity import quote_history_summary
        liquidity.update(quote_history_summary(sim.info.bid_empty, sim.info.ask_empty))
    # Full raw series is preserved even for a failed simulation.
    frame['log_return'] = np.nan
    valid = np.isfinite(prices) & (prices > 0)
    if len(prices) > 1:
        pair = valid[1:] & valid[:-1]
        values = np.full(len(prices)-1, np.nan)
        values[pair] = np.log(prices[1:][pair]) - np.log(prices[:-1][pair])
        frame.loc[1:, 'log_return'] = values
    _atomic_csv(frame, directory / 'prices.csv')
    from AgentBasedModel.utils.liquidity import empty_spells
    _atomic_csv(pd.DataFrame(empty_spells(getattr(sim.info, 'empty_books', [])),
                             columns=['start', 'duration', 'right_censored']), directory / 'empty_spells.csv')
    rows, acfs = [], []
    for burn_in in config['burn_ins']:
        base = dict(steps=steps, seed=seed, burn_in=burn_in, simulation_seconds=simulation_seconds,
                    captured_prices=len(prices), status='ok', error='', **liquidity,
                    fallback_price_calls=getattr(getattr(sim, 'exchange', None), 'fallback_calls', 0))
        try:
            if error is not None:
                raise RuntimeError(error)
            row, table = diagnose(prices, burn_in, tuple(config['test_lags']), config['acf_lags'])
            base.update(row)
            table['steps'], table['seed'] = steps, seed
            acfs.append(table)
            if burn_in == config['primary_burn_in']:
                plot_run(prices, row, table, directory / 'diagnostics', f'Single-market baseline; steps={steps}, seed={seed}')
        except Exception:
            base.update(status='failed', error=traceback.format_exc())
        rows.append(base)
    _atomic_csv(pd.DataFrame(rows), directory / 'statistics.csv')
    if acfs:
        _atomic_csv(pd.concat(acfs, ignore_index=True), directory / 'acf.csv')
    files = {p.name: _digest(p) for p in directory.iterdir() if p.is_file() and p.name != 'complete.json'}
    _atomic_json(dict(files=files, elapsed_seconds=time.perf_counter()-start), directory / 'complete.json')
    return str(directory)


def summarize(raw, alpha=.05):
    """Aggregate per-run statistics only; never concatenate price/return series."""
    corrected = raw.copy()
    p_columns = [c for c in raw if c.endswith('_p') and not c.endswith('_f_p')]
    rows = []
    for (steps, burn), group in raw.groupby(['steps', 'burn_in'], sort=True):
        valid = group[group.status == 'ok']
        row = dict(steps=steps, burn_in=burn, n_runs=len(group), n_valid=len(valid), n_failed=len(group)-len(valid))
        for name in ['excess_kurtosis', 'tail_2sigma_frequency', 'tail_3sigma_frequency',
                     'tail_4sigma_frequency', 'tail_3sigma_ratio', 'acf_returns_1',
                     'acf_abs_returns_1', 'acf_squared_returns_1', 'zero_return_fraction']:
            values = valid[name].dropna() if name in valid else pd.Series(dtype=float)
            row.update({name+'_median': values.median(), name+'_q25': values.quantile(.25),
                        name+'_q75': values.quantile(.75)})
        if len(valid):
            row['positive_excess_fraction'] = float((valid.excess_kurtosis > 0).mean())
        for name in p_columns:
            values = pd.to_numeric(valid[name], errors='coerce').dropna()
            values = values[np.isfinite(values)]
            if len(values):
                rejected, qvalues, _, _ = multipletests(values, alpha=alpha, method='fdr_bh')
                corrected.loc[values.index, name+'_bh_q'] = qvalues
                row[name+'_n_tested'] = len(values)
                row[name+'_reject_fraction'] = float((values < alpha).mean())
                row[name+'_bh_reject_fraction'] = float(rejected.mean())
        rows.append(row)
    return corrected, pd.DataFrame(rows)


def plot_summary(raw, summary, acf_data, burn_in, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    valid = raw[(raw.status == 'ok') & (raw.burn_in == burn_in)]
    if valid.empty:
        return
    horizons = sorted(valid.steps.unique())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    for ax, metric, reference, title in [
        (axes[0,0], 'excess_kurtosis', 0, 'Excess kurtosis across seeds'),
        (axes[0,1], 'tail_3sigma_ratio', 1, '3-sigma tail frequency / Gaussian')]:
        ax.boxplot([valid.loc[valid.steps == h, metric] for h in horizons], showfliers=True)
        ax.set_xticks(range(1, len(horizons)+1), [str(h) for h in horizons])
        ax.axhline(reference, color='black', ls='--', lw=.8)
        ax.set(xlabel='Simulation steps', title=title)
    part = summary[summary.burn_in == burn_in].set_index('steps')
    x = np.arange(len(horizons))
    for offset, metric, label in [(-.25, 'lb_abs_returns_20_p_bh_reject_fraction', 'LB |r| (20)'),
                                 (0, 'arch_raw_20_p_bh_reject_fraction', 'ARCH raw (20)'),
                                 (.25, 'arch_ar1_20_p_bh_reject_fraction', 'ARCH AR(1) (20)')]:
        axes[0,2].bar(x+offset, [part.loc[h, metric] for h in horizons], width=.24, label=label)
    axes[0,2].set(xticks=x, xticklabels=[str(h) for h in horizons], ylim=(0,1.08),
                  xlabel='Simulation steps', ylabel='Fraction of valid seeds', title='BH-adjusted rejections at 5%')
    axes[0,2].legend(fontsize=8, loc='lower right')
    for ax, transform in zip(axes[1], ['returns','abs_returns','squared_returns']):
        for h in horizons:
            part = acf_data[(acf_data.burn_in == burn_in) & (acf_data.steps == h) &
                            (acf_data['transform'] == transform) & (acf_data.lag > 0)]
            grouped = part.groupby('lag')['acf']
            med, low, high = grouped.median(), grouped.quantile(.25), grouped.quantile(.75)
            ax.plot(med.index, med.values, lw=1, label=str(h))
            ax.fill_between(med.index, low.values, high.values, alpha=.12)
        ax.axhline(0, color='black', lw=.5)
        ax.set(xlabel='Lag', ylabel='ACF', title=f'{transform}: median and IQR across seeds')
        ax.legend(title='Steps', fontsize=8)
    fig.suptitle(f'Single-market baseline; burn-in={burn_in}; no external shock', fontsize=12)
    fig.savefig(destination.with_suffix('.png'), dpi=250)
    fig.savefig(destination.with_suffix('.pdf'))
    plt.close(fig)


def run_study(n_runs=50, horizons=(500, 5000, 10000), workers=4, base_seed=42,
              burn_in=100, output_root='tables/stylized_facts', resume=None):
    if not 1 <= n_runs < 500:
        raise ValueError('This preliminary study supports 1–499 runs; 500+ requires separate authorization')
    if not horizons or len(set(horizons)) != len(horizons):
        raise ValueError('Horizons must be nonempty and unique')
    if workers < 1 or burn_in < 0 or min(horizons) - burn_in < 150:
        raise ValueError('Invalid workers, burn-in or horizons')
    if base_seed < 0 or base_seed+n_runs > 2**32:
        raise ValueError('Seeds must fit NumPy seed range')
    from AgentBasedModel import ExchangeAgent
    from AgentBasedModel.version import MODEL_VERSION
    root = Path(__file__).resolve().parent
    paths = [Path(__file__).resolve(), root / 'experiment_runner.py', *sorted((root/'AgentBasedModel').rglob('*.py'))]
    config = dict(liquidity_policy='last valid midpoint for one or both empty sides; no history fails explicitly',
                  liquidity_sampling='at original price capture; sampled step durations; empty-spells CSV',
                  model_version=MODEL_VERSION, n_runs=n_runs, horizons=list(horizons), base_seed=base_seed,
                  primary_burn_in=burn_in, burn_ins=sorted(set([0, burn_in])), test_lags=[10,20], acf_lags=40,
                  alpha=.05, baseline=BASELINE, cash=1000, assets=0, events=[],
                  exchange_defaults={k:v.default for k,v in inspect.signature(ExchangeAgent).parameters.items()},
                  return_definition='diff(log(prices[burn_in:])); price-only, no dividends',
                  seed_policy='base_seed + run_id; paired seeds across horizons; independent within horizon',
                  kurtosis='Fisher excess, bias=False', std_ddof=1,
                  arch='demeaned raw returns ddof=0; OLS AR(1) residuals ddof=2',
                  bh_family='seeds within each horizon, burn-in, test, transform and lag',
                  python=platform.python_version(), packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','statsmodels','matplotlib']},
                  source_sha256={str(p.relative_to(root)):_digest(p) for p in paths})
    if resume:
        directory = Path(resume)
        if json.loads((directory/'parameters.json').read_text())['config'] != config:
            raise ValueError('Resume configuration/source/dependencies differ')
    else:
        directory = Path(output_root)/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        directory.mkdir(parents=True, exist_ok=False)
        _atomic_json(dict(config=config, created_utc=datetime.now(timezone.utc).isoformat(),
                          workers=workers, platform=platform.platform()), directory/'parameters.json')
    lock = directory/'.running'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600); os.close(descriptor)
    try:
        jobs, folders = [], []
        for steps in horizons:
            for i in range(n_runs):
                seed = base_seed+i
                folder = directory/f'steps_{steps}'/f'seed_{seed}'
                folder.mkdir(parents=True, exist_ok=True)
                folders.append(folder)
                marker = folder/'complete.json'
                if marker.exists():
                    for name, digest in json.loads(marker.read_text())['files'].items():
                        if _digest(folder/name) != digest:
                            raise ValueError(f'Checkpoint checksum mismatch: {folder/name}')
                else:
                    jobs.append((steps,seed,str(folder),config))
        print(f'Results: {directory.resolve()}; pending={len(jobs)}; workers={workers}',flush=True)
        if workers == 1:
            for job in jobs:
                print('Completed',_job(job),flush=True)
        elif jobs:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                # Each task is a complete independent simulation plus diagnostics.
                futures = [pool.submit(_job,job) for job in jobs]
                for i,future in enumerate(as_completed(futures),1):
                    future.result()
                    if i % 5 == 0 or i == len(jobs):
                        print(f'Completed {i}/{len(jobs)}',flush=True)
        raw = pd.concat([pd.read_csv(f/'statistics.csv', float_precision='round_trip') for f in folders],ignore_index=True)
        corrected, summary = summarize(raw)
        _atomic_csv(corrected,directory/'per_run_statistics.csv')
        _atomic_csv(summary,directory/'summary.csv')
        acf_frames = [pd.read_csv(f/'acf.csv', float_precision='round_trip') for f in folders if (f/'acf.csv').exists()]
        if acf_frames:
            acf_data = pd.concat(acf_frames, ignore_index=True)
            _atomic_csv(acf_data,directory/'acf_per_run.csv')
            plot_summary(corrected, summary, acf_data, burn_in, directory/'ensemble_diagnostics')
        return directory, corrected, summary
    finally:
        lock.unlink()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=int,default=50)
    parser.add_argument('--horizons',type=int,nargs='+',default=[500,5000,10000])
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--base-seed',type=int,default=42)
    parser.add_argument('--burn-in',type=int,default=100)
    parser.add_argument('--output-root',default='tables/stylized_facts')
    parser.add_argument('--resume')
    args=parser.parse_args()
    started=time.perf_counter()
    directory,_,summary=run_study(args.runs,args.horizons,args.workers,args.base_seed,args.burn_in,args.output_root,args.resume)
    print(summary[['steps', 'burn_in', 'n_runs', 'n_valid', 'n_failed', 'excess_kurtosis_median']].to_string(index=False));print('Elapsed seconds:',time.perf_counter()-started)
