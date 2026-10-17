"""Observation/status bookkeeping; first-passage definitions stay in experiments.py."""
import hashlib

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

METRICS = ['avg_gap_before', 'avg_gap_after', 'max_gap_after', 'recovery_time',
           'vol_before', 'vol_after', 'final_price_a', 'final_price_b', 'final_fx',
           'final_gap', 'recovery_b', 'liquidity_recovery_time']


def observation_fields(row, *, status, requested, completed, data_valid=True):
    fields = dict(status=status, steps_requested=requested, steps_completed=completed,
                  data_valid=bool(data_valid))
    shock = row['shock_iteration']
    for metric in ['recovery_time', 'recovery_b', 'liquidity_recovery_time']:
        value = row.get(metric)
        observed = value is not None and bool(np.isfinite(value))
        eligible = completed > shock and data_valid
        if metric == 'liquidity_recovery_time':
            eligible = eligible and row.get('liquidity_recovery_available', metric in row)
        successful = status == 'completed' and eligible
        label = ('recovered' if observed else 'censored') if successful else 'unavailable'
        fields.update({metric+'_status': label,
                       metric+'_event_observed': bool(observed and eligible),
                       metric+'_censored': bool(successful and not observed),
                       metric+'_followup': max(0, completed - 1 - shock),
                       metric+'_event_before_failure': bool(observed and eligible and status == 'failed')})
    return fields


def valid_metrics(raw):
    """Never aggregate partial failed-run outcomes as full-horizon observations."""
    frame = raw.copy()
    if 'status' in frame:
        valid = frame.status.eq('completed') & frame.data_valid.fillna(False).astype(bool)
        for name in METRICS:
            if name in frame:
                frame.loc[~valid, name] = np.nan
    return frame


def recovery_summary(raw, keys):
    rows = []
    for key, group in raw.groupby(keys, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key))
        known = 'status' in group
        statuses = group.status if known else pd.Series('unknown', index=group.index)
        row.update(n_attempted=len(group) if known else np.nan,
                   n_completed=int(statuses.eq('completed').sum()),
                   n_failed=int(statuses.eq('failed').sum()),
                   n_status_unknown=int(statuses.eq('unknown').sum()),
                   n_invalid=int((~group.data_valid.astype(bool)).sum()) if known else 0,
                   n_with_empty_books=int(group.empty_book_steps.gt(0).sum()) if 'empty_book_steps' in group else 0,
                   reporting_available=known)
        for metric in ['recovery_time', 'recovery_b', 'liquidity_recovery_time']:
            labels = group[metric+'_status'] if known and metric+'_status' in group else pd.Series('unavailable', index=group.index)
            recovered, censored = int(labels.eq('recovered').sum()), int(labels.eq('censored').sum())
            denominator = recovered+censored
            row.update({metric+'_n_recovered': recovered, metric+'_n_censored': censored,
                        metric+'_n_unavailable': len(group)-denominator,
                        metric+'_n_recovered_before_failure': int(group[metric+'_event_before_failure'].sum()) if known and metric+'_event_before_failure' in group else 0,
                        metric+'_rate_denominator': denominator})
            for label, n in [('completed_valid', denominator), ('all_attempts', len(group) if known else 0)]:
                low, high = proportion_confint(recovered, n, alpha=.05, method='wilson') if n else (np.nan, np.nan)
                row.update({f'{metric}_fraction_{label}': recovered/n if n else np.nan,
                            f'{metric}_fraction_{label}_ci_low': low,
                            f'{metric}_fraction_{label}_ci_high': high})
            for label, n in [('completed_valid', denominator), ('all_attempts', len(group) if known else 0)]:
                row[f'{metric}_censored_fraction_{label}'] = censored/n if n else np.nan
            values = group.loc[labels.eq('recovered'), metric].to_numpy(dtype=float) if metric in group else np.array([])
            row[metric+'_conditional_mean'] = float(values.mean()) if len(values) else np.nan
            low, high = np.nan, np.nan
            if len(values) >= 2:
                token = repr((key, metric)).encode()
                seed = int.from_bytes(hashlib.sha256(token).digest()[:8], 'big')
                rng = np.random.default_rng(seed)
                # Bounded batches avoid O(resamples * n_runs) peak memory.
                means = []
                for start in range(0, 1999, 100):
                    count = min(100, 1999-start)
                    means.extend(values[rng.integers(0, len(values), (count, len(values)))].mean(axis=1))
                low, high = np.quantile(means, [.025, .975])
            row.update({metric+'_conditional_mean_ci_low': low,
                        metric+'_conditional_mean_ci_high': high})
        rows.append(row)
    return pd.DataFrame(rows)
