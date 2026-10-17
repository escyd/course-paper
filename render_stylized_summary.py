"""Render a compact publication figure from an existing study; no simulation."""
import argparse
import json
from pathlib import Path
import uuid

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from experiment_runner import _atomic_json, _digest
from stylized_facts import plot_summary


def render(directory):
    directory = Path(directory)
    burn_in = json.loads((directory/'parameters.json').read_text())['config']['primary_burn_in']
    raw = pd.read_csv(directory/'per_run_statistics.csv', float_precision='round_trip')
    summary = pd.read_csv(directory/'summary.csv', float_precision='round_trip')
    acfs = pd.read_csv(directory/'acf_per_run.csv', float_precision='round_trip')
    destination = directory/('publication_figures_'+uuid.uuid4().hex[:8])
    destination.mkdir(exist_ok=False)
    # Smaller type keeps every panel title inside its axes at journal figure width.
    with plt.rc_context({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8}):
        plot_summary(raw, summary, acfs, burn_in, destination/'ensemble_diagnostics')
    _atomic_json(dict(source_statistics_sha256={name:_digest(directory/name) for name in
                         ['per_run_statistics.csv','summary.csv','acf_per_run.csv']},
                      renderer_sha256=_digest(Path(__file__)),
                      analysis_script_sha256=_digest(Path(__file__).with_name('stylized_facts.py')),
                      caption=f'Burn-in {burn_in}; completed runs only: ' + ', '.join(
                          f'{int(row.steps)} steps n={int(row.n_valid)}/{int(row.n_runs)}'
                          for row in summary[summary.burn_in == burn_in].itertuples())),
                 destination/'render_parameters.json')
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    print(render(parser.parse_args().directory).resolve())
