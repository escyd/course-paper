"""Short regression checks; never run the full experiment grid."""
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import experiment_runner
from experiments import build_system, run_experiments, set_all_seeds, summarize_run
from AgentBasedModel.utils import spatial


class StageOneTests(unittest.TestCase):
    def test_cached_distance_preserves_paths_and_random_states(self):
        for tau, beta, cost in [(0., 0., 0.), (0., 0., .001), (.5, .4, .005), (1., .6, .01)]:
            for seed in [42, 43, 44]:
                results = []
                for distance in [spatial.euclidean_distance.__wrapped__, spatial.euclidean_distance]:
                    with patch.object(spatial, 'euclidean_distance', distance):
                        set_all_seeds(seed)
                        sim = build_system(tau=tau, beta=beta, fx_cost=cost)
                        sim.simulate(500)
                        results.append((vars(sim.info), [(t.cash, t.assets) for t in sim.traders],
                                        random.getstate(), np.random.get_state(),
                                        summarize_run(sim, tau, beta, 0, cost)))
                a, b = results
                self.assertEqual(a[:3], b[:3])
                np.testing.assert_equal(a[3], b[3])
                pd.testing.assert_series_equal(pd.Series(a[4]), pd.Series(b[4]), check_exact=True)
        self.assertEqual(spatial.euclidean_distance((0, 0), (3, 4)), 5.)
        self.assertEqual(spatial.euclidean_distance((0, 0), (6, 8)), 10.)

    def test_parallel_resume_and_checkpoint_integrity(self):
        with tempfile.TemporaryDirectory() as root:
            kwargs = dict(tau_values=[0., .5], beta_values=[.4], fx_cost_values=[.005],
                          n_runs=2, steps=80, shock_iteration=30, output_root=root,
                          save_full_timeseries=True)
            serial, serial_summary = run_experiments(**kwargs, workers=1)
            parallel, parallel_summary = run_experiments(**kwargs, workers=4)
            pd.testing.assert_frame_equal(serial, parallel, check_exact=True)
            pd.testing.assert_frame_equal(serial_summary, parallel_summary, check_exact=True)
            serial_dir, parallel_dir = Path(serial.attrs['output_dir']), Path(parallel.attrs['output_dir'])
            for path in (serial_dir / 'runs').glob('*_timeseries.csv'):
                self.assertEqual(path.read_bytes(), (parallel_dir / 'runs' / path.name).read_bytes())
            checkpoint = parallel_dir / 'runs' / '0000000.csv'
            modified = checkpoint.stat().st_mtime_ns
            with patch.object(experiment_runner, '_run_job', side_effect=AssertionError('Must skip saved runs')):
                resumed, _ = run_experiments(**kwargs, workers=1, resume=parallel_dir)
            pd.testing.assert_frame_equal(serial, resumed, check_exact=True)
            self.assertEqual(modified, checkpoint.stat().st_mtime_ns)
            with self.assertRaisesRegex(ValueError, 'differ'):
                run_experiments(**dict(kwargs, base_seed=142), workers=1, resume=parallel_dir)
            checkpoint.write_text('corrupted')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                run_experiments(**kwargs, workers=1, resume=parallel_dir)

    def test_interrupted_run_resumes_only_missing_jobs(self):
        with tempfile.TemporaryDirectory() as root:
            kwargs = dict(tau_values=[.5], beta_values=[.4], fx_cost_values=[.005],
                          n_runs=3, steps=80, shock_iteration=30, output_root=root, workers=1)
            original = experiment_runner._run_job
            calls = []
            def interrupted(job):
                calls.append(job['run_id'])
                if len(calls) == 2:
                    raise RuntimeError('simulated interruption')
                return original(job)
            with patch.object(experiment_runner, '_run_job', side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, 'interruption'):
                    run_experiments(**kwargs)
            directory = next(Path(root).iterdir())
            first = directory / 'runs' / '0000000.csv'
            content, modified = first.read_bytes(), first.stat().st_mtime_ns
            with patch.object(experiment_runner, '_run_job', wraps=original) as worker:
                raw, _ = run_experiments(**kwargs, resume=directory)
                self.assertEqual([c.args[0]['run_id'] for c in worker.call_args_list], [1, 2])
            self.assertEqual(len(raw), 3)
            self.assertEqual(first.read_bytes(), content)
            self.assertEqual(first.stat().st_mtime_ns, modified)
            self.assertTrue((directory / 'raw_results.csv').exists())

    def test_old_csv_and_figures_are_preserved(self):
        import experiments
        import plot_results
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            csv = root / 'existing.csv'
            csv.write_text('original results')
            with patch.object(experiments, 'OUT_DIR', str(root)):
                experiments.clear_tables()
            self.assertEqual(csv.read_text(), 'original results')
            previous_cwd, previous_output = Path.cwd(), plot_results.OUT_DIR
            try:
                os.chdir(root)
                Path('figures').mkdir()
                image = Path('figures/existing.png')
                image.write_bytes(b'original figure')
                plot_results.clear_figures()
                first = plot_results.OUT_DIR
                plot_results.clear_figures()
                self.assertNotEqual(first, plot_results.OUT_DIR)
                self.assertEqual(image.read_bytes(), b'original figure')
            finally:
                os.chdir(previous_cwd)
                plot_results.OUT_DIR = previous_output

    def test_plot_import_has_no_side_effects(self):
        import importlib
        with patch('matplotlib.pyplot.savefig', side_effect=AssertionError('Unexpected plotting')):
            importlib.import_module('plot_results')


if __name__ == '__main__':
    unittest.main()
