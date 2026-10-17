import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments import aggregate_results, compute_recovery_time_relative
from recovery_reporting import METRICS, observation_fields
from experiment_runner import _run_job, execute_jobs


class RecoveryReportingTests(unittest.TestCase):
    def row(self, run_id, recovery, status='completed', valid=True):
        row = dict(run_id=run_id, tau=0., beta=0., fx_cost=.001, shock_iteration=20,
                   **{name: 1. for name in METRICS})
        row.update(recovery_time=recovery, recovery_b=recovery)
        row.update(observation_fields(row,status=status,requested=80,completed=80 if status=='completed' else 40,data_valid=valid))
        return row

    def test_censoring_failures_and_denominators(self):
        raw=pd.DataFrame([self.row(0,0.),self.row(1,np.nan),self.row(2,np.nan,'failed'),self.row(3,8.,valid=False)])
        result=aggregate_results(raw).iloc[0]
        self.assertEqual(result.n_runs,4)
        self.assertEqual(result.n_failed,1)
        self.assertEqual(result.recovery_time_n_recovered,1)
        self.assertEqual(result.recovery_time_n_censored,1)
        self.assertEqual(result.recovery_time_n_unavailable,2)
        self.assertEqual(result.recovery_time_fraction_completed_valid,.5)
        self.assertEqual(result.recovery_time_fraction_all_attempts,.25)
        self.assertEqual(result.recovery_time_mean,0.)
        self.assertTrue(np.isnan(result.recovery_time_conditional_mean_ci_low))
        self.assertLess(result.recovery_time_fraction_completed_valid_ci_low,.5)
        self.assertGreater(result.recovery_time_fraction_completed_valid_ci_high,.5)

    def test_event_before_failure_is_not_administrative_censoring(self):
        row=self.row(0,3.,'failed')
        self.assertTrue(row['recovery_time_event_before_failure'])
        self.assertFalse(row['recovery_time_censored'])
        self.assertEqual(row['recovery_time_status'],'unavailable')
        s=aggregate_results(pd.DataFrame([row])).iloc[0]
        self.assertEqual(s.recovery_time_n_recovered_before_failure,1)
        self.assertEqual(s.recovery_time_n_censored,0)

    def test_definition_and_summary_rng_unchanged(self):
        self.assertEqual(compute_recovery_time_relative([1.]*20+[1.2,9.],20),0)
        raw=pd.DataFrame([self.row(0,2.),self.row(1,4.)])
        state=np.random.get_state()
        a=aggregate_results(raw);b=aggregate_results(raw)
        pd.testing.assert_frame_equal(a,b,check_exact=True)
        np.testing.assert_equal(state,np.random.get_state())
        self.assertEqual(a.recovery_time_conditional_mean.iloc[0],3.)

    def test_worker_records_model_error(self):
        job=dict(seed=42,run_id=0,tau=0.,beta=0.,fx_cost=.001,steps=80,shock_iteration=20)
        with patch('experiments.build_system',side_effect=RuntimeError('test failure')):
            row,series=_run_job(job)
        self.assertEqual(row['status'],'failed')
        self.assertIn('test failure',row['error'])
        self.assertEqual(row['steps_completed'],0)
        self.assertFalse(row['data_valid'])
        self.assertFalse(row['quotes_valid'])
        self.assertFalse(row['recovery_time_censored'])


    def test_failed_attempt_is_checkpointed_and_resume_does_not_drop_it(self):
        import experiment_runner
        job=dict(seed=42,run_id=0,tau=0.,beta=0.,fx_cost=.001,steps=80,shock_iteration=20)
        with tempfile.TemporaryDirectory() as root:
            with patch('experiments.build_system',autospec=True,side_effect=RuntimeError('initialization failure')):
                raw,summary=execute_jobs([job],aggregate_results,workers=1,output_root=root,save_full_timeseries=True)
            self.assertEqual(summary.n_failed.iloc[0],1)
            self.assertEqual(summary.recovery_time_n_censored.iloc[0],0)
            directory=Path(raw.attrs['output_dir'])
            self.assertTrue((directory/'runs/0000000_timeseries.csv').exists())
            with patch.object(experiment_runner,'_run_job',side_effect=AssertionError('Already attempted')):
                resumed,_=execute_jobs([job],aggregate_results,workers=1,resume=directory,save_full_timeseries=True)
            pd.testing.assert_frame_equal(raw,resumed,check_exact=True)


if __name__=='__main__':unittest.main()
