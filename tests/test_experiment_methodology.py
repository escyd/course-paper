import contextlib,io,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from experiment_methodology import liquidity_fields,methodology_tables,write_methodology_outputs,_bootstrap
from recovery_reporting import observation_fields
from final_experiment import prepare,main,DEFAULT_CONFIG


def info(flags):
    return dict(prices_a=[100.]*len(flags),liquid_a=flags,liquid_b=[True]*len(flags),
                bid_empty_a=[not x for x in flags],ask_empty_a=[False]*len(flags),
                bid_empty_b=[False]*len(flags),ask_empty_b=[False]*len(flags))


def row(seed,flags,status='completed',price=0.,liquidity=0.):
    r=dict(seed=seed,run_id=seed,tau=0.,beta=0.,fx_cost=.001,shock_iteration=3,
           avg_gap_before=1.,avg_gap_after=2.,max_gap_after=3.,final_gap=1.,
           recovery_time=price,liquidity_recovery_time=liquidity,recovery_b=0.,liquidity_recovery_available=True)
    r.update(observation_fields(r,status=status,requested=6,completed=len(flags)))
    r.update(liquidity_fields(info(flags),3,6,status));return r

class MethodologyTests(unittest.TestCase):
    def test_past_loss_and_immediate_pre_state_are_different(self):
        fields=liquidity_fields(info([True,False,True,False,True,True]),3,6)
        self.assertTrue(fields['pre_shock_liquidity_failure']);self.assertTrue(fields['pre_shock_two_sided'])
        self.assertEqual(fields['first_liquidity_loss_step'],1)
        self.assertEqual(fields['first_liquidity_loss_time_from_shock'],-2)
        self.assertTrue(fields['post_shock_liquidity_restored'])
        self.assertEqual(fields['first_post_shock_restoration_step'],4)
        self.assertEqual(fields['system_liquidity_recoveries'],2)

    def test_partial_pre_window_is_not_negative_or_administrative_censoring(self):
        fields=liquidity_fields(info([True,True]),3,6,'failed')
        self.assertFalse(fields['pre_shock_history_complete'])
        self.assertIsNone(fields['pre_shock_liquidity_failure']);self.assertIsNone(fields['pre_shock_two_sided'])
        self.assertIsNone(fields['post_shock_liquidity_restored'])
        lost=liquidity_fields(info([True,False]),3,6,'failed')
        self.assertTrue(lost['pre_shock_failure_observed'])
        self.assertTrue(lost['pre_shock_liquidity_failure'])

    def test_conditional_cohort_keeps_later_failures_and_denominators(self):
        raw=pd.DataFrame([row(1,[True]*6),row(2,[True,False,False,False,False,False],liquidity=np.nan),row(3,[True]*4,'failed',price=np.nan,liquidity=np.nan)])
        counts,desc,rates=methodology_tables(raw,199)
        allc=counts[counts.cohort=='all_attempts'].iloc[0];cond=counts[counts.cohort=='two_sided_before_shock'].iloc[0]
        self.assertEqual((allc.n_attempted,allc.n_failed,allc.n_completed_valid),(3,1,2))
        self.assertEqual((cond.n_attempted,cond.n_failed),(2,1))
        r=rates[(rates.cohort=='all_attempts')&(rates.metric=='liquidity_aware_recovery_rate')].iloc[0]
        self.assertEqual((r.numerator,r.denominator,r.estimate),(1,2,.5))
        r=rates[(rates.cohort=='all_attempts')&(rates.metric=='pre_shock_liquidity_failure_rate')].iloc[0]
        self.assertEqual((r.numerator,r.denominator),(1,3))
        self.assertEqual(allc.liquidity_recovery_time_n_censored,1)
        self.assertEqual(allc.liquidity_recovery_time_n_unavailable,1)

    def test_empty_cohort_and_zero_or_one_recovery_have_undefined_intervals(self):
        raw=pd.DataFrame([row(1,[False]*6,price=np.nan,liquidity=np.nan)])
        counts,desc,rates=methodology_tables(raw,199)
        self.assertEqual(counts[counts.cohort=='two_sided_before_shock'].n_attempted.iloc[0],0)
        self.assertTrue(desc[desc.metric=='recovery_time']['mean'].isna().all())
        self.assertTrue(rates[rates.cohort=='two_sided_before_shock'].estimate.isna().all())
        self.assertTrue(np.isnan(_bootstrap([4.],'test',199)['mean_ci_low']))
        self.assertEqual(_bootstrap([4.],'test',199)['median'],4.)

    def test_reporting_is_reproducible_and_does_not_consume_model_rng(self):
        raw=pd.DataFrame([row(1,[True]*6),row(2,[True]*6,price=2.,liquidity=2.)])
        state=np.random.get_state();a=methodology_tables(raw,199);b=methodology_tables(raw,199)
        for x,y in zip(a,b):pd.testing.assert_frame_equal(x,y,check_exact=True)
        np.testing.assert_equal(state,np.random.get_state())
        with tempfile.TemporaryDirectory() as root:
            out=write_methodology_outputs(raw,root,199);marker=json.loads((out/'complete.json').read_text())
            import hashlib
            for name,digest in marker['files'].items():self.assertEqual(hashlib.sha256((out/name).read_bytes()).hexdigest(),digest)
            self.assertTrue((out/'liquidity_periods.csv').exists())

    def test_mixed_partial_failures_preserve_unknown_pre_window_and_missing_spells(self):
        raw=pd.DataFrame([row(1,[True]*6),row(2,[True,False],'failed',price=np.nan,liquidity=np.nan)])
        raw['a_empty_periods_json']=['[]',np.nan]
        counts,desc,rates=methodology_tables(raw,199)
        full=counts[counts.cohort=='all_attempts'].iloc[0]
        self.assertEqual(full.n_pre_shock_window_unknown,1)
        self.assertEqual(full.n_pre_shock_failure_observed_in_incomplete_windows,1)
        rate=rates[(rates.cohort=='all_attempts')&(rates.metric=='pre_shock_liquidity_failure_rate')].iloc[0]
        self.assertEqual((rate.numerator,rate.denominator),(0,1))
        with tempfile.TemporaryDirectory() as root:
            self.assertTrue((write_methodology_outputs(raw,root,199)/'complete.json').exists())

    def test_preparation_and_default_dry_run_never_execute_simulations(self):
        config=json.loads(DEFAULT_CONFIG.read_text());scenarios,jobs=prepare(config)
        self.assertEqual((len(scenarios),len(jobs)),(100,50000))
        self.assertEqual(len({j['seed'] for j in jobs if j['extra']['scenario_id']=='S000'}),500)
        with patch('experiment_runner.execute_jobs',side_effect=AssertionError('No execution allowed')),patch('final_experiment.subprocess.run',return_value=SimpleNamespace(stdout='paper-revision\n')):
            with contextlib.redirect_stdout(io.StringIO()):plan=main([])
        self.assertEqual(plan['total_simulations'],50000)

if __name__=='__main__':unittest.main()
