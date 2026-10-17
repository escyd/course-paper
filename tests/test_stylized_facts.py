import random
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd

from stylized_facts import build_baseline, diagnose, log_returns, run_study


class StylizedFactsTests(unittest.TestCase):
    def test_baseline_matches_original_construction_and_rng(self):
        from AgentBasedModel import ExchangeAgent,Simulator,Random,Fundamentalist,Chartist,Universalist,MarketMaker
        random.seed(42); np.random.seed(42)
        exchange=ExchangeAgent(volume=1000)
        traders=[*[Random(exchange,1000) for _ in range(5)],
                 *[Fundamentalist(exchange,1000) for _ in range(5)],
                 *[Chartist(exchange,1000) for _ in range(5)],
                 *[Universalist(exchange,1000) for _ in range(5)],MarketMaker(exchange,1000)]
        reference=Simulator(exchange=exchange,traders=traders,events=None)
        reference.simulate(500,silent=True)
        state=random.getstate(); np_state=np.random.get_state()
        actual=build_baseline(42); actual.simulate(500,silent=True)
        self.assertEqual(reference.info.prices,actual.info.prices)
        self.assertEqual([(t.cash,t.assets,t.type) for t in reference.traders],[(t.cash,t.assets,t.type) for t in actual.traders])
        self.assertEqual(state,random.getstate());np.testing.assert_equal(np_state,np.random.get_state())

    def test_returns_no_clipping_or_boundary_bridging(self):
        prices=np.exp(np.arange(200)*.001)
        np.testing.assert_allclose(log_returns(prices,100),np.full(99,.001),atol=1e-15)
        invalid=prices.copy();invalid[150]=0
        with self.assertRaisesRegex(ValueError,'Nonpositive'):
            log_returns(invalid,100)
        with self.assertRaisesRegex(ValueError,'Constant'):
            diagnose(np.ones(200),0)
        rng=np.random.default_rng(125)
        prices=np.exp(np.cumsum(rng.normal(0,.001,1000)))
        a,_=diagnose(prices,100);b,_=diagnose(prices*5,100)
        self.assertAlmostEqual(a['excess_kurtosis'],b['excess_kurtosis'],places=9)
        self.assertEqual(a['n_returns'],899)

    def test_detects_arch_and_heavy_tail_controls(self):
        rng=np.random.default_rng(987)
        r=np.zeros(5000)
        for i in range(1,len(r)):
            r[i]=rng.normal()*np.sqrt(.000001+.8*r[i-1]**2)
        row,_=diagnose(np.exp(np.cumsum(r)),100)
        self.assertLess(row['arch_raw_20_p'],1e-8)
        self.assertGreater(row['acf_squared_returns_1'],.1)
        heavy=rng.standard_t(3,5000)*.001
        row,_=diagnose(np.exp(np.cumsum(heavy)),100)
        self.assertGreater(row['excess_kurtosis'],2)
        self.assertGreater(row['tail_3sigma_ratio'],1)

    def test_model_failure_preserves_partial_series_and_status(self):
        from stylized_facts import _job
        fake = SimpleNamespace(info=SimpleNamespace(prices=[100., 101., 102.]))
        def fail(*args, **kwargs):
            raise RuntimeError('empty ask book')
        fake.simulate = fail
        with tempfile.TemporaryDirectory() as root:
            config = dict(burn_ins=[0, 100], primary_burn_in=100)
            with patch('stylized_facts.build_baseline', return_value=fake):
                _job((500, 58, root, config))
            rows = pd.read_csv(Path(root)/'statistics.csv')
            self.assertTrue((rows.status == 'failed').all())
            self.assertTrue(rows.error.str.contains('empty ask book').all())
            self.assertEqual(pd.read_csv(Path(root)/'prices.csv').price.tolist(), [100.,101.,102.])
            self.assertTrue((Path(root)/'complete.json').exists())
            self.assertFalse((Path(root)/'diagnostics.pdf').exists())

    def test_serial_parallel_resume_exact_statistics(self):
        with tempfile.TemporaryDirectory() as root:
            a,raw_a,_=run_study(2,(500,),1,output_root=root)
            b,raw_b,_=run_study(2,(500,),4,output_root=root)
            columns=[c for c in raw_a if c!='simulation_seconds']
            pd.testing.assert_frame_equal(raw_a[columns],raw_b[columns],check_exact=True)
            for seed in [42,43]:
                relative=Path('steps_500')/f'seed_{seed}'/'prices.csv'
                self.assertEqual((a/relative).read_bytes(),(b/relative).read_bytes())
            _,resumed,_=run_study(2,(500,),1,output_root=root,resume=b)
            pd.testing.assert_frame_equal(raw_b,resumed,check_exact=True)
            self.assertTrue((b/'summary.csv').exists())


if __name__=='__main__': unittest.main()
