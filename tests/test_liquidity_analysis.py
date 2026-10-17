import unittest
import numpy as np
from analyze_liquidity_stylized import liquid_segments,marginal_statistics
from stylized_facts import diagnose

class LiquidityAnalysisTests(unittest.TestCase):
    def test_segments_preserve_original_time_gaps(self):
        self.assertEqual(liquid_segments([True,True,False,True,True,False,True],0),[(0,2),(3,5),(6,7)])
        self.assertEqual(liquid_segments([True]*10,3),[(3,10)])

    def test_constant_fallback_returns_have_undefined_standardized_statistics(self):
        row=marginal_statistics(np.zeros(100))
        self.assertEqual(row['zero_return_fraction'],1.)
        self.assertNotEqual(row['status'],'ok')
        self.assertTrue(np.isnan(row['excess_kurtosis']))
        self.assertTrue(np.isnan(row['tail_3sigma_frequency']))

    def test_marginal_definitions_match_existing_diagnostics(self):
        rng=np.random.default_rng(42);prices=100*np.exp(np.cumsum(rng.normal(0,.01,500)))
        original,_=diagnose(prices,100);subset=marginal_statistics(np.diff(np.log(prices[100:])))
        for name in ['excess_kurtosis','tail_2sigma_frequency','tail_3sigma_frequency','tail_4sigma_frequency','zero_return_fraction','std_log_return']:
            self.assertAlmostEqual(original[name],subset[name])
if __name__=='__main__':unittest.main()
