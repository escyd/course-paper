**fig01_adjusted_gap_heatmap**

Equal-weight mean over the five FX costs at each tau/beta cell. All 500 seeds retained per scenario; 2500 scenario-seed outcomes per cell, with 500 shared seed clusters. Gap averaged over recorded steps 200–249.

**fig02_liquidity_recovery_heatmap**

Equal-weight mean over the five FX costs at each tau/beta cell. All 500 seeds retained per scenario; 2500 scenario-seed outcomes per cell, with 500 shared seed clusters. Unrecovered paths remain in the denominator.

**fig03_recovery_and_censoring**

Rates average equally over beta and FX cost. Pointwise 95% percentile bootstrap intervals resample 500 whole seed clusters, B=9999. Survival curves average all 100 scenarios equally; marks at 299 denote administrative censoring, not events. Price recovery does not require two-sided liquidity.

**fig04_effects_tau**

Adjacent-level paired effects on all 500 seeds, averaged equally over the other grid parameters. Approximate simultaneous 95% Bonferroni Student-t intervals across 2192 mean effects. No automatic hypothesis decisions; constant-data intervals may degenerate. Restricted waiting time integrates survival through 299 and retains censored paths; it is not an imputed recovery event time.

**fig04_effects_beta**

Adjacent-level paired effects on all 500 seeds, averaged equally over the other grid parameters. Approximate simultaneous 95% Bonferroni Student-t intervals across 2192 mean effects. No automatic hypothesis decisions; constant-data intervals may degenerate. Restricted waiting time integrates survival through 299 and retains censored paths; it is not an imputed recovery event time.

**fig04_effects_fx_cost**

Adjacent-level paired effects on all 500 seeds, averaged equally over the other grid parameters. Approximate simultaneous 95% Bonferroni Student-t intervals across 2192 mean effects. No automatic hypothesis decisions; constant-data intervals may degenerate. Restricted waiting time integrates survival through 299 and retains censored paths; it is not an imputed recovery event time.

**fig05_saved_price_trajectories**

Fixed central scenario tau=0.3, beta=0.2, FX cost=0.003. Minimum seed in each pre-shock two-sided/illiquid stratum, not chosen by appearance; illustrative strata, not typical-path estimators. Prices converted to CUR_A; shaded recorded steps lack two-sided A/B liquidity. Shock at 0. Censored outcomes labeled; price-band crossing is not liquidity restoration.

**fig06_single_market_tail_summary**

Separate audited baseline: 50 seeds 42–91, 10000 prices, burn-in 100, original composition 5 Random + 5 Fundamentalist + 5 Chartist + 5 Universalist + 1 MarketMaker; no shock. Medians of per-run statistics, pointwise 95% seed-bootstrap CIs from prior verified analysis (B=10000). Two-sided statistics use longest continuous liquid segment per run; unequal durations and endogenous selection limit interpretation. Gaussian/zero references dashed. No returns concatenated across runs or gaps.

**fig07_single_market_volatility_clustering**

Same single-market sample and burn-in as fig06. Median seed-specific ACFs with pointwise 95% bootstrap CIs (B=1999); not simultaneous lag-wise bands. Longest contiguous liquid segment used for sensitivity analysis; no joining over missing liquidity or across seeds. Negative return ACF(1) is shown explicitly.

**fig08_single_market_distribution_qq**

Illustrative lowest seed (42), selected before inspecting statistics; 10000-price single-market run, burn-in 100, full series including fallback observations. Returns centered and divided by sample standard deviation. All empirical quantiles displayed; Gaussian reference does not imply calibrated Gaussian returns. Ensemble evidence and liquid-segment sensitivity are in fig06–07.

**table01_grid_summary**

Equal-weight grid averages; uncertainty resamples the 500 independent whole-seed clusters. All attempts retained, 50000/50000 completed valid. Restricted waiting is a survival integral to 299, not an unconditional uncensored recovery time.

**table02_parameter_effects**

Same equal-grid paired estimand and approximate simultaneous Bonferroni 95% intervals as fig04. The full numeric table also includes pre-shock liquidity loss.

**table03_single_market**

Audited single-market series, not the final two-market grid. Pointwise median bootstrap intervals; selected continuous liquid-segment analysis is descriptive. See test-rejection CSV for Ljung–Box and ARCH-LM with previously defined BH families.