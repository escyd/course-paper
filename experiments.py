import os
from pathlib import Path
import random
import numpy as np
import pandas as pd


from AgentBasedModel.agents.multi_market import AssetMarket, FXMarket
from AgentBasedModel.agents.factory import build_multi_traders
from AgentBasedModel.events.multi_events import MultiMarketPriceShock
from AgentBasedModel.simulator.multi_simulator import MultiMarketSimulator

OUT_DIR = "tables/"


def clear_tables():
    """Compatibility helper: create the directory without removing results."""
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


def build_system(
    tau=0.0,
    beta=0.0,
    fx_cost=0.001,
    shock_iteration=200,
    n_a_only_random=15,
    n_b_only_random=15,
    n_cross_random=10,
    n_a_only_fundamental=8,
    n_b_only_fundamental=8,
    n_cross_fundamental=10,
    n_fx_traders=8,
):
    market_a = AssetMarket(
        name="A",
        asset_name="ASSET_X",
        base_currency="CUR_A",
        price=100.0,
        std=10.0,
        volume=1000,
        transaction_cost=0.001,
        position=(2, 10),
    )

    market_b = AssetMarket(
        name="B",
        asset_name="ASSET_X",
        base_currency="CUR_B",
        price=102.0,
        std=10.0,
        volume=1000,
        transaction_cost=0.001,
        position=(18, 10),
    )

    fx_market = FXMarket(
        name="FX",
        base_currency="CUR_A",
        quote_currency="CUR_B",
        price=1.0,
        std=0.03,
        volume=500,
        transaction_cost=fx_cost,
        position=(10, 10),
    )

    markets = {
        "A": market_a,
        "B": market_b,
    }

    traders = build_multi_traders(
        markets=markets,
        fx_market=fx_market,
        grid_size=(20, 20),
        n_a_only_random=n_a_only_random,
        n_b_only_random=n_b_only_random,
        n_cross_random=n_cross_random,
        n_a_only_fundamental=n_a_only_fundamental,
        n_b_only_fundamental=n_b_only_fundamental,
        n_cross_fundamental=n_cross_fundamental,
        n_fx_traders=n_fx_traders,
    )

    events = [
        MultiMarketPriceShock(
            market_name="A",
            iteration=shock_iteration,
            delta_price=-10.0,
            spillover_beta=beta,
        )
    ]

    simulator = MultiMarketSimulator(
        markets=markets,
        fx_market=fx_market,
        traders=traders,
        events=events,
        tau=tau,
        beta_link=beta,
        link_strength=0.05,
    )

    return simulator


def compute_recovery_time_relative(price_gap, shock_iteration, baseline_window=50, epsilon=0.5):
    left = price_gap[max(0, shock_iteration - baseline_window):shock_iteration]

    if len(left) == 0:
        return np.nan

    baseline = sum(left) / len(left)
    threshold = baseline + epsilon

    for t in range(shock_iteration, len(price_gap)):
        if price_gap[t] <= threshold:
            return t - shock_iteration

    return np.nan


def liquidity_aware_recovery_time(price_gap, liquid_a, liquid_b, shock_iteration,
                                  baseline_window=50, epsilon=0.5):
    """Same first-hit band as price recovery, plus two-sided A and B quotes."""
    if len(liquid_a) != len(price_gap) or len(liquid_b) != len(price_gap):
        return np.nan
    left = price_gap[max(0, shock_iteration-baseline_window):shock_iteration]
    if not len(left):
        return np.nan
    threshold = sum(left)/len(left) + epsilon
    for t in range(shock_iteration, len(price_gap)):
        if price_gap[t] <= threshold and liquid_a[t] and liquid_b[t]:
            return t-shock_iteration
    return np.nan


def second_market_recovery_time(prices_b, shock_iteration, window_before=20, eps=1.0):
    left = prices_b[max(0, shock_iteration - window_before):shock_iteration]

    if len(left) == 0:
        return None

    baseline = sum(left) / len(left)

    for t in range(shock_iteration + 1, len(prices_b)):
        if abs(prices_b[t] - baseline) <= eps:
            return t - shock_iteration

    return None

def summarize_run(simulator, tau, beta, run_id, fx_cost, shock_iteration=200, window=50, ):
    prices_a = simulator.info.prices_a
    prices_b = simulator.info.prices_b
    fx_rates = simulator.info.fx_rates
    gap = simulator.info.price_gap

    left_gap = gap[max(0, shock_iteration - window):shock_iteration]
    right_gap = gap[shock_iteration:shock_iteration + window]

    vol = simulator.volatility_before_after(shock_iteration=shock_iteration, window=window)

    recovery = compute_recovery_time_relative(
        price_gap=gap,
        shock_iteration=shock_iteration,
        baseline_window=window,
        epsilon=0.5,
    )
    recovery_b = second_market_recovery_time(
        prices_b=simulator.info.prices_b,
        shock_iteration=shock_iteration,
        window_before=20,
        eps=1.0,
    )

    return {
        "run_id": run_id,
        "tau": tau,
        "beta": beta,
        "shock_iteration": shock_iteration,
        "avg_gap_before": float(np.mean(left_gap)) if len(left_gap) > 0 else np.nan,
        "avg_gap_after": float(np.mean(right_gap)) if len(right_gap) > 0 else np.nan,
        "max_gap_after": float(np.max(right_gap)) if len(right_gap) > 0 else np.nan,
        "recovery_time": recovery,
        "liquidity_recovery_time": liquidity_aware_recovery_time(
            gap, getattr(simulator.info, 'liquid_a', []), getattr(simulator.info, 'liquid_b', []),
            shock_iteration, baseline_window=window),
        "liquidity_recovery_available": (len(getattr(simulator.info, 'liquid_a', [])) == len(gap)
                                          and len(getattr(simulator.info, 'liquid_b', [])) == len(gap)),
        "vol_before": vol["before"],
        "vol_after": vol["after"],
        "final_price_a": prices_a[-1] if len(prices_a) > 0 else np.nan,
        "final_price_b": prices_b[-1] if len(prices_b) > 0 else np.nan,
        "final_fx": fx_rates[-1] if len(fx_rates) > 0 else np.nan,
        "final_gap": gap[-1] if len(gap) > 0 else np.nan,
        "fx_cost": fx_cost,
        "recovery_b": recovery_b,
    }


def save_timeseries(simulator, tau, beta, run_id, fx_cost, output_dir):
    df = pd.DataFrame({
        "price_a": simulator.info.prices_a,
        "price_b": simulator.info.prices_b,
        "fx": simulator.info.fx_rates,
        "gap": simulator.info.price_gap,
        "volume_a": simulator.info.traded_volume_a,
        "volume_b": simulator.info.traded_volume_b,
        "volume_fx": simulator.info.fx_volume,
    })

    name = f"timeseries_tau_{tau}_beta_{beta}_fx_{fx_cost}_run_{run_id}.csv"
    df.to_csv(Path(output_dir) / name, index=False)


def aggregate_results(raw_df):
    from recovery_reporting import valid_metrics, recovery_summary
    keys = ["tau", "beta", "fx_cost"]
    grouped = valid_metrics(raw_df).groupby(keys, dropna=False)

    summary_df = grouped.agg(
        n_runs=("run_id", "count"),
        avg_gap_before_mean=("avg_gap_before", "mean"),
        avg_gap_before_std=("avg_gap_before", "std"),
        avg_gap_after_mean=("avg_gap_after", "mean"),
        avg_gap_after_std=("avg_gap_after", "std"),
        max_gap_after_mean=("max_gap_after", "mean"),
        max_gap_after_std=("max_gap_after", "std"),
        recovery_time_mean=("recovery_time", "mean"),
        recovery_time_std=("recovery_time", "std"),
        vol_before_mean=("vol_before", "mean"),
        vol_before_std=("vol_before", "std"),
        vol_after_mean=("vol_after", "mean"),
        vol_after_std=("vol_after", "std"),
        final_gap_mean=("final_gap", "mean"),
        final_gap_std=("final_gap", "std"),
        recovery_b_mean=("recovery_b", "mean"),
        recovery_b_std=("recovery_b", "std"),
    ).reset_index()

    return summary_df.merge(recovery_summary(raw_df, keys), on=keys, validate="one_to_one")



def run_experiments(
    tau_values,
    beta_values,
    n_runs=50,
    steps=500,
    shock_iteration=200,
    save_full_timeseries=False,
    *,
    fx_cost_values=None,
    base_seed=42,
    workers=4,
    output_root=OUT_DIR,
    resume=None,
):
    from experiment_runner import execute_jobs

    tau_values, beta_values = list(tau_values), list(beta_values)
    fx_cost_values = list(fx_cost_values) if fx_cost_values is not None else [0.0, 0.001, 0.003, 0.005, 0.01]
    jobs = []
    for fx_cost in fx_cost_values:
        for tau in tau_values:
            for beta in beta_values:
                for run_id in range(n_runs):
                    jobs.append(dict(tau=tau, beta=beta, fx_cost=fx_cost,
                                     run_id=run_id, seed=base_seed + run_id,
                                     steps=steps, shock_iteration=shock_iteration))
    return execute_jobs(jobs, aggregate_results, workers=workers,
                        output_root=output_root, resume=resume,
                        save_full_timeseries=save_full_timeseries)


def run_robustness_checks(
    steps=500,
    shock_iteration=200,
    *, workers=4, output_root=OUT_DIR, resume=None,
):
    """
    Robustness checks:
    1) different numbers of simulation runs;
    2) different base seeds;
    3) different agent population sizes.

    The function saves:
    tables/<unique-run>/robustness_raw.csv
    tables/<unique-run>/robustness_summary.csv
    """

    selected_scenarios = [
        {"scenario": "baseline", "tau": 0.0, "beta": 0.0, "fx_cost": 0.000},
        {"scenario": "moderate_transport", "tau": 0.5, "beta": 0.0, "fx_cost": 0.001},
        {"scenario": "high_transport", "tau": 1.0, "beta": 0.0, "fx_cost": 0.001},
        {"scenario": "strong_spillover", "tau": 0.0, "beta": 0.6, "fx_cost": 0.001},
        {"scenario": "mixed", "tau": 0.5, "beta": 0.4, "fx_cost": 0.005},
        {"scenario": "stress", "tau": 1.0, "beta": 0.6, "fx_cost": 0.010},
    ]

    run_counts = [20, 50, 100]
    base_seeds = [42, 142, 242]

    agent_setups = {
        "low_agents": {
            "n_a_only_random": 8,
            "n_b_only_random": 8,
            "n_cross_random": 5,
            "n_a_only_fundamental": 4,
            "n_b_only_fundamental": 4,
            "n_cross_fundamental": 5,
            "n_fx_traders": 4,
        },
        "baseline_agents": {
            "n_a_only_random": 15,
            "n_b_only_random": 15,
            "n_cross_random": 10,
            "n_a_only_fundamental": 8,
            "n_b_only_fundamental": 8,
            "n_cross_fundamental": 10,
            "n_fx_traders": 8,
        },
        "high_agents": {
            "n_a_only_random": 23,
            "n_b_only_random": 23,
            "n_cross_random": 15,
            "n_a_only_fundamental": 12,
            "n_b_only_fundamental": 12,
            "n_cross_fundamental": 15,
            "n_fx_traders": 12,
        },
    }

    from experiment_runner import execute_jobs
    jobs = []
    for scenario in selected_scenarios:
        scenario_name = scenario["scenario"]
        tau = scenario["tau"]
        beta = scenario["beta"]
        fx_cost = scenario["fx_cost"]

        for n_runs in run_counts:
            for base_seed in base_seeds:
                for agent_setup_name, agent_kwargs in agent_setups.items():

                    for run_id in range(n_runs):
                        jobs.append(dict(
                            tau=tau, beta=beta, fx_cost=fx_cost, run_id=run_id,
                            seed=base_seed + run_id, steps=steps,
                            shock_iteration=shock_iteration, agent_kwargs=agent_kwargs,
                            extra=dict(scenario=scenario_name, n_runs_setting=n_runs,
                                       base_seed=base_seed, agent_setup=agent_setup_name),
                        ))
    return execute_jobs(jobs, aggregate_robustness, workers=workers,
                        output_root=output_root, resume=resume,
                        prefix="robustness")


def aggregate_robustness(raw_df):
    from recovery_reporting import valid_metrics, recovery_summary
    keys = ["scenario", "n_runs_setting", "base_seed", "agent_setup", "tau", "beta", "fx_cost"]
    summary_df = (
        valid_metrics(raw_df)
        .groupby(
            [
                "scenario",
                "n_runs_setting",
                "base_seed",
                "agent_setup",
                "tau",
                "beta",
                "fx_cost",
            ],
            dropna=False,
        )
        .agg(
            n_observations=("run_id", "count"),
            final_gap_mean=("final_gap", "mean"),
            final_gap_std=("final_gap", "std"),
            recovery_time_mean=("recovery_time", "mean"),
            recovery_time_std=("recovery_time", "std"),
            recovery_b_mean=("recovery_b", "mean"),
            recovery_b_std=("recovery_b", "std"),
            vol_before_mean=("vol_before", "mean"),
            vol_after_mean=("vol_after", "mean"),
        )
        .reset_index()
    )

    return summary_df.merge(recovery_summary(raw_df, keys), on=keys, validate="one_to_one")


def main():
    tau_values = [0.0, 0.1, 0.3, 0.5, 1.0]
    beta_values = [0.0, 0.2, 0.4, 0.6]

    raw_df, summary_df = run_experiments(
        tau_values=tau_values,
        beta_values=beta_values,
        n_runs=50,
        steps=500,
        shock_iteration=200,
        save_full_timeseries=False,
    )

    print()
    print("=== RAW RESULTS (first rows) ===")
    print(raw_df.head())

    print()
    print("=== SUMMARY RESULTS ===")
    print(summary_df)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reproducible multi-market experiments")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--shock-iteration", type=int, default=200)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--output-root", default=OUT_DIR)
    parser.add_argument("--resume")
    parser.add_argument("--save-timeseries", action="store_true")
    args = parser.parse_args()
    run_experiments([0.0, 0.1, 0.3, 0.5, 1.0], [0.0, 0.2, 0.4, 0.6],
                    n_runs=args.runs, steps=args.steps,
                    shock_iteration=args.shock_iteration, base_seed=args.base_seed,
                    workers=args.workers, output_root=args.output_root,
                    resume=args.resume, save_full_timeseries=args.save_timeseries)
