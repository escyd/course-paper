import os
import random

import numpy as np
import pandas as pd

from experiments import build_system, summarize_run


OUT_DIR = "tables/"


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


def run_robustness_checks(steps=500, shock_iteration=200):
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

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

    rows = []

    for scenario in selected_scenarios:
        scenario_name = scenario["scenario"]
        tau = scenario["tau"]
        beta = scenario["beta"]
        fx_cost = scenario["fx_cost"]

        for n_runs in run_counts:
            for base_seed in base_seeds:
                for agent_setup_name, agent_kwargs in agent_setups.items():

                    print(
                        f"Robustness: scenario={scenario_name}, "
                        f"n_runs={n_runs}, "
                        f"base_seed={base_seed}, "
                        f"agents={agent_setup_name}, "
                        f"tau={tau}, beta={beta}, fx_cost={fx_cost}"
                    )

                    for run_id in range(n_runs):
                        seed = base_seed + run_id
                        set_all_seeds(seed)

                        simulator = build_system(
                            tau=tau,
                            beta=beta,
                            fx_cost=fx_cost,
                            shock_iteration=shock_iteration,
                            **agent_kwargs,
                        )

                        simulator.simulate(steps)

                        row = summarize_run(
                            simulator=simulator,
                            tau=tau,
                            beta=beta,
                            run_id=run_id,
                            fx_cost=fx_cost,
                            shock_iteration=shock_iteration,
                            window=50,
                        )

                        row["scenario"] = scenario_name
                        row["n_runs_setting"] = n_runs
                        row["base_seed"] = base_seed
                        row["seed"] = seed
                        row["agent_setup"] = agent_setup_name

                        rows.append(row)

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(OUT_DIR + "robustness_raw.csv", index=False)

    summary_df = (
        raw_df
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

    summary_df.to_csv(OUT_DIR + "robustness_summary.csv", index=False)

    print()
    print("Saved:")
    print(OUT_DIR + "robustness_raw.csv")
    print(OUT_DIR + "robustness_summary.csv")
    print()
    print(summary_df.head())


if __name__ == "__main__":
    run_robustness_checks()
