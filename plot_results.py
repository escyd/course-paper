import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

OUT_DIR = "figures/"
FIG_SIZE = (9, 6)
DPI = 300

plt.style.use("seaborn-v0_8")


def clear_figures():
    """Start a new figure directory; preserve all previous figures."""
    global OUT_DIR
    from datetime import datetime, timezone
    from pathlib import Path
    import uuid
    directory = Path("figures") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    OUT_DIR = str(directory) + "/"


def make_pivot(df, value_col):
    part = df.dropna(subset=["tau", "beta", value_col]).copy()
    pivot = part.pivot_table(index="tau", columns="beta", values=value_col, aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)
    return pivot


def draw_heatmap(pivot_df, title, output_name):
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    im = ax.imshow(
        pivot_df.values,
        cmap="viridis",
        aspect="auto",
        vmin=np.nanmin(pivot_df.values),
        vmax=np.nanmax(pivot_df.values)
    )

    ax.set_title(title)
    ax.set_xlabel("beta")
    ax.set_ylabel("tau")

    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels([str(x) for x in pivot_df.columns])

    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels([str(x) for x in pivot_df.index])

    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            value = pivot_df.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(OUT_DIR + output_name, dpi=DPI)
    plt.show()


def draw_surface(df):
    pivot = df.pivot(index="tau", columns="beta", values="final_gap_mean")
    pivot = pivot.sort_index().sort_index(axis=1)

    tau_vals = pivot.index.values
    beta_vals = pivot.columns.values

    x_grid, y_grid = np.meshgrid(beta_vals, tau_vals)
    z_grid = pivot.values

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(x_grid, y_grid, z_grid)

    ax.set_xlabel("beta")
    ax.set_ylabel("tau")
    ax.set_zlabel("final_gap_mean")
    ax.set_title("Price gap surface")

    plt.tight_layout()
    plt.savefig(OUT_DIR + "surface_gap.png", dpi=DPI)
    plt.show()


def draw_tau_gap_lines(df):
    plt.figure(figsize=FIG_SIZE)

    betas = sorted(df["beta"].unique())

    for beta in betas:
        part = df[df["beta"] == beta].sort_values("tau")
        plt.plot(part["tau"], part["final_gap_mean"], marker="o", label=f"beta={beta}")

    plt.title("Final gap mean vs tau")
    plt.xlabel("tau")
    plt.ylabel("final_gap_mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR + "tau_to_gap_lines.png", dpi=DPI)
    plt.show()


def draw_tau_gap_std_lines(df):
    plt.figure(figsize=FIG_SIZE)
    betas = sorted(df["beta"].dropna().unique())
    has_any = False
    for beta in betas:
        part = df[df["beta"] == beta].copy()
        part = part.sort_values("tau")
        part = part.dropna(subset=["tau", "final_gap_std"])
        if len(part) == 0:
            continue
        has_any = True
        plt.plot(part["tau"], part["final_gap_std"], marker="o", label=f"beta={beta}")
    if not has_any:
        plt.close()
        raise ValueError("Для графика final_gap_std vs tau нет валидных данных")

    plt.title("Final gap std vs tau")
    plt.xlabel("tau")
    plt.ylabel("final_gap_std")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR + "tau_to_gap_std_lines.png", dpi=DPI)
    plt.show()


def choose_best_fx_cost(df, value_cols):
    scores = []

    for fx_cost in sorted(df["fx_cost"].dropna().unique()):
        part = df[df["fx_cost"] == fx_cost].copy()

        score = 0
        for col in value_cols:
            good = part.dropna(subset=["tau", "beta", col])
            score += len(good)

        scores.append((score, fx_cost))

    scores.sort(reverse=True)
    return scores[0][1]


def draw_recovery_b_vs_fx_cost(df, fixed_tau=0.5, fixed_beta=0.4):
    part = df[(df["tau"] == fixed_tau) & (df["beta"] == fixed_beta)].copy()
    part = part.sort_values("recovery_b_mean")

    plt.figure(figsize=FIG_SIZE)

    plt.plot(
  part["recovery_b_mean"],
        part["fx_cost"],
        marker="o"
    )

    plt.title(f"Recovery of Market B vs FX cost (tau={fixed_tau}, beta={fixed_beta})")
    plt.xlabel("recovery time of Market B")
    plt.ylabel("fx transaction cost")
    plt.tight_layout()
    plt.savefig(OUT_DIR + "recovery_b_vs_fx_cost.png", dpi=DPI)
    plt.show()


def main():
    clear_figures()

    df = pd.read_csv("tables/summary_results.csv")

    print(df[["tau", "beta", "fx_cost", "final_gap_mean", "final_gap_std"]]
          .sort_values(["fx_cost", "beta", "tau"]))

    best_fx_cost = choose_best_fx_cost(
        df,
        ["final_gap_mean", "final_gap_std"]
    )

    print("Использую fx_cost для основных графиков:", best_fx_cost)

    base_df = df[df["fx_cost"] == best_fx_cost].copy()

    pivot_gap_mean = make_pivot(base_df, "final_gap_mean")
    pivot_gap_std = make_pivot(base_df, "final_gap_std")

    draw_heatmap(
        pivot_gap_mean,
        f"Final gap mean: tau x beta (fx_cost={best_fx_cost})",
        "heatmap_final_gap_mean.png"
    )

    draw_heatmap(
        pivot_gap_std,
        f"Final gap std: tau x beta (fx_cost={best_fx_cost})",
        "heatmap_final_gap_std.png"
    )

    draw_surface(base_df)
    draw_tau_gap_lines(base_df)
    draw_tau_gap_std_lines(base_df)

    draw_recovery_b_vs_fx_cost(df, fixed_tau=0.5, fixed_beta=0.4)


if __name__ == "__main__":
    main()
