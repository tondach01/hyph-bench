#!/usr/bin/env python3
"""
Visualize GP regression at different iteration snapshots.

Creates plots similar to scikit-learn's GPR example, showing:
- Mean prediction
- 95% confidence interval
- Observed points

For each parameter dimension (bad_1, bad_2, bad_3, bad_4, threshold)
at iteration 30 by default.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
import pandas as pd
import os
import argparse

# ACL/IEEE reject Type 3 fonts; embed TrueType instead.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def score_values(history_df: pd.DataFrame) -> np.ndarray:
    """Return optimizer scores from old or validation-run history files."""
    if "score" in history_df.columns:
        return history_df["score"].values
    if "objective_score" in history_df.columns:
        return history_df["objective_score"].values
    raise KeyError("History CSV must contain either 'score' or 'objective_score'")


def fit_gp_snapshot(
    X: np.ndarray, y: np.ndarray, seed: int = 42
) -> GaussianProcessRegressor:
    """Fit a GP model on a snapshot of data."""
    kernel = Matern(
        nu=2.5, length_scale=[1.0] * X.shape[1], length_scale_bounds=(1e-3, 1e3)
    ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 1))

    gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, random_state=seed, n_restarts_optimizer=2
    )
    gp.fit(X, y)
    return gp


def predict_1d_slice(
    gp: GaussianProcessRegressor,
    X_train: np.ndarray,
    dim: int,
    x_range: tuple,
    n_points: int = 100,
) -> tuple:
    """
    Get GP predictions for 1D slice along dimension 'dim',
    holding other dimensions at their mean observed values.

    Returns: (x_test, mean, std)
    """
    # Create test points along this dimension
    x_test_1d = np.linspace(x_range[0], x_range[1], n_points)

    # Create full test array with means of other dimensions
    X_test = np.tile(X_train.mean(axis=0), (n_points, 1))
    X_test[:, dim] = x_test_1d

    # Predict
    mean, std = gp.predict(X_test, return_std=True)
    return x_test_1d, mean, std


def create_iteration_figure(
    history_df: pd.DataFrame,
    iteration: int,
    param_names: list,
    bounds: list,
    output_dir: str,
):
    """
    Create a figure with 5 subplots (one per parameter) for a given iteration.
    """
    # Get data up to this iteration
    df_snapshot = history_df.iloc[:iteration].copy()

    # Extract X and y
    param_cols = [f"param_{i}" for i in range(1, 6)]
    X = df_snapshot[param_cols].values
    y = score_values(df_snapshot)

    # Fit GP
    gp = fit_gp_snapshot(X, y)

    # Create figure with 5 subplots (2 rows, 3 cols - last spot empty or for legend)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, (name, (low, high)) in enumerate(zip(param_names, bounds)):
        ax = axes[i]

        # Get 1D predictions
        x_test, mean, std = predict_1d_slice(
            gp, X, dim=i, x_range=(low, high), n_points=100
        )

        # Plot mean prediction
        ax.plot(x_test, mean, "b-", label="Mean prediction", linewidth=2)

        # Plot 95% confidence interval
        ax.fill_between(
            x_test,
            mean - 1.96 * std,
            mean + 1.96 * std,
            alpha=0.3,
            color="tab:blue",
            label="95% confidence interval",
        )

        # Plot observed points (project to this dimension)
        ax.scatter(X[:, i], y, c="red", s=20, alpha=0.6, label="Observations", zorder=5)

        # Mark the best observation
        best_idx = np.argmax(y)
        ax.scatter(
            X[best_idx, i],
            y[best_idx],
            c="gold",
            s=100,
            marker="*",
            edgecolors="black",
            label="Best",
            zorder=10,
        )

        ax.set_xlabel(name, fontsize=11)
        ax.set_ylabel("Score" if i % 3 == 0 else "", fontsize=11)
        ax.set_title(f"{name}", fontsize=12, fontweight="bold")
        ax.set_xlim(low - 0.5, high + 0.5)
        ax.grid(True, alpha=0.3)

        # Only add legend to first subplot
        if i == 0:
            ax.legend(loc="lower right", fontsize=8)

    # Hide the 6th subplot (or use for info)
    axes[5].axis("off")

    # Add iteration info
    best_score = y.max()
    best_params = df_snapshot.iloc[np.argmax(y)][param_cols].values

    info_text = (
        f"Iteration: {iteration}\n"
        f"Best Score: {best_score:.6f}\n"
        f"Best Params:\n"
        f"  bad_1={int(best_params[0])}\n"
        f"  bad_2={int(best_params[1])}\n"
        f"  bad_3={int(best_params[2])}\n"
        f"  bad_4={int(best_params[3])}\n"
        f"  threshold={int(best_params[4])}"
    )
    axes[5].text(
        0.1,
        0.5,
        info_text,
        fontsize=12,
        family="monospace",
        verticalalignment="center",
        transform=axes[5].transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.suptitle(
        f"GP Regression at Iteration {iteration}", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, f"gp_iteration_{iteration:03d}.pdf")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    # Also save PNG for quick preview
    output_path_png = os.path.join(output_dir, f"gp_iteration_{iteration:03d}.png")
    plt.savefig(output_path_png, dpi=150, bbox_inches="tight")

    plt.close()

    return best_score, best_params


def create_combined_figure(
    history_df: pd.DataFrame,
    iterations: list,
    param_names: list,
    bounds: list,
    output_dir: str,
):
    """
    Create a single combined figure showing GP evolution across iterations.
    One row per iteration, one column per parameter.
    """
    n_iters = len(iterations)
    n_params = len(param_names)

    fig, axes = plt.subplots(n_iters, n_params, figsize=(3 * n_params, 2.5 * n_iters))

    for row, iteration in enumerate(iterations):
        # Get data up to this iteration
        df_snapshot = history_df.iloc[:iteration].copy()
        param_cols = [f"param_{i}" for i in range(1, 6)]
        X = df_snapshot[param_cols].values
        y = score_values(df_snapshot)

        # Fit GP
        gp = fit_gp_snapshot(X, y)

        for col, (name, (low, high)) in enumerate(zip(param_names, bounds)):
            ax = axes[row, col]

            # Get 1D predictions
            x_test, mean, std = predict_1d_slice(
                gp, X, dim=col, x_range=(low, high), n_points=100
            )

            # Plot
            ax.plot(x_test, mean, "b-", linewidth=1.5)
            ax.fill_between(
                x_test,
                mean - 1.96 * std,
                mean + 1.96 * std,
                alpha=0.3,
                color="tab:blue",
            )
            ax.scatter(X[:, col], y, c="red", s=10, alpha=0.4)

            # Best point
            best_idx = np.argmax(y)
            ax.scatter(
                X[best_idx, col],
                y[best_idx],
                c="gold",
                s=50,
                marker="*",
                edgecolors="black",
                zorder=10,
            )

            ax.set_xlim(low - 0.5, high + 0.5)
            ax.grid(True, alpha=0.2)

            # Labels
            if row == 0:
                ax.set_title(name, fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"Iter {iteration}", fontsize=10)
            if row == n_iters - 1:
                ax.set_xlabel(name, fontsize=9)

    fig.suptitle(
        "GP Regression Evolution Across Iterations",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    output_path = os.path.join(output_dir, "gp_evolution_combined.pdf")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")

    output_path_png = os.path.join(output_dir, "gp_evolution_combined.png")
    plt.savefig(output_path_png, dpi=150, bbox_inches="tight")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize GP regression at iteration snapshots"
    )
    parser.add_argument("--lang", default="cssk", help="Language code")
    parser.add_argument(
        "--history", type=str, help="Path to history CSV (overrides --lang)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=[30],
        help="Iterations to visualize",
    )
    parser.add_argument("--output-dir", default="figures", help="Output directory")
    parser.add_argument(
        "--combined", action="store_true", help="Also create combined figure"
    )
    args = parser.parse_args()

    # Load history
    if args.history:
        history_path = args.history
    else:
        history_path = f"results/{args.lang}_history.csv"

    print(f"Loading history from: {history_path}")
    df = pd.read_csv(history_path)
    print(f"Total iterations in history: {len(df)}")

    # Parameter names and bounds
    param_names = [
        "bad_1 (Level 1)",
        "bad_2 (Level 2)",
        "bad_3 (Level 3)",
        "bad_4 (Level 4)",
        "threshold",
    ]
    bounds = [(1, 30), (1, 30), (1, 30), (1, 30), (1, 5)]  # Assuming max_bad_weight=30

    # Detect actual bounds from data
    for i in range(5):
        col = f"param_{i + 1}"
        actual_min = df[col].min()
        actual_max = df[col].max()
        bounds[i] = (max(1, actual_min - 1), actual_max + 1)

    print(f"Detected bounds: {bounds}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Filter iterations that are available
    available_iterations = [it for it in args.iterations if it <= len(df)]
    if len(available_iterations) < len(args.iterations):
        print(f"Warning: Some iterations not available. Using: {available_iterations}")

    # Generate individual figures
    print("\nGenerating individual iteration figures...")
    for iteration in available_iterations:
        best_score, best_params = create_iteration_figure(
            df, iteration, param_names, bounds, args.output_dir
        )
        print(f"  Iteration {iteration}: best score = {best_score:.6f}")

    # Generate combined figure
    if args.combined or True:  # Always generate combined
        print("\nGenerating combined evolution figure...")
        create_combined_figure(
            df, available_iterations, param_names, bounds, args.output_dir
        )

    print(f"\nAll figures saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
