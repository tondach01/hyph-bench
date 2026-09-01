#!/usr/bin/env python3
"""
2D visualization of GP regression surfaces.

Creates heatmaps showing the interaction between parameter pairs,
revealing landscape features that 1D slices miss.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
import pandas as pd
import os
import argparse
from itertools import combinations


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


def predict_2d_slice(
    gp: GaussianProcessRegressor,
    X_train: np.ndarray,
    dim1: int,
    dim2: int,
    range1: tuple,
    range2: tuple,
    n_points: int = 50,
) -> tuple:
    """
    Get GP predictions for 2D slice along dimensions dim1 and dim2,
    holding other dimensions at their best observed values.

    Returns: (xx, yy, mean_grid, std_grid)
    """
    # Create meshgrid
    x1 = np.linspace(range1[0], range1[1], n_points)
    x2 = np.linspace(range2[0], range2[1], n_points)
    xx, yy = np.meshgrid(x1, x2)

    # Flatten for prediction
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    # Create full test array - use best point for other dimensions
    best_idx = np.argmax(gp.y_train_)
    X_test = np.tile(X_train[best_idx], (len(grid_points), 1))
    X_test[:, dim1] = grid_points[:, 0]
    X_test[:, dim2] = grid_points[:, 1]

    # Predict
    mean, std = gp.predict(X_test, return_std=True)

    # Reshape to grid
    mean_grid = mean.reshape(xx.shape)
    std_grid = std.reshape(xx.shape)

    return xx, yy, mean_grid, std_grid


def create_pairwise_heatmaps(
    history_df: pd.DataFrame,
    iteration: int,
    param_names: list,
    bounds: list,
    output_dir: str,
    param_cols: list = None,
):
    """
    Create a triangular matrix of 2D heatmaps for all parameter pairs.
    """
    # Get data up to this iteration
    df_snapshot = history_df.iloc[:iteration].copy()
    if param_cols is None:
        param_cols = [f"param_{i}" for i in range(1, 6)]
    X = df_snapshot[param_cols].values
    y = score_values(df_snapshot)

    # Fit GP
    print(f"  Fitting GP on {len(X)} points...")
    gp = fit_gp_snapshot(X, y)

    n_params = len(param_names)
    pairs = list(combinations(range(n_params), 2))

    # Create figure - triangle layout
    fig, axes = plt.subplots(n_params - 1, n_params - 1, figsize=(14, 12))

    # Track global colorbar range
    all_means = []

    # First pass: compute all predictions to get color range
    predictions = {}
    for i, j in pairs:
        xx, yy, mean_grid, std_grid = predict_2d_slice(
            gp, X, i, j, bounds[i], bounds[j], n_points=40
        )
        predictions[(i, j)] = (xx, yy, mean_grid, std_grid)
        all_means.extend(mean_grid.ravel())

    # Color normalization
    vmin, vmax = np.percentile(all_means, [2, 98])
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Best point
    best_idx = np.argmax(y)
    best_x = X[best_idx]
    best_y = y[best_idx]

    # Second pass: plot
    for i, j in pairs:
        # Position in lower triangle: row = j-1, col = i
        ax = axes[j - 1, i]

        xx, yy, mean_grid, std_grid = predictions[(i, j)]

        # Heatmap
        im = ax.contourf(xx, yy, mean_grid, levels=30, cmap="viridis", norm=norm)
        ax.contour(
            xx, yy, mean_grid, levels=10, colors="white", alpha=0.3, linewidths=0.5
        )

        # Scatter observed points
        ax.scatter(
            X[:, i],
            X[:, j],
            c="white",
            s=15,
            alpha=0.6,
            edgecolors="black",
            linewidths=0.5,
        )

        # Mark best point
        ax.scatter(
            best_x[i],
            best_x[j],
            c="gold",
            s=200,
            marker="*",
            edgecolors="black",
            linewidths=1.5,
            zorder=10,
        )

        # Labels
        if j == n_params - 1:
            ax.set_xlabel(param_names[i], fontsize=10)
        if i == 0:
            ax.set_ylabel(param_names[j], fontsize=10)

        ax.set_xlim(bounds[i])
        ax.set_ylim(bounds[j])

    # Hide upper triangle
    for i in range(n_params - 1):
        for j in range(n_params - 1):
            if j < i:
                axes[j, i].axis("off")

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Predicted Score", fontsize=11)

    # Title with info
    fig.suptitle(
        f"GP Surface: Pairwise Parameter Interactions (Iteration {iteration})\n"
        f"Best: {best_y:.6f} at [{', '.join(f'{int(v)}' for v in best_x)}]",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    # Save
    output_path = os.path.join(output_dir, f"gp_2d_heatmap_{iteration:03d}.pdf")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")

    output_path_png = os.path.join(output_dir, f"gp_2d_heatmap_{iteration:03d}.png")
    plt.savefig(output_path_png, dpi=150, bbox_inches="tight")

    plt.close()


def create_key_pairs_figure(
    history_df: pd.DataFrame,
    iteration: int,
    param_names: list,
    bounds: list,
    output_dir: str,
    key_pairs: list = None,
):
    """
    Create a focused figure showing only the most interesting parameter pairs.
    Includes uncertainty visualization.
    """
    df_snapshot = history_df.iloc[:iteration].copy()
    param_cols = [f"param_{i}" for i in range(1, 6)]
    X = df_snapshot[param_cols].values
    y = score_values(df_snapshot)

    print(f"  Fitting GP on {len(X)} points...")
    gp = fit_gp_snapshot(X, y)

    # Default key pairs: (0,1), (0,4), (2,3) - bad1 vs bad2, bad1 vs threshold, bad3 vs bad4
    if key_pairs is None:
        key_pairs = [(0, 1), (0, 4), (2, 3), (1, 2)]

    n_pairs = len(key_pairs)
    fig, axes = plt.subplots(2, n_pairs, figsize=(4 * n_pairs, 8))

    best_idx = np.argmax(y)
    best_x = X[best_idx]
    best_y = y[best_idx]

    for col, (i, j) in enumerate(key_pairs):
        xx, yy, mean_grid, std_grid = predict_2d_slice(
            gp, X, i, j, bounds[i], bounds[j], n_points=50
        )

        # Top row: Mean prediction
        ax_mean = axes[0, col]
        im_mean = ax_mean.contourf(xx, yy, mean_grid, levels=30, cmap="viridis")
        ax_mean.contour(
            xx, yy, mean_grid, levels=8, colors="white", alpha=0.4, linewidths=0.5
        )
        ax_mean.scatter(
            X[:, i],
            X[:, j],
            c="white",
            s=20,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
        )
        ax_mean.scatter(
            best_x[i],
            best_x[j],
            c="gold",
            s=250,
            marker="*",
            edgecolors="black",
            linewidths=1.5,
            zorder=10,
        )
        ax_mean.set_xlabel(param_names[i], fontsize=11)
        ax_mean.set_ylabel(param_names[j], fontsize=11)
        ax_mean.set_title(
            f"{param_names[i]} vs {param_names[j]}\n(Mean)",
            fontsize=10,
            fontweight="bold",
        )
        plt.colorbar(im_mean, ax=ax_mean, fraction=0.046, pad=0.04)

        # Bottom row: Uncertainty (std)
        ax_std = axes[1, col]
        im_std = ax_std.contourf(xx, yy, std_grid, levels=30, cmap="plasma")
        ax_std.contour(
            xx, yy, std_grid, levels=8, colors="white", alpha=0.4, linewidths=0.5
        )
        ax_std.scatter(
            X[:, i],
            X[:, j],
            c="white",
            s=20,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
        )
        ax_std.scatter(
            best_x[i],
            best_x[j],
            c="lime",
            s=250,
            marker="*",
            edgecolors="black",
            linewidths=1.5,
            zorder=10,
        )
        ax_std.set_xlabel(param_names[i], fontsize=11)
        ax_std.set_ylabel(param_names[j], fontsize=11)
        ax_std.set_title(f"(Uncertainty)", fontsize=10)
        plt.colorbar(im_std, ax=ax_std, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"GP Surface: Mean Prediction & Uncertainty (Iteration {iteration})\n"
        f"Best Score: {best_y:.6f}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"gp_2d_keypairs_{iteration:03d}.pdf")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")

    output_path_png = os.path.join(output_dir, f"gp_2d_keypairs_{iteration:03d}.png")
    plt.savefig(output_path_png, dpi=150, bbox_inches="tight")

    plt.close()


def create_acquisition_landscape(
    history_df: pd.DataFrame,
    iteration: int,
    param_names: list,
    bounds: list,
    output_dir: str,
    kappa: float = 2.0,
):
    """
    Visualize the UCB acquisition function landscape.
    Shows where the optimizer "wants" to explore next.
    """
    df_snapshot = history_df.iloc[:iteration].copy()
    param_cols = [f"param_{i}" for i in range(1, 6)]
    X = df_snapshot[param_cols].values
    y = score_values(df_snapshot)

    print(f"  Fitting GP for acquisition landscape...")
    gp = fit_gp_snapshot(X, y)

    # Pick the two most important dimensions (bad_1 and threshold tend to matter most)
    dim1, dim2 = 0, 4  # bad_1 vs threshold

    # Create grid
    n_points = 60
    x1 = np.linspace(bounds[dim1][0], bounds[dim1][1], n_points)
    x2 = np.linspace(bounds[dim2][0], bounds[dim2][1], n_points)
    xx, yy = np.meshgrid(x1, x2)

    # Build full test array
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    best_idx = np.argmax(y)
    X_test = np.tile(X[best_idx], (len(grid_points), 1))
    X_test[:, dim1] = grid_points[:, 0]
    X_test[:, dim2] = grid_points[:, 1]

    mean, std = gp.predict(X_test, return_std=True)

    # UCB acquisition function
    ucb = mean + kappa * std

    mean_grid = mean.reshape(xx.shape)
    std_grid = std.reshape(xx.shape)
    ucb_grid = ucb.reshape(xx.shape)

    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    best_x = X[best_idx]

    # Panel 1: Mean
    ax = axes[0]
    im = ax.contourf(xx, yy, mean_grid, levels=30, cmap="viridis")
    ax.contour(xx, yy, mean_grid, levels=10, colors="white", alpha=0.3, linewidths=0.5)
    ax.scatter(
        X[:, dim1],
        X[:, dim2],
        c="white",
        s=30,
        alpha=0.7,
        edgecolors="black",
        linewidths=0.5,
    )
    ax.scatter(
        best_x[dim1],
        best_x[dim2],
        c="gold",
        s=300,
        marker="*",
        edgecolors="black",
        linewidths=2,
        zorder=10,
    )
    ax.set_xlabel(param_names[dim1], fontsize=12)
    ax.set_ylabel(param_names[dim2], fontsize=12)
    ax.set_title("Mean Prediction (Exploitation)", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax)

    # Panel 2: Uncertainty
    ax = axes[1]
    im = ax.contourf(xx, yy, std_grid, levels=30, cmap="plasma")
    ax.contour(xx, yy, std_grid, levels=10, colors="white", alpha=0.3, linewidths=0.5)
    ax.scatter(
        X[:, dim1],
        X[:, dim2],
        c="white",
        s=30,
        alpha=0.7,
        edgecolors="black",
        linewidths=0.5,
    )
    ax.scatter(
        best_x[dim1],
        best_x[dim2],
        c="lime",
        s=300,
        marker="*",
        edgecolors="black",
        linewidths=2,
        zorder=10,
    )
    ax.set_xlabel(param_names[dim1], fontsize=12)
    ax.set_ylabel(param_names[dim2], fontsize=12)
    ax.set_title("Uncertainty (Exploration)", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax)

    # Panel 3: UCB (what the optimizer sees)
    ax = axes[2]
    im = ax.contourf(xx, yy, ucb_grid, levels=30, cmap="RdYlGn")
    ax.contour(xx, yy, ucb_grid, levels=10, colors="black", alpha=0.3, linewidths=0.5)
    ax.scatter(
        X[:, dim1],
        X[:, dim2],
        c="black",
        s=30,
        alpha=0.7,
        edgecolors="white",
        linewidths=0.5,
    )

    # Mark the UCB maximum
    ucb_max_idx = np.argmax(ucb_grid)
    ucb_max_x = xx.ravel()[ucb_max_idx]
    ucb_max_y = yy.ravel()[ucb_max_idx]
    ax.scatter(
        ucb_max_x,
        ucb_max_y,
        c="red",
        s=300,
        marker="X",
        edgecolors="white",
        linewidths=2,
        zorder=10,
        label="Next sample",
    )
    ax.scatter(
        best_x[dim1],
        best_x[dim2],
        c="gold",
        s=300,
        marker="*",
        edgecolors="black",
        linewidths=2,
        zorder=10,
        label="Best observed",
    )

    ax.set_xlabel(param_names[dim1], fontsize=12)
    ax.set_ylabel(param_names[dim2], fontsize=12)
    ax.set_title(f"UCB Acquisition (κ={kappa})", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    plt.colorbar(im, ax=ax)

    fig.suptitle(
        f"Acquisition Function Landscape (Iteration {iteration})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"gp_acquisition_{iteration:03d}.pdf")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")

    output_path_png = os.path.join(output_dir, f"gp_acquisition_{iteration:03d}.png")
    plt.savefig(output_path_png, dpi=150, bbox_inches="tight")

    plt.close()


def create_evolution_animation_frames(
    history_df: pd.DataFrame,
    iterations: list,
    param_names: list,
    bounds: list,
    output_dir: str,
    dim1: int = 0,
    dim2: int = 4,
):
    """
    Create frames showing GP surface evolution for a single pair of parameters.
    Good for making an animated GIF later.
    """
    param_cols = [f"param_{i}" for i in range(1, 6)]

    # Pre-compute grid
    n_points = 50
    x1 = np.linspace(bounds[dim1][0], bounds[dim1][1], n_points)
    x2 = np.linspace(bounds[dim2][0], bounds[dim2][1], n_points)
    xx, yy = np.meshgrid(x1, x2)

    # Track global range for consistent coloring
    all_means = []
    for iteration in iterations:
        df_snapshot = history_df.iloc[:iteration].copy()
        X = df_snapshot[param_cols].values
        y = score_values(df_snapshot)
        all_means.extend(y)

    vmin, vmax = min(all_means), max(all_means)
    norm = Normalize(vmin=vmin, vmax=vmax)

    frames_dir = os.path.join(output_dir, "animation_frames")
    os.makedirs(frames_dir, exist_ok=True)

    for iteration in iterations:
        df_snapshot = history_df.iloc[:iteration].copy()
        X = df_snapshot[param_cols].values
        y = score_values(df_snapshot)

        gp = fit_gp_snapshot(X, y)

        # Build test array
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        best_idx = np.argmax(y)
        X_test = np.tile(X[best_idx], (len(grid_points), 1))
        X_test[:, dim1] = grid_points[:, 0]
        X_test[:, dim2] = grid_points[:, 1]

        mean, std = gp.predict(X_test, return_std=True)
        mean_grid = mean.reshape(xx.shape)

        # Create frame
        fig, ax = plt.subplots(figsize=(8, 7))

        im = ax.contourf(xx, yy, mean_grid, levels=30, cmap="viridis", norm=norm)
        ax.contour(
            xx, yy, mean_grid, levels=10, colors="white", alpha=0.3, linewidths=0.5
        )

        # Show all observations up to this point
        ax.scatter(
            X[:, dim1],
            X[:, dim2],
            c="white",
            s=30,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
        )

        # Best point
        best_x = X[best_idx]
        ax.scatter(
            best_x[dim1],
            best_x[dim2],
            c="gold",
            s=300,
            marker="*",
            edgecolors="black",
            linewidths=2,
            zorder=10,
        )

        ax.set_xlabel(param_names[dim1], fontsize=14)
        ax.set_ylabel(param_names[dim2], fontsize=14)
        ax.set_title(
            f"Iteration {iteration} | Best: {y[best_idx]:.6f}",
            fontsize=14,
            fontweight="bold",
        )

        plt.colorbar(im, ax=ax, label="Predicted Score")
        plt.tight_layout()

        frame_path = os.path.join(frames_dir, f"frame_{iteration:03d}.png")
        plt.savefig(frame_path, dpi=100)
        plt.close()

    print(f"  Saved {len(iterations)} animation frames to {frames_dir}/")
    print(
        f"  To create GIF: convert -delay 50 -loop 0 {frames_dir}/frame_*.png {output_dir}/gp_evolution.gif"
    )


def main():
    parser = argparse.ArgumentParser(
        description="2D visualization of GP regression surfaces"
    )
    parser.add_argument("--lang", default="cssk", help="Language code")
    parser.add_argument("--history", type=str, help="Path to history CSV")
    parser.add_argument(
        "--iteration", type=int, default=30, help="Iteration to visualize"
    )
    parser.add_argument("--output-dir", default="figures", help="Output directory")
    parser.add_argument(
        "--mode",
        choices=["all", "heatmap", "keypairs", "acquisition", "animation"],
        default="all",
        help="Visualization mode",
    )
    parser.add_argument(
        "--exclude-threshold",
        action="store_true",
        help="Exclude threshold parameter (use only bad_1-4)",
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

    iteration = min(args.iteration, len(df))
    print(f"Visualizing iteration: {iteration}")

    # Parameter names and bounds
    if args.exclude_threshold:
        param_names = ["bad_1", "bad_2", "bad_3", "bad_4"]
        param_cols = [f"param_{i}" for i in range(1, 5)]
        n_params = 4
    else:
        param_names = ["bad_1", "bad_2", "bad_3", "bad_4", "threshold"]
        param_cols = [f"param_{i}" for i in range(1, 6)]
        n_params = 5

    bounds = [(1, 16)] * n_params

    # Detect actual bounds from data
    for i in range(n_params):
        col = param_cols[i]
        actual_min = df[col].min()
        actual_max = df[col].max()
        bounds[i] = (max(1, actual_min - 0.5), actual_max + 0.5)

    print(f"Detected bounds: {bounds}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode in ["all", "heatmap"]:
        print("\nCreating pairwise heatmaps...")
        create_pairwise_heatmaps(
            df, iteration, param_names, bounds, args.output_dir, param_cols
        )

    if args.mode in ["all", "keypairs"]:
        print("\nCreating key pairs figure...")
        create_key_pairs_figure(df, iteration, param_names, bounds, args.output_dir)

    if args.mode in ["all", "acquisition"]:
        print("\nCreating acquisition landscape...")
        create_acquisition_landscape(
            df, iteration, param_names, bounds, args.output_dir
        )

    if args.mode == "animation":
        print("\nCreating animation frames...")
        iterations = list(range(10, min(iteration + 1, len(df)), 5))
        create_evolution_animation_frames(
            df, iterations, param_names, bounds, args.output_dir
        )

    print(f"\nAll figures saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
