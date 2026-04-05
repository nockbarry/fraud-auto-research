"""Plot experiment results over time with improvement annotations.

Generates a multi-panel chart showing composite score, AUPRC, precision@recall,
and PSI over experiments, with annotations pointing to each improvement.

Usage:
    python3 -m harness.plot_results
    python3 -m harness.plot_results --out results.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from harness.utils import ROOT_DIR


def load_results(tsv_path: Path | None = None) -> pd.DataFrame:
    """Load results.tsv into a DataFrame."""
    if tsv_path is None:
        tsv_path = ROOT_DIR / "results.tsv"
    df = pd.read_csv(tsv_path, sep="\t")
    df.index = range(1, len(df) + 1)
    df.index.name = "experiment"
    return df


def plot_results(df: pd.DataFrame, out_path: str | None = None):
    """Create annotated metric plots over time."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle("Fraud Auto-Research: Experiment History", fontsize=16, fontweight="bold", y=0.98)

    metrics = [
        ("composite", "Composite Score", "#2563eb", True),
        ("auprc", "AUPRC (OOT)", "#059669", True),
        ("prec@recall", "Precision @ 80% Recall", "#d97706", True),
        ("psi", "PSI (Val→OOT)", "#dc2626", False),
    ]

    # Identify keeps, discards, crashes
    keeps = df[df["status"] == "keep"]
    discards = df[df["status"] == "discard"]
    crashes = df[df["status"].isin(["crash", "reject_psi"])]

    for ax, (col, title, color, higher_better) in zip(axes, metrics):
        if col not in df.columns:
            ax.set_visible(False)
            continue

        vals = pd.to_numeric(df[col], errors="coerce")

        # Plot all experiments as light dots
        ax.scatter(df.index, vals, color="#e5e7eb", s=30, zorder=2, label="_nolegend_")

        # Overlay discards
        if len(discards) > 0:
            discard_vals = pd.to_numeric(discards[col], errors="coerce")
            ax.scatter(discards.index, discard_vals, color="#fca5a5", s=30,
                      zorder=3, marker="x", linewidths=1.5, label="discard")

        # Overlay crashes
        if len(crashes) > 0:
            crash_vals = pd.to_numeric(crashes[col], errors="coerce")
            ax.scatter(crashes.index, crash_vals, color="#f87171", s=40,
                      zorder=3, marker="D", label="crash/reject")

        # Plot keeps as connected line with markers
        if len(keeps) > 0:
            keep_vals = pd.to_numeric(keeps[col], errors="coerce")
            ax.plot(keeps.index, keep_vals, color=color, linewidth=2.5,
                   marker="o", markersize=7, zorder=5, label="keep")

            # Draw running best line
            if higher_better:
                running_best = keep_vals.cummax()
            else:
                running_best = keep_vals.cummin()
            ax.step(keeps.index, running_best, color=color, linewidth=1,
                   linestyle="--", alpha=0.5, zorder=4, label="best so far")

            # Annotate improvements (when a keep improves on previous best)
            prev_best = None
            for idx, row in keeps.iterrows():
                val = pd.to_numeric(row[col], errors="coerce")
                if pd.isna(val):
                    continue

                is_improvement = False
                if prev_best is None:
                    is_improvement = True  # baseline
                elif higher_better and val > prev_best + 0.0005:
                    is_improvement = True
                elif not higher_better and val < prev_best - 0.0005:
                    is_improvement = True

                if is_improvement:
                    # Truncate hypothesis for annotation
                    hyp = str(row.get("hypothesis", ""))
                    if len(hyp) > 50:
                        hyp = hyp[:47] + "..."

                    # Alternate annotation positions to avoid overlap
                    y_offset = 15 if (list(keeps.index).index(idx) % 2 == 0) else -20

                    ax.annotate(
                        f"{val:.4f}\n{hyp}",
                        xy=(idx, val),
                        xytext=(10, y_offset),
                        textcoords="offset points",
                        fontsize=7,
                        color=color,
                        fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                 edgecolor=color, alpha=0.9),
                        zorder=10,
                    )

                    if higher_better:
                        prev_best = max(val, prev_best) if prev_best is not None else val
                    else:
                        prev_best = min(val, prev_best) if prev_best is not None else val

        # PSI threshold lines
        if col == "psi":
            ax.axhline(y=0.20, color="#fbbf24", linestyle="--", alpha=0.7, label="penalty threshold")
            ax.axhline(y=0.25, color="#ef4444", linestyle="--", alpha=0.7, label="hard reject")

        ax.set_ylabel(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left" if higher_better else "upper right", fontsize=8)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    axes[-1].set_xlabel("Experiment #", fontsize=11)

    # Summary stats at bottom
    n_total = len(df)
    n_keeps = len(keeps)
    n_discards = len(discards)
    n_crashes = len(crashes)

    if len(keeps) > 0:
        best_composite = pd.to_numeric(keeps["composite"], errors="coerce").max()
        best_auprc = pd.to_numeric(keeps["auprc"], errors="coerce").max()
        baseline_auprc = pd.to_numeric(keeps["auprc"], errors="coerce").iloc[0]
        summary = (
            f"Experiments: {n_total} total | {n_keeps} kept | {n_discards} discarded | {n_crashes} crashed    "
            f"Best AUPRC: {best_auprc:.4f} (baseline: {baseline_auprc:.4f}, "
            f"full-feature target: 0.4982)    Best Composite: {best_composite:.4f}"
        )
    else:
        summary = f"Experiments: {n_total} total | {n_keeps} kept | {n_discards} discarded | {n_crashes} crashed"

    fig.text(0.5, 0.01, summary, ha="center", fontsize=9, style="italic", color="#6b7280")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {out_path}")
    else:
        default_path = ROOT_DIR / "results.png"
        plt.savefig(default_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {default_path}")

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot auto-research experiment results")
    parser.add_argument("--tsv", type=str, default=None, help="Path to results.tsv")
    parser.add_argument("--out", type=str, default=None, help="Output image path")
    args = parser.parse_args()

    tsv_path = Path(args.tsv) if args.tsv else None
    df = load_results(tsv_path)
    print(f"Loaded {len(df)} experiments")
    plot_results(df, args.out)
