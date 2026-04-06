"""Plot experiment results over time with improvement annotations.

Generates per-dataset multi-panel charts showing composite score, AUPRC,
precision@recall, and PSI over experiments.

Usage:
    python3 -m harness.plot_results                    # plot all datasets
    python3 -m harness.plot_results --dataset ieee-cis # plot one dataset
"""

import argparse
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
    return df


def plot_dataset(df: pd.DataFrame, dataset_name: str, out_path: str):
    """Create annotated metric plots for a single dataset."""
    # Filter and re-index
    df = df.copy()
    df.index = range(1, len(df) + 1)
    df.index.name = "experiment"

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(f"Fraud Auto-Research: {dataset_name}", fontsize=16, fontweight="bold", y=0.98)

    metrics = [
        ("composite", "Composite Score (val)", "#2563eb", True),
        ("auprc", "AUPRC", "#059669", True),          # special: plots both val+OOT
        ("prec@recall", "Precision @ 80% Recall", "#d97706", True),
        ("psi", "PSI (Val\u2192OOT)", "#dc2626", False),
    ]

    keeps = df[df["status"] == "keep"]
    discards = df[df["status"] == "discard"]
    crashes = df[df["status"].isin(["crash", "reject_psi"])]

    for ax, (col, title, color, higher_better) in zip(axes, metrics):
        if col not in df.columns:
            ax.set_visible(False)
            continue

        vals = pd.to_numeric(df[col], errors="coerce")

        # All experiments as light dots
        ax.scatter(df.index, vals, color="#e5e7eb", s=30, zorder=2, label="_nolegend_")

        if len(discards) > 0:
            ax.scatter(discards.index, pd.to_numeric(discards[col], errors="coerce"),
                      color="#fca5a5", s=30, zorder=3, marker="x", linewidths=1.5, label="discard")

        if len(crashes) > 0:
            ax.scatter(crashes.index, pd.to_numeric(crashes[col], errors="coerce"),
                      color="#f87171", s=40, zorder=3, marker="D", label="crash/reject")

        if len(keeps) > 0:
            keep_vals = pd.to_numeric(keeps[col], errors="coerce")
            ax.plot(keeps.index, keep_vals, color=color, linewidth=2.5,
                   marker="o", markersize=7, zorder=5, label="keep (OOT)" if col == "auprc" else "keep")

            running_best = keep_vals.cummax() if higher_better else keep_vals.cummin()
            ax.step(keeps.index, running_best, color=color, linewidth=1,
                   linestyle="--", alpha=0.5, zorder=4, label="best so far")

            # For AUPRC panel: overlay val AUPRC as a secondary line
            if col == "auprc" and "auprc_val" in df.columns:
                keep_vals_val = pd.to_numeric(keeps["auprc_val"], errors="coerce")
                ax.plot(keeps.index, keep_vals_val, color="#34d399", linewidth=1.8,
                       marker="s", markersize=5, zorder=4, linestyle=":", alpha=0.85,
                       label="keep (val, drives selection)")
                # Also scatter val for discards
                if len(discards) > 0 and "auprc_val" in discards.columns:
                    ax.scatter(discards.index, pd.to_numeric(discards["auprc_val"], errors="coerce"),
                              color="#6ee7b7", s=20, zorder=3, marker="s", alpha=0.5, label="_nolegend_")

            # Annotate improvements (on the primary col)
            prev_best = None
            ann_idx = 0
            for idx, row in keeps.iterrows():
                val = pd.to_numeric(row[col], errors="coerce")
                if pd.isna(val):
                    continue

                improved = False
                if prev_best is None:
                    improved = True
                elif higher_better and val > prev_best + 0.0005:
                    improved = True
                elif not higher_better and val < prev_best - 0.0005:
                    improved = True

                if improved:
                    hyp = str(row.get("hypothesis", ""))
                    if len(hyp) > 45:
                        hyp = hyp[:42] + "..."

                    y_off = 18 if (ann_idx % 2 == 0) else -22
                    ax.annotate(
                        f"{val:.4f}\n{hyp}",
                        xy=(idx, val), xytext=(12, y_off),
                        textcoords="offset points", fontsize=7,
                        color=color, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                 edgecolor=color, alpha=0.9),
                        zorder=10,
                    )
                    ann_idx += 1
                    if prev_best is None:
                        prev_best = val
                    elif higher_better:
                        prev_best = max(val, prev_best)
                    else:
                        prev_best = min(val, prev_best)

        if col == "psi":
            ax.axhline(y=0.20, color="#fbbf24", linestyle="--", alpha=0.7, label="penalty threshold")
            ax.axhline(y=0.25, color="#ef4444", linestyle="--", alpha=0.7, label="hard reject")

        if col == "auprc":
            title = "AUPRC (solid=OOT held-out, dotted=val selection)"

        ax.set_ylabel(title, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left" if higher_better else "upper right", fontsize=8)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    axes[-1].set_xlabel("Experiment #", fontsize=11)

    n_total, n_keeps, n_discards = len(df), len(keeps), len(discards)
    best_auprc = pd.to_numeric(keeps["auprc"], errors="coerce").max() if len(keeps) else 0
    baseline = pd.to_numeric(keeps["auprc"], errors="coerce").iloc[0] if len(keeps) else 0
    fig.text(0.5, 0.01,
             f"{dataset_name}: {n_total} experiments | {n_keeps} kept | {n_discards} discarded | "
             f"Best AUPRC: {best_auprc:.4f} (baseline: {baseline:.4f})",
             ha="center", fontsize=9, style="italic", color="#6b7280")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_all(tsv_path: Path | None = None, out_dir: Path | None = None, dataset_filter: str | None = None):
    """Generate per-dataset plots."""
    df = load_results(tsv_path)
    out_dir = Path(out_dir) if out_dir else ROOT_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    if "dataset" not in df.columns:
        # Legacy single-dataset TSV
        plot_dataset(df, "all", str(out_dir / "results_all.png"))
        return {"all": str(out_dir / "results_all.png")}

    datasets = df["dataset"].dropna().unique()
    if dataset_filter:
        datasets = [d for d in datasets if d == dataset_filter]

    paths = {}
    for ds in datasets:
        ds_df = df[df["dataset"] == ds].reset_index(drop=True)
        ds_df.index = range(1, len(ds_df) + 1)
        out_path = str(out_dir / f"results_{ds}.png")
        print(f"Plotting {ds} ({len(ds_df)} experiments)...")
        plot_dataset(ds_df, ds, out_path)
        paths[ds] = out_path

    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    plot_all(
        tsv_path=Path(args.tsv) if args.tsv else None,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        dataset_filter=args.dataset,
    )
