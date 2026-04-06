"""Plot experiment results over time with improvement annotations.

Generates per-dataset multi-panel charts. The AUPRC panel shows three
distinct series: discard (val), keep (val, drives selection), and OOT
(held-out reporting). The line follows kept val values.

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


def _annotate_improvements(ax, keeps: pd.DataFrame, col: str, color: str, higher_better: bool = True):
    """Annotate new-best points on the keeps line."""
    prev_best = None
    ann_idx = 0
    for idx, row in keeps.iterrows():
        val = pd.to_numeric(row.get(col), errors="coerce")
        if pd.isna(val):
            continue
        improved = (prev_best is None
                    or (higher_better and val > prev_best + 0.0005)
                    or (not higher_better and val < prev_best - 0.0005))
        if improved:
            hyp = str(row.get("hypothesis", ""))[:42] + ("..." if len(str(row.get("hypothesis", ""))) > 42 else "")
            y_off = 18 if ann_idx % 2 == 0 else -22
            ax.annotate(
                f"{val:.4f}\n{hyp}",
                xy=(idx, val), xytext=(12, y_off),
                textcoords="offset points", fontsize=7,
                color=color, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, alpha=0.9),
                zorder=10,
            )
            ann_idx += 1
            prev_best = val if prev_best is None else (max(val, prev_best) if higher_better else min(val, prev_best))


def plot_dataset(df: pd.DataFrame, dataset_name: str, out_path: str):
    """Create annotated metric plots for a single dataset."""
    df = df.copy()
    df.index = range(1, len(df) + 1)
    df.index.name = "experiment"

    keeps = df[df["status"] == "keep"]
    discards = df[df["status"] == "discard"]
    crashes = df[df["status"].isin(["crash", "reject_psi"])]

    has_auroc = "auroc" in df.columns and df["auroc"].notna().any() and (df["auroc"] > 0).any()
    n_panels = 5 if has_auroc else 4
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels), sharex=True)
    fig.suptitle(f"Fraud Auto-Research: {dataset_name}", fontsize=16, fontweight="bold", y=0.99)

    # ── Panel 1: Composite score (val-based) ──────────────────────────────────
    ax = axes[0]
    col = "composite"
    vals = pd.to_numeric(df[col], errors="coerce")
    ax.scatter(df.index, vals, color="#e5e7eb", s=25, zorder=2)
    if len(discards):
        ax.scatter(discards.index, pd.to_numeric(discards[col], errors="coerce"),
                   color="#fca5a5", s=35, marker="x", linewidths=1.5, zorder=3, label="discard")
    if len(crashes):
        ax.scatter(crashes.index, pd.to_numeric(crashes[col], errors="coerce"),
                   color="#f87171", s=40, marker="D", zorder=3, label="crash/reject")
    if len(keeps):
        kv = pd.to_numeric(keeps[col], errors="coerce")
        ax.plot(keeps.index, kv, color="#2563eb", linewidth=2.5, marker="o", markersize=7, zorder=5, label="keep (val)")
        ax.step(keeps.index, kv.cummax(), color="#2563eb", linewidth=1, linestyle="--", alpha=0.4, zorder=4, label="best so far")
        _annotate_improvements(ax, keeps, col, "#2563eb")
    ax.set_ylabel("Composite Score\n(val, drives selection)", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ── Panel 2: AUPRC — 3 series: discard(val), keep(val line), OOT points ──
    ax = axes[1]
    has_val = "auprc_val" in df.columns and df["auprc_val"].notna().any()

    if has_val:
        # OOT: one amber diamond per experiment, no keep/discard distinction
        ax.scatter(df.index, pd.to_numeric(df["auprc"], errors="coerce"),
                   color="#f59e0b", s=35, marker="D", zorder=3, alpha=0.7, label="OOT (all)")

        if len(discards):
            ax.scatter(discards.index, pd.to_numeric(discards["auprc_val"], errors="coerce"),
                       color="#fca5a5", s=40, marker="x", linewidths=1.8, zorder=4, label="discard (val)")

        if len(crashes):
            ax.scatter(crashes.index, pd.to_numeric(crashes["auprc_val"], errors="coerce"),
                       color="#f87171", s=40, marker="D", zorder=3, label="crash/reject")

        if len(keeps):
            kval = pd.to_numeric(keeps["auprc_val"], errors="coerce")
            ax.plot(keeps.index, kval, color="#059669", linewidth=2.5,
                    marker="o", markersize=7, zorder=6, label="keep (val) ← selection")
            ax.step(keeps.index, kval.cummax(), color="#059669", linewidth=1,
                    linestyle="--", alpha=0.4, zorder=4)
            _annotate_improvements(ax, keeps, "auprc_val", "#059669")

        ax.set_ylabel("AUPRC\n● val (line) | ◆ OOT (one per exp)", fontsize=10, fontweight="bold")
    else:
        # Fallback: single series
        vals = pd.to_numeric(df["auprc"], errors="coerce")
        ax.scatter(df.index, vals, color="#e5e7eb", s=25, zorder=2)
        if len(keeps):
            kv = pd.to_numeric(keeps["auprc"], errors="coerce")
            ax.plot(keeps.index, kv, color="#059669", linewidth=2.5, marker="o", markersize=7, zorder=5, label="keep")
            _annotate_improvements(ax, keeps, "auprc", "#059669")
        ax.set_ylabel("AUPRC (OOT)", fontsize=10, fontweight="bold")

    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ── Panel 3: AUROC (same 3-series layout as AUPRC) ───────────────────────
    if has_auroc:
        ax = axes[2]
        has_auroc_val = "auroc_val" in df.columns and df["auroc_val"].notna().any()

        # OOT: one amber diamond per experiment, no keep/discard distinction
        ax.scatter(df.index, pd.to_numeric(df["auroc"], errors="coerce"),
                   color="#f59e0b", s=35, marker="D", zorder=3, alpha=0.7, label="OOT (all)")

        if has_auroc_val:
            if len(discards):
                ax.scatter(discards.index, pd.to_numeric(discards["auroc_val"], errors="coerce"),
                           color="#fca5a5", s=40, marker="x", linewidths=1.8, zorder=4, label="discard (val)")
            if len(keeps):
                kval = pd.to_numeric(keeps["auroc_val"], errors="coerce")
                ax.plot(keeps.index, kval, color="#6366f1", linewidth=2.5,
                        marker="o", markersize=7, zorder=6, label="keep (val) ← selection")
                ax.step(keeps.index, kval.cummax(), color="#6366f1", linewidth=1,
                        linestyle="--", alpha=0.4, zorder=4)
                _annotate_improvements(ax, keeps, "auroc_val", "#6366f1")
        elif len(keeps):
            kv = pd.to_numeric(keeps["auroc"], errors="coerce")
            ax.plot(keeps.index, kv, color="#6366f1", linewidth=2.5, marker="o", markersize=7, zorder=5, label="keep")
            _annotate_improvements(ax, keeps, "auroc", "#6366f1")

        ax.set_ylabel("AUROC\n● val (line) | ◆ OOT (one per exp)", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        prec_ax_idx = 3
        psi_ax_idx = 4
    else:
        prec_ax_idx = 2
        psi_ax_idx = 3

    # ── Panel: Precision @ Recall ─────────────────────────────────────────────
    ax = axes[prec_ax_idx]
    col = "prec@recall"
    if col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce")
        ax.scatter(df.index, vals, color="#e5e7eb", s=25, zorder=2)
        if len(discards):
            ax.scatter(discards.index, pd.to_numeric(discards[col], errors="coerce"),
                       color="#fca5a5", s=35, marker="x", linewidths=1.5, zorder=3, label="discard")
        if len(keeps):
            kv = pd.to_numeric(keeps[col], errors="coerce")
            ax.plot(keeps.index, kv, color="#d97706", linewidth=2.5, marker="o", markersize=7, zorder=5, label="keep")
            ax.step(keeps.index, kv.cummax(), color="#d97706", linewidth=1, linestyle="--", alpha=0.4, zorder=4)
            _annotate_improvements(ax, keeps, col, "#d97706")
    ax.set_ylabel("Precision @ 80% Recall\n(OOT)", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ── Panel: PSI ────────────────────────────────────────────────────────────
    ax = axes[psi_ax_idx]
    col = "psi"
    if col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce")
        ax.scatter(df.index, vals, color="#e5e7eb", s=25, zorder=2)
        if len(discards):
            ax.scatter(discards.index, pd.to_numeric(discards[col], errors="coerce"),
                       color="#fca5a5", s=35, marker="x", linewidths=1.5, zorder=3, label="discard")
        if len(keeps):
            kv = pd.to_numeric(keeps[col], errors="coerce")
            ax.plot(keeps.index, kv, color="#dc2626", linewidth=2.5, marker="o", markersize=7, zorder=5, label="keep")
            _annotate_improvements(ax, keeps, col, "#dc2626", higher_better=False)
        ax.axhline(y=0.20, color="#fbbf24", linestyle="--", alpha=0.7, label="penalty threshold (0.20)")
        ax.axhline(y=0.25, color="#ef4444", linestyle="--", alpha=0.7, label="hard reject (0.25)")
    ax.set_ylabel("PSI (Val→OOT)", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    axes[-1].set_xlabel("Experiment #", fontsize=11)

    n_total, n_keeps, n_discards = len(df), len(keeps), len(discards)
    best_auprc_oot = pd.to_numeric(keeps["auprc"], errors="coerce").max() if len(keeps) else 0
    best_auprc_val = pd.to_numeric(keeps["auprc_val"], errors="coerce").max() if len(keeps) and "auprc_val" in keeps.columns else 0
    baseline = pd.to_numeric(keeps["auprc"], errors="coerce").iloc[0] if len(keeps) else 0
    fig.text(0.5, 0.005,
             f"{dataset_name}: {n_total} experiments | {n_keeps} kept | {n_discards} discarded | "
             f"Best AUPRC val={best_auprc_val:.4f} oot={best_auprc_oot:.4f} (baseline oot={baseline:.4f})",
             ha="center", fontsize=9, style="italic", color="#6b7280")

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_all(tsv_path: Path | None = None, out_dir: Path | None = None, dataset_filter: str | None = None):
    """Generate per-dataset plots."""
    df = load_results(tsv_path)
    out_dir = Path(out_dir) if out_dir else ROOT_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    if "dataset" not in df.columns:
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
