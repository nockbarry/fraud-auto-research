"""Per-column univariate analysis. Tells the agent what raw signals exist BEFORE
any feature engineering, and re-runs periodically on the transformed feature space
so the agent can see what got created and what's now redundant.

The output is the agent's primary EDA artifact. It is NOT a leakage detector — it
exists to surface columns that the agent has been ignoring (e.g. high-NaN identity
columns whose null pattern itself is highly predictive of fraud).

CLI:
    python3 -m harness.column_analysis ieee-cis           # raw columns only
    python3 -m harness.column_analysis ieee-cis --refresh # force re-run

Cached at: experiments/{dataset}/column_analysis.json
"""

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# How many experiments before raw analysis goes stale (re-run prompt only — refreshes
# are cheap to invalidate but the raw data does not change, so the threshold is high).
RAW_STALENESS_EXPS = 50

# How many experiments between transformed-feature analyses. Set to 1 so every keep
# refreshes the picture of what features now exist.
TRANSFORMED_STALENESS_EXPS = 1


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """One-sided AUC: max(AUC, 1 - AUC). Returns 0.5 on failure."""
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(scores)) < 2:
            return 0.5
        auc = float(roc_auc_score(y_true, scores))
        return max(auc, 1.0 - auc)
    except Exception:
        return 0.5


def _binned_iv(feature: pd.Series, label: pd.Series, n_bins: int = 10) -> float:
    """Information Value with WoE binning. NaN-aware."""
    mask = feature.notna() & label.notna()
    if mask.sum() < 50:
        return 0.0
    feat = feature[mask].values
    lab = label[mask].values.astype(int)

    if len(np.unique(feat)) <= n_bins:
        binned = feat
    else:
        try:
            _, edges = pd.qcut(feat, q=n_bins, retbins=True, duplicates="drop")
            binned = np.digitize(feat, edges[1:-1])
        except ValueError:
            return 0.0

    pos_total = lab.sum()
    neg_total = len(lab) - pos_total
    if pos_total == 0 or neg_total == 0:
        return 0.0

    iv = 0.0
    for b in np.unique(binned):
        m = binned == b
        pos = lab[m].sum()
        neg = m.sum() - pos
        p = max(pos / pos_total, 1e-6)
        n = max(neg / neg_total, 1e-6)
        iv += (n - p) * np.log(n / p)
    return float(iv)


def _categorical_iv(feature: pd.Series, label: pd.Series) -> float:
    """IV for object/category columns: each level is its own bin."""
    mask = feature.notna() & label.notna()
    if mask.sum() < 50:
        return 0.0
    df = pd.DataFrame({"x": feature[mask].astype(str), "y": label[mask].astype(int)})
    grp = df.groupby("x")["y"].agg(["sum", "count"])
    pos_total = float(grp["sum"].sum())
    neg_total = float((grp["count"] - grp["sum"]).sum())
    if pos_total == 0 or neg_total == 0:
        return 0.0

    iv = 0.0
    for _, row in grp.iterrows():
        p = max(row["sum"] / pos_total, 1e-6)
        n = max((row["count"] - row["sum"]) / neg_total, 1e-6)
        iv += (n - p) * np.log(n / p)
    return float(iv)


def _grade_iv(iv: float, n_unique: int = 0) -> str:
    # High-cardinality categoricals inflate IV via per-level binning
    # (each level becomes its own bin with few samples). Downgrade them so the
    # agent doesn't mistake high cardinality for label leakage.
    if iv > 0.5 and n_unique > 50:
        return "high_card"
    if iv > 0.5:
        return "LEAK?"
    if iv > 0.3:
        return "strong"
    if iv > 0.1:
        return "medium"
    if iv > 0.02:
        return "weak"
    return "none"


def analyze_dataframe(df: pd.DataFrame, label: pd.Series, exclude: set[str] | None = None) -> list[dict]:
    """Compute per-column IV, null-flag AUC, NaN rate, n_unique, dtype.

    Returns a list of dicts (one per column), sorted by max(iv, null_flag_auc - 0.5).
    """
    exclude = exclude or set()
    rows = []
    label_arr = label.values.astype(int)

    for col in df.columns:
        if col in exclude or col == label.name:
            continue

        series = df[col]
        nan_rate = float(series.isnull().mean())
        n_unique = int(series.nunique(dropna=True))
        dtype = str(series.dtype)

        # Null flag predictivity (does the presence/absence of this col predict fraud?)
        null_flag = series.isnull().astype(int).values
        null_flag_auc = _safe_auc(label_arr, null_flag) if 0.001 < nan_rate < 0.999 else 0.5

        # IV on non-null values
        if pd.api.types.is_numeric_dtype(series):
            iv = _binned_iv(series, label)
            # Univariate AUC using fillna(median) — represents what XGBoost sees with default split
            try:
                vals = series.fillna(series.median() if series.notna().any() else 0).values.astype(float)
                uni_auc = _safe_auc(label_arr, vals)
            except Exception:
                uni_auc = 0.5
        else:
            iv = _categorical_iv(series, label)
            uni_auc = 0.5  # not meaningful for raw object columns

        rows.append({
            "column": col,
            "dtype": dtype,
            "nan_rate": round(nan_rate, 4),
            "n_unique": n_unique,
            "iv": round(iv, 4),
            "iv_grade": _grade_iv(iv, n_unique),
            "univariate_auc": round(uni_auc, 4),
            "null_flag_auc": round(null_flag_auc, 4),
        })

    # Sort by best signal: IV or null-flag predictivity (whichever is stronger)
    rows.sort(key=lambda r: max(r["iv"], (r["null_flag_auc"] - 0.5) * 2), reverse=True)
    return rows


def _experiments_dir(dataset: str) -> Path:
    from harness.experiment_tracker import EXPERIMENTS_DIR
    return Path(EXPERIMENTS_DIR) / dataset


def _cache_path(dataset: str) -> Path:
    return _experiments_dir(dataset) / "column_analysis.json"


def load_cached(dataset: str) -> dict | None:
    """Load the cached analysis JSON, or None if missing."""
    p = _cache_path(dataset)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def is_stale(dataset: str, current_n_exps: int, kind: str = "raw") -> bool:
    """True if the cached analysis is missing or older than the staleness budget."""
    cached = load_cached(dataset)
    if not cached:
        return True
    block = cached.get(kind, {})
    if not block:
        return True
    saved_at = block.get("n_experiments_at_save", -1)
    threshold = RAW_STALENESS_EXPS if kind == "raw" else TRANSFORMED_STALENESS_EXPS
    return (current_n_exps - saved_at) >= threshold


def compute_raw_analysis(dataset: str) -> dict:
    """Run univariate analysis on the raw training dataframe and cache it."""
    from harness.data_loader import load_data
    from harness.experiment_tracker import load_history
    from harness.utils import load_config

    config = load_config(f"configs/{dataset}.yaml")
    print(f"  Loading raw {dataset} data...")
    df_train, _df_val, _df_oot = load_data(config)

    label_col = config.get("fraud_type", {}).get("label_column", "label")
    if label_col not in df_train.columns:
        raise ValueError(f"Label column '{label_col}' not in train dataframe")

    label = df_train[label_col]
    label.name = label_col
    exclude = {label_col, "TransactionID", "TRANSACTION_ID", "txn_id"}

    print(f"  Computing IV / null-flag AUC across {len(df_train.columns)} columns...")
    rows = analyze_dataframe(df_train, label, exclude=exclude)

    cached = load_cached(dataset) or {}
    cached["raw"] = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "n_experiments_at_save": len(load_history(dataset)),
        "n_rows_train": int(len(df_train)),
        "n_columns": int(len(df_train.columns)),
        "rows": rows,
    }
    out = _cache_path(dataset)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cached, indent=2))
    print(f"  Saved -> {out}")
    return cached["raw"]


def compute_transformed_analysis(dataset: str, df_transformed: pd.DataFrame, label: pd.Series) -> dict:
    """Same analysis but on the post-feature-engineering dataframe.

    Called automatically after a successful keep so the context can show
    what features actually exist after transform().
    """
    from harness.experiment_tracker import load_history

    label.name = label.name or "label"
    exclude = {label.name}
    rows = analyze_dataframe(df_transformed, label, exclude=exclude)

    cached = load_cached(dataset) or {}
    cached["transformed"] = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "n_experiments_at_save": len(load_history(dataset)),
        "n_rows_train": int(len(df_transformed)),
        "n_columns": int(len(df_transformed.columns)),
        "rows": rows,
    }
    out = _cache_path(dataset)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cached, indent=2))
    return cached["transformed"]


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_for_context(dataset: str, max_rows: int = 20) -> str | None:
    """Render the cached analysis as a compact text block for context.py.

    Returns None if no cache exists.
    """
    cached = load_cached(dataset)
    if not cached:
        return None

    lines = []
    raw = cached.get("raw")
    if raw:
        lines.append(f"RAW COLUMN ANALYSIS (computed at exp #{raw['n_experiments_at_save']}, "
                     f"{raw['n_columns']} cols × {raw['n_rows_train']:,} rows):")
        lines.append(f"{'column':<26} {'IV':>7} {'grade':<7} {'univ_AUC':>9} {'null_AUC':>9} {'NaN%':>6}")
        lines.append("-" * 72)
        rows = raw["rows"][:max_rows]
        for r in rows:
            null_marker = "*" if r["null_flag_auc"] > 0.55 else " "
            lines.append(
                f"{r['column']:<26} {r['iv']:>7.4f} {r['iv_grade']:<7} "
                f"{r['univariate_auc']:>9.4f} {r['null_flag_auc']:>8.4f}{null_marker} {r['nan_rate']*100:>5.1f}"
            )
        lines.append("  (* = null-pattern itself is predictive — try col_is_null flag)")

        # Specifically call out high-IV columns at risk of being dropped
        dropped_signal = [r for r in raw["rows"]
                          if r["nan_rate"] > 0.5 and (r["iv"] > 0.05 or r["null_flag_auc"] > 0.55)]
        if dropped_signal:
            lines.append("")
            lines.append("WARNING: high-NaN columns with predictive signal — DO NOT blanket-drop these:")
            for r in dropped_signal[:10]:
                lines.append(f"  {r['column']:<24} NaN={r['nan_rate']*100:.1f}%  "
                             f"IV={r['iv']:.3f}  null_AUC={r['null_flag_auc']:.3f}")
        lines.append("")

    transformed = cached.get("transformed")
    if transformed:
        lines.append(f"TRANSFORMED FEATURE ANALYSIS (after exp #{transformed['n_experiments_at_save']}, "
                     f"{transformed['n_columns']} features):")
        lines.append(f"{'feature':<32} {'IV':>7} {'grade':<7} {'univ_AUC':>9}")
        lines.append("-" * 60)
        for r in transformed["rows"][:max_rows]:
            lines.append(
                f"{r['column']:<32} {r['iv']:>7.4f} {r['iv_grade']:<7} {r['univariate_auc']:>9.4f}"
            )
        # Surface any features that ended up with IV<0.02 (dead weight in the FE pipeline)
        dead = [r for r in transformed["rows"] if r["iv"] < 0.005 and r["nan_rate"] < 0.5]
        if dead:
            lines.append("")
            lines.append(f"DEAD FEATURES in current transform (IV<0.005, n={len(dead)}): "
                         f"{', '.join(r['column'] for r in dead[:10])}"
                         f"{'...' if len(dead) > 10 else ''}")
        lines.append("")

    return "\n".join(lines) if lines else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Per-column univariate analysis")
    parser.add_argument("dataset", help="Dataset name (ieee-cis, fdh, fraud-sim, paysim)")
    parser.add_argument("--refresh", action="store_true", help="Force re-run even if cached")
    parser.add_argument("--show", action="store_true", help="Print cached analysis without recomputing")
    args = parser.parse_args()

    if args.show:
        text = format_for_context(args.dataset, max_rows=40)
        if text:
            print(text)
        else:
            print(f"No cached analysis for {args.dataset}. Run without --show to compute.")
        return

    if args.refresh or not load_cached(args.dataset) or "raw" not in (load_cached(args.dataset) or {}):
        print(f"Computing raw column analysis for {args.dataset}...")
        compute_raw_analysis(args.dataset)
    else:
        print(f"Cached analysis exists for {args.dataset}. Use --refresh to recompute.")

    text = format_for_context(args.dataset, max_rows=40)
    if text:
        print()
        print(text)


if __name__ == "__main__":
    main()
