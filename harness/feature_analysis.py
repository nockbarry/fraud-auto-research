"""Feature-level analysis: Information Value, PSI, correlation, distributions.

Run standalone for fast iteration before committing to full model training:
    python -m harness.feature_analysis
"""

import sys
import warnings

import numpy as np
import pandas as pd

from harness.utils import ROOT_DIR, load_config

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _woe_iv_bin(events: int, non_events: int, total_events: int, total_non_events: int) -> tuple[float, float]:
    """Calculate WoE and IV for a single bin."""
    dist_events = (events / total_events) if total_events > 0 else 0
    dist_non_events = (non_events / total_non_events) if total_non_events > 0 else 0

    # Laplace smoothing to avoid log(0)
    dist_events = max(dist_events, 1e-6)
    dist_non_events = max(dist_non_events, 1e-6)

    woe = np.log(dist_non_events / dist_events)
    iv = (dist_non_events - dist_events) * woe
    return woe, iv


def information_value(feature: pd.Series, label: pd.Series, n_bins: int = 10) -> float:
    """Calculate Information Value (IV) for a single feature.

    IV interpretation:
        < 0.02: not predictive
        0.02 - 0.1: weak predictor
        0.1 - 0.3: medium predictor
        0.3 - 0.5: strong predictor
        > 0.5: suspicious (possible leakage)
    """
    mask = feature.notna() & label.notna()
    feat = feature[mask].values
    lab = label[mask].values.astype(int)

    if len(np.unique(feat)) <= n_bins:
        bins = np.unique(feat)
        binned = feat
    else:
        try:
            _, bin_edges = pd.qcut(feat, q=n_bins, retbins=True, duplicates="drop")
            binned = np.digitize(feat, bin_edges[1:-1])
        except ValueError:
            return 0.0

    total_events = lab.sum()
    total_non_events = len(lab) - total_events

    if total_events == 0 or total_non_events == 0:
        return 0.0

    iv_total = 0.0
    for b in np.unique(binned):
        mask_b = binned == b
        events = lab[mask_b].sum()
        non_events = mask_b.sum() - events
        _, iv_bin = _woe_iv_bin(events, non_events, total_events, total_non_events)
        iv_total += iv_bin

    return iv_total


def population_stability_index(expected: pd.Series, actual: pd.Series, n_bins: int = 10) -> float:
    """Calculate PSI between two distributions (e.g., train vs OOT).

    PSI interpretation:
        < 0.10: no significant shift
        0.10 - 0.25: moderate shift
        > 0.25: significant shift
    """
    mask_e = expected.notna()
    mask_a = actual.notna()
    exp = expected[mask_e].values
    act = actual[mask_a].values

    if len(exp) == 0 or len(act) == 0:
        return 0.0

    try:
        _, bin_edges = pd.qcut(exp, q=n_bins, retbins=True, duplicates="drop")
    except ValueError:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    exp_counts = np.histogram(exp, bins=bin_edges)[0]
    act_counts = np.histogram(act, bins=bin_edges)[0]

    exp_pct = exp_counts / exp_counts.sum()
    act_pct = act_counts / act_counts.sum()

    # Laplace smoothing
    exp_pct = np.maximum(exp_pct, 1e-6)
    act_pct = np.maximum(act_pct, 1e-6)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


def correlation_flags(df: pd.DataFrame, threshold: float = 0.98) -> list[tuple[str, str, float]]:
    """Find highly correlated feature pairs above threshold."""
    exclude = {"label", "txn_id", "txn_date", "customer_id"}
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

    if len(feature_cols) < 2:
        return []

    corr_matrix = df[feature_cols].corr().abs()
    pairs = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            val = corr_matrix.iloc[i, j]
            if val >= threshold:
                pairs.append((feature_cols[i], feature_cols[j], round(val, 4)))

    return sorted(pairs, key=lambda x: -x[2])


def analyze_features(
    df_train: pd.DataFrame,
    df_oot: pd.DataFrame,
    config: dict,
) -> dict:
    """Run full feature analysis: IV, PSI, correlation.

    Returns dict with keys: iv, psi, correlation_flags, summary_table
    """
    label_col = "label"
    exclude = {"label", "txn_id", "txn_date", "customer_id"}
    feature_cols = [c for c in df_train.select_dtypes(include=[np.number]).columns if c not in exclude]

    corr_threshold = config.get("validation", {}).get("max_correlation", 0.98)

    iv_scores = {}
    psi_scores = {}

    for col in feature_cols:
        iv_scores[col] = information_value(df_train[col], df_train[label_col])
        psi_scores[col] = population_stability_index(df_train[col], df_oot[col])

    corr_pairs = correlation_flags(df_train, threshold=corr_threshold)

    # Build summary table
    rows = []
    for col in feature_cols:
        iv = iv_scores[col]
        psi = psi_scores[col]
        iv_label = (
            "LEAK?" if iv > 0.5
            else "strong" if iv > 0.3
            else "medium" if iv > 0.1
            else "weak" if iv > 0.02
            else "none"
        )
        psi_label = "SHIFT!" if psi > 0.25 else "moderate" if psi > 0.1 else "stable"
        rows.append({
            "feature": col,
            "iv": round(iv, 4),
            "iv_grade": iv_label,
            "psi": round(psi, 4),
            "psi_grade": psi_label,
        })

    summary = pd.DataFrame(rows).sort_values("iv", ascending=False)

    return {
        "iv": iv_scores,
        "psi": psi_scores,
        "correlation_flags": corr_pairs,
        "summary_table": summary,
    }


def print_analysis(results: dict):
    """Pretty-print the feature analysis results."""
    summary = results["summary_table"]
    corr_pairs = results["correlation_flags"]

    print("=" * 70)
    print("FEATURE ANALYSIS")
    print("=" * 70)
    print(f"\n{'Feature':<35} {'IV':>8} {'Grade':<8} {'PSI':>8} {'Shift':<10}")
    print("-" * 70)
    for _, row in summary.iterrows():
        print(f"{row['feature']:<35} {row['iv']:>8.4f} {row['iv_grade']:<8} {row['psi']:>8.4f} {row['psi_grade']:<10}")

    if corr_pairs:
        print(f"\nHighly correlated pairs ({len(corr_pairs)}):")
        for c1, c2, val in corr_pairs[:10]:
            print(f"  {c1} <-> {c2}: {val:.4f}")

    # Summary stats
    n_features = len(summary)
    n_predictive = len(summary[summary["iv"] > 0.02])
    n_unstable = len(summary[summary["psi"] > 0.25])
    n_leak = len(summary[summary["iv"] > 0.5])

    print(f"\nSummary: {n_features} features | {n_predictive} predictive (IV>0.02) | {n_unstable} unstable (PSI>0.25) | {n_leak} possible leakage (IV>0.5)")


if __name__ == "__main__":
    from harness.data_loader import load_data

    config = load_config()
    print("Loading data...")
    df_train, df_val, df_oot = load_data(config)

    # Apply feature transforms
    sys.path.insert(0, str(ROOT_DIR))
    from features import transform

    df_train = transform(df_train, config)
    df_oot = transform(df_oot, config)

    print("\nAnalyzing features...\n")
    results = analyze_features(df_train, df_oot, config)
    print_analysis(results)
