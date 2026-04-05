"""Full evaluation pipeline: load data, apply transforms, validate, train, compute metrics.

This is the ground truth evaluation harness. The agent cannot modify this file.
Output is grep-parseable for the autonomous loop.

Usage:
    python -m harness.evaluate
"""

import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from harness.data_loader import load_data
from harness.feature_analysis import population_stability_index
from harness.utils import ROOT_DIR, load_config
from harness.validate_features import validate


def precision_at_recall(y_true: np.ndarray, y_score: np.ndarray, target_recall: float) -> tuple[float, float]:
    """Find precision at a fixed recall level.

    Returns:
        (precision, threshold) at the operating point
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)

    # precision_recall_curve returns in decreasing recall order
    # Find the point where recall >= target_recall with highest precision
    valid = recalls >= target_recall
    if not valid.any():
        # Can't achieve target recall — return precision at max recall
        return float(precisions[0]), float(thresholds[0]) if len(thresholds) > 0 else 0.5

    # Among valid points, take the one with highest precision
    idx = np.where(valid)[0]
    best_idx = idx[np.argmax(precisions[idx])]

    prec = float(precisions[best_idx])
    thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    return prec, thresh


def score_psi(y_val_score: np.ndarray, y_oot_score: np.ndarray, n_bins: int = 10) -> float:
    """PSI between validation and OOT score distributions."""
    return population_stability_index(
        pd.Series(y_val_score),
        pd.Series(y_oot_score),
        n_bins=n_bins,
    )


def composite_score(
    auprc: float,
    prec_at_recall: float,
    psi: float,
    config: dict,
) -> tuple[float, bool]:
    """Calculate the composite score and check PSI hard reject.

    Returns:
        (score, rejected) — rejected=True means PSI exceeded hard_reject threshold
    """
    metrics_cfg = config["metrics"]
    weights = metrics_cfg["composite_weights"]
    psi_threshold = metrics_cfg.get("psi_threshold", 0.20)
    psi_hard_reject = metrics_cfg.get("psi_hard_reject", 0.25)

    w_auprc = weights.get("auprc", 0.50)
    w_prec = weights.get("precision_at_recall", 0.30)
    w_psi = weights.get("psi_penalty", 0.20)

    # PSI hard reject gate
    if psi >= psi_hard_reject:
        return 0.0, True

    # PSI penalty: linear ramp from 0 at threshold to 1 at hard_reject
    if psi < psi_threshold:
        psi_penalty = 0.0
    else:
        psi_penalty = (psi - psi_threshold) / (psi_hard_reject - psi_threshold)

    score = w_auprc * auprc + w_prec * prec_at_recall - w_psi * psi_penalty
    return score, False


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-feature columns and return feature matrix."""
    drop_cols = {"label", "txn_id", "txn_date", "customer_id"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    return df[feature_cols]


def _bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, metric_fn, n_boot: int = 200, ci: float = 0.95) -> tuple[float, float]:
    """Compute bootstrapped confidence interval for a metric."""
    rng = np.random.RandomState(42)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            s = metric_fn(y_true[idx], y_score[idx])
            if np.isfinite(s):
                scores.append(s)
        except Exception:
            pass
    if not scores:
        return (0.0, 0.0)
    alpha = (1 - ci) / 2
    return (float(np.percentile(scores, 100 * alpha)), float(np.percentile(scores, 100 * (1 - alpha))))


def _check_leakage(X_val: pd.DataFrame, y_val: np.ndarray) -> list[str]:
    """Detect potential feature leakage via single-feature AUC on val set."""
    from sklearn.metrics import roc_auc_score

    warnings = []
    for col in X_val.columns:
        try:
            vals = X_val[col].fillna(0).values
            if len(np.unique(vals)) < 2:
                continue
            auc = roc_auc_score(y_val, vals)
            if auc > 0.90:
                warnings.append(f"LEAKAGE WARNING: {col} has AUC={auc:.4f} — suspiciously predictive")
            elif auc < 0.10:
                warnings.append(f"LEAKAGE WARNING: {col} has AUC={1-auc:.4f} (inverted) — suspiciously predictive")
        except Exception:
            pass
    return warnings


def _check_state_serializable(state: dict) -> list[str]:
    """Verify the fitted state is JSON-serializable."""
    import json

    warnings = []
    try:
        json.dumps(state)
    except (TypeError, ValueError) as e:
        warnings.append(f"STATE WARNING: fit() state is not JSON-serializable: {e}")
    return warnings


def _measure_transform_latency(transform_fn, df_single_row, state, config) -> float:
    """Time a single-row transform for scoring latency estimation."""
    import time as _time

    start = _time.perf_counter()
    transform_fn(df_single_row, state, config)
    return (_time.perf_counter() - start) * 1000  # ms


def run_evaluation(config: dict | None = None) -> dict:
    """Run the full evaluation pipeline with leakage-safe fit/transform separation.

    Steps:
        1. Load data
        2. Separate labels (harness controls label access)
        3. Fit feature state on train only
        4. Transform all splits without labels
        5. Validate features + leakage detection
        6. Train model
        7. Compute all metrics
    """
    if config is None:
        config = load_config()

    total_start = time.time()

    # Step 1: Load data
    print("Step 1: Loading data...")
    df_train, df_val, df_oot = load_data(config)
    base_feature_count = len(df_train.columns)

    # Step 2: Separate labels — harness controls access
    print("Step 2: Separating labels (harness-controlled)...")
    y_train = df_train.pop("label").values.astype(int)
    y_val = df_val.pop("label").values.astype(int)
    y_oot = df_oot.pop("label").values.astype(int)

    # Step 3: Fit on train only (with labels)
    print("Step 3: Fitting feature state on train...")
    sys.path.insert(0, str(ROOT_DIR))
    import importlib
    import features as features_mod
    importlib.reload(features_mod)

    fit_state = features_mod.fit(df_train.copy(), pd.Series(y_train), config)

    # Verify state serializability
    state_warnings = _check_state_serializable(fit_state)
    for w in state_warnings:
        print(f"  {w}")

    # Step 4: Transform all splits WITHOUT labels
    print("Step 4: Transforming features (no labels)...")
    df_train = features_mod.transform(df_train, fit_state, config)
    df_val = features_mod.transform(df_val, fit_state, config)
    df_oot = features_mod.transform(df_oot, fit_state, config)

    # Re-attach labels for validation (harness use only)
    df_train["label"] = y_train
    df_val["label"] = y_val
    df_oot["label"] = y_oot

    # Step 5: Validate
    print("Step 5: Validating features...")
    passed, messages = validate(df_train, df_val, df_oot, config, base_feature_count=base_feature_count)
    for msg in messages:
        print(f"  {msg}")
    if not passed:
        print("\nVALIDATION FAILED — aborting evaluation")
        return {"error": "validation_failed", "messages": messages}

    # Step 5b: Leakage detection
    X_val_check = _prepare_features(df_val)
    leakage_warnings = _check_leakage(X_val_check, y_val)
    for w in leakage_warnings:
        print(f"  {w}")

    # Step 5c: Scoring latency
    single_row = df_train.drop(columns=["label"]).head(1)
    latency_ms = _measure_transform_latency(features_mod.transform, single_row, fit_state, config)
    print(f"  Single-row transform latency: {latency_ms:.1f}ms")

    # Step 6: Prepare features and train
    print("Step 6: Training model...")
    train_start = time.time()

    X_train = _prepare_features(df_train)
    X_val = _prepare_features(df_val)
    X_oot = _prepare_features(df_oot)

    from model import train_and_evaluate

    model_result = train_and_evaluate(X_train, y_train, X_val, y_val, X_oot, y_oot, config)
    training_seconds = time.time() - train_start

    y_val_pred = model_result["y_val_pred"]
    y_oot_pred = model_result["y_oot_pred"]

    # Step 6b: Extract feature importances
    print("Step 6b: Extracting feature importances...")
    top_features = {}
    try:
        model_obj = model_result.get("model")
        if hasattr(model_obj, "feature_importances_"):
            importances = dict(zip(X_train.columns, model_obj.feature_importances_))
            top_features = dict(sorted(importances.items(), key=lambda x: -x[1])[:20])
            top_str = ", ".join(f"{k}={v:.4f}" for k, v in list(top_features.items())[:10])
            print(f"  Top 10: {top_str}")
    except Exception:
        pass

    # Step 7: Compute metrics
    print("Step 7: Computing metrics...")

    target_recall = config["metrics"].get("target_recall", 0.80)

    auprc_val = average_precision_score(y_val, y_val_pred)
    auprc_oot = average_precision_score(y_oot, y_oot_pred)
    prec_at_rec, threshold = precision_at_recall(y_oot, y_oot_pred, target_recall)
    psi = score_psi(y_val_pred, y_oot_pred)

    # FPR and review burden at operating threshold
    y_oot_binary = (y_oot_pred >= threshold).astype(int)
    fp = ((y_oot_binary == 1) & (y_oot == 0)).sum()
    tn = ((y_oot_binary == 0) & (y_oot == 0)).sum()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    flagged = y_oot_binary.sum()
    actual_fraud = y_oot.sum()
    review_burden = flagged / actual_fraud if actual_fraud > 0 else 0.0

    comp_score, psi_rejected = composite_score(auprc_oot, prec_at_rec, psi, config)

    # Step 7b: Bootstrap confidence intervals
    auprc_ci = _bootstrap_ci(y_oot, y_oot_pred, average_precision_score)
    prec_ci = _bootstrap_ci(y_oot, y_oot_pred, lambda yt, yp: precision_at_recall(yt, yp, target_recall)[0])

    total_seconds = time.time() - total_start
    n_features = X_train.shape[1]

    results = {
        "composite_score": comp_score,
        "psi_rejected": psi_rejected,
        "auprc": auprc_oot,
        "auprc_val": auprc_val,
        "precision_at_recall": prec_at_rec,
        "target_recall": target_recall,
        "operating_threshold": threshold,
        "psi": psi,
        "fpr": fpr,
        "review_burden": review_burden,
        "n_features": n_features,
        "training_seconds": training_seconds,
        "total_seconds": total_seconds,
        "n_train_rows": len(y_train),
        "n_val_rows": len(y_val),
        "n_oot_rows": len(y_oot),
        "positive_rate_train": y_train.mean(),
        "positive_rate_oot": y_oot.mean(),
        "model_info": model_result.get("train_info", {}),
        "leakage_warnings": leakage_warnings + state_warnings,
        "transform_latency_ms": latency_ms,
        "top_features": top_features,
        "auprc_ci": auprc_ci,
        "precision_ci": prec_ci,
    }

    return results


def print_results(results: dict):
    """Print results in grep-parseable format."""
    if "error" in results:
        print(f"\nerror: {results['error']}")
        return

    print("\n---")
    print(f"composite_score:     {results['composite_score']:.6f}")
    if results.get("psi_rejected"):
        print(f"psi_rejected:        true")
    print(f"auprc:               {results['auprc']:.6f}")
    print(f"auprc_val:           {results['auprc_val']:.6f}")
    print(f"precision_at_recall: {results['precision_at_recall']:.6f}")
    print(f"target_recall:       {results['target_recall']:.2f}")
    print(f"operating_threshold: {results['operating_threshold']:.6f}")
    print(f"psi:                 {results['psi']:.6f}")
    print(f"fpr:                 {results['fpr']:.6f}")
    print(f"review_burden:       {results['review_burden']:.1f}x")
    print(f"n_features:          {results['n_features']}")
    print(f"training_seconds:    {results['training_seconds']:.1f}")
    print(f"total_seconds:       {results['total_seconds']:.1f}")
    print(f"n_train_rows:        {results['n_train_rows']}")
    print(f"n_val_rows:          {results['n_val_rows']}")
    print(f"n_oot_rows:          {results['n_oot_rows']}")
    print(f"positive_rate_train: {results['positive_rate_train']:.4f}")
    print(f"positive_rate_oot:   {results['positive_rate_oot']:.4f}")
    print(f"transform_latency:   {results.get('transform_latency_ms', 0):.1f}ms")
    leakage = results.get("leakage_warnings", [])
    print(f"leakage_warnings:    {len(leakage)}")
    for w in leakage:
        print(f"  {w}")
    ci = results.get("auprc_ci", (0, 0))
    print(f"auprc_ci:            [{ci[0]:.4f}, {ci[1]:.4f}]")
    pci = results.get("precision_ci", (0, 0))
    print(f"precision_ci:        [{pci[0]:.4f}, {pci[1]:.4f}]")
    top = results.get("top_features", {})
    if top:
        top_str = ", ".join(f"{k}={v:.4f}" for k, v in list(top.items())[:10])
        print(f"top_features:        {top_str}")


def save_experiment(config: dict, results: dict, hypothesis: str, status: str, state: dict | None = None):
    """Save experiment using the directory-per-experiment tracker."""
    from harness.experiment_tracker import save_experiment as _save

    dataset = config.get("dataset_name", "unknown")
    return _save(
        dataset=dataset,
        hypothesis=hypothesis,
        status=status,
        metrics=results,
        state=state,
        config_snapshot={
            "target_recall": config.get("metrics", {}).get("target_recall"),
            "dataset_name": dataset,
        },
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--hypothesis", type=str, default=None, help="What this experiment tried")
    parser.add_argument("--save", action="store_true", help="Auto-save experiment to tracker")
    args = parser.parse_args()

    config = load_config(args.config)
    results = run_evaluation(config)
    print_results(results)

    if args.save and args.hypothesis:
        # Auto-determine status by comparing to SOTA
        from harness.experiment_tracker import get_sota
        dataset = config.get("dataset_name", "unknown")
        sota = get_sota(dataset)
        min_imp = config.get("metrics", {}).get("min_improvement", 0.001)

        if "error" in results:
            status = "crash"
        elif results.get("psi_rejected"):
            status = "reject_psi"
        elif sota is None:
            status = "keep"  # first experiment
        else:
            sota_composite = sota.get("metrics_summary", {}).get("composite_score", 0) or 0
            new_composite = results.get("composite_score", 0)
            status = "keep" if new_composite > sota_composite + min_imp else "discard"

        save_experiment(config, results, args.hypothesis, status)
        print(f"\nstatus: {status}")

        # Print experiment context for next iteration
        print()
        from harness.context import generate_context
        print(generate_context(dataset))
