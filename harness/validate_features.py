"""Feature validation tests. Run before model training to catch data issues early.

Adapted from AutoKaggle's unit test patterns for feature engineering.
"""

import sys

import numpy as np
import pandas as pd

from harness.utils import load_config


def _check_row_count(df: pd.DataFrame, expected_rows: int, name: str) -> tuple[bool, str]:
    """Verify row count hasn't changed after feature transforms."""
    actual = len(df)
    if actual != expected_rows:
        return False, f"FAIL: {name} has {actual:,} rows, expected {expected_rows:,} (rows were dropped or duplicated)"
    return True, f"PASS: {name} row count OK ({actual:,})"


def _check_duplicate_columns(df: pd.DataFrame, name: str) -> tuple[bool, str]:
    """Check for duplicate column names."""
    dupes = df.columns[df.columns.duplicated()].tolist()
    if dupes:
        return False, f"FAIL: {name} has duplicate columns: {dupes[:10]}"
    return True, f"PASS: {name} no duplicate columns"


def _check_nan_rate(df: pd.DataFrame, max_rate: float, name: str) -> tuple[bool, str]:
    """Check NaN rate per feature against threshold."""
    nan_rates = df.isna().mean()
    bad = nan_rates[nan_rates > max_rate]
    if len(bad) > 0:
        worst = bad.sort_values(ascending=False).head(5)
        details = ", ".join(f"{col}={rate:.2%}" for col, rate in worst.items())
        return False, f"FAIL: {name} has {len(bad)} features above {max_rate:.0%} NaN threshold: {details}"
    return True, f"PASS: {name} NaN rates OK (max {nan_rates.max():.2%})"


def _check_schema_alignment(df_train: pd.DataFrame, df_val: pd.DataFrame, df_oot: pd.DataFrame) -> tuple[bool, str]:
    """Verify train/val/OOT have the same columns."""
    train_cols = set(df_train.columns)
    val_cols = set(df_val.columns)
    oot_cols = set(df_oot.columns)

    if train_cols != val_cols:
        diff = train_cols.symmetric_difference(val_cols)
        return False, f"FAIL: train/val column mismatch: {diff}"
    if train_cols != oot_cols:
        diff = train_cols.symmetric_difference(oot_cols)
        return False, f"FAIL: train/oot column mismatch: {diff}"
    return True, f"PASS: schema alignment OK ({len(train_cols)} columns)"


def _check_feature_count(df: pd.DataFrame, base_count: int, config: dict, name: str) -> tuple[bool, str]:
    """Check feature count hasn't exploded beyond thresholds."""
    validation = config.get("validation", {})
    max_explosion = validation.get("max_feature_explosion", 3.0)
    max_count = validation.get("max_feature_count", 200)
    current = len(df.columns)
    explosion_limit = int(base_count * max_explosion)

    if current > max_count:
        return False, f"FAIL: {name} has {current} features, exceeds absolute max of {max_count}"
    if current > explosion_limit:
        return False, f"FAIL: {name} has {current} features, exceeds {max_explosion}x base ({base_count} -> max {explosion_limit})"
    return True, f"PASS: {name} feature count OK ({current}, base was {base_count})"


def _check_numeric_types(df: pd.DataFrame, name: str) -> tuple[bool, str]:
    """Check that all feature columns (excluding label, id, date) are numeric."""
    exclude = {"label", "txn_id", "txn_date", "customer_id"}
    feature_cols = [c for c in df.columns if c not in exclude]
    non_numeric = []
    for col in feature_cols:
        if not np.issubdtype(df[col].dtype, np.number):
            non_numeric.append(f"{col} ({df[col].dtype})")
    if non_numeric:
        return False, f"FAIL: {name} has {len(non_numeric)} non-numeric feature columns: {non_numeric[:10]}"
    return True, f"PASS: {name} all feature columns are numeric"


def _check_label_present(df: pd.DataFrame, name: str) -> tuple[bool, str]:
    """Check that label column exists and is binary."""
    if "label" not in df.columns:
        return False, f"FAIL: {name} missing 'label' column"
    unique_vals = df["label"].dropna().unique()
    if not set(unique_vals).issubset({0, 1, 0.0, 1.0, True, False}):
        return False, f"FAIL: {name} label has non-binary values: {sorted(unique_vals)[:10]}"
    return True, f"PASS: {name} label column OK"


def _check_te_overfit(df_train: pd.DataFrame, y_train, df_val: pd.DataFrame, y_val, threshold: float = 0.15) -> list[str]:
    """Detect target encoding overfit by comparing train vs val single-feature AUC."""
    from sklearn.metrics import roc_auc_score

    warnings = []
    te_cols = [c for c in df_train.columns if c.endswith("_te") or c.endswith("_target_enc") or c.endswith("_te_")]
    for col in te_cols:
        try:
            train_vals = df_train[col].fillna(0).values
            val_vals = df_val[col].fillna(0).values
            if len(np.unique(train_vals)) < 2 or len(np.unique(val_vals)) < 2:
                continue
            train_auc = roc_auc_score(y_train, train_vals)
            val_auc = roc_auc_score(y_val, val_vals)
            gap = abs(train_auc - 0.5) - abs(val_auc - 0.5)
            if gap > threshold:
                warnings.append(
                    f"TE OVERFIT: {col} train AUC={train_auc:.3f} vs val AUC={val_auc:.3f} (gap {gap:.3f})"
                )
        except Exception:
            pass
    return warnings


def validate(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_oot: pd.DataFrame,
    config: dict,
    base_feature_count: int | None = None,
) -> tuple[bool, list[str]]:
    """Run all feature validation tests.

    Args:
        df_train, df_val, df_oot: Feature DataFrames after transforms.
        config: Loaded config dict.
        base_feature_count: Number of columns in the raw data before transforms.
            If None, skips the feature explosion check.

    Returns:
        (all_passed: bool, messages: list of PASS/FAIL strings)
    """
    max_nan = config.get("validation", {}).get("max_nan_rate", 0.05)
    messages = []
    all_passed = True

    checks = []

    # Label checks
    for name, df in [("train", df_train), ("val", df_val), ("oot", df_oot)]:
        checks.append(_check_label_present(df, name))

    # Duplicate column checks
    for name, df in [("train", df_train), ("val", df_val), ("oot", df_oot)]:
        checks.append(_check_duplicate_columns(df, name))

    # NaN rate checks
    for name, df in [("train", df_train), ("val", df_val), ("oot", df_oot)]:
        checks.append(_check_nan_rate(df, max_nan, name))

    # Schema alignment
    checks.append(_check_schema_alignment(df_train, df_val, df_oot))

    # Numeric type check
    checks.append(_check_numeric_types(df_train, "train"))

    # Feature count explosion
    if base_feature_count is not None:
        checks.append(_check_feature_count(df_train, base_feature_count, config, "train"))

    for passed, msg in checks:
        messages.append(msg)
        if not passed:
            all_passed = False

    return all_passed, messages


if __name__ == "__main__":
    from harness.data_loader import load_data

    config = load_config()
    print("Loading data...")
    df_train, df_val, df_oot = load_data(config)

    # Apply feature transforms
    sys.path.insert(0, str(load_config and __import__("harness.utils", fromlist=["ROOT_DIR"]).ROOT_DIR))
    from features import transform

    base_count = len(df_train.columns)
    df_train = transform(df_train, config)
    df_val = transform(df_val, config)
    df_oot = transform(df_oot, config)

    print("\nRunning validation...\n")
    passed, messages = validate(df_train, df_val, df_oot, config, base_feature_count=base_count)

    for msg in messages:
        print(msg)

    print(f"\n{'ALL CHECKS PASSED' if passed else 'VALIDATION FAILED'}")
    sys.exit(0 if passed else 1)
