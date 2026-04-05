"""Prepare the IEEE-CIS Fraud Detection dataset for auto-research.

This script:
1. Loads the raw IEEE-CIS train data (transaction + identity)
2. Strips Vesta's derived features (V, C, D, M columns) — 377 columns
3. Keeps only raw transaction and identity attributes — ~57 columns
4. Creates a time-based train/val/OOT split using TransactionDT
5. Saves a "full" version (with derived features) for benchmarking
6. Reports baseline stats

The goal: let the auto-research agent re-derive comparable features from
the raw attributes and see how close it gets to models trained on all 431 features.

Usage:
    python3 prepare_data.py
"""

import os
import sys

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR = DATA_DIR


def load_raw():
    """Load and merge transaction + identity files."""
    print("Loading train_transaction.csv...")
    txn = pd.read_csv(os.path.join(DATA_DIR, "train_transaction.csv"))
    print(f"  Transactions: {txn.shape}")

    print("Loading train_identity.csv...")
    ident = pd.read_csv(os.path.join(DATA_DIR, "train_identity.csv"))
    print(f"  Identity: {ident.shape}")

    print("Merging on TransactionID...")
    df = txn.merge(ident, on="TransactionID", how="left")
    print(f"  Merged: {df.shape}")
    return df


def split_by_time(df: pd.DataFrame) -> dict:
    """Split into train/val/OOT based on TransactionDT quantiles.

    TransactionDT is a timedelta (seconds) from a reference point.
    We use 70/15/15 split to create meaningful OOT holdout.
    """
    dt = df["TransactionDT"]
    q70 = dt.quantile(0.70)
    q85 = dt.quantile(0.85)

    train = df[dt <= q70].copy()
    val = df[(dt > q70) & (dt <= q85)].copy()
    oot = df[dt > q85].copy()

    return {"train": train, "val": val, "oot": oot}


def strip_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Remove Vesta's derived feature columns (V, C, D, M prefixes)."""
    derived_prefixes = ("V", "C", "D", "M")
    derived_cols = [c for c in df.columns if any(c.startswith(p) and c[len(p):].isdigit() for p in derived_prefixes)]
    raw = df.drop(columns=derived_cols)
    return raw


def report_stats(splits: dict, label: str):
    """Print split stats."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for name, df in splits.items():
        n = len(df)
        fraud = df["isFraud"].sum()
        rate = fraud / n if n > 0 else 0
        print(f"  {name:>5}: {n:>9,} rows | {int(fraud):>6,} fraud ({rate:.2%}) | {len(df.columns)} cols")


def main():
    df = load_raw()

    # Time-based split
    splits = split_by_time(df)
    report_stats(splits, "FULL DATASET (all 431 features)")

    # Save full version for benchmarking
    print("\nSaving full dataset splits (for benchmarking)...")
    for name, split_df in splits.items():
        path = os.path.join(OUT_DIR, f"full_{name}.parquet")
        split_df.to_parquet(path, index=False)
        print(f"  {path} ({len(split_df):,} rows, {len(split_df.columns)} cols)")

    # Strip derived features
    print("\nStripping derived features (V, C, D, M)...")
    stripped_splits = {name: strip_derived(split_df) for name, split_df in splits.items()}
    report_stats(stripped_splits, "RAW DATASET (derived features removed)")

    # Rename isFraud -> label for harness compatibility
    for name, split_df in stripped_splits.items():
        split_df.rename(columns={"isFraud": "label"}, inplace=True)

    # Save raw version for auto-research
    print("\nSaving raw dataset splits (for auto-research agent)...")
    for name, split_df in stripped_splits.items():
        path = os.path.join(OUT_DIR, f"raw_{name}.parquet")
        split_df.to_parquet(path, index=False)
        print(f"  {path} ({len(split_df):,} rows, {len(split_df.columns)} cols)")

    # Print the columns available to the agent
    raw_cols = [c for c in stripped_splits["train"].columns if c not in ("TransactionID", "label")]
    print(f"\n{'='*60}")
    print(f"  COLUMNS AVAILABLE TO AGENT ({len(raw_cols)} features)")
    print(f"{'='*60}")
    for col in raw_cols:
        dtype = stripped_splits["train"][col].dtype
        nunique = stripped_splits["train"][col].nunique()
        null_pct = stripped_splits["train"][col].isna().mean()
        print(f"  {col:<25} {str(dtype):<12} {nunique:>6} unique  {null_pct:>6.1%} null")

    # Benchmark: train XGBoost on full features for comparison
    print(f"\n{'='*60}")
    print(f"  RUNNING BENCHMARK (full features, XGBoost)")
    print(f"{'='*60}")
    try:
        import xgboost as xgb
        from sklearn.metrics import average_precision_score

        full_train = splits["train"]
        full_oot = splits["oot"]

        # Drop non-numeric for quick benchmark
        drop_cols = {"TransactionID", "isFraud", "TransactionDT"}
        feature_cols = [c for c in full_train.columns if c not in drop_cols]
        numeric_cols = full_train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        X_tr = full_train[numeric_cols].fillna(-999)
        y_tr = full_train["isFraud"]
        X_oot = full_oot[numeric_cols].fillna(-999)
        y_oot = full_oot["isFraud"]

        pos = y_tr.sum()
        neg = len(y_tr) - pos

        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=neg / pos,
            eval_metric="aucpr",
            early_stopping_rounds=30,
            random_state=42,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_oot, y_oot)], verbose=False)

        y_pred = model.predict_proba(X_oot)[:, 1]
        auprc = average_precision_score(y_oot, y_pred)

        print(f"  Full-feature benchmark AUPRC (OOT): {auprc:.4f}")
        print(f"  Features used: {len(numeric_cols)}")
        print(f"  This is the target the agent's raw-feature model should approach.")

        # Also benchmark raw features only
        raw_oot = stripped_splits["oot"]
        raw_train = stripped_splits["train"]
        raw_numeric = raw_train.select_dtypes(include=[np.number]).columns.tolist()
        raw_numeric = [c for c in raw_numeric if c not in {"TransactionID", "label", "TransactionDT"}]

        X_tr_raw = raw_train[raw_numeric].fillna(-999)
        y_tr_raw = raw_train["label"]
        X_oot_raw = raw_oot[raw_numeric].fillna(-999)
        y_oot_raw = raw_oot["label"]

        model_raw = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=neg / pos, eval_metric="aucpr",
            early_stopping_rounds=30, random_state=42,
        )
        model_raw.fit(X_tr_raw, y_tr_raw, eval_set=[(X_oot_raw, y_oot_raw)], verbose=False)
        y_pred_raw = model_raw.predict_proba(X_oot_raw)[:, 1]
        auprc_raw = average_precision_score(y_oot_raw, y_pred_raw)

        print(f"\n  Raw-feature baseline AUPRC (OOT):   {auprc_raw:.4f}")
        print(f"  Features used: {len(raw_numeric)}")
        print(f"\n  GAP: {auprc - auprc_raw:.4f} AUPRC")
        print(f"  The agent needs to close this gap through feature engineering.")

    except Exception as e:
        print(f"  Benchmark skipped: {e}")

    print("\nDone. Data ready for auto-research.")


if __name__ == "__main__":
    main()
