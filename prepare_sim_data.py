"""Prepare the simulated fraud transaction dataset for auto-research.

Source: https://www.kaggle.com/datasets/kartik2112/fraud-detection
Simulated credit card transactions covering 1000 customers and 800 merchants.

Rich raw features: merchant, category, amount, customer demographics,
geo coordinates, timestamps — ideal for feature engineering evaluation.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "fraud-detection-sim")
OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "fraud-sim")


def load_and_clean():
    """Load train and test CSVs, combine with time ordering."""
    print("Loading fraudTrain.csv...")
    train = pd.read_csv(os.path.join(DATA_DIR, "fraudTrain.csv"))
    print(f"  Train: {train.shape}")

    print("Loading fraudTest.csv...")
    test = pd.read_csv(os.path.join(DATA_DIR, "fraudTest.csv"))
    print(f"  Test: {test.shape}")

    df = pd.concat([train, test], ignore_index=True)
    df = df.drop(columns=["Unnamed: 0", "trans_num"], errors="ignore")

    # Parse datetime
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)

    # Create timestamp as unix seconds for compatibility
    df["TransactionDT"] = (df["trans_date_trans_time"] - df["trans_date_trans_time"].min()).dt.total_seconds().astype(int)

    # Rename for consistency
    df = df.rename(columns={
        "is_fraud": "label",
        "amt": "TransactionAmt",
        "cc_num": "card_id",
    })

    # Drop raw PII names (first, last, street, dob) but keep derived features
    # We'll let the agent work with: merchant, category, amount, gender, city, state,
    # zip, lat, long, city_pop, job, merch_lat, merch_long, unix_time, card_id
    df = df.drop(columns=["first", "last", "street", "dob", "trans_date_trans_time"], errors="ignore")

    print(f"  Combined: {df.shape}")
    print(f"  Fraud rate: {df['label'].mean():.4%}")
    return df


def split_by_time(df):
    """70/15/15 time-based split."""
    dt = df["TransactionDT"]
    q70 = dt.quantile(0.70)
    q85 = dt.quantile(0.85)

    train = df[dt <= q70].copy()
    val = df[(dt > q70) & (dt <= q85)].copy()
    oot = df[dt > q85].copy()

    return {"train": train, "val": val, "oot": oot}


def report_stats(splits, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for name, df in splits.items():
        n = len(df)
        fraud = df["label"].sum()
        rate = fraud / n if n > 0 else 0
        print(f"  {name:>5}: {n:>9,} rows | {int(fraud):>5,} fraud ({rate:.2%}) | {len(df.columns)} cols")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_and_clean()
    splits = split_by_time(df)
    report_stats(splits, "FRAUD-SIM DATASET")

    # Save
    print("\nSaving splits...")
    for name, split_df in splits.items():
        path = os.path.join(OUT_DIR, f"raw_{name}.parquet")
        split_df.to_parquet(path, index=False)
        print(f"  {path} ({len(split_df):,} rows, {len(split_df.columns)} cols)")

    # Print available columns
    raw_cols = [c for c in splits["train"].columns if c not in ("label",)]
    print(f"\n{'='*60}")
    print(f"  COLUMNS AVAILABLE TO AGENT ({len(raw_cols)} features)")
    print(f"{'='*60}")
    for col in raw_cols:
        dtype = splits["train"][col].dtype
        nunique = splits["train"][col].nunique()
        null_pct = splits["train"][col].isna().mean()
        print(f"  {col:<25} {str(dtype):<12} {nunique:>8} unique  {null_pct:>6.1%} null")

    # Benchmark
    print(f"\n{'='*60}")
    print(f"  RUNNING BASELINE BENCHMARK")
    print(f"{'='*60}")
    try:
        import xgboost as xgb
        from sklearn.metrics import average_precision_score

        numeric_cols = splits["train"].select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "label"]

        X_tr = splits["train"][numeric_cols].fillna(-999)
        y_tr = splits["train"]["label"]
        X_oot = splits["oot"][numeric_cols].fillna(-999)
        y_oot = splits["oot"]["label"]

        pos, neg = y_tr.sum(), len(y_tr) - y_tr.sum()
        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=neg / pos, eval_metric="aucpr",
            early_stopping_rounds=30, random_state=42,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_oot, y_oot)], verbose=False)
        y_pred = model.predict_proba(X_oot)[:, 1]
        auprc = average_precision_score(y_oot, y_pred)
        print(f"  Numeric-only baseline AUPRC (OOT): {auprc:.4f}")
        print(f"  Features used: {len(numeric_cols)}")
    except Exception as e:
        print(f"  Benchmark skipped: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
