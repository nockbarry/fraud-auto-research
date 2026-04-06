"""Feature transforms for fraud-sim dataset. Edited by the agent.

API CONTRACT:
  fit(df_train, y_train, config) -> state
      Called ONCE on training data WITH labels.
      Return a JSON-serializable dict of fitted parameters.
  transform(df, state, config) -> df
      Called on EACH split WITHOUT labels.
      Use only the state dict from fit(). No access to labels.

Dataset: 1.8M simulated transactions, 16 raw columns.
Key columns: merchant, category, TransactionAmt, gender, city, state, zip,
             lat, long, merch_lat, merch_long, city_pop, job, TransactionDT
NOTE: 42% population shift in OOT -- avoid features that track fraud rate directly.
"""

import numpy as np
import pandas as pd


def _target_encode_fit(series, y, global_mean, min_samples=20, smoothing_width=10):
    df_tmp = pd.DataFrame({"col": series, "_y": y.values})
    stats = df_tmp.groupby("col")["_y"].agg(["mean", "count"])
    smoothing = 1 / (1 + np.exp(-(stats["count"] - min_samples) / smoothing_width))
    te = smoothing * stats["mean"] + (1 - smoothing) * global_mean
    return {str(k): float(v) for k, v in te.items()}


def fit(df_train: pd.DataFrame, y_train: pd.Series, config: dict) -> dict:
    state = {}
    global_mean = float(y_train.mean())
    state["global_mean"] = global_mean

    cfg = config.get("dataset_profile", {})
    min_s = cfg.get("recommended_te_smoothing", {}).get("min_samples", 20)
    smooth_w = cfg.get("recommended_te_smoothing", {}).get("smoothing_width", 10)

    # Find high-NaN columns to drop
    nan_rates = df_train.isnull().mean()
    drop_cols = nan_rates[nan_rates > 0.50].index.tolist()
    state["drop_cols"] = drop_cols

    # Target encoding for key high-cardinality columns
    for col in ["merchant", "category", "city", "state", "job"]:
        if col in df_train.columns and col not in drop_cols:
            state[f"{col}_te"] = _target_encode_fit(df_train[col], y_train, global_mean, min_s, smooth_w)

    # Frequency encoding for all remaining string columns
    cat_cols = df_train.drop(columns=drop_cols, errors="ignore").select_dtypes(include="object").columns.tolist()
    state["cat_cols"] = cat_cols
    for col in cat_cols:
        freq = df_train[col].value_counts(normalize=True).to_dict()
        state[f"{col}_freq"] = {str(k): float(v) for k, v in freq.items()}

    # Per-merchant/category/gender amount statistics (stable across time)
    amt_col = "TransactionAmt"
    if amt_col in df_train.columns:
        for grp_col in ["merchant", "category", "gender"]:
            if grp_col in df_train.columns:
                stats = df_train.groupby(grp_col)[amt_col].agg(["median", "std"])
                state[f"{grp_col}_amt_median"] = {str(k): float(v) for k, v in stats["median"].items()}
                state[f"{grp_col}_amt_std"] = {str(k): float(v) for k, v in stats["std"].fillna(1.0).items()}
        state["global_amt_median"] = float(df_train[amt_col].median())
        state["global_amt_std"] = float(df_train[amt_col].std())

    return state


def transform(df: pd.DataFrame, state: dict, config: dict) -> pd.DataFrame:
    df = df.copy()
    global_mean = state.get("global_mean", 0.0)
    amt_col = "TransactionAmt"

    # Drop high-NaN columns
    drop_cols = state.get("drop_cols", [])
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Target encoding
    for col in ["merchant", "category", "city", "state", "job"]:
        key = f"{col}_te"
        if key in state and col in df.columns:
            df[f"{col}_target_enc"] = df[col].map(state[key]).fillna(global_mean)

    # Frequency encoding + drop original string columns
    cat_cols = state.get("cat_cols", [])
    for col in cat_cols:
        key = f"{col}_freq"
        if key in state and col in df.columns:
            df[f"{col}_freq_enc"] = df[col].map(state[key]).fillna(0.0)
    df = df.drop(columns=[c for c in cat_cols if c in df.columns])

    # Amount deviation features
    if amt_col in df.columns:
        g_med = state.get("global_amt_median", 0.0)
        g_std = state.get("global_amt_std", 1.0)
        for grp_col in ["merchant", "category", "gender"]:
            med_key = f"{grp_col}_amt_median"
            std_key = f"{grp_col}_amt_std"
            if med_key in state and grp_col in df.columns:
                grp_med = df[grp_col].map(state[med_key]).fillna(g_med)
                grp_std = df[grp_col].map(state[std_key]).fillna(g_std)
                df[f"{grp_col}_amt_zscore"] = (df[amt_col] - grp_med) / (grp_std + 1e-6)
                df[f"{grp_col}_amt_ratio"] = df[amt_col] / (grp_med + 1e-6)
        df["log_amt"] = np.log1p(df[amt_col])

    # Time features
    if "TransactionDT" in df.columns:
        df["hour"] = (df["TransactionDT"] // 3600 % 24).astype(int)
        df["day_of_week"] = (df["TransactionDT"] // 86400 % 7).astype(int)
        df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)

    df = df.fillna(-1)
    return df
