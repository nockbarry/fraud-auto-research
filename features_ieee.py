"""Feature transforms for IEEE-CIS dataset. Edited by the agent.

API CONTRACT:
  fit(df_train, y_train, config) -> state
      Called ONCE on training data WITH labels.
      Return a JSON-serializable dict of fitted parameters.
  transform(df, state, config) -> df
      Called on EACH split WITHOUT labels.
      Use only the state dict from fit(). No access to labels.
"""

import numpy as np
import pandas as pd


def _target_encode_fit(series, y, global_mean, min_samples=50, smoothing_width=20):
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
    min_s = cfg.get("recommended_te_smoothing", {}).get("min_samples", 50)
    smooth_w = cfg.get("recommended_te_smoothing", {}).get("smoothing_width", 20)

    # Find high-NaN columns to drop (>50% missing)
    nan_rates = df_train.isnull().mean()
    drop_cols = nan_rates[nan_rates > 0.50].index.tolist()
    state["drop_cols"] = drop_cols

    # Target encoding for high-cardinality columns
    for col in ["card1", "card2", "card3", "card5", "addr1", "P_emaildomain", "R_emaildomain"]:
        if col in df_train.columns and col not in drop_cols:
            state[f"{col}_te"] = _target_encode_fit(df_train[col], y_train, global_mean, min_s, smooth_w)

    # Frequency encoding for all remaining string columns
    cat_cols = df_train.drop(columns=drop_cols, errors="ignore").select_dtypes(include="object").columns.tolist()
    state["cat_cols"] = cat_cols
    for col in cat_cols:
        freq = df_train[col].value_counts(normalize=True).to_dict()
        state[f"{col}_freq"] = {str(k): float(v) for k, v in freq.items()}

    return state


def transform(df: pd.DataFrame, state: dict, config: dict) -> pd.DataFrame:
    df = df.copy()
    global_mean = state.get("global_mean", 0.0)

    # Drop high-NaN columns
    drop_cols = state.get("drop_cols", [])
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Apply target encoding
    for col in ["card1", "card2", "card3", "card5", "addr1", "P_emaildomain", "R_emaildomain"]:
        key = f"{col}_te"
        if key in state and col in df.columns:
            df[f"{col}_target_enc"] = df[col].map(state[key]).fillna(global_mean)

    # Apply frequency encoding + drop original string columns
    cat_cols = state.get("cat_cols", [])
    for col in cat_cols:
        key = f"{col}_freq"
        if key in state and col in df.columns:
            df[f"{col}_freq_enc"] = df[col].map(state[key]).fillna(0.0)
    df = df.drop(columns=[c for c in cat_cols if c in df.columns])

    # Amount features
    if "TransactionAmt" in df.columns:
        df["log_amt"] = np.log1p(df["TransactionAmt"])
        df["amt_cents"] = (df["TransactionAmt"] * 100 % 100).round(0)
        df["amt_is_round"] = (df["amt_cents"] == 0).astype(int)

    # Time features
    if "TransactionDT" in df.columns:
        df["hour"] = (df["TransactionDT"] // 3600 % 24).astype(int)
        df["day_of_week"] = (df["TransactionDT"] // 86400 % 7).astype(int)
        df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)

    # Fill remaining NaNs with -1 (missing indicator)
    df = df.fillna(-1)

    return df
