"""Feature engineering for IEEE-CIS dataset. Agent-editable.

Fit/transform API:
  fit(df_train, y_train, config) -> state dict (JSON-serializable)
  transform(df, state, config)   -> df (all numeric, no labels)
"""
import numpy as np
import pandas as pd


def _oof_target_encode(series, y, global_mean, min_samples=30, width=15):
    """Smoothed target encoding on full training data (for val/OOT use)."""
    df_tmp = pd.DataFrame({"col": series.values, "_y": y.values})
    stats = df_tmp.groupby("col")["_y"].agg(["mean", "count"])
    smoothing = 1 / (1 + np.exp(-(stats["count"] - min_samples) / width))
    global_te = smoothing * stats["mean"] + (1 - smoothing) * global_mean
    return {str(k): float(v) for k, v in global_te.items()}


def fit(df_train: pd.DataFrame, y_train: pd.Series, config: dict) -> dict:
    """Fit on training data only. Returns JSON-serializable state."""
    state: dict = {}
    state["global_mean"] = float(y_train.mean())

    # Drop columns with >75% NaN (captures id/device cols at ~74%)
    nan_rates = df_train.isnull().mean()
    state["drop_cols"] = nan_rates[nan_rates > 0.75].index.tolist()
    df_tmp = df_train.drop(columns=state["drop_cols"], errors="ignore")

    # Identify categorical columns
    cat_cols = df_tmp.select_dtypes("object").columns.tolist()
    state["cat_cols"] = cat_cols

    # Frequency encoding for all categoricals
    for col in cat_cols:
        freq = df_tmp[col].value_counts(normalize=True).to_dict()
        state[f"{col}_freq"] = {str(k): float(v) for k, v in freq.items()}

    # Target encoding for key fields (including identity + device)
    global_mean = state["global_mean"]
    te_target_cols = [
        "P_emaildomain", "R_emaildomain", "ProductCD",
        "card1", "card2", "addr1", "addr2",
        "DeviceType", "DeviceInfo",
        "id_12", "id_15", "id_16", "id_28", "id_29",  # Found/NotFound identity signals
        "id_35", "id_36", "id_37", "id_38",  # T/F identity flags
        "id_31",  # Browser info - very strong fraud signal
    ]
    for col in te_target_cols:
        if col in df_tmp.columns:
            state[f"{col}_te"] = _oof_target_encode(
                df_tmp[col].astype(str), y_train, global_mean
            )

    # Amount behavioral: per-card1 mean/std for z-score
    if "card1" in df_tmp.columns and "TransactionAmt" in df_tmp.columns:
        card_amt = df_tmp.groupby("card1")["TransactionAmt"].agg(["mean", "std", "median"])
        state["card1_amt_mean"] = {str(k): float(v) for k, v in card_amt["mean"].items()}
        state["card1_amt_std"] = {str(k): float(v) for k, v in card_amt["std"].fillna(1).items()}
        state["card1_amt_median"] = {str(k): float(v) for k, v in card_amt["median"].items()}

    # Entity sharing: cards per DeviceInfo, addr2
    if "DeviceInfo" in df_tmp.columns and "card1" in df_tmp.columns:
        cards_per_device = df_tmp.dropna(subset=["DeviceInfo"]).groupby("DeviceInfo")["card1"].nunique()
        state["cards_per_device"] = {str(k): int(v) for k, v in cards_per_device.items()}
    if "addr2" in df_tmp.columns and "card1" in df_tmp.columns:
        cards_per_addr2 = df_tmp.groupby("addr2")["card1"].nunique()
        state["cards_per_addr2"] = {str(k): int(v) for k, v in cards_per_addr2.items()}

    # Identity cols tracking
    id_num_cols = [f"id_{str(i).zfill(2)}" for i in range(1, 12)]
    state["id_num_cols"] = [c for c in id_num_cols if c in df_tmp.columns]

    return state


def transform(df: pd.DataFrame, state: dict, config: dict) -> pd.DataFrame:
    """Transform without labels. Apply fitted state."""
    df = df.copy()
    df = df.drop(columns=state.get("drop_cols", []), errors="ignore")

    # STEP 1: Compute TE before dropping string columns
    global_mean = state.get("global_mean", 0.035)

    # Target encoding for identity + device columns (before freq enc drop)
    id_te_cols = ["id_12", "id_15", "id_16", "id_28", "id_29",
                  "id_35", "id_36", "id_37", "id_38", "DeviceType", "DeviceInfo", "id_31"]
    for col in id_te_cols:
        te_key = f"{col}_te"
        if te_key in state and col in df.columns:
            te_map = state[te_key]
            df[f"{col}_te"] = df[col].apply(lambda x: te_map.get(str(x), global_mean))

    # STEP 2: Frequency encode and drop string columns
    for col in state.get("cat_cols", []):
        if col in df.columns:
            freq_map = state.get(f"{col}_freq", {})
            df[f"{col}_freq_enc"] = df[col].apply(lambda x: freq_map.get(str(x), 0.0))
            df = df.drop(columns=[col])

    # STEP 3: Target encoding for numeric-like categoricals (card/addr/email)
    for col in ["card1", "card2", "addr1", "addr2", "P_emaildomain", "R_emaildomain", "ProductCD"]:
        te_key = f"{col}_te"
        if te_key in state and col in df.columns:
            te_map = state[te_key]
            df[f"{col}_te"] = df[col].apply(lambda x: te_map.get(str(x), global_mean))

    # Amount features
    if "TransactionAmt" in df.columns:
        amt = df["TransactionAmt"]
        df["log_amount"] = np.log1p(amt)
        df["amt_is_round_10"] = (amt % 10 == 0).astype(int)
        df["amt_is_round_100"] = (amt % 100 == 0).astype(int)
        df["amt_cents"] = ((amt * 100) % 100).astype(int)
        df["amt_has_cents"] = (df["amt_cents"] != 0).astype(int)

        if "card1" in df.columns:
            card1_str = df["card1"].astype(str)
            card_mean = card1_str.apply(lambda x: state.get("card1_amt_mean", {}).get(x, 0.0))
            card_std = card1_str.apply(lambda x: state.get("card1_amt_std", {}).get(x, 1.0)).clip(lower=0.01)
            card_median = card1_str.apply(lambda x: state.get("card1_amt_median", {}).get(x, 0.0))
            df["amt_zscore_card1"] = (amt - card_mean) / card_std
            df["amt_above_card1_median"] = (amt > card_median).astype(int)

    # Identity nan rate
    id_num_cols = state.get("id_num_cols", [])
    if id_num_cols:
        id_subset = [c for c in id_num_cols if c in df.columns]
        if id_subset:
            df["id_nan_rate"] = df[id_subset].isnull().mean(axis=1)
            df["has_identity"] = (df["id_nan_rate"] < 0.5).astype(int)

    # Entity sharing
    if "addr2" in df.columns and "cards_per_addr2" in state:
        df["cards_per_addr2"] = df["addr2"].astype(str).apply(
            lambda x: float(state["cards_per_addr2"].get(str(x), 1)))
        df["log_cards_per_addr2"] = np.log1p(df["cards_per_addr2"])

    # dist1 feature
    if "dist1" in df.columns:
        df["dist1_log"] = np.log1p(df["dist1"].clip(lower=0))
        df["dist1_is_zero"] = (df["dist1"] == 0).astype(int)
        df["dist1_is_missing"] = df["dist1"].isna().astype(int)

    # id_01 is a risk score (-100 to 0): lower = more risky
    if "id_01" in df.columns:
        df["id_01_is_low"] = (df["id_01"] <= -50).astype(int)
        df["id_01_is_missing"] = df["id_01"].isna().astype(int)

    # id_02 has huge range (30 to 999595) - log transform
    if "id_02" in df.columns:
        df["id_02_log"] = np.log1p(df["id_02"].clip(lower=0))

    return df.fillna(-1)
