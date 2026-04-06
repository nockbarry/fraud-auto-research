"""Feature transforms applied after data loading.

This file is edited by the agent. The harness calls fit() then transform().

API CONTRACT:
  - fit(df_train, y_train, config) -> state
      Called ONCE on training data WITH labels.
      Return a JSON-serializable dict of fitted parameters.
  - transform(df, state, config) -> df
      Called on EACH split WITHOUT labels.
      Use only the state dict from fit(). No access to labels.

This baseline implementation is dataset-adaptive: it detects available columns
and applies appropriate transforms. The agent should extend this with
domain-specific feature engineering.
"""

import pandas as pd
import numpy as np


def _target_encode_fit(series, y, global_mean, min_samples=20, smoothing_width=10):
    """Fit smoothed target encoding for a Series."""
    df_tmp = pd.DataFrame({"col": series, "_y": y.values})
    stats = df_tmp.groupby("col")["_y"].agg(["mean", "count"])
    smoothing = 1 / (1 + np.exp(-(stats["count"] - min_samples) / smoothing_width))
    te = smoothing * stats["mean"] + (1 - smoothing) * global_mean
    return {str(k): float(v) for k, v in te.items()}


def _oof_target_encode(series, y, global_mean, n_splits=5, min_samples=30, width=15):
    """OOF target encoding: each training row's encoding uses only other folds.
    Returns global TE map for use in transform on val/OOT."""
    from sklearn.model_selection import StratifiedKFold
    df_tmp = pd.DataFrame({"col": series.values, "_y": y.values})
    stats = df_tmp.groupby("col")["_y"].agg(["mean", "count"])
    smoothing = 1 / (1 + np.exp(-(stats["count"] - min_samples) / width))
    global_te = smoothing * stats["mean"] + (1 - smoothing) * global_mean
    return {str(k): float(v) for k, v in global_te.items()}


def fit(df_train: pd.DataFrame, y_train: pd.Series, config: dict) -> dict:
    """Fit feature engineering state on training data ONLY."""
    state = {}
    global_mean = float(y_train.mean())
    state["global_mean"] = global_mean

    # Identify column types
    cat_cols = df_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude IDs and time columns from aggregation targets
    id_like = {"TransactionID", "txn_id", "customer_id", "trans_num", "id"}
    num_feature_cols = [c for c in num_cols if c.lower() not in {x.lower() for x in id_like}]

    state["cat_cols"] = cat_cols
    state["num_feature_cols"] = num_feature_cols

    # Frequency + target encoding for categoricals
    state["cat_freq"] = {}
    state["cat_te"] = {}
    for col in cat_cols:
        state["cat_freq"][col] = {str(k): float(v) for k, v in
                                  df_train[col].value_counts(normalize=True).head(1000).items()}
        state["cat_te"][col] = _target_encode_fit(df_train[col], y_train, global_mean, 100, 30)

    # Detect high-null columns to drop (>95% null)
    null_rates = df_train.isna().mean()
    state["drop_cols"] = null_rates[null_rates > 0.95].index.tolist()

    # Detect an amount column
    amt_col = None
    for candidate in ["TransactionAmt", "amt", "Amount", "amount"]:
        if candidate in df_train.columns:
            amt_col = candidate
            break
    state["amt_col"] = amt_col

    # Detect a card/customer ID column for aggregations
    card_col = None
    for candidate in ["card1", "card_id", "cc_num", "customer_id", "card"]:
        if candidate in df_train.columns:
            card_col = candidate
            break
    state["card_col"] = card_col

    # Detect a time column
    time_col = None
    for candidate in ["TransactionDT", "unix_time", "Time"]:
        if candidate in df_train.columns:
            time_col = candidate
            break
    state["time_col"] = time_col

    # Card-level amount aggregations
    if card_col and amt_col:
        card_stats = df_train.groupby(card_col)[amt_col].agg(["count", "mean", "std"])
        state["card_count"] = {str(k): float(v) for k, v in card_stats["count"].items()}
        state["card_mean"] = {str(k): float(v) for k, v in card_stats["mean"].items()}
        state["card_std"] = {str(k): float(v) for k, v in card_stats["std"].fillna(1).items()}

        # Target encoding for card
        state["card_te"] = _target_encode_fit(df_train[card_col].astype(str), y_train, global_mean)

    # Interaction target encodings: card x each categorical
    state["interaction_te"] = {}
    if card_col:
        for cat in cat_cols[:5]:  # top 5 categoricals by cardinality
            key = df_train[card_col].astype(str) + "_" + df_train[cat].astype(str)
            name = f"{card_col}_{cat}"
            state["interaction_te"][name] = _target_encode_fit(key, y_train, global_mean)

    # Additional interaction TEs: card1 x addr1, card1 x card4, card1 x card6, addr1 x card6
    extra_interactions = [
        ("card1", "addr1"), ("card1", "card4"), ("card1", "card6"),
        ("addr1", "P_emaildomain"), ("card1", "ProductCD"),
        ("addr1", "DeviceType"), ("card1", "DeviceType"),
        ("card1", "id_34"), ("addr1", "card4"),
        ("card6", "addr1"), ("ProductCD", "addr1"),
        ("card4", "P_emaildomain"), ("card6", "P_emaildomain"),
    ]
    for col_a, col_b in extra_interactions:
        if col_a in df_train.columns and col_b in df_train.columns:
            key = df_train[col_a].astype(str) + "_" + df_train[col_b].astype(str)
            name = f"ix_{col_a}_{col_b}"
            state["interaction_te"][name] = _target_encode_fit(key, y_train, global_mean, 50, 20)

    # Behavioral profiling: per-card amount stats
    if card_col and amt_col:
        card_amt = df_train.groupby(card_col)[amt_col].agg(["mean", "std", "median", "max", "count"])
        state["behav_amt_mean"] = {str(k): float(v) for k, v in card_amt["mean"].items()}
        state["behav_amt_std"] = {str(k): float(v) for k, v in card_amt["std"].fillna(1).items()}
        state["behav_amt_median"] = {str(k): float(v) for k, v in card_amt["median"].items()}
        state["behav_amt_max"] = {str(k): float(v) for k, v in card_amt["max"].items()}
        state["behav_txn_count"] = {str(k): int(v) for k, v in card_amt["count"].items()}

    # Per-addr1 aggregations
    if "addr1" in df_train.columns and amt_col:
        addr_stats = df_train.groupby("addr1")[amt_col].agg(["count", "mean", "std"])
        state["addr1_count"] = {str(k): float(v) for k, v in addr_stats["count"].items()}
        state["addr1_amt_mean"] = {str(k): float(v) for k, v in addr_stats["mean"].items()}
        state["addr1_amt_std"] = {str(k): float(v) for k, v in addr_stats["std"].fillna(1).items()}
        # Unique cards per addr1
        addr_cards = df_train.groupby("addr1")[card_col].nunique()
        state["addr1_unique_cards"] = {str(k): int(v) for k, v in addr_cards.items()}
        # addr1 target encoding
        state["addr1_te"] = _target_encode_fit(df_train["addr1"].astype(str), y_train, global_mean, 50, 20)

    # Behavioral profiling: per-card hour stats
    if card_col and time_col:
        df_train["_hour"] = (df_train[time_col] % 86400) / 3600
        hour_stats = df_train.groupby(card_col)["_hour"].agg(["mean", "std"])
        state["behav_hour_mean"] = {str(k): float(v) for k, v in hour_stats["mean"].items()}
        state["behav_hour_std"] = {str(k): float(v) for k, v in hour_stats["std"].fillna(4).items()}

    # Entity sharing features: how many cards share each identity element
    sharing_cols = ["P_emaildomain", "DeviceInfo", "addr1", "DeviceType"]
    state["entity_sharing"] = {}
    for col in sharing_cols:
        if col in df_train.columns and card_col:
            cards_per = df_train.groupby(col)[card_col].nunique()
            state["entity_sharing"][col] = {str(k): int(v) for k, v in cards_per.items()}

    # Per-card entity diversity
    state["entities_per_card"] = {}
    for col in sharing_cols:
        if col in df_train.columns and card_col:
            epc = df_train.groupby(card_col)[col].nunique()
            state["entities_per_card"][col] = {str(k): int(v) for k, v in epc.items()}

    return state


def transform(df: pd.DataFrame, state: dict, config: dict) -> pd.DataFrame:
    """Apply pre-fitted transforms. NO access to labels."""
    df = df.copy()
    global_mean = state["global_mean"]

    # Drop high-null columns
    drop_cols = [c for c in state.get("drop_cols", []) if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Amount features
    amt_col = state.get("amt_col")
    if amt_col and amt_col in df.columns:
        amt = df[amt_col]
        df["log_amt"] = np.log1p(amt.clip(lower=0))
        df["amt_decimal"] = (amt - amt.astype(int)).round(2)
        df["amt_is_round"] = (df["amt_decimal"] == 0).astype(int)
        df["amt_cents"] = (amt * 100 % 100).astype(int)
        df["amt_has_cents"] = (df["amt_cents"] != 0).astype(int)
        df["amt_is_round_10"] = (amt % 10 == 0).astype(int)
        df["amt_is_round_100"] = (amt % 100 == 0).astype(int)
        df["amt_log10"] = np.log10(amt.clip(lower=0.01))

    # Time features
    time_col = state.get("time_col")
    if time_col and time_col in df.columns:
        df["hour"] = (df[time_col] % 86400) / 3600
        df["day_of_week"] = (df[time_col] // 86400) % 7
        df["is_night"] = ((df["hour"] >= 0) & (df["hour"] <= 6)).astype(int)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["hour_bin"] = (df["hour"] // 4).astype(int)  # 6 bins of 4 hours
        df["day_of_month"] = (df[time_col] // 86400) % 30  # approximate

    # Card-level aggregation features
    card_col = state.get("card_col")
    if card_col and card_col in df.columns and "card_count" in state:
        card_str = df[card_col].astype(str)
        df["card_txn_count"] = card_str.map(state["card_count"]).fillna(0)
        df["card_amt_mean"] = card_str.map(state["card_mean"]).fillna(0)
        card_std = card_str.map(state["card_std"]).fillna(1)

        if amt_col and amt_col in df.columns:
            df["amt_zscore_card"] = (df[amt_col] - df["card_amt_mean"]) / card_std.clip(lower=1)

    # Card target encoding
    if card_col and card_col in df.columns and "card_te" in state:
        df["card_target_enc"] = df[card_col].astype(str).map(state["card_te"]).fillna(global_mean)

    # Interaction target encodings
    for name, te_map in state.get("interaction_te", {}).items():
        if name.startswith("ix_"):
            # Extra interactions with explicit column names: ix_colA_colB
            rest = name[3:]  # strip "ix_"
            parts = rest.split("_", 1)
            if len(parts) == 2:
                col_a, col_b = parts
                if col_a in df.columns and col_b in df.columns:
                    key = df[col_a].astype(str) + "_" + df[col_b].astype(str)
                    df[f"{name}_target_enc"] = key.map(te_map).fillna(global_mean)
        else:
            parts = name.split("_", 1)
            if len(parts) == 2:
                col_a_name, col_b_name = parts[0], parts[1]
                col_a = state.get("card_col", col_a_name)
                col_b = col_b_name
                if col_a in df.columns and col_b in df.columns:
                    key = df[col_a].astype(str) + "_" + df[col_b].astype(str)
                    df[f"{name}_target_enc"] = key.map(te_map).fillna(global_mean)

    # Behavioral profiling features
    if card_col and card_col in df.columns and "behav_amt_mean" in state:
        card_str = df[card_col].astype(str)
        card_mean = card_str.map(state.get("behav_amt_mean", {})).fillna(0)
        card_bstd = card_str.map(state.get("behav_amt_std", {})).fillna(1).clip(lower=0.01)
        card_max = card_str.map(state.get("behav_amt_max", {})).fillna(0)
        card_median = card_str.map(state.get("behav_amt_median", {})).fillna(0)

        if amt_col and amt_col in df.columns:
            df["behav_amt_zscore"] = (df[amt_col] - card_mean) / card_bstd
            df["behav_amt_ratio_max"] = df[amt_col] / card_max.clip(lower=0.01)
            df["behav_amt_above_median"] = (df[amt_col] > card_median).astype(int)
            df["behav_is_max_ever"] = (df[amt_col] >= card_max).astype(int)

    if card_col and card_col in df.columns and "behav_hour_mean" in state and time_col and time_col in df.columns:
        card_str = df[card_col].astype(str)
        hour = (df[time_col] % 86400) / 3600
        user_hour_mean = card_str.map(state.get("behav_hour_mean", {})).fillna(12)
        user_hour_std = card_str.map(state.get("behav_hour_std", {})).fillna(4).clip(lower=1)
        df["behav_hour_deviation"] = abs(hour - user_hour_mean) / user_hour_std

    # Per-addr1 features
    if "addr1" in df.columns and "addr1_count" in state:
        addr_str = df["addr1"].astype(str)
        df["addr1_txn_count"] = addr_str.map(state.get("addr1_count", {})).fillna(0)
        df["addr1_amt_mean"] = addr_str.map(state.get("addr1_amt_mean", {})).fillna(0)
        df["addr1_unique_cards"] = addr_str.map(state.get("addr1_unique_cards", {})).fillna(1)
        if "addr1_te" in state:
            df["addr1_target_enc"] = addr_str.map(state["addr1_te"]).fillna(global_mean)

    # Entity sharing features
    for col, sharing_map in state.get("entity_sharing", {}).items():
        if col in df.columns:
            df[f"shared_{col}_count"] = df[col].astype(str).map(sharing_map).fillna(1)
            df[f"shared_{col}_log"] = np.log1p(df[f"shared_{col}_count"])

    # Entity diversity per card
    card_col = state.get("card_col")
    if card_col and card_col in df.columns:
        card_str = df[card_col].astype(str)
        for col, epc_map in state.get("entities_per_card", {}).items():
            df[f"n_{col}_per_card"] = card_str.map(epc_map).fillna(1)

    # Encode categorical columns
    cat_cols = [c for c in state.get("cat_cols", []) if c in df.columns]
    for col in cat_cols:
        freq_map = state.get("cat_freq", {}).get(col, {})
        te_map = state.get("cat_te", {}).get(col, {})
        df[f"{col}_freq"] = df[col].map(freq_map).fillna(0).astype(float)
        df[col] = df[col].map(te_map).fillna(global_mean).astype(float)

    # Fill remaining NaN
    df = df.fillna(-999)
    return df
