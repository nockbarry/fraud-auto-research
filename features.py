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
        state["cat_te"][col] = _target_encode_fit(df_train[col], y_train, global_mean, 10, 5)

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

    # --- Recipe 2: Velocity features ---
    if card_col and time_col and card_col in df_train.columns and time_col in df_train.columns:
        df_sorted = df_train[[card_col, time_col]].sort_values([card_col, time_col])
        df_sorted["_gap"] = df_sorted.groupby(card_col)[time_col].diff()
        gap_stats = df_sorted.groupby(card_col)["_gap"].agg(["median", "std", "min"])
        df_sorted["_burst"] = (df_sorted["_gap"] < 60).astype(int)
        burst_counts = df_sorted.groupby(card_col)["_burst"].sum()
        time_range = df_sorted.groupby(card_col)[time_col].agg(["min", "max"])
        days = ((time_range["max"] - time_range["min"]) / 86400).clip(lower=1)
        card_counts = df_sorted.groupby(card_col).size()
        daily_rate = card_counts / days
        state["vel_median_gap"] = {str(k): float(v) for k, v in gap_stats["median"].fillna(0).items()}
        state["vel_std_gap"] = {str(k): float(v) for k, v in gap_stats["std"].fillna(0).items()}
        state["vel_min_gap"] = {str(k): float(v) for k, v in gap_stats["min"].fillna(0).items()}
        state["vel_burst"] = {str(k): int(v) for k, v in burst_counts.items()}
        state["vel_daily_rate"] = {str(k): float(v) for k, v in daily_rate.items()}

    # --- Recipe 3: Behavioral profiling ---
    if card_col and amt_col and card_col in df_train.columns:
        card_amt = df_train.groupby(card_col)[amt_col].agg(["mean", "std", "median", "max"])
        state["behav_amt_mean"] = {str(k): float(v) for k, v in card_amt["mean"].items()}
        state["behav_amt_std"] = {str(k): float(v) for k, v in card_amt["std"].fillna(1).items()}
        state["behav_amt_median"] = {str(k): float(v) for k, v in card_amt["median"].items()}
        state["behav_amt_max"] = {str(k): float(v) for k, v in card_amt["max"].items()}
    if card_col and time_col and card_col in df_train.columns:
        df_train_h = df_train.copy()
        df_train_h["_hour"] = (df_train_h[time_col] % 86400) / 3600
        hour_stats = df_train_h.groupby(card_col)["_hour"].agg(["mean", "std"])
        state["behav_hour_mean"] = {str(k): float(v) for k, v in hour_stats["mean"].items()}
        state["behav_hour_std"] = {str(k): float(v) for k, v in hour_stats["std"].fillna(4).items()}

    # --- Recipe 6: Anomaly score (Mahalanobis) ---
    anomaly_exclude = {card_col, time_col, "TransactionID", "txn_id"} if card_col else {"TransactionID"}
    anomaly_cols = [c for c in num_feature_cols if c not in anomaly_exclude and df_train[c].std() > 0][:15]
    state["anomaly_cols"] = anomaly_cols
    state["anomaly_means"] = {c: float(df_train[c].fillna(0).mean()) for c in anomaly_cols}
    state["anomaly_stds"] = {c: float(df_train[c].fillna(0).std()) for c in anomaly_cols}

    # --- Recipe 5: Entity sharing counts ---
    state["entity_sharing"] = {}
    if card_col:
        for col in cat_cols[:4]:
            if col in df_train.columns:
                cards_per = df_train.groupby(col)[card_col].nunique()
                state["entity_sharing"][col] = {str(k): int(v) for k, v in cards_per.head(50000).items()}

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
        df["log_amt"] = np.log1p(df[amt_col].clip(lower=0))
        df["amt_decimal"] = (df[amt_col] - df[amt_col].astype(int)).round(2)
        df["amt_is_round"] = (df["amt_decimal"] == 0).astype(int)

    # Time features
    time_col = state.get("time_col")
    if time_col and time_col in df.columns:
        df["hour"] = (df[time_col] % 86400) / 3600
        df["day_of_week"] = (df[time_col] // 86400) % 7
        df["is_night"] = ((df["hour"] >= 0) & (df["hour"] <= 6)).astype(int)

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
        parts = name.split("_", 1)
        if len(parts) == 2:
            col_a_name, col_b_name = parts[0], parts[1]
            # Reconstruct from card_col and cat name
            col_a = state.get("card_col", col_a_name)
            col_b = col_b_name
            if col_a in df.columns and col_b in df.columns:
                key = df[col_a].astype(str) + "_" + df[col_b].astype(str)
                df[f"{name}_target_enc"] = key.map(te_map).fillna(global_mean)

    # --- Velocity features ---
    card_col = state.get("card_col")
    if card_col and card_col in df.columns and "vel_median_gap" in state:
        card_str_v = df[card_col].astype(str)
        df["vel_median_gap"] = card_str_v.map(state["vel_median_gap"]).fillna(0)
        df["vel_std_gap"] = card_str_v.map(state["vel_std_gap"]).fillna(0)
        df["vel_min_gap"] = card_str_v.map(state["vel_min_gap"]).fillna(0)
        df["vel_burst_count"] = card_str_v.map(state["vel_burst"]).fillna(0)
        df["vel_daily_rate"] = card_str_v.map(state["vel_daily_rate"]).fillna(0)

    # --- Behavioral profiling ---
    if card_col and card_col in df.columns and "behav_amt_mean" in state:
        card_str_b = df[card_col].astype(str)
        b_mean = card_str_b.map(state["behav_amt_mean"]).fillna(0)
        b_std = card_str_b.map(state["behav_amt_std"]).fillna(1).clip(lower=0.01)
        b_max = card_str_b.map(state["behav_amt_max"]).fillna(0)
        b_median = card_str_b.map(state["behav_amt_median"]).fillna(0)
        amt_col = state.get("amt_col")
        if amt_col and amt_col in df.columns:
            df["behav_amt_zscore"] = (df[amt_col] - b_mean) / b_std
            df["behav_amt_ratio_max"] = df[amt_col] / b_max.clip(lower=0.01)
            df["behav_amt_above_median"] = (df[amt_col] > b_median).astype(int)
            df["behav_is_max_ever"] = (df[amt_col] >= b_max * 0.99).astype(int)

    if card_col and card_col in df.columns and "behav_hour_mean" in state:
        time_col = state.get("time_col")
        if time_col and time_col in df.columns:
            hour = (df[time_col] % 86400) / 3600
            user_h_mean = df[card_col].astype(str).map(state["behav_hour_mean"]).fillna(12)
            user_h_std = df[card_col].astype(str).map(state["behav_hour_std"]).fillna(4).clip(lower=1)
            df["behav_hour_deviation"] = abs(hour - user_h_mean) / user_h_std

    # --- Anomaly score ---
    anomaly_cols = state.get("anomaly_cols", [])
    a_means = state.get("anomaly_means", {})
    a_stds = state.get("anomaly_stds", {})
    if anomaly_cols:
        z2_sum = pd.Series(0.0, index=df.index)
        z2_max = pd.Series(0.0, index=df.index)
        for c in anomaly_cols:
            if c in df.columns:
                z2 = ((df[c].fillna(0) - a_means.get(c, 0)) / max(a_stds.get(c, 1), 0.001)) ** 2
                z2_sum += z2
                z2_max = np.maximum(z2_max, z2)
        df["anomaly_score"] = np.sqrt(z2_sum)
        df["anomaly_max_z"] = np.sqrt(z2_max)

    # --- Entity sharing ---
    for col, sharing_map in state.get("entity_sharing", {}).items():
        if col in df.columns:
            df[f"shared_{col}_count"] = df[col].astype(str).map(sharing_map).fillna(1)

    # --- Amount patterns ---
    amt_col = state.get("amt_col")
    if amt_col and amt_col in df.columns:
        df["amt_is_round_10"] = (df[amt_col] % 10 == 0).astype(int)
        df["amt_is_round_100"] = (df[amt_col] % 100 == 0).astype(int)
        df["amt_cents"] = (df[amt_col] * 100 % 100).astype(int)

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
