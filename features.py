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

    # Categorical-level aggregations (merchant, category, etc.)
    state["cat_agg"] = {}
    if amt_col:
        for cat in cat_cols[:6]:
            grp = df_train.groupby(cat)[amt_col].agg(["count", "mean", "std"])
            state["cat_agg"][cat] = {
                "count": {str(k): float(v) for k, v in grp["count"].items()},
                "mean": {str(k): float(v) for k, v in grp["mean"].items()},
                "std": {str(k): float(v) for k, v in grp["std"].fillna(1).items()},
            }

    # Numeric column binning + target encoding
    state["num_bin_te"] = {}
    for col in num_feature_cols:
        if col == amt_col or col == time_col:
            continue
        if df_train[col].nunique() < 5:
            continue
        try:
            _, bins = pd.qcut(df_train[col].dropna(), q=10, retbins=True, duplicates="drop")
            if len(bins) < 3:
                continue
            binned = pd.cut(df_train[col], bins=bins, labels=False, include_lowest=True)
            te = _target_encode_fit(binned.astype(str), y_train, global_mean, 50, 20)
            state["num_bin_te"][col] = {"bins": [float(b) for b in bins], "te": te}
        except Exception:
            pass

    # Interaction target encodings: card x each categorical
    state["interaction_te"] = {}
    state["interaction_cols"] = {}
    if card_col:
        for cat in cat_cols[:5]:
            name = f"card_x_{cat}"
            key = df_train[card_col].astype(str) + "_" + df_train[cat].astype(str)
            state["interaction_te"][name] = _target_encode_fit(key, y_train, global_mean)
            state["interaction_cols"][name] = [card_col, cat]

    # Pairwise categorical interactions (top 3 x top 3)
    for i, a in enumerate(cat_cols[:3]):
        for b in cat_cols[i+1:4]:
            name = f"{a}_x_{b}"
            key = df_train[a].astype(str) + "_" + df_train[b].astype(str)
            state["interaction_te"][name] = _target_encode_fit(key, y_train, global_mean)
            state["interaction_cols"][name] = [a, b]

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

    # Amount ratio to card mean
    card_col = state.get("card_col")
    if card_col and card_col in df.columns and amt_col and amt_col in df.columns:
        card_mean_val = df[card_col].astype(str).map(state.get("card_mean", {})).fillna(0)
        df["amt_ratio_card"] = df[amt_col] / card_mean_val.clip(lower=0.01)
        df["amt_above_card_mean"] = (df[amt_col] > card_mean_val).astype(int)

    # Geo-distance features (if lat/long available for customer and merchant)
    if all(c in df.columns for c in ["lat", "long", "merch_lat", "merch_long"]):
        df["geo_distance"] = np.sqrt(
            (df["lat"] - df["merch_lat"]) ** 2 + (df["long"] - df["merch_long"]) ** 2
        )
        df["log_geo_distance"] = np.log1p(df["geo_distance"])

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

    # Categorical aggregation features (merchant count/mean/std, etc.)
    for cat, agg_maps in state.get("cat_agg", {}).items():
        if cat in df.columns:
            cat_str = df[cat].astype(str)
            df[f"{cat}_count"] = cat_str.map(agg_maps.get("count", {})).fillna(0)
            df[f"{cat}_amt_mean"] = cat_str.map(agg_maps.get("mean", {})).fillna(0)
            cat_std_map = agg_maps.get("std", {})
            if amt_col and amt_col in df.columns and cat_std_map:
                cat_std = cat_str.map(cat_std_map).fillna(1).clip(lower=1)
                df[f"{cat}_amt_zscore"] = (df[amt_col] - df[f"{cat}_amt_mean"]) / cat_std

    # Numeric binned target encoding
    for col, bin_info in state.get("num_bin_te", {}).items():
        if col in df.columns:
            bins = bin_info["bins"]
            te_map = bin_info["te"]
            binned = pd.cut(df[col], bins=bins, labels=False, include_lowest=True)
            df[f"{col}_bin_te"] = binned.astype(str).map(te_map).fillna(global_mean)

    # Interaction target encodings
    interaction_cols = state.get("interaction_cols", {})
    for name, te_map in state.get("interaction_te", {}).items():
        cols = interaction_cols.get(name)
        if cols and len(cols) == 2 and cols[0] in df.columns and cols[1] in df.columns:
            key = df[cols[0]].astype(str) + "_" + df[cols[1]].astype(str)
            df[f"{name}_te"] = key.map(te_map).fillna(global_mean)

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
