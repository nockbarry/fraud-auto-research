"""Feature transforms applied after data loading.

This file is edited by the agent. The harness calls fit() then transform().

API CONTRACT:
  - fit(df_train, y_train, config) -> state
      Called ONCE on training data WITH labels.
      Return a JSON-serializable dict of fitted parameters.
  - transform(df, state, config) -> df
      Called on EACH split WITHOUT labels.
      Use only the state dict from fit(). No access to labels.
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
        state["cat_te"][col] = _target_encode_fit(df_train[col], y_train, global_mean, 20, 10)

    # Detect high-null columns to drop (>95% null)
    null_rates = df_train.isna().mean()
    state["drop_cols"] = null_rates[null_rates > 0.95].index.tolist()

    # Detect key columns
    amt_col = None
    for candidate in ["TransactionAmt", "amt", "Amount", "amount"]:
        if candidate in df_train.columns:
            amt_col = candidate
            break
    state["amt_col"] = amt_col

    card_col = None
    for candidate in ["card1", "card_id", "cc_num", "customer_id", "card"]:
        if candidate in df_train.columns:
            card_col = candidate
            break
    state["card_col"] = card_col

    time_col = None
    for candidate in ["TransactionDT", "unix_time", "Time"]:
        if candidate in df_train.columns:
            time_col = candidate
            break
    state["time_col"] = time_col

    # Card-level amount aggregations
    if card_col and amt_col:
        card_stats = df_train.groupby(card_col)[amt_col].agg(["count", "mean", "std", "median", "max"])
        state["card_count"] = {str(k): float(v) for k, v in card_stats["count"].items()}
        state["card_mean"] = {str(k): float(v) for k, v in card_stats["mean"].items()}
        state["card_std"] = {str(k): float(v) for k, v in card_stats["std"].fillna(1).items()}
        state["card_median"] = {str(k): float(v) for k, v in card_stats["median"].items()}
        state["card_max"] = {str(k): float(v) for k, v in card_stats["max"].items()}

        # Target encoding for card
        state["card_te"] = _target_encode_fit(df_train[card_col].astype(str), y_train, global_mean)

    # Interaction target encodings: card x each categorical
    state["interaction_te"] = {}
    if card_col:
        for cat in cat_cols[:5]:
            key = df_train[card_col].astype(str) + "_" + df_train[cat].astype(str)
            name = f"{card_col}_{cat}"
            state["interaction_te"][name] = _target_encode_fit(key, y_train, global_mean)

    # === BEHAVIORAL PROFILING (Recipe 3) ===
    if card_col and time_col:
        df_train["_hour"] = (df_train[time_col] % 86400) / 3600
        hour_stats = df_train.groupby(card_col)["_hour"].agg(["mean", "std"])
        state["behav_hour_mean"] = {str(k): float(v) for k, v in hour_stats["mean"].items()}
        state["behav_hour_std"] = {str(k): float(v) for k, v in hour_stats["std"].fillna(4).items()}
        df_train.drop(columns=["_hour"], inplace=True)

    # === VELOCITY FEATURES (Recipe 2) ===
    if card_col and time_col:
        df_sorted = df_train.sort_values([card_col, time_col])
        df_sorted["_gap"] = df_sorted.groupby(card_col)[time_col].diff()

        gap_stats = df_sorted.groupby(card_col)["_gap"].agg(["median", "std", "min", "count"])
        df_sorted["_is_burst"] = (df_sorted["_gap"] < 60).astype(int)
        burst_counts = df_sorted.groupby(card_col)["_is_burst"].sum()

        time_range = df_sorted.groupby(card_col)[time_col].agg(["min", "max"])
        time_range["days"] = ((time_range["max"] - time_range["min"]) / 86400).clip(lower=1)
        daily_rate = gap_stats["count"] / time_range["days"]

        state["velocity_median_gap"] = {str(k): float(v) for k, v in gap_stats["median"].fillna(0).items()}
        state["velocity_std_gap"] = {str(k): float(v) for k, v in gap_stats["std"].fillna(0).items()}
        state["velocity_min_gap"] = {str(k): float(v) for k, v in gap_stats["min"].fillna(0).items()}
        state["velocity_burst_count"] = {str(k): int(v) for k, v in burst_counts.items()}
        state["velocity_daily_rate"] = {str(k): float(v) for k, v in daily_rate.items()}

    # === GEO FEATURES (per-card average distance) ===
    if "lat" in df_train.columns and "merch_lat" in df_train.columns:
        # Compute geo distance for training data
        dlat = np.radians(df_train["lat"] - df_train["merch_lat"])
        dlon = np.radians(df_train["long"] - df_train["merch_long"])
        a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(df_train["lat"])) * np.cos(np.radians(df_train["merch_lat"])) * np.sin(dlon / 2) ** 2
        df_train["_geo_dist"] = 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        if card_col:
            card_geo = df_train.groupby(card_col)["_geo_dist"].agg(["mean", "std"])
            state["card_geo_mean"] = {str(k): float(v) for k, v in card_geo["mean"].fillna(0).items()}
            state["card_geo_std"] = {str(k): float(v) for k, v in card_geo["std"].fillna(1).items()}
        df_train.drop(columns=["_geo_dist"], inplace=True)

    # === CATEGORY AMOUNT CORRIDORS (Recipe 7) ===
    if amt_col:
        state["category_amt_corridors"] = {}
        for cat in cat_cols[:3]:
            if cat in df_train.columns:
                q25 = df_train.groupby(cat)[amt_col].quantile(0.25)
                q50 = df_train.groupby(cat)[amt_col].quantile(0.50)
                q75 = df_train.groupby(cat)[amt_col].quantile(0.75)
                state["category_amt_corridors"][cat] = {
                    "median": {str(k): float(v) for k, v in q50.items()},
                    "q25": {str(k): float(v) for k, v in q25.items()},
                    "q75": {str(k): float(v) for k, v in q75.items()},
                }

    # === STATE-LEVEL TARGET ENCODING ===
    if "state" in df_train.columns:
        state["state_te"] = _target_encode_fit(df_train["state"].astype(str), y_train, global_mean, 20, 10)

    # === MERCHANT-LEVEL STATS ===
    if "merchant" in df_train.columns and amt_col:
        merch_stats = df_train.groupby("merchant")[amt_col].agg(["count", "mean"])
        state["merch_count"] = {str(k): float(v) for k, v in merch_stats["count"].items()}
        state["merch_mean_amt"] = {str(k): float(v) for k, v in merch_stats["mean"].items()}
        if card_col:
            merch_unique_cards = df_train.groupby("merchant")[card_col].nunique()
            state["merch_unique_cards"] = {str(k): float(v) for k, v in merch_unique_cards.items()}

    # === MERCHANT x CATEGORY AMOUNT CORRIDORS ===
    if "merchant" in df_train.columns and "category" in df_train.columns and amt_col:
        mc_key = df_train["merchant"].astype(str) + "||" + df_train["category"].astype(str)
        mc_median = df_train.groupby(mc_key)[amt_col].median()
        mc_std = df_train.groupby(mc_key)[amt_col].std().fillna(1.0)
        state["mc_amt_median"] = {str(k): float(v) for k, v in mc_median.items()}
        state["mc_amt_std"] = {str(k): float(v) for k, v in mc_std.items()}
        # Global fallbacks
        state["global_amt_median"] = float(df_train[amt_col].median())
        state["global_amt_std"] = float(df_train[amt_col].std())

    # === CUSTOMER x CATEGORY AMOUNT CORRIDORS ===
    if card_col and "category" in df_train.columns and amt_col:
        cc_key = df_train[card_col].astype(str) + "||" + df_train["category"].astype(str)
        cc_median = df_train.groupby(cc_key)[amt_col].median()
        cc_std = df_train.groupby(cc_key)[amt_col].std().fillna(1.0)
        state["cc_amt_median"] = {str(k): float(v) for k, v in cc_median.items()}
        state["cc_amt_std"] = {str(k): float(v) for k, v in cc_std.items()}

    # === HOUR-OF-DAY x MERCHANT x CATEGORY AMOUNT STATS ===
    # Is this transaction amount typical for this merchant-category at this hour?
    if "merchant" in df_train.columns and "category" in df_train.columns and amt_col and time_col:
        hour = (df_train[time_col] % 86400) / 3600
        is_night = (hour < 6) | (hour >= 22)
        # Night transactions within merchant-category (small key space: 800 merchants x ~15 categories x 2 = ~24k)
        mc_key = df_train["merchant"].astype(str) + "||" + df_train["category"].astype(str)
        mc_night_key = mc_key + "||" + is_night.astype(str)
        mc_night_median = df_train.groupby(mc_night_key)[amt_col].median()
        mc_night_std = df_train.groupby(mc_night_key)[amt_col].std().fillna(1.0)
        state["mc_night_amt_median"] = {str(k): float(v) for k, v in mc_night_median.items()}
        state["mc_night_amt_std"] = {str(k): float(v) for k, v in mc_night_std.items()}

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
        df["amt_small"] = (df[amt_col] < 5).astype(int)

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
        card_median = card_str.map(state.get("card_median", {})).fillna(0)
        card_max = card_str.map(state.get("card_max", {})).fillna(0)

        if amt_col and amt_col in df.columns:
            df["amt_zscore_card"] = (df[amt_col] - df["card_amt_mean"]) / card_std.clip(lower=1)
            df["behav_amt_ratio_max"] = df[amt_col] / card_max.clip(lower=0.01)
            df["behav_amt_above_median"] = (df[amt_col] > card_median).astype(int)
            df["behav_is_max_ever"] = (df[amt_col] >= card_max).astype(int)

    # Card target encoding
    if card_col and card_col in df.columns and "card_te" in state:
        df["card_target_enc"] = df[card_col].astype(str).map(state["card_te"]).fillna(global_mean)

    # Interaction target encodings
    for name, te_map in state.get("interaction_te", {}).items():
        parts = name.split("_", 1)
        if len(parts) == 2:
            col_a = state.get("card_col", parts[0])
            col_b = parts[1]
            if col_a in df.columns and col_b in df.columns:
                key = df[col_a].astype(str) + "_" + df[col_b].astype(str)
                df[f"{name}_target_enc"] = key.map(te_map).fillna(global_mean)

    # Behavioral hour deviation
    if card_col and card_col in df.columns and time_col and time_col in df.columns:
        card_str = df[card_col].astype(str)
        hour = (df[time_col] % 86400) / 3600
        user_hour_mean = card_str.map(state.get("behav_hour_mean", {})).fillna(12)
        user_hour_std = card_str.map(state.get("behav_hour_std", {})).fillna(4).clip(lower=1)
        df["behav_hour_deviation"] = abs(hour - user_hour_mean) / user_hour_std

    # Velocity features (keep only most important ones)
    if card_col and card_col in df.columns:
        card_str = df[card_col].astype(str)
        df["velocity_median_gap"] = card_str.map(state.get("velocity_median_gap", {})).fillna(0)
        df["velocity_daily_rate"] = card_str.map(state.get("velocity_daily_rate", {})).fillna(0)

    # === GEO FEATURES ===
    if "lat" in df.columns and "merch_lat" in df.columns:
        dlat = np.radians(df["lat"] - df["merch_lat"])
        dlon = np.radians(df["long"] - df["merch_long"])
        a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(df["lat"])) * np.cos(np.radians(df["merch_lat"])) * np.sin(dlon / 2) ** 2
        df["geo_distance"] = 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        df["log_geo_distance"] = np.log1p(df["geo_distance"])

        # Per-card geo deviation
        if card_col and card_col in df.columns:
            card_str = df[card_col].astype(str)
            card_geo_mean = card_str.map(state.get("card_geo_mean", {})).fillna(0)
            card_geo_std = card_str.map(state.get("card_geo_std", {})).fillna(1).clip(lower=0.1)
            df["geo_deviation"] = (df["geo_distance"] - card_geo_mean) / card_geo_std
            df["geo_far_flag"] = (df["geo_deviation"] > 2).astype(int)

    # City pop features
    if "city_pop" in df.columns:
        df["log_city_pop"] = np.log1p(df["city_pop"])
        if amt_col and amt_col in df.columns:
            df["amt_per_city_pop"] = df[amt_col] / df["city_pop"].clip(lower=1)

    # State target encoding
    if "state" in df.columns and "state_te" in state:
        df["state_target_enc"] = df["state"].astype(str).map(state["state_te"]).fillna(global_mean)

    # Merchant features
    if "merchant" in df.columns:
        merch_str = df["merchant"].astype(str)
        df["merch_txn_count"] = merch_str.map(state.get("merch_count", {})).fillna(0)
        df["merch_mean_amt"] = merch_str.map(state.get("merch_mean_amt", {})).fillna(0)
        df["merch_unique_cards"] = merch_str.map(state.get("merch_unique_cards", {})).fillna(0)

    # Merchant x Category amount deviation
    if "merchant" in df.columns and "category" in df.columns and "mc_amt_median" in state:
        mc_key = df["merchant"].astype(str) + "||" + df["category"].astype(str)
        mc_median = mc_key.map(state["mc_amt_median"]).fillna(state.get("global_amt_median", 0))
        mc_std = mc_key.map(state["mc_amt_std"]).fillna(state.get("global_amt_std", 1.0))
        if amt_col and amt_col in df.columns:
            df["mc_amt_ratio_median"] = df[amt_col] / mc_median.clip(lower=0.01)
            df["mc_amt_zscore"] = (df[amt_col] - mc_median) / mc_std.clip(lower=0.01)

    # Customer x Category amount deviation
    if card_col and card_col in df.columns and "category" in df.columns and "cc_amt_median" in state:
        cc_key = df[card_col].astype(str) + "||" + df["category"].astype(str)
        cc_median = cc_key.map(state["cc_amt_median"]).fillna(state.get("global_amt_median", 0))
        cc_std = cc_key.map(state["cc_amt_std"]).fillna(state.get("global_amt_std", 1.0))
        if amt_col and amt_col in df.columns:
            df["cc_amt_ratio_median"] = df[amt_col] / cc_median.clip(lower=0.01)
            df["cc_amt_zscore"] = (df[amt_col] - cc_median) / cc_std.clip(lower=0.01)

    # Hour x Merchant x Category amount deviation
    if "merchant" in df.columns and "category" in df.columns and "mc_night_amt_median" in state:
        if time_col and time_col in df.columns:
            hour_arr = (df[time_col] % 86400) / 3600
            is_night = (hour_arr < 6) | (hour_arr >= 22)
            mc_key = df["merchant"].astype(str) + "||" + df["category"].astype(str)
            mc_night_key = mc_key + "||" + is_night.astype(str)
            mc_night_median = mc_night_key.map(state["mc_night_amt_median"])
            mc_night_std = mc_night_key.map(state["mc_night_amt_std"])
            # Fall back to mc median
            mc_median_fb = mc_key.map(state.get("mc_amt_median", {})).fillna(state.get("global_amt_median", 0))
            mc_night_median = mc_night_median.fillna(mc_median_fb)
            mc_night_std = mc_night_std.fillna(state.get("global_amt_std", 1.0))
            if amt_col and amt_col in df.columns:
                df["mc_night_amt_zscore"] = (df[amt_col] - mc_night_median) / mc_night_std.clip(lower=0.01)
                df["mc_night_amt_ratio"] = df[amt_col] / mc_night_median.clip(lower=0.01)

    # Category amount corridors
    for cat, corridor in state.get("category_amt_corridors", {}).items():
        if cat in df.columns and amt_col and amt_col in df.columns:
            cat_median = df[cat].astype(str).map(corridor["median"]).fillna(0)
            cat_q25 = df[cat].astype(str).map(corridor["q25"]).fillna(0)
            cat_q75 = df[cat].astype(str).map(corridor["q75"]).fillna(0)
            cat_iqr = (cat_q75 - cat_q25).clip(lower=0.01)
            df[f"{cat}_amt_ratio_median"] = df[amt_col] / cat_median.clip(lower=0.01)
            df[f"{cat}_amt_outside_iqr"] = ((df[amt_col] < cat_q25) | (df[amt_col] > cat_q75)).astype(int)
            df[f"{cat}_amt_zscore"] = (df[amt_col] - cat_median) / cat_iqr

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
