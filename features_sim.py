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

    card_col = "card_id"
    time_col = "TransactionDT"
    amt_col = "TransactionAmt"

    state["card_col"] = card_col
    state["time_col"] = time_col
    state["amt_col"] = amt_col

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
    if amt_col in df_train.columns:
        for grp_col in ["merchant", "category", "gender"]:
            if grp_col in df_train.columns:
                stats = df_train.groupby(grp_col)[amt_col].agg(["median", "std"])
                state[f"{grp_col}_amt_median"] = {str(k): float(v) for k, v in stats["median"].items()}
                state[f"{grp_col}_amt_std"] = {str(k): float(v) for k, v in stats["std"].fillna(1.0).items()}
        state["global_amt_median"] = float(df_train[amt_col].median())
        state["global_amt_std"] = float(df_train[amt_col].std())

    # ----------------------------------------------------------------
    # VELOCITY FEATURES (Recipe 2) — per-card temporal statistics
    # ----------------------------------------------------------------
    if card_col in df_train.columns and time_col in df_train.columns:
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

        # Per-card transaction count
        card_txn_count = df_sorted.groupby(card_col)[time_col].count()
        state["card_txn_count"] = {str(k): int(v) for k, v in card_txn_count.items()}

    # ----------------------------------------------------------------
    # BEHAVIORAL PROFILING (Recipe 3) — per-card amount stats
    # ----------------------------------------------------------------
    if card_col in df_train.columns and amt_col in df_train.columns:
        card_amt = df_train.groupby(card_col)[amt_col].agg(["mean", "std", "median", "max", "count"])
        state["behav_amt_mean"] = {str(k): float(v) for k, v in card_amt["mean"].items()}
        state["behav_amt_std"] = {str(k): float(v) for k, v in card_amt["std"].fillna(1).items()}
        state["behav_amt_median"] = {str(k): float(v) for k, v in card_amt["median"].items()}
        state["behav_amt_max"] = {str(k): float(v) for k, v in card_amt["max"].items()}
        state["behav_txn_count"] = {str(k): int(v) for k, v in card_amt["count"].items()}

    if card_col in df_train.columns and time_col in df_train.columns:
        df_train_copy = df_train.copy()
        df_train_copy["_hour"] = (df_train_copy[time_col] % 86400) / 3600
        hour_stats = df_train_copy.groupby(card_col)["_hour"].agg(["mean", "std"])
        state["behav_hour_mean"] = {str(k): float(v) for k, v in hour_stats["mean"].items()}
        state["behav_hour_std"] = {str(k): float(v) for k, v in hour_stats["std"].fillna(4).items()}

    # ----------------------------------------------------------------
    # ANOMALY SCORE (Recipe 6) — Mahalanobis from training centroid
    # ----------------------------------------------------------------
    num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = {card_col, time_col, "TransactionID", "txn_id", "label", "is_fraud"}
    anomaly_cols = [c for c in num_cols if c not in exclude_cols and df_train[c].std() > 0][:20]
    col_means = {c: float(df_train[c].mean()) for c in anomaly_cols}
    col_stds = {c: float(df_train[c].std()) for c in anomaly_cols}
    state["anomaly_cols"] = anomaly_cols
    state["anomaly_means"] = col_means
    state["anomaly_stds"] = col_stds

    # ----------------------------------------------------------------
    # PER-CARD-CATEGORY AMOUNT STATS — "is this unusual for this card in this category?"
    # ----------------------------------------------------------------
    if card_col in df_train.columns and "category" in df_train.columns and amt_col in df_train.columns:
        cc_key = df_train[card_col].astype(str) + "_" + df_train["category"].astype(str)
        cc_stats = df_train.assign(_cc_key=cc_key).groupby("_cc_key")[amt_col].agg(["mean", "std", "count"])
        state["card_cat_amt_mean"] = {str(k): float(v) for k, v in cc_stats["mean"].items()}
        state["card_cat_amt_std"] = {str(k): float(v) for k, v in cc_stats["std"].fillna(1).items()}
        state["card_cat_txn_count"] = {str(k): int(v) for k, v in cc_stats["count"].items()}

    # ----------------------------------------------------------------
    # PER-CARD WEEKEND RATIO AND NIGHT RATIO
    # ----------------------------------------------------------------
    if card_col in df_train.columns and time_col in df_train.columns:
        df_tmp = df_train.copy()
        df_tmp["_hour"] = df_tmp[time_col] // 3600 % 24
        df_tmp["_dow"] = df_tmp[time_col] // 86400 % 7
        df_tmp["_is_night"] = ((df_tmp["_hour"] < 6) | (df_tmp["_hour"] >= 22)).astype(int)
        df_tmp["_is_weekend"] = (df_tmp["_dow"] >= 5).astype(int)
        card_night_ratio = df_tmp.groupby(card_col)["_is_night"].mean()
        card_weekend_ratio = df_tmp.groupby(card_col)["_is_weekend"].mean()
        state["card_night_ratio"] = {str(k): float(v) for k, v in card_night_ratio.items()}
        state["card_weekend_ratio"] = {str(k): float(v) for k, v in card_weekend_ratio.items()}

    # ----------------------------------------------------------------
    # VELOCITY — hour of week bins (168 bins) per card
    # ----------------------------------------------------------------
    if card_col in df_train.columns and time_col in df_train.columns:
        df_tmp2 = df_train.copy()
        df_tmp2["_hour_of_week"] = (df_tmp2[time_col] // 3600) % 168
        # Count transactions per card per 6-hour block
        df_tmp2["_6h_block"] = df_tmp2["_hour_of_week"] // 6
        card_6h = df_tmp2.groupby([card_col, "_6h_block"]).size()
        # Store most active 6h block per card
        if len(card_6h) > 0:
            card_peak_block = card_6h.groupby(level=0).idxmax().apply(lambda x: x[1] if isinstance(x, tuple) else -1)
            state["card_peak_6h_block"] = {str(k): int(v) for k, v in card_peak_block.items()}

    # ----------------------------------------------------------------
    # PER-CARD-MERCHANT STATS
    # ----------------------------------------------------------------
    if card_col in df_train.columns and "merchant" in df_train.columns and amt_col in df_train.columns:
        cm_key = df_train[card_col].astype(str) + "_" + df_train["merchant"].astype(str)
        cm_stats = df_train.assign(_cm_key=cm_key).groupby("_cm_key")[amt_col].agg(["mean", "std", "count"])
        state["card_merch_amt_mean"] = {str(k): float(v) for k, v in cm_stats["mean"].items()}
        state["card_merch_amt_std"] = {str(k): float(v) for k, v in cm_stats["std"].fillna(1).items()}
        state["card_merch_txn_count"] = {str(k): int(v) for k, v in cm_stats["count"].items()}
        # Is this card's first time at this merchant? (txn count = 0 means new in val/OOT)

    # ----------------------------------------------------------------
    # VELOCITY EXTENDED — inter-quartile gap, max gap
    # ----------------------------------------------------------------
    if card_col in df_train.columns and time_col in df_train.columns:
        df_sorted2 = df_train.sort_values([card_col, time_col])
        df_sorted2["_gap2"] = df_sorted2.groupby(card_col)[time_col].diff()
        gap_q25 = df_sorted2.groupby(card_col)["_gap2"].quantile(0.25)
        gap_q75 = df_sorted2.groupby(card_col)["_gap2"].quantile(0.75)
        gap_max = df_sorted2.groupby(card_col)["_gap2"].max()
        state["velocity_gap_q25"] = {str(k): float(v) for k, v in gap_q25.fillna(0).items()}
        state["velocity_gap_q75"] = {str(k): float(v) for k, v in gap_q75.fillna(0).items()}
        state["velocity_gap_max"] = {str(k): float(v) for k, v in gap_max.fillna(0).items()}

        # Weekly transaction count (different from daily rate)
        df_sorted2["_week"] = df_sorted2[time_col] // (86400 * 7)
        weekly_counts = df_sorted2.groupby([card_col, "_week"]).size()
        card_max_weekly = weekly_counts.groupby(level=0).max()
        card_mean_weekly = weekly_counts.groupby(level=0).mean()
        state["card_max_weekly_txn"] = {str(k): int(v) for k, v in card_max_weekly.items()}
        state["card_mean_weekly_txn"] = {str(k): float(v) for k, v in card_mean_weekly.items()}

    return state


def transform(df: pd.DataFrame, state: dict, config: dict) -> pd.DataFrame:
    df = df.copy()
    global_mean = state.get("global_mean", 0.0)
    amt_col = "TransactionAmt"
    card_col = state.get("card_col", "card_id")
    time_col = state.get("time_col", "TransactionDT")

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
    if time_col in df.columns:
        df["hour"] = (df[time_col] // 3600 % 24).astype(int)
        df["day_of_week"] = (df[time_col] // 86400 % 7).astype(int)
        df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)

    # ----------------------------------------------------------------
    # VELOCITY FEATURES
    # ----------------------------------------------------------------
    if card_col in df.columns:
        card_str = df[card_col].astype(str)
        df["velocity_median_gap"] = card_str.map(state.get("velocity_median_gap", {})).fillna(0)
        df["velocity_std_gap"] = card_str.map(state.get("velocity_std_gap", {})).fillna(0)
        df["velocity_min_gap"] = card_str.map(state.get("velocity_min_gap", {})).fillna(0)
        df["velocity_burst_count"] = card_str.map(state.get("velocity_burst_count", {})).fillna(0)
        df["velocity_daily_rate"] = card_str.map(state.get("velocity_daily_rate", {})).fillna(0)
        df["card_txn_count"] = card_str.map(state.get("card_txn_count", {})).fillna(0)
        vdr = df["velocity_daily_rate"]
        p95 = vdr.quantile(0.95) if len(vdr) > 0 else 1
        df["is_high_velocity"] = (vdr > p95).astype(int)
        df["log_velocity_daily_rate"] = np.log1p(df["velocity_daily_rate"])
        df["log_velocity_burst_count"] = np.log1p(df["velocity_burst_count"])

    # ----------------------------------------------------------------
    # BEHAVIORAL PROFILING — amount deviation from card's own history
    # ----------------------------------------------------------------
    if card_col in df.columns and amt_col in df.columns:
        card_str = df[card_col].astype(str)
        card_mean = card_str.map(state.get("behav_amt_mean", {})).fillna(0)
        card_std = card_str.map(state.get("behav_amt_std", {})).fillna(1).clip(lower=0.01)
        card_max = card_str.map(state.get("behav_amt_max", {})).fillna(0)
        card_median = card_str.map(state.get("behav_amt_median", {})).fillna(0)

        df["behav_amt_zscore"] = (df[amt_col] - card_mean) / card_std
        df["behav_amt_ratio_max"] = df[amt_col] / card_max.clip(lower=0.01)
        df["behav_amt_above_median"] = (df[amt_col] > card_median).astype(int)
        df["behav_is_max_ever"] = (df[amt_col] >= card_max).astype(int)

        # Hour deviation from user's typical pattern
        if time_col in df.columns:
            hour_float = (df[time_col] % 86400) / 3600
            user_hour_mean = card_str.map(state.get("behav_hour_mean", {})).fillna(12)
            user_hour_std = card_str.map(state.get("behav_hour_std", {})).fillna(4).clip(lower=1)
            df["behav_hour_deviation"] = abs(hour_float - user_hour_mean) / user_hour_std

    # ----------------------------------------------------------------
    # ANOMALY SCORE — Mahalanobis distance from training centroid
    # ----------------------------------------------------------------
    anomaly_cols = state.get("anomaly_cols", [])
    means = state.get("anomaly_means", {})
    stds = state.get("anomaly_stds", {})
    if anomaly_cols:
        z_scores = pd.DataFrame()
        for c in anomaly_cols:
            if c in df.columns:
                z_scores[c] = ((df[c].fillna(0) - means.get(c, 0)) / max(stds.get(c, 1), 0.001)) ** 2
        if not z_scores.empty:
            df["anomaly_mahalanobis"] = np.sqrt(z_scores.sum(axis=1))
            df["anomaly_max_zscore"] = z_scores.max(axis=1)

    # ----------------------------------------------------------------
    # PER-CARD-CATEGORY AMOUNT STATS
    # ----------------------------------------------------------------
    if card_col in df.columns and "category" in df.columns and amt_col in df.columns:
        card_str_cat = df[card_col].astype(str)
        cat_str = df["category"].astype(str) if "category" in df.columns else None
        if cat_str is not None:
            cc_key = card_str_cat + "_" + cat_str
            g_mean = float(state.get("global_mean", 0.0))
            g_amt_mean = float(state.get("global_amt_median", 0.0))
            cc_amt_mean = cc_key.map(state.get("card_cat_amt_mean", {})).fillna(g_amt_mean)
            cc_amt_std = cc_key.map(state.get("card_cat_amt_std", {})).fillna(1).clip(lower=0.01)
            cc_cnt = cc_key.map(state.get("card_cat_txn_count", {})).fillna(0)
            df["card_cat_amt_zscore"] = (df[amt_col] - cc_amt_mean) / cc_amt_std
            df["card_cat_txn_count"] = cc_cnt
            df["log_card_cat_txn_count"] = np.log1p(cc_cnt)

    # ----------------------------------------------------------------
    # PER-CARD TIME PATTERN FEATURES
    # ----------------------------------------------------------------
    if card_col in df.columns:
        card_str3 = df[card_col].astype(str)
        if "card_night_ratio" in state:
            df["card_night_ratio"] = card_str3.map(state["card_night_ratio"]).fillna(0)
        if "card_weekend_ratio" in state:
            df["card_weekend_ratio"] = card_str3.map(state["card_weekend_ratio"]).fillna(0)
        if "card_peak_6h_block" in state and time_col in df.columns:
            card_peak = card_str3.map(state["card_peak_6h_block"]).fillna(-1)
            current_6h = (df[time_col] // 3600 % 168) // 6
            df["is_off_peak_block"] = (current_6h != card_peak).astype(int)

    # ----------------------------------------------------------------
    # PER-CARD-MERCHANT AMOUNT STATS
    # ----------------------------------------------------------------
    if card_col in df.columns and "merchant" in df.columns and amt_col in df.columns:
        card_str4 = df[card_col].astype(str)
        merch_str = df["merchant"].astype(str)
        cm_key = card_str4 + "_" + merch_str
        g_amt = float(state.get("global_amt_median", 0.0))
        cm_amt_mean = cm_key.map(state.get("card_merch_amt_mean", {})).fillna(g_amt)
        cm_amt_std = cm_key.map(state.get("card_merch_amt_std", {})).fillna(1).clip(lower=0.01)
        cm_cnt = cm_key.map(state.get("card_merch_txn_count", {})).fillna(0)
        df["card_merch_amt_zscore"] = (df[amt_col] - cm_amt_mean) / cm_amt_std
        df["card_merch_txn_count"] = cm_cnt
        df["is_new_merchant_for_card"] = (cm_cnt == 0).astype(int)

    # ----------------------------------------------------------------
    # VELOCITY EXTENDED FEATURES
    # ----------------------------------------------------------------
    if card_col in df.columns:
        card_str5 = df[card_col].astype(str)
        df["velocity_gap_q25"] = card_str5.map(state.get("velocity_gap_q25", {})).fillna(0)
        df["velocity_gap_q75"] = card_str5.map(state.get("velocity_gap_q75", {})).fillna(0)
        df["velocity_gap_max"] = card_str5.map(state.get("velocity_gap_max", {})).fillna(0)
        df["card_max_weekly_txn"] = card_str5.map(state.get("card_max_weekly_txn", {})).fillna(0)
        df["card_mean_weekly_txn"] = card_str5.map(state.get("card_mean_weekly_txn", {})).fillna(0)
        # Ratio: max weekly vs mean weekly (spiky vs consistent)
        df["weekly_spikiness"] = df["card_max_weekly_txn"] / (df["card_mean_weekly_txn"].clip(lower=0.1))

    df = df.fillna(-1)
    return df
