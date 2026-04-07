"""Feature engineering for fraud-sim dataset. Agent-editable.

Fit/transform API:
  fit(df_train, y_train, config) -> state dict (JSON-serializable)
  transform(df, state, config)   -> df (all numeric, no labels)

Actual columns: card_id, merchant, category, TransactionAmt, gender, city, state, zip,
                lat, long, city_pop, job, unix_time, merch_lat, merch_long, TransactionDT
"""
import pandas as pd
import numpy as np


def _smooth_target_enc(series: pd.Series, labels: pd.Series, min_samples: int = 30, global_mean: float = 0.0) -> dict:
    """Compute smoothed target encoding: weight group mean toward global mean."""
    df = pd.DataFrame({"key": series, "label": labels})
    stats = df.groupby("key")["label"].agg(["sum", "count"]).reset_index()
    stats.columns = ["key", "sum", "count"]
    stats["rate"] = stats["sum"] / stats["count"]
    # Smoothing: blend toward global mean by min_samples
    stats["smoothed"] = (stats["sum"] + global_mean * min_samples) / (stats["count"] + min_samples)
    return {str(k): float(v) for k, v in zip(stats["key"], stats["smoothed"])}


def fit(df_train: pd.DataFrame, y_train: pd.Series, config: dict) -> dict:
    """Fit on training data only. Returns JSON-serializable state."""
    state: dict = {}
    global_mean = float(y_train.mean())
    state["global_mean"] = global_mean

    # Drop columns with >50% NaN
    nan_rates = df_train.isnull().mean()
    state["drop_cols"] = nan_rates[nan_rates > 0.50].index.tolist()
    df_tmp = df_train.drop(columns=state["drop_cols"], errors="ignore")

    # Identify categorical columns (excluding ID cols we handle separately)
    cat_cols = df_tmp.select_dtypes("object").columns.tolist()
    # Remove cols handled with target encoding or separate logic
    # Also remove city (894 unique, noisy) and job (494 unique, handled separately)
    skip_freq = {"merchant", "category", "city"}
    cat_cols = [c for c in cat_cols if c not in skip_freq]
    state["cat_cols"] = cat_cols

    # Frequency encoding for remaining categoricals
    for col in cat_cols:
        freq = df_tmp[col].value_counts(normalize=True).to_dict()
        state[f"{col}_freq"] = {str(k): float(v) for k, v in freq.items()}

    # --- Target encoding: category (smoothed fraud rate) ---
    if "category" in df_tmp.columns:
        state["category_te"] = _smooth_target_enc(df_tmp["category"], y_train, min_samples=30, global_mean=global_mean)

    # --- Target encoding: merchant (smoothed fraud rate, ~670 unique) ---
    if "merchant" in df_tmp.columns:
        state["merchant_te"] = _smooth_target_enc(df_tmp["merchant"], y_train, min_samples=30, global_mean=global_mean)

    # --- Behavioral profiling: per-card (card_id) aggregations ---
    if "card_id" in df_tmp.columns and "TransactionAmt" in df_tmp.columns:
        card_stats = df_tmp.groupby("card_id")["TransactionAmt"].agg(
            card_mean="mean",
            card_std="std",
            card_max="max",
            card_median="median",
            card_count="count",
        ).reset_index()
        card_stats["card_std"] = card_stats["card_std"].fillna(0.0)
        state["card_mean"] = {str(k): float(v) for k, v in zip(card_stats["card_id"], card_stats["card_mean"])}
        state["card_std"] = {str(k): float(v) for k, v in zip(card_stats["card_id"], card_stats["card_std"])}
        state["card_max"] = {str(k): float(v) for k, v in zip(card_stats["card_id"], card_stats["card_max"])}
        state["card_median"] = {str(k): float(v) for k, v in zip(card_stats["card_id"], card_stats["card_median"])}
        state["card_count"] = {str(k): float(v) for k, v in zip(card_stats["card_id"], card_stats["card_count"])}
        state["global_amt_mean"] = float(df_tmp["TransactionAmt"].mean())
        state["global_amt_std"] = float(df_tmp["TransactionAmt"].std())

    # --- Amount vs category median ---
    if "category" in df_tmp.columns and "TransactionAmt" in df_tmp.columns:
        cat_median = df_tmp.groupby("category")["TransactionAmt"].median().to_dict()
        state["cat_median_amt"] = {str(k): float(v) for k, v in cat_median.items()}

    # --- Merchant transaction volume (how busy is this merchant) ---
    if "merchant" in df_tmp.columns:
        merch_volume = df_tmp["merchant"].value_counts().to_dict()
        state["merchant_volume"] = {str(k): int(v) for k, v in merch_volume.items()}

    # --- Per-card: fraud-related behavioral ratios ---
    # Number of unique merchants per card (indicates spread of transactions)
    if "card_id" in df_tmp.columns and "merchant" in df_tmp.columns:
        card_merch_counts = df_tmp.groupby("card_id")["merchant"].nunique()
        state["card_n_merchants"] = {str(k): int(v) for k, v in card_merch_counts.items()}

    # --- Per-card: night transaction ratio ---
    if "card_id" in df_tmp.columns and "unix_time" in df_tmp.columns:
        tmp2 = df_tmp.copy()
        dt = pd.to_datetime(tmp2["unix_time"], unit="s")
        tmp2["_is_night"] = ((dt.dt.hour >= 22) | (dt.dt.hour < 6)).astype(int)
        card_night_ratio = tmp2.groupby("card_id")["_is_night"].mean()
        state["card_night_ratio"] = {str(k): float(v) for k, v in card_night_ratio.items()}

    # --- Per-card: unique categories (diversification) ---
    if "card_id" in df_tmp.columns and "category" in df_tmp.columns:
        card_cat_counts = df_tmp.groupby("card_id")["category"].nunique()
        state["card_n_categories"] = {str(k): int(v) for k, v in card_cat_counts.items()}

    return state


def transform(df: pd.DataFrame, state: dict, config: dict) -> pd.DataFrame:
    """Transform without labels. Apply fitted state."""
    df = df.copy()
    df = df.drop(columns=state.get("drop_cols", []), errors="ignore")

    # --- Time features from unix_time ---
    if "unix_time" in df.columns:
        dt = pd.to_datetime(df["unix_time"], unit="s")
        df["hour"] = dt.dt.hour
        df["day_of_week"] = dt.dt.dayofweek
        df["is_night"] = ((df["hour"] >= 22) | (df["hour"] < 6)).astype(int)
        # Drop is_weekend (weak) and keep only is_night + hour + day_of_week
        df = df.drop(columns=["unix_time"])

    # Drop high-PSI raw time column
    if "TransactionDT" in df.columns:
        df = df.drop(columns=["TransactionDT"])

    # --- Geo-distance: haversine between cardholder home and merchant ---
    if all(c in df.columns for c in ["lat", "long", "merch_lat", "merch_long"]):
        def haversine(row):
            lat1, lon1 = np.radians(row["lat"]), np.radians(row["long"])
            lat2, lon2 = np.radians(row["merch_lat"]), np.radians(row["merch_long"])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            return 6371.0 * 2 * np.arcsin(np.sqrt(a))
        df["home_merch_dist"] = df.apply(haversine, axis=1)

    # --- Amount features (keep only log_amt; drop is_round_amt as it's weak) ---
    if "TransactionAmt" in df.columns:
        df["log_amt"] = np.log1p(df["TransactionAmt"])

    # --- Behavioral profiling: per-card aggregations ---
    if "card_id" in df.columns and "TransactionAmt" in df.columns:
        global_mean = state.get("global_amt_mean", 0.0)
        global_std = state.get("global_amt_std", 1.0)
        if global_std == 0:
            global_std = 1.0

        card_mean_map = state.get("card_mean", {})
        card_std_map = state.get("card_std", {})
        card_max_map = state.get("card_max", {})
        card_median_map = state.get("card_median", {})
        card_count_map = state.get("card_count", {})

        cc_keys = df["card_id"].astype(str)
        df["card_mean_amt"] = cc_keys.apply(lambda x: card_mean_map.get(x, global_mean))
        df["card_std_amt"] = cc_keys.apply(lambda x: card_std_map.get(x, global_std))
        df["card_max_amt"] = cc_keys.apply(lambda x: card_max_map.get(x, global_mean))
        df["card_median_amt"] = cc_keys.apply(lambda x: card_median_map.get(x, global_mean))
        df["card_tx_count"] = cc_keys.apply(lambda x: card_count_map.get(x, 0.0))

        # Amount z-score vs own card history
        card_std_safe = df["card_std_amt"].replace(0, global_std)
        df["amt_zscore_card"] = (df["TransactionAmt"] - df["card_mean_amt"]) / card_std_safe
        # Drop weak/redundant behavioral features
        df = df.drop(columns=["card_max_amt", "card_std_amt", "card_median_amt", "card_tx_count"], errors="ignore")
        # Drop amt_vs_card_median (less informative than zscore)
        # Don't compute it

    # Number of unique merchants per card
    if "card_id" in df.columns:
        cc_keys2 = df["card_id"].astype(str)
        card_merch_map = state.get("card_n_merchants", {})
        df["card_n_merchants"] = cc_keys2.apply(lambda x: card_merch_map.get(x, 1))
        card_night_map = state.get("card_night_ratio", {})
        df["card_night_ratio"] = cc_keys2.apply(lambda x: card_night_map.get(x, 0.0))
        card_ncat_map = state.get("card_n_categories", {})
        df["card_n_categories"] = cc_keys2.apply(lambda x: card_ncat_map.get(x, 1))

    # Remove card_id and high-cardinality noisy cols after features built
    drop_noise = ["card_id", "city"]
    df = df.drop(columns=[c for c in drop_noise if c in df.columns])

    # --- Amount vs category median ---
    if "category" in df.columns and "TransactionAmt" in df.columns:
        global_amt_mean = state.get("global_amt_mean", 1.0)
        cat_median_map = state.get("cat_median_amt", {})
        cat_median_vals = df["category"].apply(lambda x: cat_median_map.get(str(x), global_amt_mean))
        df["amt_vs_cat_median"] = df["TransactionAmt"] / cat_median_vals.replace(0, global_amt_mean)

    # --- Target encoding: category ---
    if "category" in df.columns:
        te_map = state.get("category_te", {})
        global_mean = state.get("global_mean", 0.0)
        df["category_te"] = df["category"].apply(lambda x: te_map.get(str(x), global_mean))
        df = df.drop(columns=["category"])

    # --- Target encoding: merchant + merchant volume ---
    if "merchant" in df.columns:
        te_map = state.get("merchant_te", {})
        global_mean = state.get("global_mean", 0.0)
        merch_vol_map = state.get("merchant_volume", {})
        df["merchant_te"] = df["merchant"].apply(lambda x: te_map.get(str(x), global_mean))
        df["merchant_volume"] = df["merchant"].apply(lambda x: float(merch_vol_map.get(str(x), 0)))
        df = df.drop(columns=["merchant"])

    # Frequency encoding for remaining categoricals
    for col in state.get("cat_cols", []):
        if col in df.columns:
            freq_map = state.get(f"{col}_freq", {})
            df[f"{col}_freq_enc"] = df[col].apply(lambda x: freq_map.get(str(x), 0.0))
            df = df.drop(columns=[col])

    # --- Interaction features (keep only strongest, drop weak zscore_x_dist) ---
    if "is_night" in df.columns and "home_merch_dist" in df.columns:
        df["night_x_dist"] = df["is_night"] * df["home_merch_dist"]
    if "category_te" in df.columns and "amt_zscore_card" in df.columns:
        df["cate_te_x_zscore"] = df["category_te"] * df["amt_zscore_card"]
    # Drop log_home_merch_dist (correlated with home_merch_dist)

    return df.fillna(-1)
