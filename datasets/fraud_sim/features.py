"""Fraud-sim feature engineering. Agent-editable.

Fit/transform API:
  fit(df_train, y_train, config) -> state dict (JSON-serializable)
  transform(df, state, config)   -> df (all numeric, no labels)
"""
import math
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_WINDOWS_SEC = [600, 3600, 86400, 7 * 86400]  # 10min, 1h, 1d, 7d
_WIN_NAMES = ["10m", "1h", "1d", "7d"]


def _haversine_vec(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _window_velocity(df, entity_col, time_col, amt_col, windows, win_names, history=None):
    """O(N log N) rolling velocity with optional prepended training history."""
    orig_len = len(df)
    if history:
        tail_rows = []
        for ent, records in history.items():
            for t, a in records:
                tail_rows.append((ent, float(t), float(a)))
        if tail_rows:
            tail_ents, tail_times, tail_amts = zip(*tail_rows)
            tail_df = pd.DataFrame({
                entity_col: list(tail_ents),
                time_col: list(tail_times),
                amt_col: list(tail_amts),
                "_is_tail": True,
            })
            working_df = df[[entity_col, time_col, amt_col]].copy()
            working_df["_is_tail"] = False
            combined = pd.concat([tail_df, working_df], ignore_index=True)
        else:
            combined = df[[entity_col, time_col, amt_col]].copy()
            combined["_is_tail"] = False
    else:
        combined = df[[entity_col, time_col, amt_col]].copy()
        combined["_is_tail"] = False

    combined = combined.sort_values(time_col)
    times_all = combined[time_col].values.astype(np.float64)
    ents_all = combined[entity_col].astype(str).values
    amts_all = combined[amt_col].values.astype(np.float64)
    is_tail = combined["_is_tail"].values
    n_total = len(combined)

    result = {}
    for wname in win_names:
        result[f"{entity_col}_{wname}_count"] = np.zeros(n_total)
        result[f"{entity_col}_{wname}_sum"] = np.zeros(n_total)

    unique_ents = np.unique(ents_all)
    for ent in unique_ents:
        mask = ents_all == ent
        idxs = np.where(mask)[0]
        t = times_all[idxs]
        a = amts_all[idxs]
        n = len(t)
        cum_amt = np.concatenate([[0.0], np.cumsum(a)])
        for w, wname in zip(windows, win_names):
            start_pos = np.searchsorted(t, t - w, side="left")
            counts = np.arange(n) - start_pos
            sums = cum_amt[np.arange(n)] - cum_amt[start_pos]
            for k, global_idx in enumerate(idxs):
                result[f"{entity_col}_{wname}_count"][global_idx] = counts[k]
                result[f"{entity_col}_{wname}_sum"][global_idx] = sums[k]

    result_df = pd.DataFrame(result, index=np.arange(n_total))
    non_tail = ~is_tail
    result_trimmed = result_df[non_tail].copy()
    result_trimmed.index = df.index
    return result_trimmed


# ---------------------------------------------------------------------------
# fit / transform
# ---------------------------------------------------------------------------

def fit(df_train: pd.DataFrame, y_train: pd.Series, config: dict) -> dict:
    state: dict = {}
    state["global_mean"] = float(y_train.mean())

    nan_rates = df_train.isnull().mean()
    state["drop_cols"] = nan_rates[nan_rates > 0.50].index.tolist()
    df_tmp = df_train.drop(columns=state["drop_cols"], errors="ignore")

    cat_cols = df_tmp.select_dtypes("object").columns.tolist()
    state["cat_cols"] = cat_cols
    for col in cat_cols:
        freq = df_tmp[col].value_counts(normalize=True).to_dict()
        state[f"{col}_freq"] = {str(k): float(v) for k, v in freq.items()}

    cfg = config.get("dataset_profile", {})
    amt_col = cfg.get("amt_col", "TransactionAmt")
    cust_col = cfg.get("key_entity_col", "card_id")
    time_col = cfg.get("time_col", "TransactionDT")

    # Per-card behavioral stats
    cust_grp = df_tmp.groupby(cust_col)[amt_col]
    state["cust_mean"] = cust_grp.mean().to_dict()
    state["cust_std"] = cust_grp.std().fillna(1.0).to_dict()
    state["cust_p90"] = cust_grp.quantile(0.90).to_dict()
    state["global_mean_amt"] = float(df_tmp[amt_col].mean())
    state["global_std_amt"] = float(df_tmp[amt_col].std())

    # Per-card home location (for geo features, should be stable per card)
    home_loc = df_tmp.groupby(cust_col)[["lat", "long"]].first()
    state["cust_home_lat"] = home_loc["lat"].to_dict()
    state["cust_home_long"] = home_loc["long"].to_dict()

    # Smoothed category TE (category has ~14 unique values — stable)
    global_rate = float(y_train.mean())
    smoothing = 20
    category_col = "category"
    if category_col in df_tmp.columns:
        df_tmp_lab = df_tmp[[category_col]].copy()
        df_tmp_lab["_label"] = y_train.values
        grp = df_tmp_lab.groupby(category_col)["_label"].agg(["sum", "count"])
        smoothed = (grp["sum"] + smoothing * global_rate) / (grp["count"] + smoothing)
        state["category_te"] = {str(k): float(v) for k, v in smoothed.items()}

    # Per-card behavioral stats: distinct merchant/category counts
    state["cust_n_merchants"] = df_tmp.groupby(cust_col)["merchant"].nunique().to_dict()
    state["cust_n_categories"] = df_tmp.groupby(cust_col)["category"].nunique().to_dict()
    state["global_n_merchants"] = float(df_tmp["merchant"].nunique()) / len(df_tmp[cust_col].unique())

    # Remove merchant from cat_cols so we don't get redundant freq_enc for it
    # (merchant has 800 unique values; TE is better than freq enc)
    merchant_col = "merchant"
    if merchant_col in df_tmp.columns:
        df_tmp_lab2 = df_tmp[[merchant_col]].copy()
        df_tmp_lab2["_label"] = y_train.values
        grp_m = df_tmp_lab2.groupby(merchant_col)["_label"].agg(["sum", "count"])
        smoothed_m = (grp_m["sum"] + smoothing * global_rate) / (grp_m["count"] + smoothing)
        state["merchant_te"] = {str(k): float(v) for k, v in smoothed_m.items()}

    # Per-card typical hour distribution (for behavioral fingerprint)
    if "unix_time" in df_tmp.columns:
        df_tmp["_hour"] = pd.to_datetime(df_tmp["unix_time"], unit="s").dt.hour
        cust_hour_mean = df_tmp.groupby(cust_col)["_hour"].mean()
        state["cust_hour_mean"] = {str(k): float(v) for k, v in cust_hour_mean.items()}

    # Per-card typical haversine distance
    df_tmp["_geo_dist"] = _haversine_vec(
        df_tmp["lat"].values, df_tmp["long"].values,
        df_tmp["merch_lat"].values, df_tmp["merch_long"].values
    )
    cust_dist_mean = df_tmp.groupby(cust_col)["_geo_dist"].mean()
    cust_dist_std = df_tmp.groupby(cust_col)["_geo_dist"].std().fillna(1.0)
    cust_dist_p90 = df_tmp.groupby(cust_col)["_geo_dist"].quantile(0.90)
    state["cust_dist_mean"] = {str(k): float(v) for k, v in cust_dist_mean.items()}
    state["cust_dist_std"] = {str(k): float(v) for k, v in cust_dist_std.items()}
    state["cust_dist_p90"] = {str(k): float(v) for k, v in cust_dist_p90.items()}
    state["global_dist_mean"] = float(df_tmp["_geo_dist"].mean())
    state["global_dist_std"] = float(df_tmp["_geo_dist"].std())

    # Store last 7 days of training for velocity continuity
    max_window = max(_WINDOWS_SEC)
    max_train_t = float(df_tmp[time_col].max())
    tail_mask = df_tmp[time_col] >= (max_train_t - max_window)
    tail = df_tmp.loc[tail_mask, [cust_col, time_col, amt_col]]
    cust_tail: dict = {}
    for ent, grp in tail.groupby(cust_col):
        cust_tail[str(ent)] = [
            [float(t), float(a)] for t, a in zip(grp[time_col], grp[amt_col])
        ]
    state["cust_tail"] = cust_tail

    return state


def transform(df: pd.DataFrame, state: dict, config: dict) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=state.get("drop_cols", []), errors="ignore")

    cfg = config.get("dataset_profile", {})
    amt_col = cfg.get("amt_col", "TransactionAmt")
    cust_col = cfg.get("key_entity_col", "card_id")
    time_col = cfg.get("time_col", "TransactionDT")

    # --- TE for merchant and category (before categorical freq enc drop) ---
    global_fraud_rate = float(state.get("global_mean", 0.005))
    if "category" in df.columns:
        df["category_te"] = df["category"].astype(str).map(state.get("category_te", {})).fillna(global_fraud_rate)
    if "merchant" in df.columns:
        df["merchant_te"] = df["merchant"].astype(str).map(state.get("merchant_te", {})).fillna(global_fraud_rate)

    # Frequency encoding for other categoricals (skip merchant — TE replaces it)
    for col in state.get("cat_cols", []):
        if col in df.columns and col != "merchant":
            freq_map = state.get(f"{col}_freq", {})
            df[f"{col}_freq_enc"] = df[col].apply(lambda x: freq_map.get(str(x), 0.0))
            df = df.drop(columns=[col])
    # Drop merchant (already TE'd above)
    df = df.drop(columns=["merchant"], errors="ignore")

    # --- Haversine distance: home to merchant ---
    if "lat" in df.columns and "merch_lat" in df.columns:
        home_lat = df[cust_col].map(state.get("cust_home_lat", {})).fillna(df["lat"])
        home_long = df[cust_col].map(state.get("cust_home_long", {})).fillna(df["long"])
        df["haversine_km"] = _haversine_vec(
            home_lat.values, home_long.values,
            df["merch_lat"].values, df["merch_long"].values
        )
        df["log_haversine_km"] = np.log1p(df["haversine_km"])

    # --- Customer behavioral deviation ---
    df["cust_mean_amt"] = df[cust_col].map(state["cust_mean"]).fillna(state["global_mean_amt"])
    df["cust_std_amt"] = df[cust_col].map(state["cust_std"]).fillna(state["global_std_amt"])
    df["cust_p90_amt"] = df[cust_col].map(state["cust_p90"]).fillna(state["global_mean_amt"])
    df["amt_zscore_cust"] = (df[amt_col] - df["cust_mean_amt"]) / df["cust_std_amt"].clip(lower=0.01)
    df["amt_vs_p90"] = df[amt_col] / df["cust_p90_amt"].clip(lower=1.0)
    df = df.drop(columns=["cust_mean_amt", "cust_std_amt", "cust_p90_amt"], errors="ignore")

    # --- Time features from unix_time ---
    if "unix_time" in df.columns:
        dt = pd.to_datetime(df["unix_time"], unit="s")
        df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
        df["is_night"] = ((dt.dt.hour >= 22) | (dt.dt.hour <= 6)).astype(float)

    # --- Velocity stack with training history ---
    cust_history = {k: [tuple(v2) for v2 in v] for k, v in state.get("cust_tail", {}).items()}
    cust_vel = _window_velocity(df, cust_col, time_col, amt_col, _WINDOWS_SEC, _WIN_NAMES, cust_history)
    df = pd.concat([df, cust_vel], axis=1)

    # --- Per-card behavioral fingerprint: unusual hour + unusual distance ---
    if "unix_time" in df.columns:
        curr_hour = pd.to_datetime(df["unix_time"], unit="s").dt.hour.values
        cust_hour_mean_map = state.get("cust_hour_mean", {})
        df["cust_hour_mean"] = df[cust_col].map(cust_hour_mean_map).fillna(12.0)
        # Angular distance (hours are cyclic — use abs circular diff)
        hour_diff = np.abs(curr_hour - df["cust_hour_mean"].values)
        df["hour_deviation"] = np.minimum(hour_diff, 24 - hour_diff)
        df = df.drop(columns=["cust_hour_mean"], errors="ignore")

    # Per-card haversine deviation
    df["cust_dist_mean_map"] = df[cust_col].map(state.get("cust_dist_mean", {})).fillna(state.get("global_dist_mean", 50.0))
    df["cust_dist_std_map"] = df[cust_col].map(state.get("cust_dist_std", {})).fillna(state.get("global_dist_std", 50.0))
    df["cust_dist_p90_map"] = df[cust_col].map(state.get("cust_dist_p90", {})).fillna(state.get("global_dist_mean", 50.0))
    if "haversine_km" in df.columns:
        df["dist_zscore_cust"] = (df["haversine_km"] - df["cust_dist_mean_map"]) / df["cust_dist_std_map"].clip(lower=0.1)
        df["dist_vs_p90"] = df["haversine_km"] / df["cust_dist_p90_map"].clip(lower=0.1)
    df = df.drop(columns=["cust_dist_mean_map", "cust_dist_std_map", "cust_dist_p90_map"], errors="ignore")

    # Burst ratios
    df[f"{cust_col}_burst_10m_1h"] = (
        (df[f"{cust_col}_10m_count"] + 1) / (df[f"{cust_col}_1h_count"] + 1)
    )
    df[f"{cust_col}_burst_1h_1d"] = (
        (df[f"{cust_col}_1h_count"] + 1) / (df[f"{cust_col}_1d_count"] + 1)
    )

    # --- Key interactions ---
    if "haversine_km" in df.columns:
        # High amount far from home
        df["log_amt_x_log_dist"] = np.log1p(df[amt_col]) * df["log_haversine_km"]
        # Rapid velocity far from home (card-testing while traveling)
        df[f"vel_1h_x_dist"] = df[f"{cust_col}_1h_count"] * df["log_haversine_km"]

    # --- Per-card diversity features (fraud uses fewer merchants; new merchant = suspicious) ---
    df["cust_n_merchants_train"] = df[cust_col].map(state.get("cust_n_merchants", {})).fillna(state.get("global_n_merchants", 5.0))
    df["cust_n_categories_train"] = df[cust_col].map(state.get("cust_n_categories", {})).fillna(5.0)

    # --- Demographic features ---
    if "city_pop" in df.columns:
        df["log_city_pop"] = np.log1p(df["city_pop"])
        # Relative amount for city size (large txn in small city = more suspicious)
        df["amt_per_city_pop"] = df[amt_col] / df["city_pop"].clip(lower=1)

    # Drop remaining object columns
    for col in df.select_dtypes("object").columns:
        df = df.drop(columns=[col])

    return df.fillna(-1)
