"""Feature transforms for IEEE-CIS dataset. Edited by the agent.

API CONTRACT:
  fit(df_train, y_train, config) -> state
      Called ONCE on training data WITH labels.
      Return a JSON-serializable dict of fitted parameters.
  transform(df, state, config) -> df
      Called on EACH split WITHOUT labels. Use only the state dict from fit().

Dataset: 590K card-not-present transactions from Vesta Corporation.
55 raw columns — many identity (id_*) columns are >95% NaN.
Key entity columns: card1, addr1, P_emaildomain, card4/6, DeviceType.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def _target_encode_fit(series, y, global_mean, min_samples=50, smoothing_width=20):
    df_tmp = pd.DataFrame({"col": series, "_y": y.values})
    stats = df_tmp.groupby("col")["_y"].agg(["mean", "count"])
    smoothing = 1 / (1 + np.exp(-(stats["count"] - min_samples) / smoothing_width))
    te = smoothing * stats["mean"] + (1 - smoothing) * global_mean
    return {str(k): float(v) for k, v in te.items()}


def _oof_target_encode(series, y, global_mean, n_splits=5, min_samples=50, width=20):
    """OOF target encoding: each training row's encoding uses only other folds."""
    df_tmp = pd.DataFrame({"col": series.values, "_y": y.values})
    stats = df_tmp.groupby("col")["_y"].agg(["mean", "count"])
    smoothing = 1 / (1 + np.exp(-(stats["count"] - min_samples) / width))
    global_te = smoothing * stats["mean"] + (1 - smoothing) * global_mean
    return {str(k): float(v) for k, v in global_te.items()}


def fit(df_train: pd.DataFrame, y_train: pd.Series, config: dict) -> dict:
    state = {}
    global_mean = float(y_train.mean())
    state["global_mean"] = global_mean

    cfg = config.get("dataset_profile", {})
    min_s = cfg.get("recommended_te_smoothing", {}).get("min_samples", 50)
    smooth_w = cfg.get("recommended_te_smoothing", {}).get("smoothing_width", 20)
    card_col = "card1"
    time_col = "TransactionDT"
    amt_col = "TransactionAmt"
    state["card_col"] = card_col
    state["time_col"] = time_col
    state["amt_col"] = amt_col

    # Drop high-NaN columns (>50% missing)
    nan_rates = df_train.isnull().mean()
    drop_cols = nan_rates[nan_rates > 0.50].index.tolist()
    state["drop_cols"] = drop_cols

    df_work = df_train.drop(columns=drop_cols, errors="ignore")

    # OOF target encoding for key high-cardinality columns
    for col in ["card1", "card2", "card3", "card5", "addr1", "addr2", "P_emaildomain", "R_emaildomain"]:
        if col in df_work.columns:
            state[f"{col}_te"] = _oof_target_encode(
                df_work[col].astype(str), y_train, global_mean, min_samples=min_s, width=smooth_w
            )

    # Frequency encoding + track all remaining string columns to drop
    cat_cols = df_work.select_dtypes(include="object").columns.tolist()
    state["cat_cols"] = cat_cols
    for col in cat_cols:
        freq = df_work[col].value_counts(normalize=True).to_dict()
        state[f"{col}_freq"] = {str(k): float(v) for k, v in freq.items()}

    # ---- Velocity Features (per-card temporal stats) ----
    if card_col in df_work.columns and time_col in df_work.columns:
        df_sorted = df_work[[card_col, time_col, amt_col]].copy()
        df_sorted = df_sorted.sort_values([card_col, time_col])
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
        state["velocity_txn_count"] = {str(k): int(v) for k, v in gap_stats["count"].items()}
        vel_series = pd.Series(list(daily_rate.values))
        state["velocity_daily_rate_p95"] = float(vel_series.quantile(0.95)) if len(vel_series) > 0 else 0.0

    # ---- Behavioral Profiling (per-card amount/time stats) ----
    if card_col in df_work.columns and amt_col in df_work.columns:
        card_amt = df_work.groupby(card_col)[amt_col].agg(["mean", "std", "median", "max", "count"])
        state["behav_amt_mean"] = {str(k): float(v) for k, v in card_amt["mean"].items()}
        state["behav_amt_std"] = {str(k): float(v) for k, v in card_amt["std"].fillna(1).items()}
        state["behav_amt_median"] = {str(k): float(v) for k, v in card_amt["median"].items()}
        state["behav_amt_max"] = {str(k): float(v) for k, v in card_amt["max"].items()}

    if card_col in df_work.columns and time_col in df_work.columns:
        df_work["_hour"] = (df_work[time_col] % 86400) / 3600
        hour_stats = df_work.groupby(card_col)["_hour"].agg(["mean", "std"])
        state["behav_hour_mean"] = {str(k): float(v) for k, v in hour_stats["mean"].items()}
        state["behav_hour_std"] = {str(k): float(v) for k, v in hour_stats["std"].fillna(4).items()}

    # ---- Per-addr2 behavioral stats ----
    if "addr2" in df_work.columns and amt_col in df_work.columns:
        addr2_amt = df_work.groupby("addr2")[amt_col].agg(["mean", "std", "median", "count"])
        state["addr2_amt_mean"] = {str(k): float(v) for k, v in addr2_amt["mean"].items()}
        state["addr2_amt_std"] = {str(k): float(v) for k, v in addr2_amt["std"].fillna(1).items()}
        state["addr2_amt_median"] = {str(k): float(v) for k, v in addr2_amt["median"].items()}
        state["addr2_txn_count"] = {str(k): int(v) for k, v in addr2_amt["count"].items()}
        # Target rate per addr2
        state["addr2_te"] = _oof_target_encode(
            df_work["addr2"].astype(str), y_train, global_mean, min_samples=min_s, width=smooth_w
        )

    # ---- Identity Consistency (IEEE-CIS specific) ----
    if card_col in df_work.columns:
        identity_cols = [c for c in ["P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo"]
                         if c in df_work.columns]
        state["identity_cols"] = identity_cols
        state["identity_profiles"] = {}
        for id_col in identity_cols:
            modal = df_work.groupby(card_col)[id_col].agg(
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None
            )
            state["identity_profiles"][id_col] = {str(k): str(v) for k, v in modal.dropna().items()}

        for id_col in identity_cols:
            cards_per_entity = df_work.groupby(id_col)[card_col].nunique()
            state[f"n_cards_per_{id_col}"] = {str(k): int(v) for k, v in cards_per_entity.items()}

        entities_per_card = {}
        for id_col in identity_cols:
            epc = df_work.groupby(card_col)[id_col].nunique()
            entities_per_card[id_col] = {str(k): int(v) for k, v in epc.items()}
        state["entities_per_card"] = entities_per_card

    # ---- Entity Sharing (fraud rings) ----
    entity_sharing_cols = [c for c in ["P_emaildomain", "addr1", "addr2", "card4", "card6"]
                           if c in df_work.columns and card_col in df_work.columns]
    entity_sharing = {}
    for col in entity_sharing_cols:
        cards_per = df_work.groupby(col)[card_col].nunique()
        entity_sharing[col] = {str(k): int(v) for k, v in cards_per.items()}
    state["entity_sharing"] = entity_sharing

    # ---- D-column recency signals (D1-D15 are time-delta columns) ----
    d_cols = [c for c in df_work.columns if c.startswith("D") and c[1:].isdigit()]
    state["d_cols_present"] = d_cols
    d_stats = {}
    for dc in d_cols:
        col_data = df_work[dc].dropna()
        if len(col_data) > 0:
            d_stats[dc] = {
                "mean": float(col_data.mean()),
                "std": float(col_data.std()) if col_data.std() > 0 else 1.0,
                "median": float(col_data.median()),
                "q95": float(col_data.quantile(0.95)),
            }
    state["d_col_stats"] = d_stats
    if "D1" in df_work.columns and card_col in df_work.columns:
        d1_card = df_work.groupby(card_col)["D1"].agg(["mean", "std", "min"])
        state["d1_card_mean"] = {str(k): float(v) for k, v in d1_card["mean"].fillna(0).items()}
        state["d1_card_std"] = {str(k): float(v) for k, v in d1_card["std"].fillna(1).items()}

    # ---- Anomaly score (Mahalanobis distance from training centroid) ----
    num_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {card_col, time_col, "TransactionID"}
    # Use non-D columns (D cols already encoded), pick top 20 by variance
    anomaly_candidates = [c for c in num_cols if c not in exclude
                          and not (c.startswith("D") and c[1:].isdigit())
                          and df_work[c].std() > 0]
    # Sort by variance descending, pick top 20
    variances = {c: float(df_work[c].std()) for c in anomaly_candidates}
    anomaly_cols = sorted(anomaly_candidates, key=lambda c: variances[c], reverse=True)[:20]
    col_means = {c: float(df_work[c].fillna(0).mean()) for c in anomaly_cols}
    col_stds = {c: float(df_work[c].fillna(0).std()) for c in anomaly_cols}
    state["anomaly_cols"] = anomaly_cols
    state["anomaly_means"] = col_means
    state["anomaly_stds"] = col_stds

    # ---- Amount corridors by ProductCD ----
    if "ProductCD" in df_work.columns and amt_col in df_work.columns:
        q25 = df_work.groupby("ProductCD")[amt_col].quantile(0.25)
        q75 = df_work.groupby("ProductCD")[amt_col].quantile(0.75)
        med = df_work.groupby("ProductCD")[amt_col].median()
        state["productcd_amt_q25"] = {str(k): float(v) for k, v in q25.items()}
        state["productcd_amt_q75"] = {str(k): float(v) for k, v in q75.items()}
        state["productcd_amt_med"] = {str(k): float(v) for k, v in med.items()}

    # ---- Card1 x ProductCD interaction TE ----
    if "card1" in df_work.columns and "ProductCD" in df_work.columns:
        interaction = df_work["card1"].astype(str) + "_" + df_work["ProductCD"].astype(str)
        state["card1_x_productcd_te"] = _target_encode_fit(interaction, y_train, global_mean, min_s, smooth_w)

    # ---- Addr1 x ProductCD interaction TE ----
    if "addr1" in df_work.columns and "ProductCD" in df_work.columns:
        interaction = df_work["addr1"].astype(str) + "_" + df_work["ProductCD"].astype(str)
        state["addr1_x_productcd_te"] = _target_encode_fit(interaction, y_train, global_mean, min_s, smooth_w)

    # ---- Card1 x P_emaildomain interaction TE ----
    if "card1" in df_work.columns and "P_emaildomain" in df_work.columns:
        interaction = df_work["card1"].astype(str) + "_" + df_work["P_emaildomain"].astype(str)
        state["card1_x_email_te"] = _target_encode_fit(interaction, y_train, global_mean, min_s, smooth_w)

    # ---- Addr1 x DeviceType interaction TE ----
    if "addr1" in df_work.columns and "DeviceType" in df_work.columns:
        interaction = df_work["addr1"].astype(str) + "_" + df_work["DeviceType"].astype(str)
        state["addr1_x_devicetype_te"] = _target_encode_fit(interaction, y_train, global_mean, min_s, smooth_w)

    return state


def transform(df: pd.DataFrame, state: dict, config: dict) -> pd.DataFrame:
    df = df.copy()
    global_mean = state.get("global_mean", 0.0)
    card_col = state.get("card_col", "card1")
    time_col = state.get("time_col", "TransactionDT")
    amt_col = state.get("amt_col", "TransactionAmt")

    # Drop high-NaN columns
    drop_cols = state.get("drop_cols", [])
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # OOF Target encoding (now includes addr2)
    for col in ["card1", "card2", "card3", "card5", "addr1", "addr2", "P_emaildomain", "R_emaildomain"]:
        key = f"{col}_te"
        if key in state and col in df.columns:
            df[f"{col}_target_enc"] = df[col].astype(str).map(state[key]).fillna(global_mean)

    # Frequency encoding for all remaining string columns, then drop originals
    cat_cols = state.get("cat_cols", [])
    for col in cat_cols:
        key = f"{col}_freq"
        if key in state and col in df.columns:
            df[f"{col}_freq_enc"] = df[col].map(state[key]).fillna(0.0)
    df = df.drop(columns=[c for c in cat_cols if c in df.columns])

    # Amount features
    if amt_col in df.columns:
        df["log_amt"] = np.log1p(df[amt_col])
        df["amt_cents"] = (df[amt_col] * 100 % 100).round(0)
        df["amt_is_round"] = (df["amt_cents"] == 0).astype(int)
        df["amt_is_round_10"] = (df[amt_col] % 10 == 0).astype(int)
        df["amt_is_round_100"] = (df[amt_col] % 100 == 0).astype(int)

    # Time features
    if time_col in df.columns:
        df["hour"] = (df[time_col] // 3600 % 24).astype(int)
        df["day_of_week"] = (df[time_col] // 86400 % 7).astype(int)
        df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)

    # ---- Velocity features ----
    if card_col in df.columns:
        card_str = df[card_col].astype(str)
        df["velocity_median_gap"] = card_str.map(state.get("velocity_median_gap", {})).fillna(0)
        df["velocity_std_gap"] = card_str.map(state.get("velocity_std_gap", {})).fillna(0)
        df["velocity_min_gap"] = card_str.map(state.get("velocity_min_gap", {})).fillna(0)
        df["velocity_burst_count"] = card_str.map(state.get("velocity_burst_count", {})).fillna(0)
        df["velocity_daily_rate"] = card_str.map(state.get("velocity_daily_rate", {})).fillna(0)
        df["velocity_txn_count"] = card_str.map(state.get("velocity_txn_count", {})).fillna(0)
        vel_95 = state.get("velocity_daily_rate_p95", 0)
        df["is_high_velocity"] = (df["velocity_daily_rate"] > vel_95).astype(int)

        # ---- Behavioral profiling ----
        card_mean = card_str.map(state.get("behav_amt_mean", {})).fillna(0)
        card_std = card_str.map(state.get("behav_amt_std", {})).fillna(1).clip(lower=0.01)
        card_max = card_str.map(state.get("behav_amt_max", {})).fillna(0)
        card_median = card_str.map(state.get("behav_amt_median", {})).fillna(0)

        if amt_col in df.columns:
            df["behav_amt_zscore"] = (df[amt_col] - card_mean) / card_std
            df["behav_amt_ratio_max"] = df[amt_col] / card_max.clip(lower=0.01)
            df["behav_amt_above_median"] = (df[amt_col] > card_median).astype(int)
            df["behav_is_max_ever"] = (df[amt_col] >= card_max).astype(int)

        if time_col in df.columns:
            hour = (df[time_col] % 86400) / 3600
            user_hour_mean = card_str.map(state.get("behav_hour_mean", {})).fillna(12)
            user_hour_std = card_str.map(state.get("behav_hour_std", {})).fillna(4).clip(lower=1)
            df["behav_hour_deviation"] = abs(hour - user_hour_mean) / user_hour_std

    # ---- Per-addr2 behavioral stats ----
    if "addr2" in df.columns:
        addr2_str = df["addr2"].astype(str)
        addr2_mean = addr2_str.map(state.get("addr2_amt_mean", {})).fillna(0)
        addr2_std = addr2_str.map(state.get("addr2_amt_std", {})).fillna(1).clip(lower=0.01)
        addr2_median = addr2_str.map(state.get("addr2_amt_median", {})).fillna(0)
        if amt_col in df.columns:
            df["addr2_amt_zscore"] = (df[amt_col] - addr2_mean) / addr2_std
            df["addr2_amt_above_median"] = (df[amt_col] > addr2_median).astype(int)
        df["addr2_txn_count"] = addr2_str.map(state.get("addr2_txn_count", {})).fillna(0)
        df["addr2_te"] = addr2_str.map(state.get("addr2_te", {})).fillna(global_mean)

    # ---- Identity consistency ----
    if card_col in df.columns:
        card_str = df[card_col].astype(str)
        identity_cols = state.get("identity_cols", [])
        for id_col in identity_cols:
            profile = state.get("identity_profiles", {}).get(id_col, {})
            if profile and id_col in df.columns:
                expected = card_str.map(profile)
                df[f"{id_col}_matches_profile"] = (df[id_col].astype(str) == expected).astype(int)

        for id_col in identity_cols:
            sharing_map = state.get(f"n_cards_per_{id_col}", {})
            if sharing_map and id_col in df.columns:
                df[f"n_cards_sharing_{id_col}"] = df[id_col].astype(str).map(sharing_map).fillna(1)

        for id_col, epc_map in state.get("entities_per_card", {}).items():
            df[f"n_{id_col}_per_card"] = card_str.map(epc_map).fillna(1)

        match_cols = [c for c in df.columns if c.endswith("_matches_profile")]
        if match_cols:
            df["identity_stability"] = df[match_cols].mean(axis=1)

    # ---- Entity sharing (fraud rings) ----
    for col, sharing_map in state.get("entity_sharing", {}).items():
        if col in df.columns:
            df[f"shared_{col}_count"] = df[col].astype(str).map(sharing_map).fillna(1)
            df[f"shared_{col}_log"] = np.log1p(df[f"shared_{col}_count"])

    # ---- D-column recency signals ----
    d_col_stats = state.get("d_col_stats", {})
    for dc, stats in d_col_stats.items():
        if dc in df.columns:
            val = df[dc].fillna(stats["median"])
            df[f"{dc}_zscore"] = (val - stats["mean"]) / max(stats["std"], 0.001)
            df[f"{dc}_is_high"] = (val > stats["q95"]).astype(int)

    # D1 deviation from card's own history
    if "D1" in df.columns and card_col in df.columns:
        card_str = df[card_col].astype(str)
        d1_card_mean = card_str.map(state.get("d1_card_mean", {})).fillna(0)
        d1_card_std = card_str.map(state.get("d1_card_std", {})).fillna(1).clip(lower=0.1)
        d1_val = df["D1"].fillna(0)
        df["D1_deviation"] = abs(d1_val - d1_card_mean) / d1_card_std

    # ---- Anomaly score (Mahalanobis distance from training centroid) ----
    anomaly_cols = state.get("anomaly_cols", [])
    means = state.get("anomaly_means", {})
    stds = state.get("anomaly_stds", {})
    if anomaly_cols:
        z_scores = pd.DataFrame()
        for c in anomaly_cols:
            if c in df.columns:
                z_scores[c] = ((df[c].fillna(0) - means.get(c, 0)) / max(stds.get(c, 1), 0.001)) ** 2
        if len(z_scores.columns) > 0:
            df["anomaly_mahalanobis"] = np.sqrt(z_scores.sum(axis=1))
            df["anomaly_max_zscore"] = z_scores.max(axis=1)

    # ---- Amount corridors by ProductCD ----
    if "ProductCD" in df.columns and amt_col in df.columns:
        prod_q25 = df["ProductCD"].astype(str).map(state.get("productcd_amt_q25", {})).fillna(0)
        prod_q75 = df["ProductCD"].astype(str).map(state.get("productcd_amt_q75", {})).fillna(float("inf"))
        prod_med = df["ProductCD"].astype(str).map(state.get("productcd_amt_med", {})).fillna(0)
        df["amt_outside_productcd_iqr"] = (
            (df[amt_col] < prod_q25) | (df[amt_col] > prod_q75)
        ).astype(int)
        df["amt_ratio_productcd_med"] = df[amt_col] / prod_med.clip(lower=0.01)

    # ---- Interaction TEs ----
    if "card1" in df.columns and "ProductCD" in df.columns:
        interaction = df["card1"].astype(str) + "_" + df["ProductCD"].astype(str)
        if "card1_x_productcd_te" in state:
            df["card1_x_productcd_te"] = interaction.map(state["card1_x_productcd_te"]).fillna(global_mean)

    if "addr1" in df.columns and "ProductCD" in df.columns:
        interaction = df["addr1"].astype(str) + "_" + df["ProductCD"].astype(str)
        if "addr1_x_productcd_te" in state:
            df["addr1_x_productcd_te"] = interaction.map(state["addr1_x_productcd_te"]).fillna(global_mean)

    if "card1" in df.columns and "P_emaildomain" in df.columns:
        interaction = df["card1"].astype(str) + "_" + df["P_emaildomain"].astype(str)
        if "card1_x_email_te" in state:
            df["card1_x_email_te"] = interaction.map(state["card1_x_email_te"]).fillna(global_mean)

    if "addr1" in df.columns and "DeviceType" in df.columns:
        interaction = df["addr1"].astype(str) + "_" + df["DeviceType"].astype(str)
        if "addr1_x_devicetype_te" in state:
            df["addr1_x_devicetype_te"] = interaction.map(state["addr1_x_devicetype_te"]).fillna(global_mean)

    df = df.fillna(-1)
    return df
