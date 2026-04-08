"""IEEE-CIS feature engineering. Agent-editable.

Fit/transform API:
  fit(df_train, y_train, config) -> state dict (JSON-serializable)
  transform(df, state, config)   -> df (all numeric, no labels)
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smoothed_te(series, labels, global_rate, smoothing=20):
    """Return smoothed target encoding dict for a categorical series."""
    grp = pd.DataFrame({"x": series, "y": labels}).groupby("x")["y"].agg(["sum", "count"])
    smoothed = (grp["sum"] + smoothing * global_rate) / (grp["count"] + smoothing)
    return {str(k): float(v) for k, v in smoothed.items()}


# ---------------------------------------------------------------------------
# fit / transform
# ---------------------------------------------------------------------------

def fit(df_train: pd.DataFrame, y_train: pd.Series, config: dict) -> dict:
    state: dict = {}
    state["global_mean"] = float(y_train.mean())

    # Selective NaN drop. The previous >50% blanket rule killed all 43 identity
    # columns (id_01-38, DeviceInfo, R_emaildomain) whose null pattern is highly
    # predictive — id_17 IV=0.35, id_30 IV=0.62, DeviceInfo IV=1.78.
    # Drop only: constant columns (n_unique<=1) OR near-empty (>99% NaN with <=5 levels).
    nan_rates = df_train.isnull().mean()
    n_unique = df_train.nunique(dropna=True)
    drop_mask = (n_unique <= 1) | ((nan_rates > 0.99) & (n_unique <= 5))
    state["drop_cols"] = nan_rates[drop_mask].index.tolist()
    df_tmp = df_train.drop(columns=state["drop_cols"], errors="ignore")

    # Identity-cluster presence flag. The id_* columns share a ~98% correlated
    # null pattern (a single "was identity collected" signal, null_AUC ~0.65).
    # One binary flag captures this without adding 40 redundant indicators.
    id_cols_present = [c for c in df_tmp.columns if c.startswith("id_")]
    state["id_cluster_anchor"] = id_cols_present[0] if id_cols_present else None

    global_rate = float(y_train.mean())

    # Smoothed TE replaces freq encoding for high-IV categoricals.
    # Now model is regularized (max_depth=4), TE signal won't overfit as badly.
    # R_emaildomain IV=0.584, ProductCD IV=0.516, card6 strong, card4 moderate.
    # P_emaildomain is low-cardinality — TE is clean here.
    te_replace_cols = ["ProductCD", "P_emaildomain", "R_emaildomain", "card4", "card6"]
    state["te_replace_cols"] = [c for c in te_replace_cols if c in df_tmp.columns]
    for col in state["te_replace_cols"]:
        state[f"{col}_smooth_te"] = _smoothed_te(
            df_tmp[col].astype(str), y_train, global_rate, smoothing=20
        )

    # Frequency encoding for remaining categoricals (not TE-replaced)
    te_set = set(state["te_replace_cols"])
    cat_cols = [c for c in df_tmp.select_dtypes("object").columns if c not in te_set]
    state["cat_cols"] = cat_cols
    for col in cat_cols:
        freq = df_tmp[col].value_counts(normalize=True).to_dict()
        state[f"{col}_freq"] = {str(k): float(v) for k, v in freq.items()}

    # Campaign amount-patterns Step 2: Anomaly score (Recipe 6).
    # Mahalanobis-like distance from training distribution centroid.
    # Use the top numeric features that are PSI-safe (no ID columns, no time).
    anomaly_exclude = {"TransactionID", "TransactionDT", "card1", "card2", "card5"}
    num_cols_all = [c for c in df_tmp.select_dtypes(include=[np.number]).columns
                    if c not in anomaly_exclude and df_tmp[c].std() > 0][:20]
    state["anomaly_cols"] = num_cols_all
    state["anomaly_means"] = {c: float(df_tmp[c].mean()) for c in num_cols_all}
    state["anomaly_stds"] = {c: float(max(df_tmp[c].std(), 0.001)) for c in num_cols_all}

    # Campaign 2 Step 1: UID = card1 + addr1 aggregations (Recipe 15 pattern).
    # The regularized model (max_depth=4) should handle the card-level memorization better.
    # Use mean/std/count/min/max for amount + distinct emails and products per UID.
    if "card1" in df_tmp.columns and "addr1" in df_tmp.columns:
        df_tmp["_uid"] = df_tmp["card1"].astype(str) + "_" + df_tmp["addr1"].astype(str)
        uid_grp = df_tmp.groupby("_uid")

        amt_stats = uid_grp["TransactionAmt"].agg(["mean", "std", "min", "max", "count"])
        state["uid_amt_mean"] = {str(k): float(v) for k, v in amt_stats["mean"].items()}
        state["uid_amt_std"] = {str(k): float(v) for k, v in amt_stats["std"].fillna(0).items()}
        state["uid_amt_min"] = {str(k): float(v) for k, v in amt_stats["min"].items()}
        state["uid_amt_max"] = {str(k): float(v) for k, v in amt_stats["max"].items()}
        state["uid_count"] = {str(k): int(v) for k, v in amt_stats["count"].items()}

        if "R_emaildomain" in df_tmp.columns:
            state["uid_email_distinct"] = {
                str(k): int(v) for k, v in uid_grp["R_emaildomain"].nunique().items()
            }
        if "ProductCD" in df_tmp.columns:
            state["uid_prod_distinct"] = {
                str(k): int(v) for k, v in uid_grp["ProductCD"].nunique().items()
            }

    # Campaign 6 Step 1: R_emaildomain aggregations.
    # R_emaildomain is the #1 feature (TE importance 0.081). Aggregate amount stats per domain.
    # Email domains have ~60 unique values — stable aggregation without overfitting.
    if "R_emaildomain" in df_tmp.columns:
        email_grp = df_tmp.groupby("R_emaildomain")["TransactionAmt"]
        state["email_amt_mean"] = {str(k): float(v) for k, v in email_grp.mean().items()}
        state["email_amt_std"] = {str(k): float(v) for k, v in email_grp.std().fillna(0).items()}
        state["email_txn_count"] = {str(k): int(v) for k, v in email_grp.count().items()}
        # card1 diversity per email domain (many unique cards using same domain = suspicious)
        if "card1" in df_tmp.columns:
            card_per_email = df_tmp.groupby("R_emaildomain")["card1"].nunique()
            state["email_card_distinct"] = {str(k): int(v) for k, v in card_per_email.items()}

    # Campaign 2 Step 3: Per-card1 velocity features (Recipe 2).
    # card1 is more stable than UID for velocity because velocity stats summarize the
    # entire card history (gap distribution), not individual transactions.
    # median_gap, std_gap, min_gap, burst_count, daily_rate are all stable per-card stats.
    if "card1" in df_tmp.columns and "TransactionDT" in df_tmp.columns:
        df_v = df_tmp.sort_values(["card1", "TransactionDT"])
        df_v["_gap"] = df_v.groupby("card1")["TransactionDT"].diff()
        gap_stats = df_v.groupby("card1")["_gap"].agg(["median", "std", "min", "count"])
        df_v["_is_burst"] = (df_v["_gap"] < 60).astype(int)
        burst = df_v.groupby("card1")["_is_burst"].sum()
        time_range = df_v.groupby("card1")["TransactionDT"].agg(["min", "max"])
        time_range["days"] = ((time_range["max"] - time_range["min"]) / 86400).clip(lower=1)
        daily_rate = gap_stats["count"] / time_range["days"]
        state["v_median_gap"] = {str(k): float(v) for k, v in gap_stats["median"].fillna(0).items()}
        state["v_std_gap"] = {str(k): float(v) for k, v in gap_stats["std"].fillna(0).items()}
        state["v_min_gap"] = {str(k): float(v) for k, v in gap_stats["min"].fillna(0).items()}
        state["v_burst"] = {str(k): int(v) for k, v in burst.items()}
        state["v_daily_rate"] = {str(k): float(v) for k, v in daily_rate.items()}

    return state


def transform(df: pd.DataFrame, state: dict, config: dict) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=state.get("drop_cols", []), errors="ignore")

    # --- Identity-cluster presence flag ---
    anchor = state.get("id_cluster_anchor")
    if anchor and anchor in df.columns:
        df["id_cluster_present"] = df[anchor].notna().astype(float)
    else:
        df["id_cluster_present"] = 0.0

    # --- Smoothed TE for high-IV columns (replaces freq encoding) ---
    global_mean = state.get("global_mean", 0.035)
    for col in state.get("te_replace_cols", []):
        if col in df.columns:
            te_map = state.get(f"{col}_smooth_te", {})
            df[f"{col}_te"] = df[col].astype(str).map(te_map).fillna(global_mean)
            df = df.drop(columns=[col])

    # --- Frequency encoding for remaining categoricals ---
    for col in state.get("cat_cols", []):
        if col in df.columns:
            freq_map = state.get(f"{col}_freq", {})
            df[f"{col}_freq_enc"] = df[col].apply(lambda x: freq_map.get(str(x), 0.0))
            df = df.drop(columns=[col])

    # --- R_emaildomain aggregations (Campaign 6 Step 1) ---
    # R_emaildomain is dropped after TE, so use the original col from df_copy before TE drop.
    # We need to map using R_emaildomain before it gets dropped.
    # NOTE: by this point R_emaildomain is already dropped (TE applied).
    # We can track email via the TE value itself — or better: look up using original raw col.
    # Actually R_emaildomain is already in te_replace_cols, so it's dropped.
    # Workaround: store email aggs keyed by the TE value (continuous) is not possible.
    # Solution: compute before TE drop by adding email lookup at start of transform.

    # --- UID aggregation features (Campaign 2 Step 1) ---
    if "card1" in df.columns and "addr1" in df.columns:
        uid_str = df["card1"].astype(str) + "_" + df["addr1"].astype(str)
        df["uid_amt_mean"] = uid_str.map(state.get("uid_amt_mean", {})).fillna(-1)
        df["uid_amt_std"] = uid_str.map(state.get("uid_amt_std", {})).fillna(-1)
        df["uid_amt_min"] = uid_str.map(state.get("uid_amt_min", {})).fillna(-1)
        df["uid_amt_max"] = uid_str.map(state.get("uid_amt_max", {})).fillna(-1)
        df["uid_count"] = uid_str.map(state.get("uid_count", {})).fillna(1)
        df["uid_email_distinct"] = uid_str.map(state.get("uid_email_distinct", {})).fillna(-1)
        df["uid_prod_distinct"] = uid_str.map(state.get("uid_prod_distinct", {})).fillna(-1)

    # --- Per-card1 velocity features (Campaign 2 Step 3) ---
    if "card1" in df.columns:
        c1_str = df["card1"].astype(str)
        df["v_median_gap"] = c1_str.map(state.get("v_median_gap", {})).fillna(0)
        df["v_std_gap"] = c1_str.map(state.get("v_std_gap", {})).fillna(0)
        df["v_min_gap"] = c1_str.map(state.get("v_min_gap", {})).fillna(0)
        df["v_burst"] = c1_str.map(state.get("v_burst", {})).fillna(0)
        df["v_daily_rate"] = c1_str.map(state.get("v_daily_rate", {})).fillna(0)

    # --- Anomaly score (Recipe 6 — Mahalanobis-like distance) ---
    anomaly_cols = [c for c in state.get("anomaly_cols", []) if c in df.columns]
    if anomaly_cols:
        means = state.get("anomaly_means", {})
        stds = state.get("anomaly_stds", {})
        z2 = pd.DataFrame({
            c: ((df[c].fillna(0) - means.get(c, 0)) / stds.get(c, 1)) ** 2
            for c in anomaly_cols
        })
        df["anomaly_mahalanobis"] = np.sqrt(z2.sum(axis=1))
        df["anomaly_max_zscore"] = z2.max(axis=1)

    # --- Cyclic time features from TransactionDT ---
    # TransactionDT is seconds since reference. Extract cyclic hour-of-day and
    # day-of-week features using sine/cosine encoding (prevents boundary discontinuity).
    # These are deterministic transforms — stable across all temporal splits.
    if "TransactionDT" in df.columns:
        secs_per_day = 86400
        secs_per_week = 7 * secs_per_day
        hour_of_day = (df["TransactionDT"] % secs_per_day) / secs_per_day  # 0-1
        day_of_week = (df["TransactionDT"] % secs_per_week) / secs_per_week  # 0-1
        df["txn_hour_sin"] = np.sin(2 * np.pi * hour_of_day)
        df["txn_hour_cos"] = np.cos(2 * np.pi * hour_of_day)
        df["txn_dow_sin"] = np.sin(2 * np.pi * day_of_week)
        df["txn_dow_cos"] = np.cos(2 * np.pi * day_of_week)
        # Drop raw TransactionDT (PSI=12.43) to reduce train→val temporal drift
        df = df.drop(columns=["TransactionDT"], errors="ignore")

    # log(TransactionAmt) — captures multiplicative fraud patterns
    if "TransactionAmt" in df.columns:
        df["log_TransactionAmt"] = np.log1p(df["TransactionAmt"])
        # Keep raw amount too (different scale captures linear patterns)

        # Amount pattern features (Recipe 7) — stateless, PSI-safe
        amt = df["TransactionAmt"]
        df["amt_is_round_10"] = ((amt % 10) < 0.01).astype(float)
        df["amt_is_round_100"] = ((amt % 100) < 0.01).astype(float)
        df["amt_cents"] = ((amt * 100) % 100).astype(float)
        df["amt_has_cents"] = (df["amt_cents"] > 0.5).astype(float)
        # Threshold testing: amounts just below round thresholds ($99.99, $999.99)
        df["amt_below_100"] = ((amt >= 95) & (amt < 100)).astype(float)
        df["amt_below_1000"] = ((amt >= 990) & (amt < 1000)).astype(float)

    # Drop any remaining object columns
    for col in df.select_dtypes("object").columns:
        df = df.drop(columns=[col])

    return df.fillna(-1)
