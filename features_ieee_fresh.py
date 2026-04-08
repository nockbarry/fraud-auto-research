"""IEEE-CIS feature engineering — Track B fresh start. Agent-editable.

Same data as ieee-cis Track A but clean-slate FE so the agent builds independently.

Fit/transform API:
  fit(df_train, y_train, config) -> state dict (JSON-serializable)
  transform(df, state, config)   -> df (all numeric, no labels)
"""
import numpy as np
import pandas as pd


def _smoothed_te(series, labels, global_rate, smoothing=20):
    """Return smoothed target encoding dict for a categorical series."""
    grp = pd.DataFrame({"x": series, "y": labels}).groupby("x")["y"].agg(["sum", "count"])
    smoothed = (grp["sum"] + smoothing * global_rate) / (grp["count"] + smoothing)
    return {str(k): float(v) for k, v in smoothed.items()}


def fit(df_train: pd.DataFrame, y_train: pd.Series, config: dict) -> dict:
    state: dict = {}
    state["global_mean"] = float(y_train.mean())
    global_mean = state["global_mean"]

    # Selective NaN drop: only drop constant or near-empty columns.
    nan_rates = df_train.isnull().mean()
    n_unique = df_train.nunique(dropna=True)
    drop_mask = (n_unique <= 1) | ((nan_rates > 0.99) & (n_unique <= 5))
    drop_cols = nan_rates[drop_mask].index.tolist()
    state["drop_cols"] = drop_cols
    df_tmp = df_train.drop(columns=drop_cols, errors="ignore")

    # Identity-cluster presence flag
    id_cols = [c for c in df_tmp.columns if c.startswith("id_")]
    state["id_cluster_anchor"] = id_cols[0] if id_cols else None

    # === IDENTITY CONSISTENCY (Recipe 4) ===
    if "card1" in df_tmp.columns:
        if "R_emaildomain" in df_tmp.columns:
            cnts = df_tmp.groupby("card1")["R_emaildomain"].nunique()
            state["card1_n_email"] = {str(k): int(v) for k, v in cnts.items()}
            modal = df_tmp.groupby("card1")["R_emaildomain"].agg(
                lambda x: x.value_counts().index[0] if len(x.dropna()) > 0 else "__NAN__"
            )
            state["card1_modal_email"] = {str(k): str(v) for k, v in modal.items()}

        if "P_emaildomain" in df_tmp.columns:
            cnts = df_tmp.groupby("card1")["P_emaildomain"].nunique()
            state["card1_n_p_email"] = {str(k): int(v) for k, v in cnts.items()}

        if "DeviceInfo" in df_tmp.columns:
            cnts = df_tmp.groupby("card1")["DeviceInfo"].nunique()
            state["card1_n_device"] = {str(k): int(v) for k, v in cnts.items()}
            modal = df_tmp.groupby("card1")["DeviceInfo"].agg(
                lambda x: x.value_counts().index[0] if len(x.dropna()) > 0 else "__NAN__"
            )
            state["card1_modal_device"] = {str(k): str(v) for k, v in modal.items()}

        if "addr1" in df_tmp.columns:
            cnts = df_tmp.groupby("card1")["addr1"].nunique()
            state["card1_n_addr"] = {str(k): int(v) for k, v in cnts.items()}

        # Known (card1, R_emaildomain) pairs — new pair = fraud signal
        if "R_emaildomain" in df_tmp.columns:
            known_pairs = set(
                df_tmp[["card1", "R_emaildomain"]].dropna().apply(
                    lambda r: f"{r['card1']}_{r['R_emaildomain']}", axis=1
                )
            )
            state["known_card_email_pairs"] = list(known_pairs)

    # === AMOUNT FEATURES ===
    # Per-card1 amount statistics for behavioral deviation
    if "card1" in df_tmp.columns and "TransactionAmt" in df_tmp.columns:
        amt_stats = df_tmp.groupby("card1")["TransactionAmt"].agg(["mean", "std", "median", "count"])
        state["card1_amt_mean"] = {str(k): float(v) for k, v in amt_stats["mean"].items()}
        state["card1_amt_std"] = {str(k): float(v) for k, v in amt_stats["std"].fillna(0).items()}
        state["card1_amt_median"] = {str(k): float(v) for k, v in amt_stats["median"].items()}
        state["card1_txn_count"] = {str(k): int(v) for k, v in amt_stats["count"].items()}
        state["global_amt_mean"] = float(df_tmp["TransactionAmt"].mean())
        state["global_amt_std"] = float(df_tmp["TransactionAmt"].std())

    # Amount interaction with card6 (debit vs credit fraud patterns differ)
    if "card6" in df_tmp.columns and "TransactionAmt" in df_tmp.columns:
        c6_stats = df_tmp.groupby("card6")["TransactionAmt"].agg(["mean", "std"])
        state["card6_amt_mean"] = {str(k): float(v) for k, v in c6_stats["mean"].items()}
        state["card6_amt_std"] = {str(k): float(v) for k, v in c6_stats["std"].fillna(0).items()}

    # Fraud rate per card1 hour (hour-of-day distribution per card1 bucket)
    # addr1 × card1 amount stats
    if "addr1" in df_tmp.columns and "TransactionAmt" in df_tmp.columns:
        addr_stats = df_tmp.groupby("addr1")["TransactionAmt"].agg(["mean", "std", "count"])
        state["addr1_amt_mean"] = {str(k): float(v) for k, v in addr_stats["mean"].items()}
        state["addr1_amt_std"] = {str(k): float(v) for k, v in addr_stats["std"].fillna(0).items()}
        state["addr1_txn_count"] = {str(k): int(v) for k, v in addr_stats["count"].items()}

    # R_emaildomain fraud rate by amount quintile
    # (some email domains are associated with high-amount fraud)
    if "R_emaildomain" in df_tmp.columns and "TransactionAmt" in df_tmp.columns:
        # Smoothed TE on R_emaildomain → already done via freq, but add amount context
        state["r_email_amt_mean"] = {
            str(k): float(v) for k, v in
            df_tmp.groupby("R_emaildomain")["TransactionAmt"].mean().items()
        }

    # ProductCD × card6 interaction frequency (product type + payment method)
    if "ProductCD" in df_tmp.columns and "card6" in df_tmp.columns:
        combo = df_tmp["ProductCD"].fillna("__NAN__").astype(str) + "_" + df_tmp["card6"].fillna("__NAN__").astype(str)
        freq = combo.value_counts(normalize=True).to_dict()
        state["productcd_card6_freq"] = {str(k): float(v) for k, v in freq.items()}

    # card4 × card6 interaction (network × type)
    if "card4" in df_tmp.columns and "card6" in df_tmp.columns:
        combo = df_tmp["card4"].fillna("__NAN__").astype(str) + "_" + df_tmp["card6"].fillna("__NAN__").astype(str)
        freq = combo.value_counts(normalize=True).to_dict()
        state["card4_card6_freq"] = {str(k): float(v) for k, v in freq.items()}

    # R_emaildomain × card6 (email domain + payment method)
    if "R_emaildomain" in df_tmp.columns and "card6" in df_tmp.columns:
        combo = df_tmp["R_emaildomain"].fillna("__NAN__").astype(str) + "_" + df_tmp["card6"].fillna("__NAN__").astype(str)
        freq = combo.value_counts(normalize=True).to_dict()
        state["r_email_card6_freq"] = {str(k): float(v) for k, v in freq.items()}

    # R_emaildomain × ProductCD (email domain + product type)
    if "R_emaildomain" in df_tmp.columns and "ProductCD" in df_tmp.columns:
        combo = df_tmp["R_emaildomain"].fillna("__NAN__").astype(str) + "_" + df_tmp["ProductCD"].fillna("__NAN__").astype(str)
        freq = combo.value_counts(normalize=True).to_dict()
        state["r_email_productcd_freq"] = {str(k): float(v) for k, v in freq.items()}

    # addr1 × card6 (billing address + payment method)
    if "addr1" in df_tmp.columns and "card6" in df_tmp.columns:
        combo = df_tmp["addr1"].fillna(-1).astype(str) + "_" + df_tmp["card6"].fillna("__NAN__").astype(str)
        freq = combo.value_counts(normalize=True).to_dict()
        state["addr1_card6_freq"] = {str(k): float(v) for k, v in freq.items()}

    # card3 × card6 (card3 is important numeric — bucket it into quintiles for interaction)
    if "card3" in df_tmp.columns and "card6" in df_tmp.columns:
        # card3 is numeric but has categorical structure (screen dpi/type)
        # Bucket into quintiles for freq interaction
        try:
            card3_q = pd.qcut(df_tmp["card3"].fillna(-1), q=10, labels=False, duplicates="drop").fillna(-1).astype(str)
            combo = card3_q + "_" + df_tmp["card6"].fillna("__NAN__").astype(str)
            freq = combo.value_counts(normalize=True).to_dict()
            state["card3q_card6_freq"] = {str(k): float(v) for k, v in freq.items()}
        except Exception:
            pass

    # P_emaildomain × card6 (sender email domain + payment method)
    if "P_emaildomain" in df_tmp.columns and "card6" in df_tmp.columns:
        combo = df_tmp["P_emaildomain"].fillna("__NAN__").astype(str) + "_" + df_tmp["card6"].fillna("__NAN__").astype(str)
        freq = combo.value_counts(normalize=True).to_dict()
        state["p_email_card6_freq"] = {str(k): float(v) for k, v in freq.items()}

    # id_12 × card6 (identity found/not found × payment method)
    if "id_12" in df_tmp.columns and "card6" in df_tmp.columns:
        combo = df_tmp["id_12"].fillna("__NAN__").astype(str) + "_" + df_tmp["card6"].fillna("__NAN__").astype(str)
        freq = combo.value_counts(normalize=True).to_dict()
        state["id12_card6_freq"] = {str(k): float(v) for k, v in freq.items()}

    # Frequency encoding for all categoricals
    cat_cols = df_tmp.select_dtypes("object").columns.tolist()
    state["cat_cols"] = cat_cols
    for col in cat_cols:
        freq = df_tmp[col].value_counts(normalize=True).to_dict()
        state[f"{col}_freq"] = {str(k): float(v) for k, v in freq.items()}

    # Record TransactionDT reference for cyclic features
    state["has_transaction_dt"] = "TransactionDT" in df_tmp.columns

    return state


def transform(df: pd.DataFrame, state: dict, config: dict) -> pd.DataFrame:
    df = df.copy()

    # Cyclic time features from TransactionDT BEFORE dropping
    if state.get("has_transaction_dt") and "TransactionDT" in df.columns:
        t = df["TransactionDT"].fillna(0)
        seconds_in_day = t % 86400
        hour = seconds_in_day / 3600
        df["txn_hour"] = hour
        df["txn_hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["txn_hour_cos"] = np.cos(2 * np.pi * hour / 24)
        day_of_week = (t // 86400) % 7
        df["txn_dow"] = day_of_week
        df["txn_dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
        df["txn_dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
        df = df.drop(columns=["TransactionDT"])

    df = df.drop(columns=state.get("drop_cols", []), errors="ignore")

    # Identity-cluster presence flag
    anchor = state.get("id_cluster_anchor")
    if anchor and anchor in df.columns:
        df["id_cluster_present"] = df[anchor].notna().astype(float)
    else:
        df["id_cluster_present"] = 0.0

    # Identity consistency features
    if "card1" in df.columns:
        card1_str = df["card1"].astype(str)

        if state.get("card1_n_email"):
            df["card1_n_email"] = card1_str.map(state["card1_n_email"]).fillna(0)
        if state.get("card1_n_p_email"):
            df["card1_n_p_email"] = card1_str.map(state["card1_n_p_email"]).fillna(0)
        if state.get("card1_n_device"):
            df["card1_n_device"] = card1_str.map(state["card1_n_device"]).fillna(0)
        if state.get("card1_n_addr"):
            df["card1_n_addr"] = card1_str.map(state["card1_n_addr"]).fillna(0)

        # New (card1, R_emaildomain) pair flag
        if state.get("known_card_email_pairs") and "R_emaildomain" in df.columns:
            known = set(state["known_card_email_pairs"])
            pair_key = card1_str + "_" + df["R_emaildomain"].fillna("__NAN__").astype(str)
            df["card_email_is_new"] = (~pair_key.isin(known)).astype(float)

        # Is current device the modal device for this card?
        if state.get("card1_modal_device") and "DeviceInfo" in df.columns:
            modal = card1_str.map(state["card1_modal_device"]).fillna("__NAN__")
            df["is_modal_device"] = (df["DeviceInfo"].fillna("__NAN__").astype(str) == modal).astype(float)

    # Amount behavioral deviation from card1 baseline
    if "card1" in df.columns and "TransactionAmt" in df.columns:
        card1_str = df["card1"].astype(str)
        global_mean_amt = state.get("global_amt_mean", 134.0)
        global_std_amt = state.get("global_amt_std", 238.0)

        df["card1_amt_mean"] = card1_str.map(state.get("card1_amt_mean", {})).fillna(global_mean_amt)
        df["card1_amt_std"] = card1_str.map(state.get("card1_amt_std", {})).fillna(global_std_amt)
        df["card1_txn_count"] = card1_str.map(state.get("card1_txn_count", {})).fillna(0)

        amt = df["TransactionAmt"]
        df["card1_amt_zscore"] = (amt - df["card1_amt_mean"]) / df["card1_amt_std"].clip(lower=0.01)
        df["card1_amt_ratio"] = amt / df["card1_amt_mean"].clip(lower=0.01)
        df["amt_log"] = np.log1p(amt)
        df["amt_is_round"] = (amt % 1 == 0).astype(float)
        df["amt_cents"] = amt % 1

    # Amount vs card6 profile
    if "card6" in df.columns and "TransactionAmt" in df.columns:
        c6_str = df["card6"].astype(str).fillna("__NAN__")
        amt = df["TransactionAmt"]
        c6_mean = c6_str.map(state.get("card6_amt_mean", {})).fillna(state.get("global_amt_mean", 134.0))
        c6_std = c6_str.map(state.get("card6_amt_std", {})).fillna(state.get("global_amt_std", 238.0))
        df["card6_amt_zscore"] = (amt - c6_mean) / c6_std.clip(lower=0.01)

    # addr1-level amount features
    if "addr1" in df.columns and "TransactionAmt" in df.columns:
        addr_str = df["addr1"].astype(str)
        amt = df["TransactionAmt"]
        global_mean_amt = state.get("global_amt_mean", 134.0)
        df["addr1_amt_mean"] = addr_str.map(state.get("addr1_amt_mean", {})).fillna(global_mean_amt)
        df["addr1_amt_std"] = addr_str.map(state.get("addr1_amt_std", {})).fillna(state.get("global_amt_std", 238.0))
        df["addr1_txn_count"] = addr_str.map(state.get("addr1_txn_count", {})).fillna(0)
        df["addr1_amt_zscore"] = (amt - df["addr1_amt_mean"]) / df["addr1_amt_std"].clip(lower=0.01)

    # R_emaildomain amount context
    if "R_emaildomain" in df.columns and "TransactionAmt" in df.columns:
        email_str = df["R_emaildomain"].fillna("__NAN__").astype(str)
        global_mean_amt = state.get("global_amt_mean", 134.0)
        email_mean = email_str.map(state.get("r_email_amt_mean", {})).fillna(global_mean_amt)
        df["r_email_amt_ratio"] = df["TransactionAmt"] / email_mean.clip(lower=0.01)

    # Interaction frequency features
    if "ProductCD" in df.columns and "card6" in df.columns:
        combo = df["ProductCD"].fillna("__NAN__").astype(str) + "_" + df["card6"].fillna("__NAN__").astype(str)
        df["productcd_card6_freq"] = combo.map(state.get("productcd_card6_freq", {})).fillna(0.0)

    if "card4" in df.columns and "card6" in df.columns:
        combo = df["card4"].fillna("__NAN__").astype(str) + "_" + df["card6"].fillna("__NAN__").astype(str)
        df["card4_card6_freq"] = combo.map(state.get("card4_card6_freq", {})).fillna(0.0)

    if "R_emaildomain" in df.columns and "card6" in df.columns:
        combo = df["R_emaildomain"].fillna("__NAN__").astype(str) + "_" + df["card6"].fillna("__NAN__").astype(str)
        df["r_email_card6_freq"] = combo.map(state.get("r_email_card6_freq", {})).fillna(0.0)

    if "R_emaildomain" in df.columns and "ProductCD" in df.columns:
        combo = df["R_emaildomain"].fillna("__NAN__").astype(str) + "_" + df["ProductCD"].fillna("__NAN__").astype(str)
        df["r_email_productcd_freq"] = combo.map(state.get("r_email_productcd_freq", {})).fillna(0.0)

    if "addr1" in df.columns and "card6" in df.columns:
        combo = df["addr1"].fillna(-1).astype(str) + "_" + df["card6"].fillna("__NAN__").astype(str)
        df["addr1_card6_freq"] = combo.map(state.get("addr1_card6_freq", {})).fillna(0.0)

    if "P_emaildomain" in df.columns and "card6" in df.columns:
        combo = df["P_emaildomain"].fillna("__NAN__").astype(str) + "_" + df["card6"].fillna("__NAN__").astype(str)
        df["p_email_card6_freq"] = combo.map(state.get("p_email_card6_freq", {})).fillna(0.0)

    if "id_12" in df.columns and "card6" in df.columns:
        combo = df["id_12"].fillna("__NAN__").astype(str) + "_" + df["card6"].fillna("__NAN__").astype(str)
        df["id12_card6_freq"] = combo.map(state.get("id12_card6_freq", {})).fillna(0.0)

    # dist1 log transform (62% NaN, AUC=0.54 when present)
    if "dist1" in df.columns:
        df["dist1_log"] = np.log1p(df["dist1"].clip(lower=0))

    # Frequency encoding for all categoricals
    for col in state.get("cat_cols", []):
        if col in df.columns:
            freq_map = state.get(f"{col}_freq", {})
            df[f"{col}_freq_enc"] = df[col].apply(lambda x: freq_map.get(str(x), 0.0))
            df = df.drop(columns=[col])

    # Drop any remaining object columns
    for col in df.select_dtypes("object").columns:
        df = df.drop(columns=[col])

    return df.fillna(-1)
