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
    # Per-card1: count of distinct identity values seen (more distinct = higher risk)
    if "card1" in df_tmp.columns:
        c1 = df_tmp["card1"].astype(str)

        # Count distinct R_emaildomain per card1
        if "R_emaildomain" in df_tmp.columns:
            cnts = df_tmp.groupby("card1")["R_emaildomain"].nunique()
            state["card1_n_email"] = {str(k): int(v) for k, v in cnts.items()}
            # Modal email per card1
            modal = df_tmp.groupby("card1")["R_emaildomain"].agg(
                lambda x: x.value_counts().index[0] if len(x.dropna()) > 0 else "__NAN__"
            )
            state["card1_modal_email"] = {str(k): str(v) for k, v in modal.items()}

        # Count distinct P_emaildomain per card1
        if "P_emaildomain" in df_tmp.columns:
            cnts = df_tmp.groupby("card1")["P_emaildomain"].nunique()
            state["card1_n_p_email"] = {str(k): int(v) for k, v in cnts.items()}

        # Count distinct DeviceInfo per card1
        if "DeviceInfo" in df_tmp.columns:
            cnts = df_tmp.groupby("card1")["DeviceInfo"].nunique()
            state["card1_n_device"] = {str(k): int(v) for k, v in cnts.items()}
            modal = df_tmp.groupby("card1")["DeviceInfo"].agg(
                lambda x: x.value_counts().index[0] if len(x.dropna()) > 0 else "__NAN__"
            )
            state["card1_modal_device"] = {str(k): str(v) for k, v in modal.items()}

        # Count distinct addr1 per card1
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
