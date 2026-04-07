"""Feature engineering for FDH dataset. Agent-editable.

Fit/transform API:
  fit(df_train, y_train, config) -> state dict (JSON-serializable)
  transform(df, state, config)   -> df (all numeric, no labels)
"""
import numpy as np
import pandas as pd


def fit(df_train: pd.DataFrame, y_train: pd.Series, config: dict) -> dict:
    """Fit on training data only. Returns JSON-serializable state."""
    state: dict = {}
    state["global_mean"] = float(y_train.mean())

    # Drop columns with >50% NaN
    nan_rates = df_train.isnull().mean()
    state["drop_cols"] = nan_rates[nan_rates > 0.50].index.tolist()
    df_tmp = df_train.drop(columns=state["drop_cols"], errors="ignore")

    # ----- Customer behavioral profiling -----
    cust_stats = (
        df_tmp.groupby("CUSTOMER_ID")["TX_AMOUNT"]
        .agg(["mean", "std", "median", "max", "count"])
        .reset_index()
    )
    cust_stats.columns = ["CUSTOMER_ID", "cust_mean_amt", "cust_std_amt",
                          "cust_median_amt", "cust_max_amt", "cust_tx_count"]

    cust_p90 = (
        df_tmp.groupby("CUSTOMER_ID")["TX_AMOUNT"]
        .quantile(0.90)
        .reset_index()
        .rename(columns={"TX_AMOUNT": "cust_p90_amt"})
    )
    cust_stats = cust_stats.merge(cust_p90, on="CUSTOMER_ID", how="left")

    if "TX_TIME_DAYS" in df_tmp.columns:
        day_span = df_tmp.groupby("CUSTOMER_ID")["TX_TIME_DAYS"].agg(lambda x: max(x.max() - x.min(), 1))
        day_span = day_span.reset_index().rename(columns={"TX_TIME_DAYS": "cust_day_span"})
        cust_stats = cust_stats.merge(day_span, on="CUSTOMER_ID", how="left")
        cust_stats["cust_daily_rate"] = cust_stats["cust_tx_count"] / cust_stats["cust_day_span"].clip(lower=1)
    else:
        cust_stats["cust_daily_rate"] = np.nan

    for col in ["cust_mean_amt", "cust_std_amt", "cust_median_amt",
                "cust_max_amt", "cust_tx_count", "cust_p90_amt", "cust_daily_rate"]:
        state[col] = {str(k): float(v) for k, v in zip(cust_stats["CUSTOMER_ID"], cust_stats[col])}

    state["cust_defaults"] = {
        "cust_mean_amt": float(df_tmp["TX_AMOUNT"].mean()),
        "cust_std_amt": float(df_tmp["TX_AMOUNT"].std()),
        "cust_median_amt": float(df_tmp["TX_AMOUNT"].median()),
        "cust_max_amt": float(df_tmp["TX_AMOUNT"].max()),
        "cust_tx_count": 1.0,
        "cust_p90_amt": float(df_tmp["TX_AMOUNT"].quantile(0.90)),
        "cust_daily_rate": float(cust_stats["cust_daily_rate"].median()),
    }

    # ----- Terminal amount profile -----
    term_amt = (
        df_tmp.groupby("TERMINAL_ID")["TX_AMOUNT"]
        .agg(["mean", "std", "median"])
        .reset_index()
    )
    term_amt.columns = ["TERMINAL_ID", "term_mean_amt", "term_std_amt", "term_median_amt"]

    for col in ["term_mean_amt", "term_std_amt", "term_median_amt"]:
        state[col] = {str(k): float(v) for k, v in zip(term_amt["TERMINAL_ID"], term_amt[col])}

    state["term_defaults"] = {
        "term_mean_amt": float(df_tmp["TX_AMOUNT"].mean()),
        "term_std_amt": float(df_tmp["TX_AMOUNT"].std()),
        "term_median_amt": float(df_tmp["TX_AMOUNT"].median()),
    }

    # ----- Customer-terminal diversity -----
    cust_unique_terms = (
        df_tmp.groupby("CUSTOMER_ID")["TERMINAL_ID"]
        .nunique()
        .reset_index()
        .rename(columns={"TERMINAL_ID": "cust_unique_terminals"})
    )
    state["cust_unique_terminals"] = {
        str(k): float(v)
        for k, v in zip(cust_unique_terms["CUSTOMER_ID"], cust_unique_terms["cust_unique_terminals"])
    }
    state["cust_unique_terminals_default"] = float(cust_unique_terms["cust_unique_terminals"].median())

    # ----- Terminal amount Q25/Q75 -----
    term_q75 = df_tmp.groupby("TERMINAL_ID")["TX_AMOUNT"].quantile(0.75).reset_index()
    term_q75.columns = ["TERMINAL_ID", "term_q75_amt"]
    term_q25 = df_tmp.groupby("TERMINAL_ID")["TX_AMOUNT"].quantile(0.25).reset_index()
    term_q25.columns = ["TERMINAL_ID", "term_q25_amt"]

    state["term_q75_amt"] = {str(k): float(v) for k, v in zip(term_q75["TERMINAL_ID"], term_q75["term_q75_amt"])}
    state["term_q25_amt"] = {str(k): float(v) for k, v in zip(term_q25["TERMINAL_ID"], term_q25["term_q25_amt"])}
    state["term_q75_default"] = float(df_tmp["TX_AMOUNT"].quantile(0.75))
    state["term_q25_default"] = float(df_tmp["TX_AMOUNT"].quantile(0.25))

    return state


def transform(df: pd.DataFrame, state: dict, config: dict) -> pd.DataFrame:
    """Transform without labels. Apply fitted state."""
    df = df.copy()
    df = df.drop(columns=state.get("drop_cols", []), errors="ignore")

    # Convert any datetime columns to unix seconds
    for col in df.select_dtypes(include=["datetime64"]).columns:
        df[col] = df[col].astype("int64") // 10**9

    # Drop raw ID columns — no signal, high PSI
    df = df.drop(columns=["TRANSACTION_ID"], errors="ignore")

    # ----- Amount features -----
    df["log_amount"] = np.log1p(df["TX_AMOUNT"])
    df["is_round"] = (df["TX_AMOUNT"] % 1 == 0).astype(float)
    df["is_small"] = (df["TX_AMOUNT"] < 20).astype(float)

    # ----- Time features (cyclic, stable across splits) -----
    if "TX_DATETIME" in df.columns:
        tx_ts = df["TX_DATETIME"].astype(float)
        hour = (tx_ts % 86400) / 3600
        df["tx_hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["tx_hour_cos"] = np.cos(2 * np.pi * hour / 24)
        dow = (tx_ts // 86400) % 7
        df["tx_dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["tx_dow_cos"] = np.cos(2 * np.pi * dow / 7)
        df["is_night"] = ((hour >= 22) | (hour < 6)).astype(float)

    # ----- Customer behavioral features -----
    defaults = state.get("cust_defaults", {})
    cust_str = df["CUSTOMER_ID"].astype(str)
    for col in ["cust_mean_amt", "cust_std_amt", "cust_median_amt",
                "cust_max_amt", "cust_tx_count", "cust_p90_amt", "cust_daily_rate"]:
        mapping = state.get(col, {})
        default_val = defaults.get(col, 0.0)
        df[col] = cust_str.apply(lambda x: mapping.get(x, default_val))

    std_safe = df["cust_std_amt"].clip(lower=0.01)
    df["amt_zscore_cust"] = (df["TX_AMOUNT"] - df["cust_mean_amt"]) / std_safe
    df["amt_vs_p90"] = df["TX_AMOUNT"] / df["cust_p90_amt"].clip(lower=0.01)

    # ----- Terminal amount features -----
    term_defaults = state.get("term_defaults", {})
    term_str = df["TERMINAL_ID"].astype(str)
    for col in ["term_mean_amt", "term_std_amt", "term_median_amt"]:
        mapping = state.get(col, {})
        default_val = term_defaults.get(col, 0.0)
        df[col] = term_str.apply(lambda x: mapping.get(x, default_val))

    term_std_safe = df["term_std_amt"].clip(lower=0.01)
    df["amt_zscore_term"] = (df["TX_AMOUNT"] - df["term_mean_amt"]) / term_std_safe
    df["amt_vs_term_median"] = df["TX_AMOUNT"] / df["term_median_amt"].clip(lower=0.01)

    # Q75 of terminal (for above-iqr flag)
    q75_map = state.get("term_q75_amt", {})
    q75_default = state.get("term_q75_default", 100.0)
    df["term_q75_amt"] = term_str.apply(lambda x: q75_map.get(x, q75_default))
    df["amt_above_term_q75"] = (df["TX_AMOUNT"] - df["term_q75_amt"]).clip(lower=0)

    # ----- Customer-terminal diversity -----
    cut_map = state.get("cust_unique_terminals", {})
    cut_default = state.get("cust_unique_terminals_default", 1.0)
    df["cust_unique_terminals"] = cust_str.apply(lambda x: cut_map.get(x, cut_default))
    df["cust_term_diversity"] = df["cust_unique_terminals"] / df["cust_tx_count"].clip(lower=1)

    # ----- Interaction features (KEY: captures multi-scenario fingerprints) -----
    # High customer amount vs p90 AND high terminal amount variance = scenario 2 fingerprint
    df["amt_vs_p90_x_term_std"] = df["amt_vs_p90"].clip(0, 20) * np.log1p(df["term_std_amt"])

    # Double anomaly: simultaneously anomalous for customer AND terminal
    df["double_anomaly"] = (
        df["amt_vs_p90"].clip(0, 20) * df["amt_zscore_term"].clip(-5, 20)
    )

    # High amount vs customer p90 AND above terminal Q75 = even more suspicious
    df["amt_vs_p90_x_above_q75"] = df["amt_vs_p90"].clip(0, 20) * np.log1p(df["amt_above_term_q75"])

    # Drop raw CUSTOMER_ID and TERMINAL_ID
    df = df.drop(columns=["CUSTOMER_ID", "TERMINAL_ID"], errors="ignore")

    # Drop high-PSI time index columns
    df = df.drop(columns=["TX_TIME_DAYS", "TX_DATETIME", "TX_TIME_SECONDS"], errors="ignore")

    return df.fillna(-1)
