"""Feature transforms for FDH (Fraud Detection Handbook) dataset. Edited by the agent.

Exp_011: Focused minimal feature set — keep only features with proven importance.
Strategy: start from exp_008 SOTA but remove low-importance features
to see if reduction improves OOT generalization.

Key features to keep (from exp_008 top features):
1. term_fraud_rate (0.39)
2. is_large_for_cust (0.19)
3. amount_vs_cust_median_ratio (0.16)
4. term_fr_28d (0.09)
5. amount_cust_zscore (0.04)
6. amount_vs_cust_mean (0.02)
7. cust_std_amt (0.02)
8. tx_time_days_num (0.02)
9. term_fr_14d (0.01)
10. term_fr_7d (0.01)
+ others that may add incremental value

Remove: low-importance velocity stats that don't appear in top 10
"""

import numpy as np
import pandas as pd


def fit(df_train: pd.DataFrame, y_train: pd.Series, config: dict) -> dict:
    """Fit feature state on training data. Return JSON-serializable dict."""
    df = df_train.copy()
    df["label"] = y_train.values

    global_mean_amt = float(df["TX_AMOUNT"].mean())
    global_std_amt = float(df["TX_AMOUNT"].std())
    global_fraud_rate = float(y_train.mean())

    # ── Customer stats ────────────────────────────────────────────────────────
    cust_stats = df.groupby("CUSTOMER_ID")["TX_AMOUNT"].agg(
        ["mean", "std", "count", "median"]
    ).reset_index()
    cust_stats.columns = ["CUSTOMER_ID", "cust_mean_amt", "cust_std_amt", "cust_tx_count", "cust_median_amt"]
    cust_stats["cust_std_amt"] = cust_stats["cust_std_amt"].fillna(1.0).clip(lower=0.1)

    cust_p10_amt = df.groupby("CUSTOMER_ID")["TX_AMOUNT"].quantile(0.10).to_dict()
    cust_p90_amt = df.groupby("CUSTOMER_ID")["TX_AMOUNT"].quantile(0.90).to_dict()
    cust_unique_terminals = df.groupby("CUSTOMER_ID")["TERMINAL_ID"].nunique().to_dict()

    # Customer velocity stats
    df_cs = df.sort_values(["CUSTOMER_ID", "TX_TIME_SECONDS"])
    df_cs["_gap"] = df_cs.groupby("CUSTOMER_ID")["TX_TIME_SECONDS"].diff()
    cust_gap_median = df_cs.groupby("CUSTOMER_ID")["_gap"].median().fillna(86400).to_dict()
    cust_gap_min = df_cs.groupby("CUSTOMER_ID")["_gap"].min().fillna(86400).to_dict()
    df_cs["_is_burst"] = (df_cs["_gap"] < 300).astype(int)
    cust_burst = df_cs.groupby("CUSTOMER_ID")["_is_burst"].sum().to_dict()
    train_days = float((df["TX_TIME_SECONDS"].max() - df["TX_TIME_SECONDS"].min()) / 86400)
    train_days = max(train_days, 1.0)
    cust_daily_rate = (df.groupby("CUSTOMER_ID")["TRANSACTION_ID"].count() / train_days).to_dict()

    # ── Terminal stats ────────────────────────────────────────────────────────
    term_stats = df.groupby("TERMINAL_ID").agg(
        term_fraud_rate=("label", "mean"),
        term_tx_count=("TRANSACTION_ID", "count"),
        term_mean_amt=("TX_AMOUNT", "mean"),
    ).reset_index()

    # Terminal multi-window fraud rates
    train_max_time = float(df["TX_TIME_SECONDS"].max())
    term_window_fraud = {}
    for wd in [7, 14, 28]:
        df_w = df[df["TX_TIME_SECONDS"] >= train_max_time - wd * 86400]
        stats_w = df_w.groupby("TERMINAL_ID").agg(
            fr=("label", "mean"), cnt=("TRANSACTION_ID", "count")
        )
        min_s = 20
        stats_w["fr_s"] = (
            stats_w["cnt"] / (stats_w["cnt"] + min_s) * stats_w["fr"]
            + min_s / (stats_w["cnt"] + min_s) * global_fraud_rate
        )
        term_window_fraud[f"term_fr_{wd}d"] = {str(k): float(v) for k, v in stats_w["fr_s"].items()}

    # Customer-terminal visit count
    cust_term_counts = df.groupby(["CUSTOMER_ID", "TERMINAL_ID"]).size()
    cust_term_count_dict = {f"{k[0]}_{k[1]}": int(v) for k, v in cust_term_counts.items()}

    return {
        "cust_mean_amt": {str(k): float(v) for k, v in cust_stats.set_index("CUSTOMER_ID")["cust_mean_amt"].items()},
        "cust_std_amt": {str(k): float(v) for k, v in cust_stats.set_index("CUSTOMER_ID")["cust_std_amt"].items()},
        "cust_count": {str(k): int(v) for k, v in cust_stats.set_index("CUSTOMER_ID")["cust_tx_count"].items()},
        "cust_median_amt": {str(k): float(v) for k, v in cust_stats.set_index("CUSTOMER_ID")["cust_median_amt"].items()},
        "cust_p10_amt": {str(k): float(v) for k, v in cust_p10_amt.items()},
        "cust_p90_amt": {str(k): float(v) for k, v in cust_p90_amt.items()},
        "cust_unique_terminals": {str(k): int(v) for k, v in cust_unique_terminals.items()},
        "cust_median_gap": {str(k): float(v) for k, v in cust_gap_median.items()},
        "cust_min_gap": {str(k): float(v) for k, v in cust_gap_min.items()},
        "cust_burst_count": {str(k): int(v) for k, v in cust_burst.items()},
        "cust_daily_rate": {str(k): float(v) for k, v in cust_daily_rate.items()},
        "term_fraud_rate": {str(k): float(v) for k, v in term_stats.set_index("TERMINAL_ID")["term_fraud_rate"].items()},
        "term_tx_count": {str(k): int(v) for k, v in term_stats.set_index("TERMINAL_ID")["term_tx_count"].items()},
        "term_mean_amt": {str(k): float(v) for k, v in term_stats.set_index("TERMINAL_ID")["term_mean_amt"].items()},
        "term_fr_7d": term_window_fraud["term_fr_7d"],
        "term_fr_14d": term_window_fraud["term_fr_14d"],
        "term_fr_28d": term_window_fraud["term_fr_28d"],
        "cust_term_count_dict": cust_term_count_dict,
        "global_mean_amt": global_mean_amt,
        "global_std_amt": global_std_amt,
        "global_fraud_rate": global_fraud_rate,
    }


def transform(df: pd.DataFrame, state: dict, config: dict) -> pd.DataFrame:
    """Apply feature transforms. No label access."""
    df = df.copy()

    global_mean = state["global_mean_amt"]
    global_std = max(state["global_std_amt"], 0.01)
    global_fr = state["global_fraud_rate"]

    # ── Time features ─────────────────────────────────────────────────────────
    dt = pd.to_datetime(df["TX_DATETIME"])
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(np.int8)
    df["is_night"] = ((dt.dt.hour < 6) | (dt.dt.hour >= 22)).astype(np.int8)
    df["tx_time_days_num"] = pd.to_numeric(df["TX_TIME_DAYS"], errors="coerce").fillna(0)

    # ── Amount features ───────────────────────────────────────────────────────
    df["log_amount"] = np.log1p(df["TX_AMOUNT"])
    df["amount_global_zscore"] = (df["TX_AMOUNT"] - global_mean) / global_std

    # ── Customer behavioral deviation ─────────────────────────────────────────
    cust_mean = state["cust_mean_amt"]
    cust_std = state["cust_std_amt"]
    cust_count = state["cust_count"]
    cust_median = state["cust_median_amt"]
    cust_p10 = state["cust_p10_amt"]
    cust_p90 = state["cust_p90_amt"]
    cust_unique_terms = state["cust_unique_terminals"]

    df["cust_mean_amt"] = df["CUSTOMER_ID"].map(lambda x: float(cust_mean.get(str(x), global_mean)))
    df["cust_std_amt"] = df["CUSTOMER_ID"].map(lambda x: float(cust_std.get(str(x), global_std)))
    df["cust_tx_count"] = df["CUSTOMER_ID"].map(lambda x: float(cust_count.get(str(x), 0)))
    df["cust_median_amt"] = df["CUSTOMER_ID"].map(lambda x: float(cust_median.get(str(x), global_mean)))

    df["amount_vs_cust_mean"] = df["TX_AMOUNT"] - df["cust_mean_amt"]
    df["amount_cust_zscore"] = df["amount_vs_cust_mean"] / df["cust_std_amt"]
    df["amount_vs_cust_median_ratio"] = df["TX_AMOUNT"] / (df["cust_median_amt"] + 0.01)
    df["log_cust_tx_count"] = np.log1p(df["cust_tx_count"])

    df["is_small_for_cust"] = (df["TX_AMOUNT"] <= df["CUSTOMER_ID"].map(
        lambda x: float(cust_p10.get(str(x), 0)))).astype(np.int8)
    df["is_large_for_cust"] = (df["TX_AMOUNT"] >= df["CUSTOMER_ID"].map(
        lambda x: float(cust_p90.get(str(x), global_mean * 2)))).astype(np.int8)

    df["log_cust_unique_terminals"] = np.log1p(df["CUSTOMER_ID"].map(
        lambda x: float(cust_unique_terms.get(str(x), 0))))
    df["log_cust_median_gap"] = np.log1p(df["CUSTOMER_ID"].map(
        lambda x: float(state["cust_median_gap"].get(str(x), 86400))))
    df["log_cust_min_gap"] = np.log1p(df["CUSTOMER_ID"].map(
        lambda x: float(state["cust_min_gap"].get(str(x), 86400))))
    df["log_cust_burst_count"] = np.log1p(df["CUSTOMER_ID"].map(
        lambda x: float(state["cust_burst_count"].get(str(x), 0))))
    df["log_cust_daily_rate"] = np.log1p(df["CUSTOMER_ID"].map(
        lambda x: float(state["cust_daily_rate"].get(str(x), 1.0))))

    # ── Terminal risk features ────────────────────────────────────────────────
    term_fraud = state["term_fraud_rate"]
    term_count = state["term_tx_count"]
    term_mean = state["term_mean_amt"]

    df["term_fraud_rate"] = df["TERMINAL_ID"].map(lambda x: float(term_fraud.get(str(x), global_fr)))
    df["term_tx_count"] = df["TERMINAL_ID"].map(lambda x: float(term_count.get(str(x), 0)))
    df["term_mean_amt"] = df["TERMINAL_ID"].map(lambda x: float(term_mean.get(str(x), global_mean)))
    df["log_term_tx_count"] = np.log1p(df["term_tx_count"])
    df["amount_vs_term_mean"] = df["TX_AMOUNT"] - df["term_mean_amt"]

    for wd in [7, 14, 28]:
        key = f"term_fr_{wd}d"
        df[key] = df["TERMINAL_ID"].map(lambda x, k=key: float(state[k].get(str(x), global_fr)))

    # ── Customer-terminal novelty (vectorized) ────────────────────────────────
    cust_term_count_dict = state.get("cust_term_count_dict", {})
    ct_key = df["CUSTOMER_ID"].astype(str) + "_" + df["TERMINAL_ID"].astype(str)
    df["cust_term_visit_count"] = ct_key.map(lambda k: float(cust_term_count_dict.get(k, 0)))
    df["is_new_terminal"] = (df["cust_term_visit_count"] == 0).astype(np.int8)

    # ── Drop raw string/datetime columns ──────────────────────────────────────
    drop_cols = ["TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID", "TRANSACTION_ID",
                 "TX_TIME_SECONDS", "TX_TIME_DAYS"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df
