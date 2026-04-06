"""Feature transforms for PaySim mobile money fraud dataset. Edited by the agent.

API CONTRACT:
  fit(df_train, y_train, config) -> state
      Called ONCE on training data WITH labels.
      Return a JSON-serializable dict of fitted parameters.
  transform(df, state, config) -> df
      Called on EACH split WITHOUT labels.
      Use only the state dict from fit(). No access to labels.

Dataset: PaySim synthetic mobile money transactions, 6.3M rows.
Key columns: step (hour 1-743), type, amount, nameOrig, oldbalanceOrg,
             newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, label
CRITICAL: fraud only occurs in TRANSFER and CASH_OUT types.
CRITICAL: isFlaggedFraud is NOT present (was dropped — it is leaky).
Key fraud signal: balance error = abs((oldbalanceOrg - newbalanceOrig) - amount)
"""

import numpy as np
import pandas as pd


TYPE_ORDER = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
TYPE_MAP = {t: i for i, t in enumerate(TYPE_ORDER)}


def _compute_smoothed_te(series: pd.Series, labels: pd.Series, min_samples: int = 10, smoothing_width: float = 5.0) -> dict:
    """Compute smoothed target encoding (OOF not needed for fit — apply globally)."""
    global_mean = float(labels.mean())
    df_tmp = pd.DataFrame({"entity": series, "label": labels})
    agg = df_tmp.groupby("entity")["label"].agg(["sum", "count"])
    # Smoothed estimate
    smoothing = 1.0 / (1.0 + np.exp(-(agg["count"] - min_samples) / smoothing_width))
    agg["smoothed"] = smoothing * (agg["sum"] / agg["count"]) + (1 - smoothing) * global_mean
    return {str(k): float(v) for k, v in agg["smoothed"].to_dict().items()}, global_mean


def fit(df_train: pd.DataFrame, y_train: pd.Series, config: dict) -> dict:
    """Fit feature state on training data.

    Exp 5: Target encoding for nameDest and nameOrig.
    nameDest fraud rate: if a destination account receives fraud money in train,
    it will likely do so again. Similarly for nameOrig.
    Use smoothed TE to prevent overfitting.
    """
    # Amount stats per type
    type_amount_stats = {}
    for t in TYPE_ORDER:
        mask = df_train["type"] == t
        if mask.sum() > 0:
            vals = df_train.loc[mask, "amount"]
            type_amount_stats[t] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()) if vals.std() > 0 else 1.0,
            }
        else:
            type_amount_stats[t] = {"mean": 1.0, "std": 1.0}

    amt_mean = float(df_train["amount"].mean())
    amt_std = float(df_train["amount"].std()) if df_train["amount"].std() > 0 else 1.0

    # nameOrig/nameDest transaction counts
    orig_counts = df_train["nameOrig"].value_counts().to_dict()
    dest_counts = df_train["nameDest"].value_counts().to_dict()

    # Target encoding for nameDest (fraud rate as destination)
    dest_te, dest_global_mean = _compute_smoothed_te(
        df_train["nameDest"], y_train, min_samples=10, smoothing_width=5.0
    )

    # Target encoding for nameOrig (fraud rate as origin)
    orig_te, orig_global_mean = _compute_smoothed_te(
        df_train["nameOrig"], y_train, min_samples=10, smoothing_width=5.0
    )

    # Type-conditional TE for nameDest: fraud rate when receiving a TRANSFER
    transfer_mask = df_train["type"] == "TRANSFER"
    if transfer_mask.sum() > 0:
        dest_te_transfer, dest_te_transfer_global = _compute_smoothed_te(
            df_train.loc[transfer_mask, "nameDest"],
            y_train[transfer_mask],
            min_samples=5, smoothing_width=3.0
        )
    else:
        dest_te_transfer, dest_te_transfer_global = {}, 0.0

    # Type-conditional TE for nameOrig: fraud rate when doing CASH_OUT
    cashout_mask = df_train["type"] == "CASH_OUT"
    if cashout_mask.sum() > 0:
        orig_te_cashout, orig_te_cashout_global = _compute_smoothed_te(
            df_train.loc[cashout_mask, "nameOrig"],
            y_train[cashout_mask],
            min_samples=5, smoothing_width=3.0
        )
    else:
        orig_te_cashout, orig_te_cashout_global = {}, 0.0

    return {
        "type_amount_stats": type_amount_stats,
        "amt_mean": amt_mean,
        "amt_std": amt_std,
        "orig_counts": {str(k): int(v) for k, v in orig_counts.items()},
        "dest_counts": {str(k): int(v) for k, v in dest_counts.items()},
        "dest_te": dest_te,
        "dest_global_mean": dest_global_mean,
        "orig_te": orig_te,
        "orig_global_mean": orig_global_mean,
        "dest_te_transfer": dest_te_transfer,
        "dest_te_transfer_global": dest_te_transfer_global,
        "orig_te_cashout": orig_te_cashout,
        "orig_te_cashout_global": orig_te_cashout_global,
    }


def transform(df: pd.DataFrame, state: dict, config: dict) -> pd.DataFrame:
    """Apply features with target encoding for entity IDs."""
    df = df.copy()

    # ── Type encoding ─────────────────────────────────────────────────────────
    df["type_enc"] = df["type"].map(TYPE_MAP).fillna(-1).astype(int)
    df["is_transfer"] = (df["type"] == "TRANSFER").astype(np.int8)
    df["is_cash_out"] = (df["type"] == "CASH_OUT").astype(np.int8)
    df["is_fraud_type"] = ((df["type"] == "TRANSFER") | (df["type"] == "CASH_OUT")).astype(np.int8)

    # ── Amount features ────────────────────────────────────────────────────────
    df["log_amount"] = np.log1p(df["amount"])
    amt_mean = state["amt_mean"]
    amt_std = state["amt_std"]
    df["amount_zscore"] = (df["amount"] - amt_mean) / amt_std

    type_stats = state["type_amount_stats"]
    type_mean_map = {t: v["mean"] for t, v in type_stats.items()}
    type_std_map = {t: max(v["std"], 1e-6) for t, v in type_stats.items()}
    t_mean = df["type"].map(type_mean_map).fillna(amt_mean)
    t_std = df["type"].map(type_std_map).fillna(max(amt_std, 1e-6))
    df["amount_type_zscore"] = (df["amount"] - t_mean) / t_std

    # ── Balance error features (most important) ────────────────────────────────
    df["orig_balance_error"] = np.abs(
        (df["oldbalanceOrg"] - df["newbalanceOrig"]) - df["amount"]
    )
    df["log_orig_balance_error"] = np.log1p(df["orig_balance_error"])

    df["dest_balance_error"] = np.abs(
        (df["newbalanceDest"] - df["oldbalanceDest"]) - df["amount"]
    )
    df["log_dest_balance_error"] = np.log1p(df["dest_balance_error"])

    # ── Balance flags ──────────────────────────────────────────────────────────
    df["orig_balance_after_zero"] = (df["newbalanceOrig"] == 0).astype(np.int8)
    df["orig_balance_was_zero"] = (df["oldbalanceOrg"] == 0).astype(np.int8)
    df["dest_balance_was_zero"] = (df["oldbalanceDest"] == 0).astype(np.int8)
    df["amount_exceeds_balance"] = (df["amount"] > df["oldbalanceOrg"] + 1).astype(np.int8)
    df["amount_to_orig_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1.0)
    df["log_orig_balance"] = np.log1p(df["oldbalanceOrg"])
    df["log_dest_balance_before"] = np.log1p(df["oldbalanceDest"])

    # ── Time features ──────────────────────────────────────────────────────────
    df["step_mod_24"] = df["step"] % 24
    df["step_mod_168"] = df["step"] % 168
    df["step_bin_day"] = df["step"] // 24

    # ── Entity frequency features ──────────────────────────────────────────────
    orig_counts = state["orig_counts"]
    dest_counts = state["dest_counts"]
    df["nameOrig_freq"] = df["nameOrig"].apply(lambda x: float(orig_counts.get(str(x), 0)))
    df["nameDest_freq"] = df["nameDest"].apply(lambda x: float(dest_counts.get(str(x), 0)))
    df["log_nameOrig_freq"] = np.log1p(df["nameOrig_freq"])
    df["log_nameDest_freq"] = np.log1p(df["nameDest_freq"])

    # ── Target encoding features ───────────────────────────────────────────────
    dest_te = state["dest_te"]
    dest_global = state["dest_global_mean"]
    orig_te = state["orig_te"]
    orig_global = state["orig_global_mean"]
    dest_te_transfer = state["dest_te_transfer"]
    dest_te_transfer_global = state["dest_te_transfer_global"]
    orig_te_cashout = state["orig_te_cashout"]
    orig_te_cashout_global = state["orig_te_cashout_global"]

    df["nameDest_fraud_rate"] = df["nameDest"].apply(
        lambda x: float(dest_te.get(str(x), dest_global))
    )
    df["nameOrig_fraud_rate"] = df["nameOrig"].apply(
        lambda x: float(orig_te.get(str(x), orig_global))
    )
    df["nameDest_transfer_fraud_rate"] = df["nameDest"].apply(
        lambda x: float(dest_te_transfer.get(str(x), dest_te_transfer_global))
    )
    df["nameOrig_cashout_fraud_rate"] = df["nameOrig"].apply(
        lambda x: float(orig_te_cashout.get(str(x), orig_te_cashout_global))
    )

    # ── Drop raw columns ───────────────────────────────────────────────────────
    drop_cols = ["type", "nameOrig", "nameDest"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df
