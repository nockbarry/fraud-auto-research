"""FDH feature engineering. Agent-editable.

Fit/transform API:
  fit(df_train, y_train, config) -> state dict (JSON-serializable)
  transform(df, state, config)   -> df (all numeric, no labels)
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_WINDOWS_SEC = [3600, 6 * 3600, 86400, 7 * 86400]  # 1h, 6h, 1d, 7d
_WIN_NAMES = ["1h", "6h", "1d", "7d"]


def _window_velocity(df, entity_col, time_col, amt_col, windows, win_names, history=None):
    """O(N log N) rolling velocity with optional prepended training history.

    history: dict {entity: [[time, amount], ...]} (last N seconds of training data)
    Returns DataFrame (index = df.index) with count and sum per window.
    """
    # Build combined df: training tail + current batch
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

    # Sort by time globally
    combined = combined.sort_values(time_col)
    times_all = combined[time_col].values.astype(np.float64)
    ents_all = combined[entity_col].values
    amts_all = combined[amt_col].values.astype(np.float64)
    is_tail = combined["_is_tail"].values
    n_total = len(combined)

    result = {}
    for wname in win_names:
        result[f"{entity_col}_{wname}_count"] = np.zeros(n_total)
        result[f"{entity_col}_{wname}_sum"] = np.zeros(n_total)

    # Per-entity rolling windows
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
            counts = np.arange(n) - start_pos  # rows before current in window
            sums = cum_amt[np.arange(n)] - cum_amt[start_pos]
            for k, global_idx in enumerate(idxs):
                result[f"{entity_col}_{wname}_count"][global_idx] = counts[k]
                result[f"{entity_col}_{wname}_sum"][global_idx] = sums[k]

    result_df = pd.DataFrame(result, index=np.arange(n_total))
    # Keep only non-tail rows, restore original index
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

    dt_cols = df_train.select_dtypes("datetime").columns.tolist()
    state["drop_dt_cols"] = dt_cols

    cfg = config.get("dataset_profile", {})
    amt_col = cfg.get("amt_col", "TX_AMOUNT")
    cust_col = cfg.get("key_entity_col", "CUSTOMER_ID")
    time_col = "TX_TIME_SECONDS"

    # Per-customer amount stats
    cust_grp = df_train.groupby(cust_col)[amt_col]
    state["cust_mean"] = cust_grp.mean().to_dict()
    state["cust_std"] = cust_grp.std().fillna(1.0).to_dict()
    state["cust_p90"] = cust_grp.quantile(0.90).to_dict()
    state["global_mean_amt"] = float(df_train[amt_col].mean())
    state["global_std_amt"] = float(df_train[amt_col].std())

    # Per-terminal amount stats
    term_grp = df_train.groupby("TERMINAL_ID")[amt_col]
    state["term_mean"] = term_grp.mean().to_dict()
    state["term_std"] = term_grp.std().fillna(1.0).to_dict()
    state["term_count"] = term_grp.count().astype(int).to_dict()

    # Store last 7 days of training for velocity continuity in val/oot
    max_window = max(_WINDOWS_SEC)
    max_train_t = float(df_train[time_col].max())
    tail_mask = df_train[time_col] >= (max_train_t - max_window)
    tail = df_train.loc[tail_mask, [cust_col, "TERMINAL_ID", time_col, amt_col]]

    cust_tail: dict = {}
    for ent, grp in tail.groupby(cust_col):
        cust_tail[str(ent)] = [
            [float(t), float(a)] for t, a in zip(grp[time_col], grp[amt_col])
        ]
    state["cust_tail"] = cust_tail

    term_tail: dict = {}
    for ent, grp in tail.groupby("TERMINAL_ID"):
        term_tail[str(ent)] = [
            [float(t), float(a)] for t, a in zip(grp[time_col], grp[amt_col])
        ]
    state["term_tail"] = term_tail

    # Rolling terminal fraud rate (28-day window) — targets Scenario 2
    # Build per-terminal per-day fraud counts from training data
    df_train_tmp = df_train.copy()
    df_train_tmp["_day"] = (df_train_tmp[time_col] // 86400).astype(int)
    df_train_tmp["_fraud"] = y_train.values

    term_day_fraud = {}  # {terminal: {day: [fraud_count, total_count]}}
    for (term, day), grp in df_train_tmp.groupby(["TERMINAL_ID", "_day"]):
        term_key = str(term)
        if term_key not in term_day_fraud:
            term_day_fraud[term_key] = {}
        term_day_fraud[term_key][int(day)] = [
            int(grp["_fraud"].sum()),
            int(len(grp))
        ]
    state["term_day_fraud"] = term_day_fraud

    # Global fraud rate for smoothing
    state["global_fraud_rate"] = float(y_train.mean())

    # Static per-terminal fraud rate (entire training period, stable OOT)
    term_fraud_static = df_train.groupby("TERMINAL_ID").apply(
        lambda g: float(y_train.loc[g.index].mean())
    )
    state["term_static_fraud_rate"] = {str(k): float(v) for k, v in term_fraud_static.items()}

    # Per-customer distinct-terminal count in training (baseline for novelty)
    cust_term_count = df_train.groupby(cust_col)["TERMINAL_ID"].nunique()
    state["cust_n_terminals"] = {str(k): int(v) for k, v in cust_term_count.items()}

    return state


def transform(df: pd.DataFrame, state: dict, config: dict) -> pd.DataFrame:
    df = df.copy()

    cfg = config.get("dataset_profile", {})
    amt_col = cfg.get("amt_col", "TX_AMOUNT")
    cust_col = cfg.get("key_entity_col", "CUSTOMER_ID")
    time_col = "TX_TIME_SECONDS"

    # --- Cyclic time features (BEFORE dropping TX_DATETIME) ---
    if "TX_DATETIME" in df.columns:
        tx_dt = pd.to_datetime(df["TX_DATETIME"])
        hours = tx_dt.dt.hour.values
        dow = tx_dt.dt.dayofweek.values
        df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        df["is_night"] = ((hours >= 22) | (hours <= 6)).astype(float)
        df["is_weekend"] = (dow >= 5).astype(float)

    # Drop datetime columns after extracting time features
    df = df.drop(columns=state.get("drop_dt_cols", []), errors="ignore")

    # --- Customer behavioral deviation (Scenario 1) ---
    df["cust_mean_amt"] = df[cust_col].map(state["cust_mean"]).fillna(state["global_mean_amt"])
    df["cust_std_amt"] = df[cust_col].map(state["cust_std"]).fillna(state["global_std_amt"])
    df["cust_p90_amt"] = df[cust_col].map(state["cust_p90"]).fillna(state["global_mean_amt"])

    df["amt_zscore_cust"] = (df[amt_col] - df["cust_mean_amt"]) / df["cust_std_amt"].clip(lower=0.01)
    df["amt_vs_p90"] = df[amt_col] / df["cust_p90_amt"].clip(lower=1.0)
    # Drop intermediate helpers — not useful as standalone features
    df = df.drop(columns=["cust_mean_amt", "cust_std_amt", "cust_p90_amt"], errors="ignore")

    # --- Terminal stats ---
    df["term_mean_amt"] = df["TERMINAL_ID"].map(state["term_mean"]).fillna(state["global_mean_amt"])
    df["term_std_amt"] = df["TERMINAL_ID"].map(state["term_std"]).fillna(state["global_std_amt"])
    df["term_tx_count"] = df["TERMINAL_ID"].map(state["term_count"]).fillna(0)
    df["amt_zscore_term"] = (df[amt_col] - df["term_mean_amt"]) / df["term_std_amt"].clip(lower=0.01)

    # Normalize entity columns to str to match state dict keys
    df[cust_col] = df[cust_col].astype(str)
    df["TERMINAL_ID"] = df["TERMINAL_ID"].astype(str)

    # --- Velocity stack with training history prepended (Scenario 3) ---
    cust_history = {k: [tuple(v2) for v2 in v] for k, v in state.get("cust_tail", {}).items()}
    term_history = {k: [tuple(v2) for v2 in v] for k, v in state.get("term_tail", {}).items()}

    cust_vel = _window_velocity(df, cust_col, time_col, amt_col, _WINDOWS_SEC, _WIN_NAMES, cust_history)
    term_vel = _window_velocity(df, "TERMINAL_ID", time_col, amt_col, _WINDOWS_SEC, _WIN_NAMES, term_history)

    df = pd.concat([df, cust_vel, term_vel], axis=1)

    # Burst ratio: 1h vs 1d
    df["cust_burst_1h_1d"] = (
        (df[f"{cust_col}_1h_count"] + 1) / (df[f"{cust_col}_1d_count"] + 1)
    )
    df["term_burst_1h_1d"] = (
        (df["TERMINAL_ID_1h_count"] + 1) / (df["TERMINAL_ID_1d_count"] + 1)
    )

    # --- Rolling terminal fraud rate 28-day window (Scenario 2) ---
    term_day_fraud = state.get("term_day_fraud", {})
    global_fraud_rate = float(state.get("global_fraud_rate", 0.008))
    min_samples = 10  # Bayesian smoothing weight
    window_days = 28

    # Vectorized: unique (terminal, day) pairs → O(unique pairs × 28) not O(N × 28)
    terminals_arr = df["TERMINAL_ID"].values
    days_arr = (df[time_col].values // 86400).astype(int)

    unique_pairs = {}
    for i, (term, day) in enumerate(zip(terminals_arr, days_arr)):
        key = (term, day)
        if key not in unique_pairs:
            tfd = term_day_fraud.get(str(term), {})
            fraud_sum = 0
            total_sum = 0
            for d in range(day - window_days, day):
                if d in tfd:
                    vals = tfd[d]
                    fraud_sum += vals[0]
                    total_sum += vals[1]
            unique_pairs[key] = (fraud_sum + min_samples * global_fraud_rate) / (total_sum + min_samples)

    term_fraud_rates = np.array([unique_pairs[(t, d)] for t, d in zip(terminals_arr, days_arr)])
    df["term_fraud_rate_28d"] = term_fraud_rates

    # Static terminal fraud rate (stable OOT — full training history, no rolling window)
    term_static = state.get("term_static_fraud_rate", {})
    df["term_static_fraud_rate"] = df["TERMINAL_ID"].map(term_static).fillna(global_fraud_rate)

    # Per-customer terminal diversity from training (scenario 3: CNP hits many terminals)
    cust_n_term = state.get("cust_n_terminals", {})
    df["cust_n_term_train"] = df[cust_col].map(cust_n_term).fillna(1.0)

    # Small amount flag (scenario 3: CNP repeated small amounts)
    df["amt_is_small"] = (df[amt_col] < 50).astype(float)
    df["amt_is_round"] = (df[amt_col] % 10 < 0.01).astype(float)

    # Drop raw categoricals
    for col in df.select_dtypes("object").columns:
        df = df.drop(columns=[col])

    return df.fillna(-1)
