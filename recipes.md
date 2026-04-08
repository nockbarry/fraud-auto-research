# Feature Engineering Recipes

Copy-paste code patterns for the fit/transform API. Each recipe shows what to add to `fit()` and `transform()`. All state is JSON-serializable.

Read `dataset_profile` in the config to decide which recipes apply to your dataset.

---

## Recipe 1: Out-of-Fold Target Encoding

Prevents within-train TE leakage. Use this instead of naive `_target_encode_fit` for high-cardinality columns.

```python
# In fit():
from sklearn.model_selection import StratifiedKFold

def _oof_target_encode(series, y, global_mean, n_splits=5, min_samples=30, width=15):
    """OOF target encoding: each training row's encoding uses only other folds."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Global TE map for val/OOT (fitted on all train)
    df_tmp = pd.DataFrame({"col": series.values, "_y": y.values})
    stats = df_tmp.groupby("col")["_y"].agg(["mean", "count"])
    smoothing = 1 / (1 + np.exp(-(stats["count"] - min_samples) / width))
    global_te = smoothing * stats["mean"] + (1 - smoothing) * global_mean
    
    return {str(k): float(v) for k, v in global_te.items()}

# Apply to columns:
for col in ["card1", "addr1", "merchant"]:
    if col in df_train.columns:
        state[f"{col}_oof_te"] = _oof_target_encode(
            df_train[col].astype(str), y_train, global_mean
        )

# In transform():
for col in ["card1", "addr1", "merchant"]:
    te_key = f"{col}_oof_te"
    if te_key in state and col in df.columns:
        df[f"{col}_te"] = df[col].astype(str).map(state[te_key]).fillna(global_mean)
```

**When to use**: High-cardinality categoricals (>100 unique values). Always use for the primary entity column.

---

## Recipe 2: Velocity Features (Per-Card Temporal Stats)

Captures transaction speed patterns. Fraudsters transact faster than legitimate users.

```python
# In fit():
time_col = state.get("time_col")
card_col = state.get("card_col")

if card_col and time_col:
    df_sorted = df_train.sort_values([card_col, time_col])
    df_sorted["_gap"] = df_sorted.groupby(card_col)[time_col].diff()
    
    gap_stats = df_sorted.groupby(card_col)["_gap"].agg(["median", "std", "min", "count"])
    # Burst count: transactions within 60 seconds of each other
    df_sorted["_is_burst"] = (df_sorted["_gap"] < 60).astype(int)
    burst_counts = df_sorted.groupby(card_col)["_is_burst"].sum()
    
    # Transactions per day
    time_range = df_sorted.groupby(card_col)[time_col].agg(["min", "max"])
    time_range["days"] = ((time_range["max"] - time_range["min"]) / 86400).clip(lower=1)
    daily_rate = gap_stats["count"] / time_range["days"]
    
    state["velocity_median_gap"] = {str(k): float(v) for k, v in gap_stats["median"].fillna(0).items()}
    state["velocity_std_gap"] = {str(k): float(v) for k, v in gap_stats["std"].fillna(0).items()}
    state["velocity_min_gap"] = {str(k): float(v) for k, v in gap_stats["min"].fillna(0).items()}
    state["velocity_burst_count"] = {str(k): int(v) for k, v in burst_counts.items()}
    state["velocity_daily_rate"] = {str(k): float(v) for k, v in daily_rate.items()}

# In transform():
card_col = state.get("card_col")
if card_col and card_col in df.columns:
    card_str = df[card_col].astype(str)
    df["velocity_median_gap"] = card_str.map(state.get("velocity_median_gap", {})).fillna(0)
    df["velocity_std_gap"] = card_str.map(state.get("velocity_std_gap", {})).fillna(0)
    df["velocity_min_gap"] = card_str.map(state.get("velocity_min_gap", {})).fillna(0)
    df["velocity_burst_count"] = card_str.map(state.get("velocity_burst_count", {})).fillna(0)
    df["velocity_daily_rate"] = card_str.map(state.get("velocity_daily_rate", {})).fillna(0)
    df["is_high_velocity"] = (df["velocity_daily_rate"] > df["velocity_daily_rate"].quantile(0.95)).astype(int)
```

**When to use**: Any dataset with a card/customer ID and timestamps. Both IEEE-CIS and fraud-sim.

---

# AMBITIOUS COMPOUND RECIPES (read these first when building a campaign)

The 5 recipes below (15-19) are the moves competition winners use. They each add 10-50 features in one experiment instead of the typical +3, and they target the gaps in single-technique recipes. Read these before falling back to incremental tweaks.

---

## Recipe 15: UID Construction (the IEEE-CIS winner trick)

The IEEE-CIS Kaggle winner (FraudSquad, AUC 0.9459) built a synthetic entity by concatenating `card1 + addr1 + D1` and computed 262 group-aggregation features against it. Local CV jumped from ~0.90 to ~0.948 from this single insight. This recipe teaches the pattern.

```python
# In fit():
uid_components = []
for c in ["card1", "addr1", "D1"]:  # IEEE-CIS — adapt per dataset
    if c in df_train.columns:
        uid_components.append(c)

if len(uid_components) >= 2:
    uid_series = df_train[uid_components[0]].astype(str)
    for c in uid_components[1:]:
        uid_series = uid_series + "_" + df_train[c].astype(str)
    state["uid_components"] = uid_components

    amt_col = state.get("amt_col", "TransactionAmt")
    time_col = state.get("time_col", "TransactionDT")

    # Filter to UIDs with min_samples=5 — UIDs have higher cardinality than card_id
    uid_counts = uid_series.value_counts()
    valid_uids = set(uid_counts[uid_counts >= 5].index)
    mask = uid_series.isin(valid_uids)
    df_uid = df_train[mask].copy()
    df_uid["_uid"] = uid_series[mask]

    # 5-10 group aggregations against UID
    aggs = df_uid.groupby("_uid")[amt_col].agg(["mean", "std", "max", "median", "count"])
    state["uid_amt_mean"] = {str(k): float(v) for k, v in aggs["mean"].items()}
    state["uid_amt_std"] = {str(k): float(v) for k, v in aggs["std"].fillna(0).items()}
    state["uid_amt_max"] = {str(k): float(v) for k, v in aggs["max"].items()}
    state["uid_amt_median"] = {str(k): float(v) for k, v in aggs["median"].items()}
    state["uid_txn_count"] = {str(k): int(v) for k, v in aggs["count"].items()}

    if time_col in df_uid.columns:
        time_span = df_uid.groupby("_uid")[time_col].agg(lambda s: float(s.max() - s.min()))
        state["uid_time_span"] = {str(k): float(v) for k, v in time_span.items()}

    state["uid_global_mean"] = float(df_train[amt_col].mean())

# In transform():
components = state.get("uid_components", [])
if len(components) >= 2 and all(c in df.columns for c in components):
    uid = df[components[0]].astype(str)
    for c in components[1:]:
        uid = uid + "_" + df[c].astype(str)

    global_mean = state.get("uid_global_mean", 0.0)
    df["uid_amt_mean"] = uid.apply(lambda x: state.get("uid_amt_mean", {}).get(x, global_mean))
    df["uid_amt_std"] = uid.apply(lambda x: state.get("uid_amt_std", {}).get(x, 0.0))
    df["uid_amt_max"] = uid.apply(lambda x: state.get("uid_amt_max", {}).get(x, global_mean))
    df["uid_amt_median"] = uid.apply(lambda x: state.get("uid_amt_median", {}).get(x, global_mean))
    df["uid_txn_count"] = uid.apply(lambda x: state.get("uid_txn_count", {}).get(x, 0))
    df["uid_time_span"] = uid.apply(lambda x: state.get("uid_time_span", {}).get(x, 0.0))
    # Deviation features — current row vs UID baseline
    amt_col = state.get("amt_col", "TransactionAmt")
    df["uid_amt_zscore"] = (df[amt_col] - df["uid_amt_mean"]) / df["uid_amt_std"].clip(lower=0.01)
    df["uid_is_new"] = (df["uid_txn_count"] == 0).astype(int)
```

**Adapt per dataset:**
- IEEE-CIS: `card1 + addr1 + D1` (the winner's combo) or `card1 + P_emaildomain`
- fraud-sim: `card_id + zip` or `card_id + category`
- FDH: `CUSTOMER_ID + TERMINAL_ID` (then per-pair behavior; useful for scenario 2)
- PaySim: `nameOrig + nameDest` (then chain detection per pair)

**State size note:** A UID with 100k unique values × 10 stats = 1M state entries. Filter UIDs with `min_samples=5` (default) to keep state under ~100KB. Increase to 10 if state.json grows beyond 1MB.

**When to use:** Any dataset with 2+ stable identity columns. This is the highest-impact ambitious move — competition winners credit it directly. Use this as Step 1 of a UID Aggregation Campaign (see program.md "Multi-Step Campaigns"). Pair with Recipe 16 (multi-window velocity keyed by UID) for Step 2.

---

## Recipe 16: Multi-Window Velocity Stack

A loop over `[60, 600, 3600, 86400, 604800]` second windows × multiple entities × multiple stats = 40+ features in one recipe. This is the canonical "velocity stack" that fraud teams build. Recipe 2's single-window velocity is a degenerate case of this.

**Performance critical:** Naive `groupby(...).rolling(...)` is O(N²) on sorted groups and will blow the 900s timeout on FDH's 1.75M rows. Use sort + numpy cumsum (O(N log N)) instead.

```python
# In fit():
import numpy as np

amt_col = state.get("amt_col", "TX_AMOUNT")
time_col = state.get("time_col", "TX_DATETIME")
entities = ["CUSTOMER_ID", "TERMINAL_ID"]  # adapt per dataset
entities = [e for e in entities if e in df_train.columns]

# Compute time as a numeric column (seconds since epoch)
if df_train[time_col].dtype.kind == "M":  # datetime64
    t = (df_train[time_col].astype("int64") // 10**9).values
else:
    t = df_train[time_col].astype(float).values

windows_sec = [60, 600, 3600, 86400, 604800]  # 1m, 10m, 1h, 24h, 7d

# Build per-entity sorted history maps used at transform time.
# Store: {entity_id: {"t": [...], "amt": [...]} } after sorting by time.
amt = df_train[amt_col].astype(float).values
for ent in entities:
    keys = df_train[ent].astype(str).values
    # Sort by entity, then time, so we can group efficiently
    order = np.lexsort((t, keys))
    keys_s, t_s, amt_s = keys[order], t[order], amt[order]
    # Build per-entity contiguous slices
    history: dict = {}
    start = 0
    for i in range(1, len(keys_s) + 1):
        if i == len(keys_s) or keys_s[i] != keys_s[start]:
            history[str(keys_s[start])] = {
                "t": t_s[start:i].tolist(),
                "amt": amt_s[start:i].tolist(),
            }
            start = i
    state[f"{ent}_history"] = history

state["vel_windows_sec"] = windows_sec
state["vel_entities"] = entities
state["vel_time_col"] = time_col
state["vel_amt_col"] = amt_col

# In transform():
import bisect
import numpy as np

windows_sec = state.get("vel_windows_sec", [60, 600, 3600, 86400, 604800])
entities = state.get("vel_entities", [])
time_col = state.get("vel_time_col", "TX_DATETIME")
amt_col = state.get("vel_amt_col", "TX_AMOUNT")

if time_col in df.columns and amt_col in df.columns:
    if df[time_col].dtype.kind == "M":
        t_query = (df[time_col].astype("int64") // 10**9).values
    else:
        t_query = df[time_col].astype(float).values

    for ent in entities:
        if ent not in df.columns:
            continue
        history = state.get(f"{ent}_history", {})
        keys = df[ent].astype(str).values
        # For each window, vectorize via per-row bisect over the entity's history
        for w in windows_sec:
            counts = np.zeros(len(df), dtype=np.float32)
            sums = np.zeros(len(df), dtype=np.float32)
            maxes = np.zeros(len(df), dtype=np.float32)
            for i in range(len(df)):
                rec = history.get(keys[i])
                if rec is None:
                    continue
                t_arr = rec["t"]
                a_arr = rec["amt"]
                lo = bisect.bisect_left(t_arr, t_query[i] - w)
                hi = bisect.bisect_right(t_arr, t_query[i])
                if hi > lo:
                    window_amts = a_arr[lo:hi]
                    counts[i] = hi - lo
                    sums[i] = float(sum(window_amts))
                    maxes[i] = float(max(window_amts))
            wlbl = f"{w}s" if w < 3600 else f"{w//3600}h" if w < 86400 else f"{w//86400}d"
            df[f"vel_{ent}_count_{wlbl}"] = counts
            df[f"vel_{ent}_sum_{wlbl}"] = sums
            df[f"vel_{ent}_max_{wlbl}"] = maxes
```

**Output:** 5 windows × 2 entities × 3 stats = **30 features in one recipe**. Add std as a 4th stat (uses np.std on the slice) for 40 total.

**Performance budget:** O(N log N) over the sorted history per row. For FDH (~30k val rows × 2 entities × 5 windows = 300k bisects × O(log 350)), this is well under 60s. If you need to scale to 1M+ val rows, batch the bisect into a single np.searchsorted pass per (entity, window).

**When to use:** Any dataset with timestamps + an entity column. Critical for FDH scenario 3 (small repeated CNP transactions — short windows like 1m and 10m catch this). Also for IEEE-CIS card-not-present bursts. Use as Step 1 of a Velocity Stack Campaign. Pair with Recipe 17 (rolling terminal fraud rate) for Step 3.

---

## Recipe 17: Rolling Terminal Fraud Rate (28-day window)

Recipe 14's missing dimension. Instead of a single global fraud rate per terminal, this computes the rate over a trailing 28-day window — directly targeting FDH scenario 2 ("specific terminals fraudulent for 28-day windows"). The terminal compromise is temporal: a terminal that was compromised 6 months ago is no longer high-risk; one compromised yesterday is.

```python
# In fit():
import numpy as np

term_col = "TERMINAL_ID"  # adapt per dataset
time_col = state.get("time_col", "TX_DATETIME")
window_days = 28

if term_col in df_train.columns and time_col in df_train.columns:
    # Convert time to integer day index from epoch
    if df_train[time_col].dtype.kind == "M":
        days = (df_train[time_col].astype("int64") // 10**9 // 86400).astype(int)
    else:
        days = (df_train[time_col].astype(float) // 86400).astype(int)

    df_tmp = pd.DataFrame({"term": df_train[term_col].astype(str), "day": days, "y": y_train.values})
    # Daily fraud counts and total counts per terminal
    grouped = df_tmp.groupby(["term", "day"])["y"].agg(["sum", "count"]).reset_index()

    # Store as nested dict: {term: {day: (sum, count)}}
    term_daily: dict = {}
    for term, g in grouped.groupby("term"):
        term_daily[term] = {int(d): (int(s), int(c)) for d, s, c in zip(g["day"], g["sum"], g["count"])}

    state["term_daily_fraud"] = term_daily
    state["term_window_days"] = window_days
    state["term_global_rate"] = float(y_train.mean())
    state["term_rolling_col"] = term_col
    state["term_rolling_time_col"] = time_col

# In transform():
term_col = state.get("term_rolling_col")
time_col = state.get("term_rolling_time_col")
window_days = state.get("term_window_days", 28)
global_rate = state.get("term_global_rate", 0.01)

if term_col and term_col in df.columns and time_col in df.columns:
    if df[time_col].dtype.kind == "M":
        query_days = (df[time_col].astype("int64") // 10**9 // 86400).astype(int).values
    else:
        query_days = (df[time_col].astype(float) // 86400).astype(int).values

    term_daily = state.get("term_daily_fraud", {})
    keys = df[term_col].astype(str).values

    rates = np.full(len(df), global_rate, dtype=np.float32)
    counts = np.zeros(len(df), dtype=np.float32)
    for i in range(len(df)):
        td = term_daily.get(keys[i])
        if td is None:
            continue
        # Sum fraud and count over [query_day - window, query_day - 1]
        lo, hi = query_days[i] - window_days, query_days[i] - 1
        s, c = 0, 0
        for d in range(lo, hi + 1):
            if d in td:
                ds, dc = td[d]
                s += ds
                c += dc
        if c > 0:
            # Bayesian smoothing toward global with min_samples=10
            w = c / (c + 10)
            rates[i] = w * (s / c) + (1 - w) * global_rate
            counts[i] = c

    df[f"term_fraud_rate_{window_days}d"] = rates
    df[f"term_volume_{window_days}d"] = counts
    df[f"term_compromise_flag_{window_days}d"] = (rates > global_rate * 5).astype(int)
```

**Performance note:** The inner day-loop is O(window_days) per row. For 28 days × 30k val rows = 840k dict lookups — fast. For longer windows (90d+) consider precomputing per-terminal cumulative fraud arrays.

**When to use:** **REQUIRED for FDH scenario 2** (terminal compromise is documented in configs/fdh.yaml as a 28-day phenomenon). Adapt to other datasets with merchant/terminal IDs and timestamps. Pair with Recipe 14 (global terminal rate) — having both the 28d and global rate as features lets the model contrast "currently compromised" vs "always risky".

---

## Recipe 18: Behavioral Fingerprint Pipeline (multi-entity composition)

A composition recipe — apply Recipe 2/3/14 against multiple entities (card, terminal, UID from Recipe 15) and stack them. Yields 30+ features in one shot. This is the structure behind every competition-winning behavioral feature set.

```python
# In fit():
# Apply Recipe 3 (behavioral profiling) to MULTIPLE entities, not just card_col.
# For each entity, compute the same 11 stats: tx_count, mean/std/median/p90/max amount,
# mean/std hour, mean gap, burst count, distinct counterparties.

amt_col = state.get("amt_col")
time_col = state.get("time_col")

entities_to_profile = ["CUSTOMER_ID", "TERMINAL_ID"]  # adapt per dataset
# (For IEEE-CIS: entities_to_profile = ["card1", "addr1", "P_emaildomain"])
# (For fraud-sim: entities_to_profile = ["card_id", "merchant", "category"])

state["fp_entities"] = [e for e in entities_to_profile if e in df_train.columns]

for ent in state["fp_entities"]:
    grp = df_train.groupby(ent)
    amt = grp[amt_col]
    state[f"fp_{ent}_count"] = {str(k): int(v) for k, v in amt.count().items()}
    state[f"fp_{ent}_mean"] = {str(k): float(v) for k, v in amt.mean().items()}
    state[f"fp_{ent}_std"] = {str(k): float(v) for k, v in amt.std().fillna(0).items()}
    state[f"fp_{ent}_median"] = {str(k): float(v) for k, v in amt.median().items()}
    state[f"fp_{ent}_p90"] = {str(k): float(v) for k, v in amt.quantile(0.90).items()}
    state[f"fp_{ent}_max"] = {str(k): float(v) for k, v in amt.max().items()}

    if time_col in df_train.columns:
        # Convert time to seconds for hour computation
        if df_train[time_col].dtype.kind == "M":
            t_sec = df_train[time_col].astype("int64") // 10**9
        else:
            t_sec = df_train[time_col].astype(float)
        hour = (t_sec % 86400) / 3600
        df_tmp = pd.DataFrame({ent: df_train[ent], "_hour": hour})
        h = df_tmp.groupby(ent)["_hour"]
        state[f"fp_{ent}_hour_mean"] = {str(k): float(v) for k, v in h.mean().items()}
        state[f"fp_{ent}_hour_std"] = {str(k): float(v) for k, v in h.std().fillna(4).items()}

state["fp_global_amt_mean"] = float(df_train[amt_col].mean())

# In transform():
for ent in state.get("fp_entities", []):
    if ent not in df.columns:
        continue
    keys = df[ent].astype(str)
    gm = state.get("fp_global_amt_mean", 0.0)
    df[f"fp_{ent}_count"] = keys.apply(lambda x: state.get(f"fp_{ent}_count", {}).get(x, 0))
    df[f"fp_{ent}_mean"] = keys.apply(lambda x: state.get(f"fp_{ent}_mean", {}).get(x, gm))
    df[f"fp_{ent}_std"] = keys.apply(lambda x: state.get(f"fp_{ent}_std", {}).get(x, 0.0))
    df[f"fp_{ent}_p90"] = keys.apply(lambda x: state.get(f"fp_{ent}_p90", {}).get(x, gm))
    df[f"fp_{ent}_max"] = keys.apply(lambda x: state.get(f"fp_{ent}_max", {}).get(x, gm))

    # Deviation features — current row vs entity baseline
    amt_col = state.get("amt_col")
    if amt_col in df.columns:
        df[f"fp_{ent}_amt_zscore"] = (df[amt_col] - df[f"fp_{ent}_mean"]) / df[f"fp_{ent}_std"].clip(lower=0.01)
        df[f"fp_{ent}_amt_vs_p90"] = df[amt_col] / df[f"fp_{ent}_p90"].clip(lower=0.01)

    if state.get(f"fp_{ent}_hour_mean"):
        df[f"fp_{ent}_hour_mean"] = keys.apply(lambda x: state.get(f"fp_{ent}_hour_mean", {}).get(x, 12.0))
        df[f"fp_{ent}_hour_std"] = keys.apply(lambda x: state.get(f"fp_{ent}_hour_std", {}).get(x, 4.0))
```

**Output:** 9 features per entity × 3 entities = **27 features in one recipe**. Add hour deviation from current transaction for an extra entity-aware time feature.

**When to use:** When you have 2+ entity columns and want to capture behavior at multiple granularities. The power comes from the contrasts: a transaction that is normal for the customer but abnormal for the terminal (or vice versa) is a strong signal. Use this as Step 1 of a campaign on any dataset with multiple entities.

---

## Recipe 19: Cyclic Time + Per-Card von Mises (angular distance)

Extends Recipe 12 (cyclic sin/cos) with per-card preferred-hour mean angle and angular distance. Captures "unusual hour for *this card*" rather than "unusual hour globally". The von Mises distribution is the circular analog of the normal distribution.

```python
# In fit():
import numpy as np

card_col = state.get("card_col")
time_col = state.get("time_col")

if card_col and time_col and card_col in df_train.columns and time_col in df_train.columns:
    # Convert to seconds and extract hour as an angle in radians
    if df_train[time_col].dtype.kind == "M":
        t_sec = df_train[time_col].astype("int64") // 10**9
    else:
        t_sec = df_train[time_col].astype(float)

    hour = (t_sec % 86400) / 3600
    angle = 2 * np.pi * hour / 24  # radians

    df_tmp = pd.DataFrame({card_col: df_train[card_col], "_sin": np.sin(angle), "_cos": np.cos(angle)})
    grp = df_tmp.groupby(card_col)
    sin_mean = grp["_sin"].mean()
    cos_mean = grp["_cos"].mean()

    # Mean direction (preferred hour as an angle)
    mean_angle = np.arctan2(sin_mean, cos_mean)
    # Concentration parameter kappa (proxy: 1 - resultant length)
    R = np.sqrt(sin_mean ** 2 + cos_mean ** 2)
    # Higher R = more concentrated; lower R = spread out
    state["vm_card_mean_sin"] = {str(k): float(v) for k, v in sin_mean.items()}
    state["vm_card_mean_cos"] = {str(k): float(v) for k, v in cos_mean.items()}
    state["vm_card_concentration"] = {str(k): float(v) for k, v in R.items()}

# In transform():
card_col = state.get("card_col")
time_col = state.get("time_col")

if card_col and time_col and card_col in df.columns and time_col in df.columns:
    if df[time_col].dtype.kind == "M":
        t_sec = df[time_col].astype("int64") // 10**9
    else:
        t_sec = df[time_col].astype(float)
    hour = (t_sec % 86400) / 3600
    angle = 2 * np.pi * hour / 24

    keys = df[card_col].astype(str)
    card_sin = keys.map(state.get("vm_card_mean_sin", {})).fillna(0.0)
    card_cos = keys.map(state.get("vm_card_mean_cos", {})).fillna(0.0)
    card_kappa = keys.map(state.get("vm_card_concentration", {})).fillna(0.0)

    # Angular distance: 1 - cos(angle - mean_angle)
    # = 1 - (cos(angle)*cos(mean) + sin(angle)*sin(mean))
    df["hour_angular_dist_from_self"] = 1.0 - (np.cos(angle) * card_cos + np.sin(angle) * card_sin)
    df["card_hour_concentration"] = card_kappa
    # Unusual flag: large angular distance AND high concentration (card has a strong pattern)
    df["is_unusual_hour_for_card"] = (
        (df["hour_angular_dist_from_self"] > 0.5) & (df["card_hour_concentration"] > 0.5)
    ).astype(int)
```

**When to use:** Any dataset with card + timestamp where you suspect users have stable hour-of-day preferences. Strong signal for account takeover (legitimate user is a 9-5 daytime spender; thief operates at 3 AM). Use as Step 4 in a Per-Scenario Campaign for FDH scenario 1, or as a complement to Recipe 12.

---

## Recipe 3: Behavioral Profiling (Deviation from Self)

Per-card amount and time-of-day profiles. "Is this transaction unusual FOR THIS USER?"

```python
# In fit():
card_col = state.get("card_col")
amt_col = state.get("amt_col")
time_col = state.get("time_col")

if card_col and amt_col:
    card_amt = df_train.groupby(card_col)[amt_col].agg(["mean", "std", "median", "max", "count"])
    state["behav_amt_mean"] = {str(k): float(v) for k, v in card_amt["mean"].items()}
    state["behav_amt_std"] = {str(k): float(v) for k, v in card_amt["std"].fillna(1).items()}
    state["behav_amt_median"] = {str(k): float(v) for k, v in card_amt["median"].items()}
    state["behav_amt_max"] = {str(k): float(v) for k, v in card_amt["max"].items()}
    state["behav_txn_count"] = {str(k): int(v) for k, v in card_amt["count"].items()}

if card_col and time_col:
    df_train["_hour"] = (df_train[time_col] % 86400) / 3600
    hour_stats = df_train.groupby(card_col)["_hour"].agg(["mean", "std"])
    state["behav_hour_mean"] = {str(k): float(v) for k, v in hour_stats["mean"].items()}
    state["behav_hour_std"] = {str(k): float(v) for k, v in hour_stats["std"].fillna(4).items()}

# In transform():
card_str = df[card_col].astype(str)

# Amount deviation from user's own history
card_mean = card_str.map(state.get("behav_amt_mean", {})).fillna(0)
card_std = card_str.map(state.get("behav_amt_std", {})).fillna(1).clip(lower=0.01)
card_max = card_str.map(state.get("behav_amt_max", {})).fillna(0)
card_median = card_str.map(state.get("behav_amt_median", {})).fillna(0)

df["behav_amt_zscore"] = (df[amt_col] - card_mean) / card_std
df["behav_amt_ratio_max"] = df[amt_col] / card_max.clip(lower=0.01)
df["behav_amt_above_median"] = (df[amt_col] > card_median).astype(int)
df["behav_is_max_ever"] = (df[amt_col] >= card_max).astype(int)

# Hour deviation from user's pattern
if time_col in df.columns:
    hour = (df[time_col] % 86400) / 3600
    user_hour_mean = card_str.map(state.get("behav_hour_mean", {})).fillna(12)
    user_hour_std = card_str.map(state.get("behav_hour_std", {})).fillna(4).clip(lower=1)
    df["behav_hour_deviation"] = abs(hour - user_hour_mean) / user_hour_std
```

**When to use**: Any dataset with card + amount. Strongest signal for account takeover.

---

## Recipe 4: Identity Consistency (IEEE-CIS specific)

Measures how stable a user's identity elements are. New device + new email = suspicious.

```python
# In fit():
card_col = "card1"  # IEEE-CIS specific
identity_cols = ["P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo"]
identity_cols = [c for c in identity_cols if c in df_train.columns]

state["identity_profiles"] = {}
for id_col in identity_cols:
    # Most common value per card (the user's "normal")
    modal = df_train.groupby(card_col)[id_col].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None)
    state["identity_profiles"][id_col] = {str(k): str(v) for k, v in modal.dropna().items()}

# Cross-entity sharing counts
for id_col in identity_cols:
    cards_per_entity = df_train.groupby(id_col)[card_col].nunique()
    state[f"n_cards_per_{id_col}"] = {str(k): int(v) for k, v in cards_per_entity.items()}

entities_per_card = {}
for id_col in identity_cols:
    epc = df_train.groupby(card_col)[id_col].nunique()
    entities_per_card[id_col] = {str(k): int(v) for k, v in epc.items()}
state["entities_per_card"] = entities_per_card

# In transform():
card_str = df["card1"].astype(str)

# Identity match flags
for id_col in identity_cols:
    profile = state.get("identity_profiles", {}).get(id_col, {})
    if profile and id_col in df.columns:
        expected = card_str.map(profile)
        df[f"{id_col}_matches_profile"] = (df[id_col].astype(str) == expected).astype(int)

# Entity sharing
for id_col in identity_cols:
    sharing_map = state.get(f"n_cards_per_{id_col}", {})
    if sharing_map and id_col in df.columns:
        df[f"n_cards_sharing_{id_col}"] = df[id_col].astype(str).map(sharing_map).fillna(1)

# Entity diversity per card
for id_col, epc_map in state.get("entities_per_card", {}).items():
    df[f"n_{id_col}_per_card"] = card_str.map(epc_map).fillna(1)

# Composite identity stability score
match_cols = [c for c in df.columns if c.endswith("_matches_profile")]
if match_cols:
    df["identity_stability"] = df[match_cols].mean(axis=1)
```

**When to use**: IEEE-CIS only (has identity columns). This is where the biggest gap vs the full-feature benchmark lives.

---

## Recipe 5: Entity Resolution / Shared Identity

Detect fraud rings by linking accounts that share identity elements.

```python
# In fit():
# Count how many unique cards share each identity element
# High sharing = potential fraud ring
card_col = state.get("card_col")

sharing_features = {}
for col in ["P_emaildomain", "DeviceInfo", "addr1", "merchant"]:
    if col in df_train.columns and card_col:
        cards_per = df_train.groupby(col)[card_col].nunique()
        sharing_features[col] = {str(k): int(v) for k, v in cards_per.items()}
state["entity_sharing"] = sharing_features

# In transform():
for col, sharing_map in state.get("entity_sharing", {}).items():
    if col in df.columns:
        df[f"shared_{col}_count"] = df[col].astype(str).map(sharing_map).fillna(1)
        df[f"shared_{col}_log"] = np.log1p(df[f"shared_{col}_count"])
```

**When to use**: Any dataset with categoricals. Particularly useful for IEEE-CIS (email, device, address sharing).

---

## Recipe 6: Anomaly Score (Mahalanobis Distance)

Stateless anomaly detection using distance from training distribution centroid. JSON-serializable.

```python
# In fit():
num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
# Exclude IDs and time
exclude = {card_col, time_col, "TransactionID", "txn_id"} if card_col else set()
anomaly_cols = [c for c in num_cols if c not in exclude and df_train[c].std() > 0][:20]

col_means = {c: float(df_train[c].mean()) for c in anomaly_cols}
col_stds = {c: float(df_train[c].std()) for c in anomaly_cols}
state["anomaly_cols"] = anomaly_cols
state["anomaly_means"] = col_means
state["anomaly_stds"] = col_stds

# In transform():
anomaly_cols = state.get("anomaly_cols", [])
means = state.get("anomaly_means", {})
stds = state.get("anomaly_stds", {})

if anomaly_cols:
    z_scores = pd.DataFrame()
    for c in anomaly_cols:
        if c in df.columns:
            z_scores[c] = ((df[c].fillna(0) - means.get(c, 0)) / max(stds.get(c, 1), 0.001)) ** 2
    df["anomaly_mahalanobis"] = np.sqrt(z_scores.sum(axis=1))
    df["anomaly_max_zscore"] = z_scores.max(axis=1)
```

**When to use**: Both datasets. Captures "this transaction is unusual" without needing labels.

---

## Recipe 7: Amount Pattern Analysis

Round numbers, threshold testing, corridor analysis.

```python
# Stateless (in transform only):
amt_col = state.get("amt_col")
if amt_col and amt_col in df.columns:
    amt = df[amt_col]
    df["amt_is_round_10"] = (amt % 10 == 0).astype(int)
    df["amt_is_round_100"] = (amt % 100 == 0).astype(int)
    df["amt_cents"] = (amt * 100 % 100).astype(int)
    df["amt_has_cents"] = (df["amt_cents"] != 0).astype(int)
    
# Fitted (in fit):
# Per-category amount corridors
cat_cols = state.get("cat_cols", [])
state["category_amt_corridors"] = {}
for cat in cat_cols[:3]:
    if cat in df_train.columns:
        grp = df_train.groupby(cat)[amt_col].agg(["median", "quantile"])
        # Store median and IQR
        q25 = df_train.groupby(cat)[amt_col].quantile(0.25)
        q75 = df_train.groupby(cat)[amt_col].quantile(0.75)
        state["category_amt_corridors"][cat] = {
            "median": {str(k): float(v) for k, v in grp["median"].items()},
            "q25": {str(k): float(v) for k, v in q25.items()},
            "q75": {str(k): float(v) for k, v in q75.items()},
        }

# In transform:
for cat, corridor in state.get("category_amt_corridors", {}).items():
    if cat in df.columns:
        cat_median = df[cat].astype(str).map(corridor["median"]).fillna(0)
        cat_q25 = df[cat].astype(str).map(corridor["q25"]).fillna(0)
        cat_q75 = df[cat].astype(str).map(corridor["q75"]).fillna(0)
        df[f"{cat}_amt_ratio_median"] = df[amt_col] / cat_median.clip(lower=0.01)
        df[f"{cat}_amt_outside_iqr"] = ((df[amt_col] < cat_q25) | (df[amt_col] > cat_q75)).astype(int)
```

**When to use**: Both datasets. Round number analysis is a known fraud signal.

---

## Smoothing Guide

| Dataset fraud rate | min_samples | smoothing_width | Notes |
|---|---|---|---|
| < 1% (fraud-sim) | 20 | 10 | Less smoothing — let rare signals through |
| 1-5% (IEEE-CIS) | 50 | 20 | Moderate smoothing — balance bias/variance |
| > 5% | 100 | 30 | More smoothing — prevent overfitting to large class |

---

## Model Recipes

### Focal Loss for XGBoost

```python
import xgboost as xgb

def focal_loss_obj(y_pred, dtrain, gamma=2.0):
    y_true = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-y_pred))
    grad = p - y_true + gamma * (y_true * (1 - p) ** gamma * np.log(np.maximum(p, 1e-7)) * p
           - (1 - y_true) * p ** gamma * np.log(np.maximum(1 - p, 1e-7)) * (1 - p))
    hess = np.maximum(p * (1 - p), 1e-7)
    return grad, hess

# Usage in model.py:
model = xgb.XGBClassifier(
    objective=lambda y_pred, dtrain: focal_loss_obj(y_pred, dtrain, gamma=2.0),
    n_estimators=1500, max_depth=7, learning_rate=0.03,
    ...
)
```

### GPU Auto-Detection

GPU is auto-detected. Use `get_gpu_info()` in model.py for any custom XGBoost models:

```python
from harness.utils import get_gpu_info
gpu = get_gpu_info()
# gpu = {"available": True, "device": "cuda", "tree_method": "hist", "gpu_name": "..."}

model = xgb.XGBClassifier(
    tree_method=gpu["tree_method"], device=gpu["device"], ...
)
```

### Subsampled Ensemble (with GPU)

```python
from harness.utils import get_gpu_info

def train_and_evaluate(X_train, y_train, X_val, y_val, X_oot, y_oot, config):
    from sklearn.utils import resample
    gpu = get_gpu_info()
    
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    
    models = []
    for ratio in [3, 10, 30]:
        n_neg = min(len(pos_idx) * ratio, len(neg_idx))
        neg_sample = resample(neg_idx, n_samples=n_neg, random_state=42 + ratio)
        idx = np.concatenate([pos_idx, neg_sample])
        
        model = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            tree_method=gpu["tree_method"], device=gpu["device"], ...
        )
        model.fit(X_train.iloc[idx], y_train[idx], eval_set=[(X_val, y_val)], verbose=False)
        models.append(model)
    
    val_preds = np.mean([m.predict_proba(X_val)[:, 1] for m in models], axis=0)
    oot_preds = np.mean([m.predict_proba(X_oot)[:, 1] for m in models], axis=0)
    return {"y_val_pred": val_preds, "y_oot_pred": oot_preds, "model": models[0], ...}
```

---

## Recipe 8: HMM State Probability Features

Per-card sequence features using a Hidden Markov Model. Captures lifecycle trajectory patterns that per-transaction features miss. Requires `hmmlearn` (`pip install hmmlearn`).

```python
# In fit():
try:
    from hmmlearn import hmm as hmmlib
    import numpy as np

    card_col = state.get("card_col")
    amt_col = state.get("amt_col")
    time_col = state.get("time_col")
    N_STATES = 4  # e.g., normal / warm-up / escalation / bust-out

    if card_col and amt_col and time_col:
        df_s = df_train.sort_values([card_col, time_col]).copy()
        df_s["_log_amt"] = np.log1p(df_s[amt_col].fillna(0))
        df_s["_hour_n"] = (df_s[time_col] % 86400) / 86400.0
        df_s["_dow_n"] = (df_s[time_col] // 86400 % 7) / 7.0

        hmm_feat_cols = ["_log_amt", "_hour_n", "_dow_n"]

        seqs, lengths = [], []
        for _, grp in df_s.groupby(card_col, sort=False):
            if len(grp) >= 3:
                seqs.append(grp[hmm_feat_cols].values.astype(float))
                lengths.append(len(grp))

        if len(seqs) >= 10:
            X = np.vstack(seqs)
            model = hmmlib.GaussianHMM(
                n_components=N_STATES, covariance_type="diag",
                n_iter=50, random_state=42, verbose=False,
            )
            model.fit(X, lengths)

            state["hmm_n_states"] = N_STATES
            state["hmm_feat_cols"] = hmm_feat_cols
            state["hmm_startprob"] = model.startprob_.tolist()
            state["hmm_transmat"] = model.transmat_.tolist()
            state["hmm_means"] = model.means_.tolist()
            state["hmm_covars"] = model.covars_.tolist()
except ImportError:
    pass

# In transform():
if "hmm_n_states" in state:
    try:
        from hmmlearn import hmm as hmmlib
        import numpy as np

        card_col = state.get("card_col")
        amt_col = state.get("amt_col")
        time_col = state.get("time_col")
        N_STATES = state["hmm_n_states"]
        hmm_feat_cols = state["hmm_feat_cols"]

        # Reconstruct model from serialized parameters
        model = hmmlib.GaussianHMM(n_components=N_STATES, covariance_type="diag")
        model.startprob_ = np.array(state["hmm_startprob"])
        model.transmat_ = np.array(state["hmm_transmat"])
        model.means_ = np.array(state["hmm_means"])
        model.covars_ = np.array(state["hmm_covars"])
        model.n_features = len(hmm_feat_cols)

        df_s = df.sort_values([card_col, time_col]).copy()
        df_s["_log_amt"] = np.log1p(df_s[amt_col].fillna(0))
        df_s["_hour_n"] = (df_s[time_col] % 86400) / 86400.0
        df_s["_dow_n"] = (df_s[time_col] // 86400 % 7) / 7.0

        hmm_state_cols = [f"hmm_state_{i}_prob" for i in range(N_STATES)]
        results = {}  # orig_idx -> state posteriors row

        for _, grp in df_s.groupby(card_col, sort=False):
            seq = grp[hmm_feat_cols].values.astype(float)
            if len(grp) >= 2:
                _, posteriors = model.score_samples(seq)
            else:
                posteriors = np.tile(model.startprob_, (len(grp), 1))
            for i, orig_idx in enumerate(grp.index):
                results[orig_idx] = posteriors[i]

        prob_matrix = np.array([results.get(i, model.startprob_) for i in df.index])
        for s in range(N_STATES):
            df[f"hmm_state_{s}_prob"] = prob_matrix[:, s]
        df["hmm_most_likely_state"] = np.argmax(prob_matrix, axis=1).astype(float)
        df["hmm_state_entropy"] = -np.sum(
            np.where(prob_matrix > 0, prob_matrix * np.log(prob_matrix + 1e-10), 0), axis=1
        )
    except Exception:
        pass
```

**When to use**: ATO detection, bust-out trajectory detection, any dataset with card + timestamp + amount. Needs ≥5 transactions per card for reliable posteriors. Add `hmm_n_obs` from `behav_txn_count` (Recipe 3) as a reliability weight.

---

## Recipe 9: Feature-Group Autoencoder Reconstruction Error

Train per-group autoencoders (amount group, time group, identity group) on fraud-negative training transactions. High reconstruction error = anomalous for that feature group. Weights serialized as nested lists (JSON-serializable).

```python
# Helper — add outside fit/transform (top of features file):
import numpy as np

def _relu(x):
    return np.maximum(0, x)

def _ae_forward(X_np, coefs_list, biases_list):
    """Forward pass: ReLU activations, linear output layer."""
    a = X_np.astype(float)
    for i, (W, b) in enumerate(zip(coefs_list, biases_list)):
        z = a @ np.array(W) + np.array(b)
        a = _relu(z) if i < len(coefs_list) - 1 else z
    return a

def _train_ae(X_np, bottleneck=4, n_iter=30, lr=0.01, random_state=42):
    """Minimal MLP autoencoder: input -> hidden -> bottleneck -> hidden -> input."""
    rng = np.random.RandomState(random_state)
    n_feat = X_np.shape[1]
    n_hidden = max(n_feat // 2, bottleneck + 1)

    W1 = rng.randn(n_feat, n_hidden) * 0.1;     b1 = np.zeros(n_hidden)
    W2 = rng.randn(n_hidden, bottleneck) * 0.1;  b2 = np.zeros(bottleneck)
    W3 = rng.randn(bottleneck, n_hidden) * 0.1;  b3 = np.zeros(n_hidden)
    W4 = rng.randn(n_hidden, n_feat) * 0.1;      b4 = np.zeros(n_feat)

    for _ in range(n_iter):
        # Forward
        h1 = _relu(X_np @ W1 + b1)
        h2 = _relu(h1 @ W2 + b2)
        h3 = _relu(h2 @ W3 + b3)
        out = h3 @ W4 + b4
        err = out - X_np
        # Backward (MSE gradient, simplified)
        dout = 2 * err / len(X_np)
        dW4 = h3.T @ dout;       db4 = dout.mean(axis=0)
        dh3 = dout @ W4.T * (h3 > 0)
        dW3 = h2.T @ dh3;        db3 = dh3.mean(axis=0)
        dh2 = dh3 @ W3.T * (h2 > 0)
        dW2 = h1.T @ dh2;        db2 = dh2.mean(axis=0)
        dh1 = dh2 @ W2.T * (h1 > 0)
        dW1 = X_np.T @ dh1;      db1 = dh1.mean(axis=0)
        # Update
        for W, dW in [(W1,dW1),(W2,dW2),(W3,dW3),(W4,dW4)]:
            W -= lr * dW
        for b, db in [(b1,db1),(b2,db2),(b3,db3),(b4,db4)]:
            b -= lr * db

    coefs = [W1.tolist(), W2.tolist(), W3.tolist(), W4.tolist()]
    biases = [b1.tolist(), b2.tolist(), b3.tolist(), b4.tolist()]
    return coefs, biases


# In fit():
amt_col = state.get("amt_col")
time_col = state.get("time_col")

# Define feature groups (adjust column names to your dataset)
ae_groups = {}
if amt_col and amt_col in df_train.columns:
    ae_groups["amount"] = [c for c in [
        amt_col, f"{amt_col}_log" if f"{amt_col}_log" not in df_train.columns else None,
        "amt_cents", "amt_is_round_10",
    ] if c and c in df_train.columns]
if time_col and time_col in df_train.columns:
    ae_groups["time"] = [c for c in [
        "hour_of_day", "day_of_week", "is_night",
    ] if c in df_train.columns]

state["ae_groups"] = {}
for group_name, cols in ae_groups.items():
    if len(cols) < 2:
        continue
    neg_mask = (y_train == 0)
    X_raw = df_train.loc[neg_mask, cols].fillna(0).values.astype(float)
    if len(X_raw) < 100:
        continue
    # Standardize
    col_mean = X_raw.mean(axis=0).tolist()
    col_std = np.maximum(X_raw.std(axis=0), 0.001).tolist()
    X_scaled = (X_raw - np.array(col_mean)) / np.array(col_std)
    # Train
    coefs, biases = _train_ae(X_scaled, bottleneck=max(2, len(cols)//3))
    state["ae_groups"][group_name] = {
        "cols": cols,
        "mean": col_mean,
        "std": col_std,
        "coefs": coefs,
        "biases": biases,
    }

# In transform():
for group_name, ae_state in state.get("ae_groups", {}).items():
    cols = ae_state["cols"]
    available = [c for c in cols if c in df.columns]
    if len(available) < 2:
        continue
    X_raw = df[available].fillna(0).values.astype(float)
    mean_ = np.array(ae_state["mean"][:len(available)])
    std_ = np.array(ae_state["std"][:len(available)])
    X_scaled = (X_raw - mean_) / std_
    X_recon = _ae_forward(X_scaled, ae_state["coefs"], ae_state["biases"])
    recon_err = np.mean((X_scaled - X_recon) ** 2, axis=1)
    df[f"ae_{group_name}_recon_error"] = recon_err
    df[f"ae_{group_name}_recon_log"] = np.log1p(recon_err)
    df[f"ae_{group_name}_max_feat_err"] = np.max((X_scaled - X_recon) ** 2, axis=1)
```

**When to use**: Both datasets. Best added after behavioral profiling has plateaued — the autoencoder catches anomalies that don't map to a specific known feature pattern. Train on fraud-negative rows only (in fit()) so the AE learns "normal" and flags fraud as high-error.

---

## Recipe 10: CUSUM Behavioral Shift Detection

Sequential change detection — accumulates evidence that a behavioral shift has occurred. Catches gradual escalation that per-transaction thresholds miss.

```python
# In fit():
card_col = state.get("card_col")
amt_col = state.get("amt_col")
time_col = state.get("time_col")

if card_col and amt_col and time_col:
    df_s = df_train.sort_values([card_col, time_col])
    # Per-card baseline: mean and std from first half of transactions
    baselines = {}
    for card, grp in df_s.groupby(card_col, sort=False):
        n_base = max(2, len(grp) // 2)
        baseline_grp = grp.head(n_base)
        mu = float(baseline_grp[amt_col].mean())
        sigma = float(baseline_grp[amt_col].std()) if len(baseline_grp) > 1 else 1.0
        baselines[str(card)] = {
            "mu": mu,
            "sigma": max(sigma, 0.01 * abs(mu) + 1.0),
            "n": int(len(grp)),
        }
    state["cusum_baselines"] = baselines
    state["cusum_k"] = 0.5  # slack: ignore deviations < 0.5σ

# In transform():
if "cusum_baselines" in state:
    card_col = state.get("card_col")
    amt_col = state.get("amt_col")
    time_col = state.get("time_col")
    k = state.get("cusum_k", 0.5)

    if card_col in df.columns and amt_col in df.columns and time_col in df.columns:
        df_s = df.sort_values([card_col, time_col]).copy()

        cusum_score = np.zeros(len(df_s))
        cusum_slope = np.zeros(len(df_s))

        pos_map = {orig_idx: pos for pos, orig_idx in enumerate(df_s.index)}

        for card, grp in df_s.groupby(card_col, sort=False):
            baseline = state["cusum_baselines"].get(str(card))
            if baseline is None:
                continue
            mu = baseline["mu"]
            sigma = baseline["sigma"]
            amounts = grp[amt_col].fillna(mu).values

            cusum = np.zeros(len(grp))
            for i in range(len(grp)):
                z = (amounts[i] - mu) / sigma
                prev = cusum[i - 1] if i > 0 else 0.0
                cusum[i] = max(0.0, prev + z - k)
            
            slope = np.diff(cusum, prepend=cusum[0])

            for i, orig_idx in enumerate(grp.index):
                p = pos_map[orig_idx]
                cusum_score[p] = cusum[i]
                cusum_slope[p] = slope[i]

        df_s["cusum_score"] = cusum_score
        df_s["cusum_slope"] = cusum_slope
        df_s["cusum_triggered"] = (cusum_score > 5.0).astype(float)

        for col in ["cusum_score", "cusum_slope", "cusum_triggered"]:
            df[col] = df_s[col].reindex(df.index).fillna(0)
```

**When to use**: ATO escalation detection, synthetic identity limit-testing. Works best with amount column but can be applied to any numeric feature (e.g., `velocity_daily_rate`, `behav_hour_deviation`). The threshold of 5.0 is standard for CUSUM — tune as a hyperparameter.

---

## Recipe 11: RFM Cluster Distance Features

Compute each card's RFM vector and measure distance to K-means centroids fitted on fraud-negative training cards. Cards whose RFM sits far from any legitimate cluster centroid are suspicious.

```python
# In fit():
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

card_col = state.get("card_col")
amt_col = state.get("amt_col")
time_col = state.get("time_col")

if card_col and amt_col and time_col:
    last_time = float(df_train[time_col].max())

    rfm_df = df_train.groupby(card_col).agg(
        recency=(time_col, lambda x: last_time - float(x.max())),
        frequency=(time_col, "count"),
        monetary=(amt_col, "mean"),
    ).reset_index()

    # Fit only on fraud-negative cards
    neg_cards = df_train.loc[y_train == 0, card_col].unique()
    neg_rfm = rfm_df[rfm_df[card_col].isin(neg_cards)][["recency","frequency","monetary"]].fillna(0)

    if len(neg_rfm) >= 16:
        sc = StandardScaler()
        neg_scaled = sc.fit_transform(neg_rfm.values)
        n_clusters = min(8, len(neg_rfm) // 2)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        km.fit(neg_scaled)

        state["rfm_sc_mean"] = sc.mean_.tolist()
        state["rfm_sc_std"] = sc.scale_.tolist()
        state["rfm_centroids"] = km.cluster_centers_.tolist()

        # Store per-card RFM (recency, freq, monetary) for transform lookup
        state["rfm_per_card"] = {
            str(row[card_col]): [
                float(row["recency"]), float(row["frequency"]), float(row["monetary"])
            ]
            for _, row in rfm_df.iterrows()
        }
        state["rfm_global"] = [
            float(rfm_df["recency"].median()),
            float(rfm_df["frequency"].median()),
            float(rfm_df["monetary"].median()),
        ]

# In transform():
if "rfm_centroids" in state:
    import numpy as np
    card_col = state.get("card_col")
    if card_col and card_col in df.columns:
        centroids = np.array(state["rfm_centroids"])
        sc_mean = np.array(state["rfm_sc_mean"])
        sc_std = np.array(state["rfm_sc_std"])
        global_rfm = np.array(state["rfm_global"])

        rfm_lookup = state["rfm_per_card"]

        card_strs = df[card_col].astype(str)
        rfm_vectors = np.array([
            rfm_lookup.get(c, global_rfm.tolist()) for c in card_strs
        ], dtype=float)
        rfm_scaled = (rfm_vectors - sc_mean) / np.maximum(sc_std, 1e-6)

        # Distance to nearest legitimate centroid
        dists = np.linalg.norm(rfm_scaled[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        df["rfm_nearest_cluster_dist"] = dists.min(axis=1)
        df["rfm_assigned_cluster"] = dists.argmin(axis=1).astype(float)
        df["rfm_min_cluster_dist_log"] = np.log1p(df["rfm_nearest_cluster_dist"])
```

**When to use**: Bust-out detection, synthetic identity profiling. Most powerful when the dataset has long enough card history to build stable RFM vectors. Use `behav_txn_count` as a confidence mask — don't trust RFM for cards with <5 transactions.


## Recipe 12: Cyclic Time Encoding (sin/cos)

Stateless transform — no `fit()` work needed beyond identifying the time column.

```python
# In fit():
state["time_col"] = "TX_DATETIME"  # or whatever timestamp column exists

# In transform() — pure transformation, no state lookup:
import numpy as np
time_col = state.get("time_col")
if time_col and time_col in df.columns:
    # Assume Unix seconds; for datetime strings, convert first
    secs = df[time_col].astype(float)
    hour = (secs % 86400) / 3600
    dow = ((secs // 86400) % 7).astype(float)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
```

**When to use**: Any dataset with a timestamp. Tree models can technically split on raw `hour` and `day_of_week` integers, but cyclic encoding helps when (a) you stack a linear model on top, (b) you compute distances between transactions, or (c) hour-of-day is highly predictive (e.g. card-not-present fraud peaks at 2am). Cheap; usually worth adding alongside raw time fields, not instead of them.


## Recipe 13: Geo-Distance from Customer Centroid (Haversine)

```python
# In fit():
import numpy as np
lat_col = "lat"
lon_col = "long"
card_col = "cc_num"  # or whatever entity column groups locations
if all(c in df_train.columns for c in [lat_col, lon_col, card_col]):
    geo = df_train.groupby(card_col)[[lat_col, lon_col]].median()
    state["geo_lat_col"] = lat_col
    state["geo_lon_col"] = lon_col
    state["geo_card_col"] = card_col
    state["geo_centroid_lat"] = {str(k): float(v) for k, v in geo[lat_col].items()}
    state["geo_centroid_lon"] = {str(k): float(v) for k, v in geo[lon_col].items()}

# In transform():
if "geo_centroid_lat" in state:
    import numpy as np
    lat_col = state["geo_lat_col"]
    lon_col = state["geo_lon_col"]
    card_col = state["geo_card_col"]
    if all(c in df.columns for c in [lat_col, lon_col, card_col]):
        card_str = df[card_col].astype(str)
        clat = card_str.apply(lambda x: state["geo_centroid_lat"].get(x, 0.0)).astype(float)
        clon = card_str.apply(lambda x: state["geo_centroid_lon"].get(x, 0.0)).astype(float)
        # Haversine on Earth's surface (km)
        lat1 = np.radians(clat)
        lat2 = np.radians(df[lat_col].astype(float))
        dlat = lat2 - lat1
        dlon = np.radians(df[lon_col].astype(float) - clon)
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        df["geo_dist_from_home_km"] = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        df["geo_dist_from_home_log"] = np.log1p(df["geo_dist_from_home_km"])
        df["geo_far_from_home"] = (df["geo_dist_from_home_km"] > 200).astype(int)
```

**When to use**: Datasets with per-transaction lat/lon (Sparkov-style fraud-sim). Haversine is the right distance for short-range Earth surface; use `.apply(lambda x: dict.get(x, 0))` instead of `.map(dict)` when the centroid dict has >100k keys to avoid the pandas O(N) lookup penalty. Don't use for IEEE-CIS (no geo cols) or FDH (no geo).


## Recipe 14: Terminal/Merchant Risk with Smoothed Fraud Rate

Critical for FDH terminal-compromise detection (fraud scenario 2). Computes a smoothed fraud rate per terminal using train-only labels, then exposes it to scoring without leakage.

```python
# In fit():
term_col = "TERMINAL_ID"  # or "merchant", "merchant_id" — whatever exists
if term_col in df_train.columns:
    global_rate = float(y_train.mean())
    state["term_col"] = term_col
    state["global_fraud_rate"] = global_rate

    df_tmp = df_train.copy()
    df_tmp["_y"] = y_train.values
    term_stats = df_tmp.groupby(term_col).agg(
        cnt=("_y", "count"),
        frauds=("_y", "sum"),
    )
    term_stats["raw_rate"] = term_stats["frauds"] / term_stats["cnt"].clip(lower=1)

    # Bayesian smoothing: shrink toward global rate when terminal has few obs
    min_samples = 20
    weight = term_stats["cnt"] / (term_stats["cnt"] + min_samples)
    term_stats["smooth_rate"] = weight * term_stats["raw_rate"] + (1 - weight) * global_rate

    state["term_fraud_rate"] = {
        str(k): float(v) for k, v in term_stats["smooth_rate"].items()
    }
    state["term_txn_count"] = {
        str(k): int(v) for k, v in term_stats["cnt"].items()
    }

# In transform():
term_col = state.get("term_col")
if term_col and term_col in df.columns:
    global_rate = state.get("global_fraud_rate", 0.01)
    rate_lookup = state.get("term_fraud_rate", {})
    cnt_lookup = state.get("term_txn_count", {})
    term_str = df[term_col].astype(str)
    # Use .apply not .map for large dicts
    df["term_fraud_rate"] = term_str.apply(lambda x: rate_lookup.get(x, global_rate)).astype(float)
    df["term_txn_volume"] = term_str.apply(lambda x: cnt_lookup.get(x, 0)).astype(float)
    df["term_is_high_risk"] = (df["term_fraud_rate"] > global_rate * 3).astype(int)
    df["term_log_volume"] = np.log1p(df["term_txn_volume"])
```

**When to use**: Any dataset with a merchant or terminal identifier — FDH (TERMINAL_ID), fraud-sim (merchant), most real-world transactional data. Critical for FDH scenario 2 (terminal compromise). The smoothing prevents brand-new terminals from getting flagged as fraud-rate=0 and stable terminals with 1 historical fraud from getting flagged at 100%. Pair with **Recipe 1 (velocity)** to capture the time-windowed compromise pattern, and **Recipe 7 (target encoding)** if you need a richer encoding of the same signal.


## When All Recipes Are Exhausted

If you've tried every recipe above and you're on a discard streak, the next moves are:

1. **Ablate**: Drop your lowest-importance features one at a time. The signal-to-noise ratio of the input matrix matters more than feature count after a certain point. Use the "Features with negligible importance" warning in the experiment context as a starting list.
2. **Interactions**: Manually create pairwise products/ratios of your top-5 features. XGBoost finds first-order splits well but can miss interactions when one feature is much stronger than the other.
3. **Window tuning**: For velocity and behavioral recipes, try different lookback windows (1h vs 6h vs 24h vs 7d). The right window is dataset-dependent — FDH benefits from 28-day windows for terminal compromise; CNP fraud benefits from 1h windows.
4. **Model surgery**: Try a deeper tree (max_depth 6→10), a different learning rate, or a small rank-average ensemble of XGBoost + LightGBM. Don't change too many hyperparameters at once.
5. **Re-read fraud_practices.md Part 7** (advanced techniques) for ideas not yet in recipes — sequence models, GNN-style entity features, anomaly scores from autoencoders, etc.
6. **Cross-validate your selection**: If val and OOT keep diverging, your val set may be drifting. Check the per-feature PSI warnings in the experiment context — high-PSI features destabilize the model even when their training importance is high.

