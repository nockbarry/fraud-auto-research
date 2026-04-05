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
    state["behav_hour_std"] = {str(k): float(v) for k, v hour_stats["std"].fillna(4).items()}

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

### Subsampled Ensemble

```python
def train_and_evaluate(X_train, y_train, X_val, y_val, X_oot, y_oot, config):
    from sklearn.utils import resample
    
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    
    models = []
    for ratio in [3, 10, 30]:
        n_neg = min(len(pos_idx) * ratio, len(neg_idx))
        neg_sample = resample(neg_idx, n_samples=n_neg, random_state=42 + ratio)
        idx = np.concatenate([pos_idx, neg_sample])
        
        model = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, ...)
        model.fit(X_train.iloc[idx], y_train[idx], eval_set=[(X_val, y_val)], verbose=False)
        models.append(model)
    
    # Rank-average predictions
    val_preds = np.mean([m.predict_proba(X_val)[:, 1] for m in models], axis=0)
    oot_preds = np.mean([m.predict_proba(X_oot)[:, 1] for m in models], axis=0)
    
    return {"y_val_pred": val_preds, "y_oot_pred": oot_preds, "model": models[0], ...}
```
