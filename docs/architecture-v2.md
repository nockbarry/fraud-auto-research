# Architecture v2: Leakage Prevention + Multi-Dataset Results

## 1. Leakage-Safe Feature API

### Current (v1) — leaky
```python
def transform(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    # Agent sees label in all splits, can compute target stats
```

### Proposed (v2) — fit/transform separation
```python
def fit(df_train: pd.DataFrame, y_train: pd.Series, config: dict) -> dict:
    """Fit feature engineering state on training data ONLY.
    
    Args:
        df_train: Training features (NO label column).
        y_train: Training labels (separate Series).
        config: Config dict.
    
    Returns:
        state: dict of fitted parameters (encodings, scalers, means, etc.)
              Must be JSON-serializable for auditability.
    """
    state = {}
    
    # Example: target encoding fitted on train
    global_mean = y_train.mean()
    state["global_mean"] = global_mean
    
    df_train["_label"] = y_train.values
    for col in ["card1", "addr1"]:
        stats = df_train.groupby(col)["_label"].agg(["mean", "count"])
        smoothing = 1 / (1 + np.exp(-(stats["count"] - 20) / 10))
        te = smoothing * stats["mean"] + (1 - smoothing) * global_mean
        state[f"{col}_target_enc"] = te.to_dict()
    
    return state


def transform(df: pd.DataFrame, state: dict, config: dict) -> pd.DataFrame:
    """Apply pre-fitted transforms. NO access to labels.
    
    Args:
        df: Features for any split (NO label column).
        state: Fitted state from fit().
        config: Config dict.
    
    Returns:
        Transformed DataFrame. All features numeric. Row count preserved.
    """
    df = df.copy()
    global_mean = state["global_mean"]
    
    for col in ["card1", "addr1"]:
        te_map = state.get(f"{col}_target_enc", {})
        df[f"{col}_target_enc"] = df[col].map(te_map).fillna(global_mean)
    
    return df
```

### Harness enforcement (evaluate.py changes)
```python
# Strip label before passing to agent code
y_train = df_train.pop("label")
y_val = df_val.pop("label")
y_oot = df_oot.pop("label")

# Fit on train only
state = features.fit(df_train, y_train, config)

# Transform all splits with fitted state (no labels visible)
df_train = features.transform(df_train, state, config)
df_val = features.transform(df_val, state, config)
df_oot = features.transform(df_oot, state, config)
```

### What this prevents
- Agent cannot compute target statistics on val/OOT (no labels passed)
- Agent cannot compute population statistics across splits (called separately)
- State is serializable and auditable (can inspect what was fitted)
- State can be saved to disk for production deployment

---

## 2. Automated Leakage Detection

Add to `harness/validate_features.py`:

### Feature-level leakage checks
```python
def check_leakage(df_train, y_train, df_val, y_val, config):
    """Detect potential feature leakage."""
    warnings = []
    
    # 1. Suspiciously high single-feature AUC
    from sklearn.metrics import roc_auc_score
    for col in df_train.columns:
        try:
            auc = roc_auc_score(y_val, df_val[col].fillna(0))
            if auc > 0.90 or auc < 0.10:  # >0.9 or <0.1 (inverted)
                warnings.append(f"LEAKAGE WARNING: {col} has AUC={auc:.4f} on val — suspiciously predictive")
        except:
            pass
    
    # 2. Perfect separation check
    for col in df_train.columns:
        if df_val[col].nunique() <= 2:
            fraud_rate_0 = y_val[df_val[col] == df_val[col].min()].mean()
            fraud_rate_1 = y_val[df_val[col] == df_val[col].max()].mean()
            if abs(fraud_rate_0 - fraud_rate_1) > 0.5:
                warnings.append(f"LEAKAGE WARNING: {col} nearly separates classes")
    
    # 3. Feature not available at prediction time
    # (requires a manifest of available-at-scoring features)
    
    return warnings
```

### Temporal consistency check
```python
def check_temporal_consistency(df, y, state, config):
    """Shuffle time ordering and check if metric changes suspiciously."""
    # If features are time-independent, shuffling rows shouldn't matter
    # If metric IMPROVES after shuffle, temporal leakage likely
    pass
```

---

## 3. Multi-Dataset Results Storage

### Directory structure
```
results/
├── index.json                          # Registry of all experiments
├── ieee-cis/                           # Dataset name
│   ├── dataset.json                    # Dataset metadata (source, description, columns)
│   ├── all_transactions/               # Segment name
│   │   ├── experiments.jsonl           # One JSON object per experiment (append-only)
│   │   ├── best_state.json             # Fitted state from best experiment
│   │   ├── best_features.py            # Snapshot of features.py at best score
│   │   ├── best_model.py              # Snapshot of model.py at best score  
│   │   └── plots/
│   │       ├── metrics_over_time.png
│   │       └── feature_importance.png
│   ├── new_customers/                  # Another segment
│   │   ├── experiments.jsonl
│   │   └── ...
│   └── high_value/
│       └── ...
├── internal-fraud/                     # Another dataset
│   ├── dataset.json
│   ├── ato/
│   │   └── ...
│   └── cnp/
│       └── ...
└── synthetic-benchmark/
    └── ...
```

### experiments.jsonl format (one line per experiment)
```json
{
  "id": "exp_001",
  "timestamp": "2026-04-05T14:23:00Z",
  "git_commit": "ab445bf",
  "dataset": "ieee-cis",
  "segment": "all_transactions",
  "hypothesis": "add smoothed target encoding for card1, card2, card5, addr1",
  "experiment_type": "feature_engineering",
  "files_changed": ["features.py"],
  "status": "keep",
  "metrics": {
    "composite_score": 0.2683,
    "auprc_oot": 0.4377,
    "auprc_val": 0.4510,
    "precision_at_recall": 0.1648,
    "target_recall": 0.80,
    "psi": 0.0050,
    "fpr": 0.0320,
    "review_burden": 5.2,
    "n_features": 62,
    "training_seconds": 4.1
  },
  "leakage_warnings": [],
  "feature_summary": {
    "top_5_by_importance": ["card1_target_enc", "uid_target_enc", "addr1_target_enc", "card1_email_target_enc", "TransactionAmt"],
    "n_added": 4,
    "n_removed": 0
  },
  "config_snapshot": {
    "metrics.target_recall": 0.80,
    "validation.max_nan_rate": 0.50
  },
  "parent_experiment": "exp_000",
  "is_sota": true
}
```

### index.json (registry)
```json
{
  "datasets": {
    "ieee-cis": {
      "description": "Vesta CNP fraud, 590K txns, derived features stripped",
      "segments": ["all_transactions", "new_customers", "high_value"],
      "total_experiments": 120,
      "best_auprc": 0.7954
    },
    "internal-fraud": {
      "description": "Internal platform transaction fraud",
      "segments": ["ato", "cnp", "first_party"],
      "total_experiments": 85,
      "best_auprc": null
    }
  }
}
```

### Why JSONL over SQLite or TSV

| | TSV | SQLite | JSONL |
|---|---|---|---|
| Append-only writes | Yes | Yes | Yes |
| Human readable | Yes | No | Yes |
| Git-friendly diffs | Yes | No | Yes |
| Nested data (metrics, features) | No | Awkward | Natural |
| Query across datasets | No | Yes | Via jq or pandas |
| Concurrent writes | Fragile | Yes | Append-safe |
| Schema evolution | Breaking | Migration needed | Flexible |

JSONL is the sweet spot: human-readable, git-friendly, supports nested experiment metadata, and trivially loadable with `pd.read_json(path, lines=True)`. 

For cross-dataset analysis, load all JSONL files into a single DataFrame:
```python
import glob, pandas as pd
dfs = [pd.read_json(f, lines=True) for f in glob.glob("results/*/experiments.jsonl")]
all_experiments = pd.concat(dfs)
```

---

## 4. Production Environment Simulation Checklist

The harness should enforce these conditions:

### At data loading time
- [ ] Train/val/OOT are strictly time-ordered (no temporal overlap)
- [ ] Label maturation gap respected (OOT end + lag < today)
- [ ] No future data in any split

### At feature engineering time
- [ ] fit() only sees training data and labels
- [ ] transform() never sees labels
- [ ] Each split transformed independently (no cross-split stats)
- [ ] State is serializable (can be deployed to production)

### At evaluation time
- [ ] Leakage detector runs on val set
- [ ] Features with AUC > 0.90 flagged
- [ ] PSI check between val and OOT (distribution shift)
- [ ] Single-row transform latency measured (scoring SLA)

### At results storage time
- [ ] Full config snapshot saved with each experiment
- [ ] Feature importance saved
- [ ] Leakage warnings recorded
- [ ] Code snapshot of features.py and model.py saved
