# fraud-auto-research

Autonomous feature engineering and model evaluation for transaction fraud monitoring, inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch). An LLM agent proposes hypotheses, implements features, evaluates results, and iterates — the harness just keeps score.

## How It Works

The system has two halves:

1. **The harness** (`harness/`) — fixed infrastructure that loads data, validates features, trains models, computes metrics, tracks experiments, and generates dashboards. The agent cannot modify this.

2. **The editable files** (`features_{dataset}.py`, `model_{dataset}.py`) — the agent's workspace. It edits these to engineer features and tune models, then runs the harness to see if the change improved things.

Each iteration takes 30–120 seconds. The agent reads structured feedback (what worked, what didn't, which techniques are untried, which features are growing), proposes a hypothesis, implements it, and evaluates. No human in the loop.

```
LOOP:
  1. Read experiment context (SOTA, history, technique success rates)
  2. Propose hypothesis ("add merchant x category amount corridors")
  3. Implement — edit features_{dataset}.py and/or model_{dataset}.py
  4. Run: python3 -m harness.evaluate --config configs/{dataset}.yaml --save --hypothesis "..."
  5. Harness auto-determines keep/discard, saves code snapshot, updates SOTA
  6. Read feedback, repeat
```

## Project Structure

```
fraud-auto-research/
├── program.md                  # Agent instructions — the full prompt
├── recipes.md                  # Copy-paste feature engineering patterns (fit/transform)
├── config.yaml                 # Default config
│
├── configs/
│   ├── ieee-cis.yaml           # IEEE-CIS dataset config + profile
│   └── fraud-sim.yaml          # Fraud-sim dataset config + profile
│
├── features_ieee.py            # AGENT-EDITABLE — IEEE-CIS feature transforms
├── features_sim.py             # AGENT-EDITABLE — fraud-sim feature transforms
├── model_ieee.py               # AGENT-EDITABLE — IEEE-CIS model definition
├── model_sim.py                # AGENT-EDITABLE — fraud-sim model definition
│
├── harness/                    # READ-ONLY — evaluation infrastructure
│   ├── evaluate.py             # Full pipeline: load → fit → transform → validate → train → metrics
│   ├── experiment_tracker.py   # Directory-per-experiment tracking, SOTA symlinks, index.jsonl
│   ├── context.py              # Structured agent memory: history, success rates, recommendations
│   ├── dashboard.py            # Auto-updating HTML dashboard with embedded plots
│   ├── plot_results.py         # Per-dataset annotated metric plots
│   ├── data_loader.py          # Parquet loading, train/val/OOT splitting
│   ├── validate_features.py    # Feature validation (NaN rates, schema alignment, count limits)
│   ├── feature_analysis.py     # IV, PSI per feature, correlation analysis
│   └── utils.py                # Config loader, GPU detection, file hashing
│
├── experiments/                # Auto-generated — one directory per experiment per dataset
│   ├── ieee-cis/
│   │   ├── exp_000_baseline/
│   │   │   ├── features.py     # Code snapshot
│   │   │   ├── model.py        # Code snapshot
│   │   │   ├── metrics.json    # Full metrics + feature importances + CIs
│   │   │   ├── state.json      # Fitted feature state (deployable artifact)
│   │   │   └── metadata.json   # Hypothesis, status, timestamp, parent
│   │   ├── exp_001_.../
│   │   ├── index.jsonl         # Append-only experiment log
│   │   └── sota -> exp_001_... # Symlink to current best
│   └── fraud-sim/
│       └── ...
│
├── data/                       # Source parquet files (not in repo)
│   ├── raw_train.parquet
│   ├── raw_val.parquet
│   ├── raw_oot.parquet
│   └── fraud-sim/
│       ├── raw_train.parquet
│       ├── raw_val.parquet
│       └── raw_oot.parquet
│
├── reports/                    # Auto-generated dashboard and plots
│   ├── dashboard.html          # Self-contained HTML with embedded plot images
│   ├── plot_ieee-cis.png
│   └── plot_fraud-sim.png
│
└── pyproject.toml
```

## Key Design Decisions

### Leakage-safe fit/transform API

The agent's feature code must implement two functions:

```python
def fit(df_train: pd.DataFrame, y_train: pd.Series, config: dict) -> dict:
    """Called ONCE on training data WITH labels. Returns JSON-serializable state."""

def transform(df: pd.DataFrame, state: dict, config: dict) -> pd.DataFrame:
    """Called on EACH split WITHOUT labels. Uses only the state dict from fit()."""
```

The harness strips labels before calling `transform()`. Target encoding, frequency stats, and any label-dependent computation must happen in `fit()` and be stored in `state` as plain dicts/lists/numbers. The harness also runs a leakage detector — any single feature with AUC > 0.90 on validation is flagged.

### Composite score

```
composite_score = 0.50 * AUPRC + 0.30 * Precision@Recall(80%) - 0.20 * PSI_penalty
```

AUPRC on the out-of-time (OOT) holdout is the primary metric. Precision at 80% recall captures operating-point performance. PSI (Population Stability Index) between validation and OOT score distributions penalizes models that don't generalize across time — a PSI above the hard-reject threshold (0.25) is auto-rejected.

### Experiment tracking without git

Instead of git keep/revert, experiments are tracked as directories. Every experiment — kept or discarded — is preserved with full code snapshots, metrics, and fitted state. A `sota` symlink always points to the current best. The agent never needs to run git during the loop.

### Structured agent context

After each experiment, the harness prints a structured context block:

- **Current SOTA** with top features, confidence intervals, feature count
- **Last 10 experiments** with AUPRC and hypothesis (kept and discarded)
- **Technique success rates** — which categories of changes have historically worked
- **Untried techniques** — cross-referenced against `recipes.md`
- **Feature importance trends** — which features are growing, declining, or new
- **Recommendations** — concrete next steps

This is the agent's memory between iterations. It prevents repeating failed approaches and guides exploration toward untried territory.

## Included Datasets

### IEEE-CIS Fraud Detection

590K card-not-present transactions from the [Vesta Corporation IEEE-CIS dataset](https://www.kaggle.com/c/ieee-fraud-detection). The original V1–V339, C1–C14, D1–D15, and M1–M9 derived features are stripped — the agent works from 55 raw columns (transaction amount, card info, addresses, email domains, device/identity fields). 3.5% fraud rate.

### Fraud-Sim

1.8M simulated credit card transactions from [Sparkov fraud simulation](https://www.kaggle.com/datasets/kartik2112/fraud-detection). 16 raw columns: merchant, category, amount, geographic coordinates, demographics. 0.5% fraud rate with a 42% population shift in the OOT period (fraud rate drops from 0.58% to 0.33%), making this a challenging test for model stability.

## Example: Run 3 Results (In Progress)

Both agents running simultaneously on an RTX 4070 Ti SUPER, starting from seeded baselines:

### Fraud-Sim — AUPRC 0.037 to 0.609 (+1549%)

| # | Status | AUPRC | Hypothesis |
|---|--------|-------|------------|
| 0 | keep | 0.0369 | baseline: v3 features + 6-model ensemble |
| 1 | keep | 0.5432 | seed: v2 best features + GPU ensemble |
| 2 | discard | 0.4697 | Extended merchant/customer/category behavioral profiling |
| 3 | discard | 0.5051 | Add merchant std-based zscore and global median stats |
| 4 | **keep** | **0.6043** | merchant x category amount corridors |
| 5 | discard | 0.5329 | gender x category amount corridors |
| 6 | **keep** | **0.6089** | customer x category amount corridors |

Top features at SOTA: `mc_amt_ratio_median` (0.19), `mc_amt_zscore` (0.13), `category_amt_ratio_median` (0.12), `gender_amt_zscore` (0.09)

### IEEE-CIS — AUPRC 0.201 to 0.264 (+32%)

| # | Status | AUPRC | Hypothesis |
|---|--------|-------|------------|
| 0 | keep | 0.2009 | baseline: v3 features + 6-model ensemble |
| 1 | **keep** | **0.2644** | seed: v2 best features + GPU ensemble |
| 2 | discard | 0.2644 | velocity features: per-card transaction gap stats |
| 3 | discard | 0.2646 | identity consistency: modal profile per card |
| 4 | discard | 0.1196 | extended interaction TEs (too many, overfit) |
| 5 | discard | 0.2643 | M-field features + ProductCD analysis |
| 6 | discard | 0.2666 | XGBoost + LightGBM ensemble |

The IEEE-CIS dataset is harder — the raw columns are less informative without the stripped derived features, so improvements are incremental. The agent's context system flagged the 5-experiment discard streak and recommended trying a radically different approach.

## Adapting to a New Dataset

### 1. Prepare data

Split your data into train/val/OOT parquet files with a `label` column (0/1). Place them in a directory:

```
data/my-dataset/
├── raw_train.parquet
├── raw_val.parquet
└── raw_oot.parquet
```

### 2. Create a config

Copy an existing config and edit it:

```yaml
dataset_name: "my-dataset"
features_file: "features_mydataset.py"
model_file: "model_mydataset.py"

local_data:
  enabled: true
  data_dir: "./data/my-dataset"
  prefix: "raw"

metrics:
  target_recall: 0.80
  composite_weights:
    auprc: 0.50
    precision_at_recall: 0.30
    psi_penalty: 0.20
  psi_threshold: 0.20
  psi_hard_reject: 0.25
  min_improvement: 0.001

dataset_profile:
  fraud_rate: 0.01
  n_rows: 500000
  n_raw_features: 30
  # ... describe what columns are available
```

### 3. Create baseline feature/model files

Start with a simple passthrough features file:

```python
def fit(df_train, y_train, config):
    return {"global_mean": float(y_train.mean())}

def transform(df, state, config):
    return df.copy()
```

And a basic model file (GPU auto-detected):

```python
import xgboost as xgb
from harness.utils import get_gpu_info

def train_and_evaluate(X_train, y_train, X_val, y_val, X_oot, y_oot, config):
    gpu = get_gpu_info()
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        tree_method=gpu["tree_method"], device=gpu["device"],
        eval_metric="aucpr", early_stopping_rounds=50, random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_val_pred = model.predict_proba(X_val)[:, 1]
    y_oot_pred = model.predict_proba(X_oot)[:, 1]
    return {"y_val_pred": y_val_pred, "y_oot_pred": y_oot_pred,
            "model": model, "train_info": {}}
```

### 4. Run the baseline

```bash
python3 -m harness.evaluate --config configs/my-dataset.yaml --save --hypothesis "baseline"
```

### 5. Point the agent at it

Give the agent `program.md` and the dataset config. It handles the rest.

## Adapting to a New Domain

The repo targets fraud detection, but the architecture is domain-agnostic. To apply it to a different binary classification problem:

### What to change

- **Config YAML** — set your label column, metrics weights, date splits, dataset profile
- **`recipes.md`** — replace fraud-specific feature patterns with domain-relevant ones
- **`program.md`** — update the feature engineering strategy section with domain context
- **Feature/model files** — start fresh with baseline implementations

### What stays the same

The entire `harness/` directory is domain-agnostic:

| File | Purpose | Domain-dependent? |
|------|---------|-------------------|
| `evaluate.py` | Pipeline orchestration, metric computation | No — AUPRC/precision/PSI are general |
| `experiment_tracker.py` | Directory-per-experiment tracking | No |
| `context.py` | Structured agent memory | No — reads from recipes.md for technique names |
| `dashboard.py` | HTML dashboard generation | No |
| `plot_results.py` | Metric plots | No |
| `data_loader.py` | Parquet loading, splitting | No |
| `validate_features.py` | Feature validation | No |
| `feature_analysis.py` | IV, PSI, correlation | No |
| `utils.py` | Config, GPU, file hashing | No |

The composite score formula is configurable via YAML weights. If you need different metrics entirely (e.g., log loss, F1, custom business metric), you'd modify `evaluate.py` — that's the one harness file worth customizing.

## Running the Agent

The system is designed for Claude Code, Cursor, or any LLM coding agent that can read files and execute shell commands.

```bash
# Install dependencies
pip install -e .

# Single evaluation
python3 -m harness.evaluate --config configs/ieee-cis.yaml --save --hypothesis "my change"

# View experiment history
python3 -m harness.experiment_tracker ieee-cis

# View agent context (what the agent sees between iterations)
python3 -m harness.context ieee-cis

# Regenerate dashboard
python3 -m harness.dashboard --open
```

To launch an autonomous agent, provide `program.md` as the system prompt and point it at a dataset config. The agent reads the config, the current features/model files, `recipes.md`, and the experiment context — then loops.

## Dependencies

- Python 3.10+
- pandas, numpy, scikit-learn, xgboost, matplotlib, pyarrow, pyyaml
- Optional: google-cloud-bigquery (for BigQuery data sources), lightgbm, scipy
- GPU: CUDA-compatible GPU auto-detected for XGBoost acceleration
