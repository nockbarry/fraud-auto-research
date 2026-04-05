# fraud-auto-research

Autonomous feature engineering and model evaluation for transaction fraud monitoring.
You are an autonomous researcher. You propose features, build models, evaluate results, and iterate — indefinitely.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date and dataset (e.g. `apr5-ieee`). Branch `research/<tag>` must not exist.
2. **Create the branch**: `git checkout -b research/<tag>` from current main.
3. **Read the in-scope files**:
   - The config file for this run (e.g., `configs/ieee-cis.yaml` or `configs/fraud-sim.yaml`). Do not modify.
   - `features.py` — feature transforms. You modify this. **Uses fit/transform API.**
   - `model.py` — model training. You modify this.
   - `harness/evaluate.py` — evaluation pipeline. Do not modify.
4. **Establish baseline**: Run the pipeline as-is to record the baseline score.
5. **Begin the loop**.

## Scope

**What you CAN modify:**
- `features.py` — Your primary workspace. Contains `fit()` and `transform()`:
  - `fit(df_train, y_train, config) -> state`: Called ONCE on training data WITH labels. Return a JSON-serializable dict.
  - `transform(df, state, config) -> df`: Called on EACH split WITHOUT labels. Use only the state dict.
  - **YOU CANNOT ACCESS LABELS IN transform()**. The harness strips them. Any target-dependent features must be computed in `fit()` and stored in `state`.
- `model.py` — Change hyperparameters, algorithms, ensemble. Signature must not change.

**What you CANNOT modify:**
- Anything in `harness/` — fixed evaluation infrastructure.
- Config files — the human sets these.
- Files in `data/` — source parquet files are fixed.

## Goal

**Maximize `composite_score` on the out-of-time (OOT) holdout.**

```
composite_score = 0.50 * AUPRC + 0.30 * Precision@Recall - 0.20 * PSI_penalty
```

## Leakage Prevention (CRITICAL)

The harness enforces leakage-safe feature engineering:

1. **Labels are stripped** before calling `transform()`. You cannot compute target statistics inside transform.
2. **fit() only sees training data**. Stats computed there will generalize honestly to val/OOT.
3. **State must be JSON-serializable** — no fitted sklearn objects, no closures. Dict of primitives.
4. **The harness runs a leakage detector** — any single feature with AUC > 0.90 on validation is flagged.
5. **Scoring latency is measured** — single-row transform time is reported.

If you need target encoding, frequency encoding, or aggregation stats: compute them in `fit()`, store the mapping in `state`, apply in `transform()`.

## Running Experiments

```bash
# Run against a specific dataset config:
python3 -m harness.evaluate --config configs/ieee-cis.yaml > run.log 2>&1
python3 -m harness.evaluate --config configs/fraud-sim.yaml > run.log 2>&1

# Feature analysis (fast, ~30s):
python3 -m harness.feature_analysis

# Extract key metric:
grep "^composite_score:" run.log
```

## Available Datasets

### IEEE-CIS (configs/ieee-cis.yaml)
- 590K card-not-present transactions from Vesta Corporation
- 55 raw columns (V/C/D/M derived features stripped)
- 3.5% fraud rate, chargeback-based labels
- Baseline AUPRC: ~0.20 | Full-feature ceiling: 0.4982

### Fraud-Sim (configs/fraud-sim.yaml)
- 1.8M simulated credit card transactions
- 16 raw columns: merchant, category, amount, geo, demographics
- 0.5% fraud rate
- Baseline AUPRC: ~0.04 (lots of room for improvement)

## Feature Engineering Strategy

1. **Target encoding is the most powerful technique** — but must be done in `fit()`, stored in `state`, applied in `transform()`.
2. **Interaction target encodings** compound: card x merchant, card x category, card x email, etc.
3. **Aggregation stats** (mean, std, count per card/customer/merchant) — compute in `fit()`.
4. **Time features** (hour, day_of_week, is_night) — simple, no leakage risk.
5. **Amount features** (log, decimal, z-score vs card mean) — use stats from `fit()`.
6. **Always check IV** before adding features. IV < 0.02 = noise.
7. **Watch PSI** — unstable features get auto-rejected.

## The Experiment Loop

**LOOP FOREVER:**

1. **Read context**: Check results (results.tsv or results/ dir). Identify SOTA.
2. **Propose hypothesis**: Specific, testable idea.
3. **Implement**: Edit `features.py` fit() and/or transform(), or `model.py`.
4. **Commit**: `git commit -m "hypothesis description"`
5. **Run**: `python3 -m harness.evaluate --config configs/<dataset>.yaml > run.log 2>&1`
6. **Extract**: `grep "^composite_score:\|^auprc:\|^leakage_warnings:" run.log`
7. **Decide**:
   - If leakage_warnings > 0 → investigate, fix, and re-run.
   - If composite > SOTA + min_improvement → **KEEP**
   - Otherwise → **REVERT** (`git reset --hard HEAD~1`)
8. **Log**: Record in results.tsv
9. **Plot**: `python3 -m harness.plot_results`
10. **NEVER STOP**

## Anti-Patterns

- **DO NOT compute target statistics in transform()** — the harness won't give you labels there.
- **DO NOT use non-serializable objects in state** — it must be a dict of strings, numbers, and nested dicts/lists.
- **DO NOT one-hot encode high-cardinality features** — use target or frequency encoding.
- **DO NOT ignore leakage warnings** — investigate and fix before proceeding.
