# fraud-auto-research

Autonomous feature engineering and model evaluation for transaction fraud monitoring.
You are an autonomous researcher. You propose features, build models, evaluate results, and iterate — indefinitely.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date and dataset (e.g. `apr5-ieee`). Branch `research/<tag>` must not exist.
2. **Create the branch**: `git checkout -b research/<tag>` from current main.
3. **Read the in-scope files**:
   - The config file for this run (e.g., `configs/ieee-cis.yaml` or `configs/fraud-sim.yaml`). Do not modify.
   - The features file for your dataset — specified in the config as `features_file`:
     - IEEE-CIS: `features_ieee.py`
     - Fraud-Sim: `features_sim.py`
   - The model file for your dataset — specified in the config as `model_file`:
     - IEEE-CIS: `model_ieee.py`
     - Fraud-Sim: `model_sim.py`
   - `harness/evaluate.py` — evaluation pipeline. Do not modify.
   - **IMPORTANT**: Edit ONLY your dataset's specific files. Do not touch the other dataset's files.
4. **Establish baseline**: Run the pipeline as-is to record the baseline score.
5. **Begin the loop**.

## Scope

**What you CAN modify:**
- `features_{dataset}.py` — Your primary workspace (e.g., `features_ieee.py` or `features_sim.py`). Contains `fit()` and `transform()`:
  - `fit(df_train, y_train, config) -> state`: Called ONCE on training data WITH labels. Return a JSON-serializable dict.
  - `transform(df, state, config) -> df`: Called on EACH split WITHOUT labels. Use only the state dict.
  - **YOU CANNOT ACCESS LABELS IN transform()**. The harness strips them. Any target-dependent features must be computed in `fit()` and stored in `state`.
- `model_{dataset}.py` — Change hyperparameters, algorithms, ensemble. Signature must not change. GPU is auto-detected.

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
# Run and auto-save to experiment tracker:
python3 -m harness.evaluate --config configs/ieee-cis.yaml --save --hypothesis "add velocity features"

# Or run without saving (manual review):
python3 -m harness.evaluate --config configs/ieee-cis.yaml > run.log 2>&1
grep "^composite_score:\|^auprc:\|^top_features:" run.log

# View experiment status:
python3 -m harness.experiment_tracker
python3 -m harness.experiment_tracker ieee-cis

# Feature analysis (fast, ~30s):
python3 -m harness.feature_analysis
```

## Experiment Tracking

Experiments are saved to `experiments/{dataset}/` directories. Each experiment preserves:
- **features.py** and **model.py** code snapshots
- **metrics.json** — all metrics, feature importances, CIs
- **state.json** — fitted feature state (the deployable artifact)
- **metadata.json** — hypothesis, status, timestamp, parent experiment

The `sota` symlink always points to the current best. Use `--save` flag to auto-save and auto-determine keep/discard based on composite score vs SOTA.

**No git operations needed during the loop.** Just edit features.py/model.py, run with `--save`, and the tracker handles everything. You can review discarded experiments later — nothing is lost.

## Available Datasets

### IEEE-CIS (configs/ieee-cis.yaml) — features_ieee.py / model_ieee.py
- 590K card-not-present transactions from Vesta Corporation
- 55 raw columns (V/C/D/M derived features stripped)
- 3.5% fraud rate, chargeback-based labels
- Current SOTA AUPRC: ~0.264 | Full-feature ceiling: 0.4982

### Fraud-Sim (configs/fraud-sim.yaml) — features_sim.py / model_sim.py
- 1.8M simulated credit card transactions
- 16 raw columns: merchant, category, amount, geo, demographics
- 0.5% fraud rate, 42% population shift in OOT
- Current SOTA AUPRC: ~0.543

## Feature Engineering Strategy

Read `recipes.md` for copy-paste code patterns. Read `fraud_practices.md` for SOTA techniques by fraud type. Read `dataset_profile` in the config for dataset characteristics.

**Priority order** (by expected impact):

1. **Velocity features** (Recipe 2) — per-card median gap, burst count, daily rate. Fraud has strong temporal patterns. Both datasets.
2. **Behavioral profiling** (Recipe 3) — deviation-from-self: amount z-score vs user's own history, hour deviation. Both datasets.
3. **OOF target encoding** (Recipe 1) — prevents within-train leakage. Use for all high-cardinality columns. Both datasets.
4. **Identity consistency** (Recipe 4) — per-card modal email/device, entity sharing. IEEE-CIS only.
5. **Entity resolution** (Recipe 5) — shared identity counts. Both datasets.
6. **Anomaly score** (Recipe 6) — Mahalanobis distance from train centroid. Both datasets.
7. **Amount patterns** (Recipe 7) — round numbers, corridor analysis. Both datasets.
8. **Interaction TEs** — card x category, card x email, card x device. Use fitted TEs from Recipe 1.

**Per-dataset strategy:**
- **IEEE-CIS** (3.5% fraud, identity-heavy): Focus on identity consistency, OOF TE, entity sharing. Use min_samples=50.
- **Fraud-Sim** (0.5% fraud, geo-heavy, high population shift): Focus on velocity, behavioral profiling, geo features. Use min_samples=20. Don't over-smooth.

**Anti-leakage rules:**
- All target-dependent stats MUST be computed in `fit()`, stored in `state`
- `transform()` has NO labels — use only `state` dict
- Check `top_features:` output after each run to verify new features contribute
- If feature has importance < 0.001, consider removing it (adds noise)

## The Experiment Loop

**LOOP FOREVER:**

1. **Read context**: After each `--save` run, a full experiment context is printed automatically. Or run `python3 -m harness.context <dataset>` to see:
   - Current SOTA with top features and confidence intervals
   - Last 10 experiments (kept and discarded) with AUPRC
   - Technique success rates (which categories of changes work)
   - Untried techniques from recipes.md
   - Feature importance trends (growing vs declining features)
   - Recommended next steps
2. **Propose hypothesis**: Use the context to make an informed choice. Build on growing features, try untried techniques, avoid repeating failed categories.
3. **Implement**: Edit `features.py` fit() and/or transform(), or `model.py`.
4. **Run and save**:
   ```bash
   python3 -m harness.evaluate --config configs/<dataset>.yaml --save --hypothesis "your hypothesis"
   ```
   The tracker auto-determines keep/discard, saves code snapshots, and updates SOTA.
5. **Read feedback**: The output shows status, top_features, auprc_ci. Use this to guide the next experiment.
6. **If discarded**: The code is preserved in `experiments/<dataset>/exp_NNN/`. You can review it later. Just edit features.py again and try something new.
7. **If kept**: The `sota` symlink is updated. Build on this success.
10. **NEVER STOP**

## Anti-Patterns

- **DO NOT compute target statistics in transform()** — the harness won't give you labels there.
- **DO NOT use non-serializable objects in state** — it must be a dict of strings, numbers, and nested dicts/lists.
- **DO NOT one-hot encode high-cardinality features** — use target or frequency encoding.
- **DO NOT ignore leakage warnings** — investigate and fix before proceeding.
