# fraud-auto-research

Autonomous feature engineering and model evaluation for transaction fraud monitoring.
You are an autonomous researcher. You propose features, build models, evaluate results, and iterate — indefinitely.

## Setup

To set up a new experiment run:

1. **Agree on a run tag**: propose a tag based on today's date and segment (e.g. `apr5-new-cust`). The branch `research/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b research/<tag>` from current main.
3. **Read the in-scope files**:
   - `config.yaml` — current segment target, date ranges, metric thresholds. Do not modify.
   - `features.sql` — BigQuery feature SQL. You modify this.
   - `features.py` — Python feature transforms. You modify this.
   - `model.py` — Model training. You modify this.
   - `harness/evaluate.py` — evaluation pipeline. Do not modify.
   - `harness/feature_analysis.py` — feature analysis tools. Do not modify.
   - `harness/validate_features.py` — feature validation. Do not modify.
4. **Initialize results.tsv**: Create with header row only.
5. **Establish baseline**: Run the pipeline as-is to record the baseline score.
6. **Begin the loop**.

## Scope

**What you CAN modify:**
- `features.py` — Python transforms. This is your primary workspace. Add encoding, scaling, interactions, aggregations, derived features. The `transform(df, config)` function must preserve row count and output all-numeric features.
- `model.py` — Model training. Change hyperparameters, try different algorithms, add calibration, ensemble. The `train_and_evaluate()` function signature must not change.
- `features.sql` — BigQuery feature extraction (only used in BigQuery mode, not for IEEE-CIS evaluation).

**What you CANNOT modify:**
- Anything in `harness/` — this is the fixed evaluation infrastructure.
- `config.yaml` — the human sets this before launching you.
- `pyproject.toml` — do not add dependencies.
- Files in `data/` — the source parquet files are fixed.

## Goal

**Maximize `composite_score` on the out-of-time (OOT) holdout.**

The composite score is:
```
composite_score = 0.50 * AUPRC + 0.30 * Precision@Recall - 0.20 * PSI_penalty
```

Where:
- **AUPRC**: Area under precision-recall curve on OOT. Higher = better.
- **Precision@Recall**: Precision when recall is fixed at the target (see config). Higher = better.
- **PSI penalty**: Population Stability Index between val and OOT score distributions.
  - PSI < 0.20: no penalty
  - 0.20 <= PSI < 0.25: linear penalty ramp
  - PSI >= 0.25: **automatic rejection** regardless of other metrics

## Dataset: IEEE-CIS Fraud Detection

This is the Vesta Corporation card-not-present fraud dataset from Kaggle. The original dataset has 431 features including 377 Vesta-derived features (V, C, D, M columns). Those derived features have been **stripped** — your job is to engineer new features from the 55 raw attributes and close the performance gap.

**Benchmarks** (from `prepare_data.py`):
- Full-feature XGBoost (all 431 columns): **AUPRC = 0.4982** — this is the ceiling
- Raw-feature baseline (55 columns, freq encoding): **AUPRC = 0.1899** — this is where you start
- **Gap to close: 0.3083 AUPRC**

**Available raw columns** (55 features after stripping V/C/D/M):
- `TransactionDT` — seconds from a reference time point (use for time-based features)
- `TransactionAmt` — transaction dollar amount
- `ProductCD` — product code (W, H, C, S, R)
- `card1-card6` — card attributes (card1=number hash, card4=visa/mastercard, card6=debit/credit)
- `addr1, addr2` — address info (addr1=billing region, addr2=country code)
- `dist1` — distance measure
- `P_emaildomain, R_emaildomain` — purchaser and recipient email domains
- `id_01 to id_38` — identity verification features (many high-null)
- `DeviceType` — desktop/mobile
- `DeviceInfo` — device model/browser

**Data setup**: Pre-split into `data/raw_{train,val,oot}.parquet` (70/15/15 time-based split). The harness loads these directly — no BigQuery needed for this evaluation dataset.

**Key insights from feature analysis** (`python -m harness.feature_analysis`):
- 37 of 46 features are predictive (IV > 0.02)
- Strong predictors: ProductCD, identity flags (id_15-id_38), card3, addr2, DeviceType
- High correlation pairs to watch: id_03↔id_04 (1.0), id_09↔id_10 (1.0), id_05↔id_06 (1.0)
- Unstable features (PSI > 0.25): TransactionID, TransactionDT, P_emaildomain, card6, id_13

## Domain Context: Feature Engineering Ideas

Since the V/C/D/M columns were derived from these raw attributes, you can try to reconstruct similar features:

**Transaction amount features** (features.py):
- Log transform of TransactionAmt (highly skewed)
- Amount bins / quantile buckets
- Amount relative to card1 average (z-score within card groups)
- Amount decimal pattern (cents = 0 vs non-zero, known fraud signal)

**Time-based features** (features.py using TransactionDT):
- Hour of day (TransactionDT mod 86400 / 3600)
- Day of week
- Time since previous transaction per card1 (requires sort + groupby)
- Transaction velocity: count per card1 in last N seconds
- Time-of-day deviation from card1's typical pattern

**Card-level aggregations** (features.py):
- Transaction count per card1
- Mean/std amount per card1
- Unique addr1 count per card1
- Unique ProductCD count per card1
- card1 + addr1 combination frequency

**Email domain features** (features.py):
- Email domain risk score (frequency encoding is baseline — try target encoding)
- P_email == R_email match flag
- Free email provider flag (gmail, yahoo, hotmail vs corporate)
- Email domain + card combination frequency

**Identity signal aggregation** (features.py):
- Count of non-null identity fields (more verification = lower risk)
- Identity field consistency flags
- Device type + identity match patterns

**Interaction features** (features.py):
- card1 x ProductCD
- card1 x addr1
- TransactionAmt x ProductCD
- DeviceType x card6 (mobile debit vs desktop credit)

**Python-side transforms** (features.py):
- Frequency encoding for high-cardinality categoricals (baseline already does this)
- Target encoding with smoothing (be careful — fit on train only)
- Log transforms for skewed distributions
- Ratio features (amount / card_avg_amount)
- Binning continuous features into risk buckets

## Feature Engineering Strategy

1. **Prefer BQ SQL for heavy aggregations** — window functions and GROUP BY run on BigQuery's engine, much faster than Python.
2. **Use Python for encoding and transforms** — things that need sklearn or pandas logic.
3. **Check IV before adding a feature**: Run `python -m harness.feature_analysis` (fast, ~30s). If a feature has IV < 0.02, it's likely noise — don't keep it.
4. **Check PSI of new features**: If a feature has PSI > 0.25 between train and OOT, it's unstable over time. Drop it or stabilize it.
5. **Watch for leakage**: If any single feature has IV > 0.5, investigate — it may leak future information.
6. **Simplicity criterion**: All else being equal, fewer features is better. A model with 30 predictive features beats a model with 150 features and the same score.

## Anti-Patterns (Avoid These)

- **Target leakage**: Never use future information. No features derived from chargeback data, dispute flags, or labels.
- **High-cardinality one-hot encoding**: Do NOT one-hot encode merchant_id, customer_id, IP address, etc. Use frequency or target encoding instead.
- **Feature explosion without metric improvement**: Adding 50 features that don't improve composite_score is waste. Keep features lean.
- **Ignoring PSI**: A model that scores well on validation but has PSI > 0.25 on OOT is worthless. The harness will auto-reject it.
- **Overfitting to validation**: Use OOT as the truth. If val AUPRC is great but OOT degrades, you're overfitting.

## Output Format

The evaluation harness prints:

```
---
composite_score:     0.4523
auprc:               0.4891
precision_at_recall: 0.3210
psi:                 0.0834
fpr:                 0.0523
n_features:          47
training_seconds:    142.3
total_seconds:       287.1
```

Extract the key metric: `grep "^composite_score:" run.log`

## Logging Results

Log every experiment to `results.tsv` (tab-separated). Do not commit this file.

Header and columns:
```
commit	composite	auprc	prec@recall	psi	status	hypothesis
```

- **commit**: git short hash (7 chars)
- **composite**: composite_score (e.g. 0.4523) — use 0.000000 for crashes
- **auprc**: AUPRC on OOT
- **prec@recall**: precision at target recall on OOT
- **psi**: PSI between val and OOT scores
- **status**: `keep`, `discard`, `crash`, or `reject_psi`
- **hypothesis**: short description of what was tried

Example:
```
commit	composite	auprc	prec@recall	psi	status	hypothesis
a1b2c3d	0.3200	0.3800	0.2100	0.0520	keep	baseline
b2c3d4e	0.3450	0.4100	0.2300	0.0610	keep	add 24h txn velocity per customer
c3d4e5f	0.3420	0.4050	0.2250	0.0580	discard	add merchant category frequency encoding
d4e5f6g	0.0000	0.0000	0.0000	0.0000	crash	add IP geolocation distance (SQL syntax error)
e5f6g7h	0.3500	0.4200	0.2400	0.2600	reject_psi	add device age feature (unstable over time)
```

## The Experiment Loop

Run on a dedicated branch (e.g. `research/apr5-new-cust`).

**LOOP FOREVER:**

1. **Read context**: Look at `results.tsv` to understand what's been tried and what worked. Identify the current SOTA (last row with status=keep, or baseline). Read the current state of features.sql, features.py, model.py.

2. **Propose hypothesis**: Formulate a specific, testable idea. Examples:
   - "Adding 7-day transaction velocity per customer will improve AUPRC by capturing ATO patterns"
   - "Log-transforming transaction_amount will reduce skew and improve model discrimination"
   - "Increasing max_depth to 8 with lower learning_rate will capture more complex interactions"
   Choose ONE of:
   a) Feature SQL change (new BQ aggregation, window, join)
   b) Feature Python change (new encoding, scaling, interaction)
   c) Model change (hyperparameter, algorithm, calibration)
   d) Feature analysis only (`python -m harness.feature_analysis` — fast check before committing)

3. **Implement**: Edit features.sql, features.py, and/or model.py.

4. **Commit**: `git commit` with the hypothesis as the message.

5. **Validate features** (optional but recommended for SQL/feature changes):
   ```
   python -m harness.validate_features > validate.log 2>&1
   ```
   If FAIL: read the error, fix, and re-commit. Do not proceed to full training.

6. **Run evaluation**:
   ```
   python -m harness.evaluate > run.log 2>&1
   ```

7. **Extract results**: `grep "^composite_score:\|^auprc:\|^precision_at_recall:\|^psi:" run.log`
   - If empty: the run crashed. Run `tail -n 50 run.log` to read the error.

8. **Decide**:
   - If `psi_rejected: true` in output → status=`reject_psi`, revert.
   - If `composite_score` > SOTA composite_score (by at least min_improvement in config) → status=`keep`. This commit is the new SOTA.
   - If `composite_score` <= SOTA or improvement < min_improvement → status=`discard`. Revert: `git reset --hard HEAD~1`.
   - If crash → status=`crash`. If easy fix (typo, import), fix and retry. If broken idea, revert and move on.

9. **Log to results.tsv**: Record the result regardless of outcome.

10. **NEVER STOP**: Do not pause. Do not ask if you should continue. The human may be asleep. Run indefinitely until manually interrupted. If you run out of ideas:
    - Re-read results.tsv for patterns (what categories of features helped?)
    - Try combining features from successful experiments
    - Try removing features to simplify
    - Try different model configurations
    - Run feature analysis to find high-IV features you haven't exploited
    - Try features from a different category (if velocity worked, try behavioral deviation)
    - Try interaction features between your best predictors

**Timeout**: Each experiment should complete within 15 minutes. If it exceeds this, kill and treat as crash.

**Crashes**: If a run crashes, use judgment. Typo or import error? Fix and retry. SQL syntax error? Fix the SQL. Fundamentally broken idea? Log crash and move on.

**Pacing**: Each experiment takes roughly 2-10 minutes depending on BQ caching and model complexity. Expect ~10-30 experiments per hour. An overnight run (8 hours) yields 80-240 experiments.
