# fraud-auto-research

Autonomous feature engineering and model evaluation for transaction fraud monitoring.
You are an autonomous researcher. You propose features, build models, evaluate results, and iterate — indefinitely.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date and dataset (e.g. `apr5-ieee`). Branch `research/<tag>` must not exist.
2. **Create the branch**: `git checkout -b research/<tag>` from current main.
3. **Read the in-scope files**:
   - The config file for this run (e.g., `configs/ieee-cis.yaml`). Do not modify.
   - The features and model files for your dataset live in `datasets/<name>/`. The config's `features_file` / `model_file` keys point at them. Examples:
     - IEEE-CIS: `datasets/ieee_cis/features.py` and `datasets/ieee_cis/model.py`
     - IEEE-CIS-Fresh: `datasets/ieee_cis_fresh/features.py` and `datasets/ieee_cis_fresh/model.py`
     - Fraud-Sim: `datasets/fraud_sim/features.py` and `datasets/fraud_sim/model.py`
     - FDH: `datasets/fdh/features.py` and `datasets/fdh/model.py`
     - PaySim: `datasets/paysim/features.py` and `datasets/paysim/model.py`
   - Your journal: `journals/<dataset>.md` (e.g. `journals/ieee-cis.md`).
   - `harness/evaluate.py` — evaluation pipeline. Do not modify.
   - **IMPORTANT**: Edit ONLY your dataset's directory under `datasets/`. Do not touch other datasets.
4. **Establish baseline**: Run the pipeline as-is to record the baseline score.
5. **Begin the loop**.

## Scope

**What you CAN modify:**
- `datasets/<name>/features.py` — Your primary workspace. Contains `fit()` and `transform()`:
  - `fit(df_train, y_train, config) -> state`: Called ONCE on training data WITH labels. Return a JSON-serializable dict.
  - `transform(df, state, config) -> df`: Called on EACH split WITHOUT labels. Use only the state dict.
  - **YOU CANNOT ACCESS LABELS IN transform()**. The harness strips them. Any target-dependent features must be computed in `fit()` and stored in `state`.
- `datasets/<name>/model.py` — Change hyperparameters, algorithms, ensemble. Signature must not change. GPU is auto-detected.
- `journals/<dataset>.md` — Your campaign journal. Update before/after each experiment.

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

### IEEE-CIS (configs/ieee-cis.yaml) — datasets/ieee_cis/{features,model}.py
- 590K card-not-present transactions from Vesta Corporation
- 55 raw columns (V/C/D/M derived features stripped)
- 3.5% fraud rate, chargeback-based labels
- Current SOTA AUPRC: ~0.264 | Full-feature ceiling: 0.4982

### Fraud-Sim (configs/fraud-sim.yaml) — datasets/fraud_sim/{features,model}.py
- 1.8M simulated credit card transactions
- 16 raw columns: merchant, category, amount, geo, demographics
- 0.5% fraud rate, 42% population shift in OOT
- Current SOTA AUPRC: ~0.543

### FDH (configs/fdh.yaml) — datasets/fdh/{features,model}.py
- Fraud Detection Handbook simulated transactions, 1.75M rows
- 6 raw columns: TRANSACTION_ID, TX_DATETIME, CUSTOMER_ID, TERMINAL_ID, TX_AMOUNT, label
- 0.84% fraud rate, 4990 customers, 10000 terminals, stable fraud rate across splits
- THREE FRAUD SCENARIOS (not labeled — must engineer features to capture each):
  1. High-amount: fraud amount >> customer's typical range → behavioral deviation features
  2. Terminal compromise: specific terminals fraudulent for 28-day windows → terminal risk/velocity
  3. Card-not-present: small repeated amounts across many terminals → velocity + amount pattern features
- Baseline AUPRC: 0.248 (strong engineering needed to reach 0.65+)
- term_fraud_rate dominates baseline — next priority: velocity (rolling window counts/sums)

### PaySim (configs/paysim.yaml) — datasets/paysim/{features,model}.py
- PaySim synthetic mobile money transactions, 6.3M rows, 1-month simulation
- 9 raw columns: step (hour), type, amount, origin/dest balances, entity IDs
- 0.13% overall fraud rate — fraud ONLY in TRANSFER (0.77%) and CASH_OUT (0.18%) types
- Temporal split: steps 1-500 train, 501-600 val, 601-743 OOT
- Near-perfect baseline AUPRC (~1.0) — key signal: balance error feature
- Agent should focus on: reducing feature count, improving PSI robustness, latency optimization, exploring temporal sequence patterns
- CRITICAL: `isFlaggedFraud` is NOT in the data (was dropped — it is leaky). Do not reference it.
- Key fraud pattern: TRANSFER to new account → CASH_OUT. Balance drain (orig goes to 0). Balance error = |(old - new) - amount| ≈ 0 for legit, large for fraud.

## Feature Engineering Strategy

Read `recipes.md` for copy-paste code patterns.

For SOTA technique reference: **never read `fraud_practices.md` whole** (it's 107 KB / ~25K tokens). The experiment context already includes the `fraud_practices_index.md` table-of-contents — find the relevant section there, then call `Read fraud_practices.md offset=<start> limit=<count>` to pull only that section. Each section is 30-100 lines. The `when_to_use:` annotation in the index tells you which sections apply to your current dataset and campaign.

Also read `dataset_profile` in the config for dataset characteristics.

**Priority order** (by expected impact):

**Phase 1 — Tabular features (high ROI, fast to implement):**
1. **Velocity features** (Recipe 2) — per-card median gap, burst count, daily rate. Fraud has strong temporal patterns. Both datasets.
2. **Behavioral profiling** (Recipe 3) — deviation-from-self: amount z-score vs user's own history, hour deviation. Both datasets.
3. **OOF target encoding** (Recipe 1) — prevents within-train leakage. Use for all high-cardinality columns. Both datasets.
4. **Identity consistency** (Recipe 4) — per-card modal email/device, entity sharing. IEEE-CIS only.
5. **Entity resolution** (Recipe 5) — shared identity counts. Both datasets.
6. **Anomaly score** (Recipe 6) — Mahalanobis distance from train centroid. Both datasets.
7. **Amount patterns** (Recipe 7) — round numbers, corridor analysis. Both datasets.
8. **Interaction TEs** — card x category, card x email, card x device. Use fitted TEs from Recipe 1.

**Phase 2 — Sequence & trajectory features (add once Phase 1 plateaus):**
9. **RFM cluster distance** (Recipe 11) — distance from card's RFM vector to legitimate cluster centroids. Detects bust-out trajectories. Requires sklearn KMeans (already available).
10. **CUSUM behavioral shift** (Recipe 10) — sequential change detection on amount sequences. Catches gradual escalation. Both datasets.
11. **HMM state features** (Recipe 8) — hidden Markov model posterior state probabilities. Requires `hmmlearn` (`pip install hmmlearn`). Most powerful for ATO and bust-out patterns.
12. **Feature-group autoencoders** (Recipe 9) — reconstruction error per feature group (amount, time, identity). Catches anomalies with no single-feature signal. Both datasets.

**Phase 2 reference:** Use `fraud_practices_index.md` to locate Part 7 (Advanced Sequence Modeling, lines ~1414-1789) and the Five-Tier Architecture Synthesis (Part 7.9, lines ~1718-1760). Then `Read fraud_practices.md offset=1414 limit=376` for Part 7 in full, or pull just the subsection you need.

**Per-dataset strategy:**
- **IEEE-CIS** (3.5% fraud, identity-heavy): Focus on identity consistency, OOF TE, entity sharing. Use min_samples=50.
- **Fraud-Sim** (0.5% fraud, geo-heavy, high population shift): Focus on velocity, behavioral profiling, geo features. Use min_samples=20. Don't over-smooth.
- **PaySim** (0.13% fraud, mobile money, near-perfect baseline): Balance error already near-perfect. Focus on: (1) velocity/sequence features on nameOrig/nameDest entity pairs — TRANSFER→CASH_OUT chain detection; (2) feature compression — maintain AUPRC with fewer features; (3) model latency; (4) robustness without balance features (try model that excludes balance columns as ablation). Use type-conditional features extensively.
- **FDH** (0.84% fraud, 6 raw columns, hard — needs engineering): Baseline AUPRC=0.248. Priority order: (1) velocity features — rolling 1h/6h/24h/7d counts and sums per CUSTOMER_ID and TERMINAL_ID (captures scenario 3: repeated small txns); (2) terminal compromise signal — rolling fraud rate per terminal over recent window (captures scenario 2); (3) behavioral deviation — amount z-score vs customer's own history (captures scenario 1); (4) customer-terminal novelty — is this customer's first visit to this terminal? Time since last transaction. Use TX_TIME_DAYS or TX_DATETIME for window computations. Expected ceiling AUPRC ~0.80 with good velocity features.

**Anti-leakage rules:**
- All target-dependent stats MUST be computed in `fit()`, stored in `state`
- `transform()` has NO labels — use only `state` dict
- Check `top_features:` output after each run to verify new features contribute
- If feature has importance < 0.001, consider removing it (adds noise)

## Multi-Step Campaigns (read before every experiment)

A "feature leap" is a 3-5 experiment campaign aimed at a specific signal, not a single tweak. Single-feature experiments asymptote at +0.005 composite. Campaigns can move +0.05. Kaggle competition winners (IEEE-CIS, AMEX) all used compound moves: UID construction → 200+ aggregations, multi-window velocity stacks, behavioral fingerprints across multiple entities. The agent's job is to attempt those moves here.

### Campaign templates

1. **UID Aggregation Campaign** (4 experiments — IEEE-CIS / fraud-sim)
   - Step 1: build UID via Recipe 15 (e.g., `card1 + addr1 + D1`), add 5 basic UID aggregations (mean amt, std amt, count, distinct merchants, distinct days)
   - Step 2: add 10 more UID aggregations — time-of-day stats, tx gap stats, amount percentiles, distinct terminals
   - Step 3: add UID-conditioned velocity stack (Recipe 16 keyed by UID instead of card1)
   - Step 4: add UID-level OOF target encoding (Recipe 1 with UID as the column, smoothing=10)

2. **Velocity Stack Campaign** (4 experiments — all datasets)
   - Step 1: add 1h/24h/7d count and sum per primary entity (Recipe 16, 12 features)
   - Step 2: add std and burst (count_1h / count_7d) features for the same windows (8 more features)
   - Step 3: cross with rolling terminal/merchant fraud rate (Recipe 17, 5 more features)
   - Step 4: add velocity ratios and acceleration (count_1h / count_24h, sudden activity spikes)

3. **Per-Scenario Campaign for FDH** (3 experiments — REQUIRED for FDH)
   - `configs/fdh.yaml` lines 79-83 documents three fraud scenarios — re-read them before starting.
   - Scenario 1 (high amount, 973 cases): Recipe 18 customer fingerprints + Recipe 19 von Mises hour
   - Scenario 2 (terminal compromise 28d, 9077 cases): Recipe 17 rolling terminal fraud rate + Recipe 16 windowed velocity per terminal
   - Scenario 3 (CNP small repeated, 4631 cases): Recipe 16 with 1m/10m windows + customer-distinct-terminals count

### Persistence rules

- A step that loses `< 0.005` composite is **not a failure** — keep going to the next step.
- Only abandon a campaign after **3 consecutive steps each lose `> 0.005` composite**, OR if the journal's "Abandon criteria" is hit.
- Before discarding a step, **try one variation** of the same recipe (different window, different smoothing, different entity).
- The hypothesis field of every experiment must reference its campaign step using the exact pattern:
  `"velocity-stack campaign step 2/4: add std + burst on top of step 1's 1h/24h/7d counts"`. The harness scans for `step X/Y` to track campaigns.

### Updating the journal

- BEFORE every experiment: re-read `journals/{dataset}.md`. If the thesis is contradicted by the last 3 keeps, rewrite it.
- BEFORE starting a campaign: write all 3-5 planned steps into "Active Campaign" with status `planned`.
- AFTER every experiment: append exactly one line to "Lessons Learned". Prune oldest if over 20.
- A campaign is "done" when 4 of its planned steps have status `done(exp_NNN)` or `failed`.
- Keep the journal under 4 KB total. Discarded Theses capped at 5 entries — graveyard, not history.

## The Experiment Loop

**LOOP FOREVER:**

0. **READ THE COLUMN ANALYSIS FIRST** (before journal, before SOTA, before anything else). Every context dump includes a `RAW COLUMN ANALYSIS` block from `harness/column_analysis.py` showing per-column IV, univariate AUC, null-flag AUC, and NaN%. This is the EDA the agent has historically skipped — Run 3 spent 10 experiments chasing UID aggregations while ignoring `id_17` (IV=0.35), `id_30` (IV=0.62), `DeviceInfo` (IV=1.78). The block also lists `TRANSFORMED FEATURE ANALYSIS` after every keep — this is what the model currently sees, including any DEAD FEATURES (IV<0.005) that should be removed. Rules:
   - If a column has IV ≥ 0.1 and is not in `top_features`, the FE pipeline is dropping/destroying it. Investigate.
   - If a column has `null_AUC > 0.55` (marked with `*`), the missingness itself is predictive — add a `col_is_null` flag.
   - **Do NOT blanket-drop columns by NaN rate.** High-NaN identity columns often carry the strongest signal via their null pattern. The IEEE-CIS `datasets/ieee_cis/features.py` `fit()` now uses a selective rule (drop only `n_unique<=1` or `>99% NaN with <=5 levels`) — preserve this pattern in any FE you write.
   - The column analysis is recomputed automatically: raw is cached for 50 experiments, transformed refreshes after every keep. If something feels stale, run `python3 -m harness.column_analysis {dataset} --refresh`.

1. **Read context AND journal**: After each `--save` run, a full experiment context is printed automatically — and the top of that output now includes `journals/{dataset}.md` followed by the column analysis. If you don't see your journal there, create it (use the template in any existing `journals/*.md`). The context shows:
   - Your journal: thesis, active campaign, lessons learned, discarded theses
   - Raw column analysis (univariate IV / null-AUC / NaN%) — STEP 0 above
   - Transformed feature analysis (post-FE IV) — surfaces dead features
   - Current SOTA with top features and confidence intervals
   - Last 10 experiments (kept and discarded) with AUPRC
   - Active campaign tracking (which campaigns have stalled vs progressing)
   - Technique success rates (which categories of changes work)
   - Untried techniques from recipes.md
   - Feature importance trends (growing vs declining features)
   - Recommended next steps
   Confirm before proposing: (a) is there an active campaign? (b) is this experiment a step in it? (c) does the campaign target a documented fraud scenario from `configs/{dataset}.yaml`? (d) does the column analysis show any high-IV columns that the SOTA top_features is ignoring?
1.5. **Update journal BEFORE proposing**: If you're starting a new campaign, write all 3-5 planned steps into "Active Campaign" now (use the Edit tool on `journals/{dataset}.md`). If the thesis is contradicted by recent results, rewrite it. Prune Lessons Learned to 20 max. Move dead theses to Discarded Theses (max 5).
2. **Propose hypothesis**: Use the context AND journal to make an informed choice. Build on growing features, try untried techniques, avoid repeating failed categories. If running a campaign step, the hypothesis MUST contain `step X/Y` (e.g., `"uid-aggregation campaign step 2/4: add 10 more UID time/distinct stats"`).
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

- **DO NOT abandon a thesis after one losing experiment.** A campaign is 3-5 experiments. Re-read `journals/{dataset}.md` "Active Campaign" — only abandon if 3 consecutive steps each lose >0.005 composite. Run one variation (different window, different entity, different smoothing) before declaring a step failed. Single-tweak mode caps you at +0.005 per experiment; campaigns can move +0.05.
- **DO NOT propose a single-feature tweak when a campaign is active.** If your journal's Active Campaign has incomplete steps, do the next step. Drift back to single tweaks is the failure mode that kills ambitious feature engineering.
- **DO NOT compute target statistics in transform()** — the harness won't give you labels there.
- **DO NOT use non-serializable objects in state** — it must be a dict of strings, numbers, and nested dicts/lists. Numpy arrays → `.tolist()`, sklearn objects → serialize their parameters manually.
- **DO NOT one-hot encode high-cardinality features** — use target or frequency encoding.
- **DO NOT blanket-drop columns by NaN rate.** A `nan_rates > 0.50` filter killed all 43 IEEE-CIS identity columns through 11 experiments — including DeviceInfo (IV=1.78), id_30 (IV=0.62), id_17 (IV=0.35). The null pattern is often the signal. Use the column analysis (Step 0) to decide what to drop, and prefer leaving NaN intact for XGBoost's native handling, or adding a single `cluster_present` binary flag for correlated null clusters.
- **DO NOT ignore the column analysis block** — if a column has IV ≥ 0.1 and is missing from `top_features`, the FE pipeline is destroying signal. Trace why.
- **DO NOT ignore leakage warnings** — investigate and fix before proceeding. Note: an `IV grade` of `high_card` (n_unique > 50) is NOT leakage — it's an artifact of per-level binning on high-cardinality categoricals. Only `LEAK?` (IV > 0.5 with n_unique ≤ 50) is suspicious.
- **DO NOT use graph-based GNN features** — graph construction and GNN training require external infrastructure. Use entity resolution (Recipe 5) and cross-entity aggregations (fraud_practices.md Part 5.4 — see fraud_practices_index.md, lines ~1156-1235) as graph-free alternatives.
- **For HMM/autoencoder features**: wrap in `try/except` with `pass` so missing dependencies don't crash the harness. Check that card has ≥3 observations before HMM decoding — single-observation sequences produce unreliable posteriors.

## Crash Recovery

If your experiment is logged with status `crash` (or `timeout`), the working `features_{dataset}.py` and/or `model_{dataset}.py` is in a broken state. The previous SOTA's code is preserved as a snapshot — restore it before continuing:

```bash
# Replace the corrupted file with the last known-good SOTA snapshot
cp experiments/{dataset}/sota/features.py features_{dataset}.py
cp experiments/{dataset}/sota/model.py model_{dataset}.py

# Verify it reproduces the SOTA score before trying again
python3 -m harness.evaluate --config configs/{dataset}.yaml
```

Once the SOTA score is reproduced, make a smaller, more incremental change and try again. Common crash causes:

- **Missing column**: referencing `df["foo"]` where `foo` doesn't exist in this dataset. Always check `dataset_profile.raw_columns` in the config first.
- **Non-serializable state**: numpy arrays, sklearn objects, or pandas Series stored in the state dict. The harness checks JSON-serializability on every run — convert with `.tolist()`, `float()`, `int()`, or nested dict comprehensions.
- **Missing import**: forgetting `import numpy as np` inside the features file.
- **Division by zero**: dividing by a `std`, `count`, or `denominator` that can be zero. Always `.clip(lower=0.01)` denominators.
- **Timeout**: a fit() or transform() step that doesn't scale linearly. Check for accidental cross-joins, full pairwise distance matrices, or `.map()` on dictionaries with millions of keys (use `.apply(lambda x: d.get(x, default))` for large dicts).
