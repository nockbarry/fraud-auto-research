# Fraud Practices Index

`fraud_practices.md` is ~1928 lines / 107 KB. **Do not read it whole.** This index
maps each section to its line range and a one-line `when_to_use`. To pull a
section into context, call:

  `Read fraud_practices.md offset=<start> limit=<count>`

where `count = end - start + 1`. Most sections are 30-100 lines.

Re-generate this index whenever fraud_practices.md changes:
  `python3 -m harness.context --rebuild-index`  (TODO — currently hand-curated)

---

## Part 1 — Current Landscape (lines 10-132)

### 1.1 GBDT vs. Deep Learning — Resolved (lines 12-44, 33 lines)
when_to_use: ALWAYS read first when starting a new dataset. Resolves the model-class debate
(GBDT wins on tabular fraud) so you don't waste experiments on TabNet/FT-Transformer/MLP.

### 1.2 Foundation Models for Tabular Data (lines 45-74, 30 lines)
when_to_use: Only if you've exhausted XGBoost gains and want to try a TabPFN/TabuLa-style
embedding-as-feature pattern. Skip otherwise — these add infrastructure cost.

### 1.3 Competition-Validated Techniques (lines 75-108, 34 lines)
when_to_use: Read when planning a UID/aggregation campaign. Documents the IEEE-CIS winner
trick (card1+addr1+D1 → AUC 0.90→0.948), TalkingData click-rate trick, etc.

### 1.4 What Production Systems Actually Deploy (lines 109-132, 24 lines)
when_to_use: Read when making latency-vs-accuracy tradeoffs. Confirms entity aggregations
+ velocity windows + identity stability are the production-deployed feature families.

---

## Part 2 — Public Datasets (lines 133-294)

### 2 Dataset selection criteria + tiered list (lines 133-294, 162 lines)
when_to_use: Reference. Skip during experiments — the agent already knows which dataset
it's working on. Useful only when comparing dataset characteristics.

---

## Part 3 — Fraud Type Taxonomy (lines 295-498)

### Type 1: Card Not Present / E-Commerce (lines 301-330, 30 lines)
when_to_use: IEEE-CIS, fraud-sim, paysim. Covers the dominant CNP signals: device/IP,
amount-velocity, merchant-novelty, behavioral fingerprinting.

### Type 2: Account Takeover / ATO (lines 331-360, 30 lines)
when_to_use: Datasets with login/session events. Identity instability × behavioral deviation
patterns. Useful for fraud-sim (has device + customer history).

### Type 3: First-Party Fraud / Bust-Out / Friendly (lines 361-400, 40 lines)
when_to_use: Application-time data, long-history datasets. Less applicable to current 4 datasets.

### Type 4: Third-Party / Stolen Cards (lines 401-446, 46 lines)
when_to_use: ALL CNP datasets. Geo-distance, MCC novelty, amount z-score. Has copy-paste code.
**High value for IEEE-CIS, fraud-sim, FDH.**

### Type 5: Synthetic Identity (lines 447-475, 29 lines)
when_to_use: Application data with PII. Less useful for transaction-only datasets.

### Type 6: Application Fraud (lines 476-498, 23 lines)
when_to_use: Skip — none of the current 4 datasets have application data.

---

## Part 4 — Cross-Cutting Feature Engineering Techniques (lines 499-873)

### 4.1 Velocity / Aggregation Windows (lines 501-563, 63 lines)
when_to_use: **First move on FDH, fraud-sim, paysim.** Multi-window count/sum/std per entity.
Implementation: Recipe 2 (single window) and Recipe 16 (multi-window stack).

### 4.2 Behavioral Profiling (Self/Peer Deviation) (lines 564-612, 49 lines)
when_to_use: When you have stable entity IDs and history. Self-z-score, peer-group baselines,
proxy UID construction (the IEEE-CIS pattern). Implementation: Recipe 3, 18.

### 4.3 Entity Resolution / Graph Features (lines 613-681, 69 lines)
when_to_use: Cross-entity sharing counts (graph-FREE). Use when raw identity columns let you
count "how many cards share this addr" or "how many devices share this email". **High value for IEEE-CIS.**
Skip the GNN/PageRank parts — those need external graph infra.

### 4.4 Amount Pattern Analysis (lines 682-718, 37 lines)
when_to_use: ALWAYS. Cents-component, log-amount, amount-vs-customer-distribution, threshold-just-below.
Cheap, no state, no PSI. Implementation: see exp_013 in IEEE-CIS journal — `amt_is_round` and
`amt_cents` were top features.

### 4.5 Temporal Pattern Features (lines 719-773, 55 lines)
when_to_use: Datasets with TX_DATETIME / TransactionDT. Cyclic hour/dow encoding. **CRITICAL —
modular extraction (`% 86400`) avoids the PSI=12.43 trap that killed Track A early experiments.**
Implementation: Recipe 12, exp_008 in IEEE-CIS journal.

### 4.6 Categorical Feature Encoding (lines 774-815, 42 lines)
when_to_use: Every dataset. Frequency vs target-encoded vs WoE comparison with leakage warnings.
Lesson: TE for low/medium cardinality (<50 levels); freq encoding for high cardinality.
**Don't TE high-NaN columns** — they collapse to global_mean.

### 4.7 Population Stability and OOT Generalization (lines 816-873, 58 lines)
when_to_use: Read when train_val_psi > 0.10 or auroc_train_val_gap > 0.10. PSI thresholds and
diagnosis. Composite score penalizes both, so this is foundational reading.

---

## Part 5 — Email, Device, Identity Deep Reference (lines 874-1235)

### 5.1 Email Feature Engineering — 30+ Patterns (lines 880-962, 83 lines)
when_to_use: Datasets with email columns. IEEE-CIS has R_emaildomain / P_emaildomain — those
are top features already. Read for ideas like entropy, char composition, disposable-domain flag.

### 5.2 Device Signal Features (lines 963-1040, 78 lines)
when_to_use: Datasets with device fingerprint columns. IEEE-CIS DeviceInfo, fraud-sim device_id.
Battery/charging, canvas hash, JA3 — most apply only with rich raw signals.

### 5.3 Identity Stability — 4-Layer Framework (lines 1041-1155, 115 lines)
when_to_use: **High value for IEEE-CIS, fraud-sim** — any dataset with rich identity columns.
First-seen flags, modal-identity Jaccard, novelty-burst patterns. Implementation backbone for
Recipe 5 and Recipe 18.

### 5.4 Recipient-Side and Cross-Entity Aggregations (lines 1156-1235, 80 lines)
when_to_use: When you have a destination/recipient/terminal column. FDH (TERMINAL_ID), PaySim
(nameDest). Aggregate BY destination, not sender. Highest-leverage move for FDH.

---

## Part 6 — Modeling Methodology for Extreme Imbalance (lines 1236-1413)

### 6.1 Evaluation Hierarchy (lines 1242-1277, 36 lines)
when_to_use: Reference. AUPRC > AUROC at <1% positive rate, why CIs matter at small fraud counts,
how to read the composite score.

### 6.2 Imbalance Treatment Comparison (lines 1278-1321, 44 lines)
when_to_use: When tempted to try SMOTE/undersampling/class weights. Lessons: scale_pos_weight
+ focal loss outperform resampling. **Don't SMOTE.**

### 6.3 Recommended Ensemble Strategy (lines 1322-1371, 50 lines)
when_to_use: When XGBoost single-model has plateaued and you want to try a 2-3 model ensemble
with rank-averaging. Code patterns included.

### 6.4 Practical XGBoost Settings for Fraud (lines 1372-1413, 42 lines)
when_to_use: When tuning the model. Reference XGBoost hyperparameters that work on fraud data
(max_depth=6, lr=0.01, lambda=3, gamma=0.3 — exactly what Track B converged on).

---

## Part 7 — Advanced Sequence Modeling & Lifecycle (lines 1414-1789)

### 7.1 RFM Recency-Frequency-Monetary (lines 1420-1490, 71 lines)
when_to_use: When you have stable entity + ≥30 historical transactions per entity. RFM cluster
distance is a known fraud signal. Implementation patterns included.

### 7.2 HMM Fraud Lifecycle Modeling (lines 1491-1548, 58 lines)
when_to_use: Advanced — needs hmmlearn dependency. Models fraud as state sequence (normal →
testing → escalation → bust). Skip unless you've exhausted simpler features.

### 7.3 Transformer / Attention Sequence Models (lines 1549-1580, 32 lines)
when_to_use: Frontier — train a small per-card sequence transformer, extract embeddings as
features. Only attempt if you have a strong machine and 24h to spare.

### 7.4 State-Space Models (Mamba / S4) (lines 1581-1594, 14 lines)
when_to_use: Frontier reference only. No production examples in fraud yet. Read for awareness.

### 7.5 Feature-Group Autoencoder Anomaly Detection (lines 1595-1630, 36 lines)
when_to_use: When you want unsupervised anomaly score as a feature. Train a small AE on normals,
use reconstruction error as input. Implementation pattern included.

### 7.6 Dynamic Time Warping (DTW) Trajectory (lines 1631-1660, 30 lines)
when_to_use: Per-card transaction trajectories. Compute slope/variance/max-jump in fit(),
look up at transform(). Niche but cheap.

### 7.7 CUSUM Behavioral Shift Detection (lines 1661-1687, 27 lines)
when_to_use: When you suspect ATO (sudden behavior shift on a stable account). Cheap to compute,
gives a clean signal.

### 7.8 PU Learning and Adversarial Adaptation (lines 1688-1717, 30 lines)
when_to_use: When labeled fraud is rare and noisy. Less applicable to current datasets which
have cleaner labels.

### 7.9 Five-Tier Architecture Synthesis 2026 (lines 1718-1760, 43 lines)
when_to_use: **Read at the start of any new dataset.** Maps the 5 feature tiers (raw → entity
aggs → behavioral → graph/identity → embeddings) to which to attempt first based on data shape.

### 7.10 Multi-Source Joins (lines 1761-1789, 29 lines)
when_to_use: When the dataset has multiple tables (transaction + identity + device events).
IEEE-CIS has this structure. Reference for join-time feature design.

---

## Part 8 — Feature Engineering Checklist for LLM Agent (lines 1790-1863, 74 lines)

when_to_use: **Read at the start of every campaign.** Lettered checklist (A-I) of which feature
families to try in order. The most direct mapping from "what dataset is this?" to "what to build first?".

---

## References (lines 1864-1928, 65 lines)

when_to_use: Skip during experiments. Useful only for tracking down a specific paper or dataset URL.
