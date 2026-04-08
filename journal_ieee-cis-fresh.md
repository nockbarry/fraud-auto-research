# Journal: ieee-cis-fresh (Track B)

## Current Thesis (max 5 sentences)
IEEE-CIS is CNP fraud with 57 raw columns. The identity cluster (id_01-38, DeviceInfo, R_emaildomain) is the dominant raw signal — their null pattern (null_AUC ~0.65) indicates "was identity collected" and DeviceInfo has IV=1.78. SOTA has regularized XGBoost (max_depth=4) + cyclic time features (hour/dow from TransactionDT mod 86400). Next priority: add entity-sharing (identity consistency) and interaction features between high-IV identity columns. The freq encoding approach works well; focus on adding stable cross-entity signals.

## Active Campaign
- Goal: Regularization → TE → UID aggs → velocity stack (20-exp campaign)
- Step 1/5: Save baseline — status: done(exp_000)
- Step 2/5: Model regularization (max_depth=4, subsampling=0.8) + drop TransactionDT — status: done(exp_001) AUPRC_val=0.321 (+4.2%), PSI killed
- Step 3a/5: OOF TE (DeviceInfo, id_33, id_30, id_31, R_emaildomain, card1, P_emaildomain) + all id_* null flags — status: failed(exp_002) composite -0.021
- Step 3b/5: OOF TE variation — only ProductCD, card4, card6, card1, R_emaildomain (lower cardinality) without null flags — status: failed(exp_003/004) composite -0.004. Freq encoding of R_emaildomain already performing well; TE doesn't improve with current regularization.
- Step 4a/5: UID aggs card1+addr1 with 8 aggs + card1 aggs — status: failed(exp_005) AUPRC_val 0.321→0.280. UID without temporal context adds noise not signal.
- Step 5/5: Velocity stack (Recipe 16, card1, 1h/24h/7d windows) — status: planned
- Abandon criteria: abandon step only if 3 consecutive variations each lose >0.005 composite

## Column Analysis Key Findings
- DeviceInfo IV=1.78 (high_card), null_AUC=0.61 — most predictive raw column
- R_emaildomain IV=0.58 (high_card), null_AUC=0.65 — strong null signal
- id_31 IV=0.54, null_AUC=0.65 — strong null signal
- ProductCD IV=0.52 — flagged LEAK? but is low-card categorical (5 unique), not actual leak
- card3 IV=0.44, univ_AUC=0.63 — strong numeric signal
- id_17 IV=0.35, null_AUC=0.65 — important identity column
- id_cluster presence (null_AUC ~0.65 across many id_* cols) — already captured by id_cluster_present flag
- 27+ id_* columns with null_AUC > 0.55 — need null flags for each, or rely on id_cluster_present

## Open Questions
- Will the fresh-start agent independently discover that regularization is the first unlock?
- Which approach to model regularization will the agent choose (depth vs subsampling vs lambda)?
- Will the agent add TransactionDT cyclic encoding without being told about the PSI=12.43 problem?

## Lessons Learned (append-only, max 20 — prune oldest when full)
- exp_000: Baseline AUPRC_val=0.308, train_val_psi=0.294 (HIGH). TransactionDT PSI=12.43 is the dominant instability. DeviceInfo (IV=1.78) is being destroyed by freq encoding as high-cardinality — needs TE instead. Dead features: id_33_freq_enc, id_30_freq_enc (high-IV but high-card cols being killed by freq encoding).
- exp_001: Regularization (max_depth=4, subsample=0.8, min_child_weight=10) + drop TransactionDT → AUPRC_val 0.308→0.321 (+4.2%), composite 0.071→0.129, train_val_psi 0.294→0.121. Critical unlock: TransactionDT was key PSI source.
- exp_002: OOF TE for 7 high-card cols + all id_* null flags → discarded, composite -0.021. Too many new features (88 total) adding noise; null flags redundant with id_cluster_present. Try TE for fewer, lower-card columns only.
- exp_003/004: Focused OOF TE for 6 medium-card cols → discarded both, composite -0.004. Freq encoding already capturing signal; TE not improving with current regularization. Move to UID aggregation campaign.
- TE LESSON: Freq encoding is competitive with TE when model is well-regularized. Don't force TE — move to aggregation features instead.
- exp_005: UID aggs (card1+addr1, 8 aggs + amount zscore) → discarded, AUPRC_val 0.321→0.280 (-12%). Amount aggs from train data cause distribution shift when val has different spending patterns. Pure amount-based UID aggs are unstable.
- exp_006: Per-card1 D1 velocity + amount behavioral deviation → discarded, AUPRC_val 0.314 (SOTA 0.321). D columns are time-delta features. Slight PSI from per-card baseline lookups. New features not improving top features list.
- KEY INSIGHT: Current SOTA features are id_* freq encodings. The issue is these are already capturing most signal. Need model improvements or completely different feature types to break through.
- exp_007: LightGBM switch → massive fail, AUPRC_val 0.145 (vs XGBoost 0.321). LightGBM CPU much slower for this dataset, XGBoost GPU is clearly superior here.
- exp_008: Cyclic time features (hour_sin/cos, dow_sin/cos from TransactionDT % 86400) → KEPT, composite 0.1287→0.1330. PSI dropped from 0.121 to 0.043. Time-of-day is a real fraud signal and the modular extraction avoids PSI.
- exp_009: Identity consistency (per-card1 distinct email/device/addr counts + card_email_is_new flag + is_modal_device) → KEPT, composite 0.1330→0.1353. card_email_is_new importance=0.032. New card-email pairs are fraud-risky.
- exp_010: Extended identity (card_device_is_new + card_addr_is_new + card1_n_card6) → discarded, composite 0.1313. DeviceInfo 78% NaN means device pair flag mostly compares NaN values. addr1 11% NaN more useful but not sufficient.
- DISCOVERY: id_29/id_35/id_36-38 are binary (Found/NotFound, T/F) — freq encoding loses the actual binary signal! id_30=OS, id_31=browser, id_33=screen are high-card (71/108/183 levels) — need TE. id_23=proxy type is a direct fraud signal.
- exp_011: Binary id_* → 0/1 + id_30/31/33/DeviceInfo → TE → FAILED, AUPRC 0.218 (-0.015). TE on 85%+ NaN columns collapses to global_mean. Freq encoding of binary cols actually works fine (gives slight variant signal). DO NOT TE identity columns.
- exp_012: Deeper XGBoost (max_depth=5, lr=0.02, n_est=3000, gamma=0.1, lambda=2.0) → KEPT, AUPRC_val 0.319→0.344 (+8%), composite 0.1353→0.1393. Deeper trees extract more from existing features. Model capacity improvement.

## Discarded Theses (graveyard, max 5)
- (fresh start)
