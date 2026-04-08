# Journal: ieee-cis

## Current Thesis
IEEE-CIS is CNP fraud with 57 raw columns. Identity cluster (id_*/DeviceInfo/R_emaildomain) is dominant signal. Model regularization (max_depth=4, subsampling) was the key unlock — baseline PSI was maxed out preventing any FE gains. On regularized model: TE, UID aggs, and velocity all improve composite. High-card entity TE (card1) causes OOT regression — TE only works for low-to-medium cardinality.

## Campaign 1 (complete)
- Regularization + TE. Final: composite=0.1357, AUPRC_val=0.3227

## Campaign 2: UID + Velocity (complete)
- UID aggs: train_val_psi dropped 0.132→0.032; velocity: P@R_val 0.083→0.095
- Failed: card1 TE (OOT regression), amount zscore (PSI spike)

## Campaign 3: Interaction Features + Identity Depth (COMPLETE — all steps failed/discarded)
- All steps discarded. addr1_te marginally useful in top10 but not enough to overcome PSI penalty.

## Active Campaign 4: DeviceInfo TE + High-IV id columns
- NOTE: D/M/C columns were stripped from IEEE-CIS (not available). Pivoting to identity signals.
- Goal: DeviceInfo (IV=1.778, highest in dataset, currently freq-encoded) should get OOF TE with high smoothing. id_30 (IV=0.619, 71 unique), id_33 (IV=0.879, 183 unique), id_31 (IV=0.538, 108 unique) — all freq-encoded, try smoothed TE.
- Step 1/4: DeviceInfo OOF TE with high smoothing (1546 unique — need K-fold TE or strong smoothing ~50) — status: planned
- Step 2/4: id_30 smoothed TE (71 unique, smoothing=20) + id_33 smoothed TE (183 unique, smoothing=30) — status: planned
- Step 3/4: Dead feature pruning — remove 13 dead features (IV<0.005): id_33_freq_enc, txn_dow_cos, id_30_freq_enc, dist2, id_11, id_18, id_21, id_22, id_07, id_10 — status: planned
- Step 4/4: id_31 TE (108 unique) + card2 TE (500 unique, smoothing=50) + card5 TE (110 unique) — status: planned
- Abandon criteria: abandon step only if 3 consecutive variations each lose >0.005 composite

## Active Campaign 5: UID Level 2 + Velocity Depth
- Goal: UID2 = card1+card5 for finer entity resolution; additional velocity features (card5-level or email-domain-level)
- Step 1/3: UID2 = card1+card5 aggregations (mean/std/min/max/count for amount) — status: planned
- Step 2/3: card1 OOF TE using K-fold (Recipe 1) — careful re-test now PSI is 0.013 — status: planned
- Step 3/3: Combined UID1 + UID2 aggregations — status: planned
- Abandon criteria: abandon step only if 3 consecutive variations each lose >0.005 composite

## Lessons Learned
- exp_000: Baseline composite=0.0705; TransactionDT PSI=12.43; id_17 top feature.
- exp_001-003: All TE/FE changes discarded; root cause was PSI maxed out from baseline overfit.
- exp_004: Model regularization composite 0.0705→0.1111 (+57%); key unlock; tv_psi 0.294→0.155.
- exp_005: TE replaces freq for 5 cols: composite 0.1111→0.1240; R_emaildomain_te top; tv_psi→0.132.
- exp_006: UID=card1+addr1 7 aggs: composite 0.1240→0.1296; tv_psi 0.132→0.032! UID aggs are highly stable.
- exp_007: UID zscore+IQR+q25/q75: discard; PSI 0.032→0.125; card-specific features cause temporal drift.
- exp_008: Per-card1 velocity (5 features): composite 0.1296→0.1357; P@R_val 0.083→0.095.
- exp_009: card1 TE: discard; AUPRC_oot 0.246→0.202; high-card TE causes OOT regression; avoid.
- exp_010-013: TE experiments for medium-card cols: all discard; TE hurts OOT even for 60-300 unique cols.
- exp_014: Cyclic time (sin/cos hour/dow + log_amt + drop TransactionDT): KEEP; composite 0.1357→0.1424; tv_psi 0.085→0.013! AUPRC_oot 0.246→0.264.
- exp_015-023: Various TE, pruning, UID2 — all discard; id_30/31/33 TE displaces id_17; drop-TransactionID improves OOT (0.277!) but loses val; amt_pattern features are PSI-stable.
- exp_024: Amount patterns (has_cents, cents, round flags, threshold tests): KEEP composite 0.1424→0.1479 (+3.8%); amt_has_cents=0.0543 top feature; tv_psi dropped 0.013→0.005! Key insight: stateless deterministic features are maximally PSI-stable.
- RULE: Stateless deterministic transforms (cyclic time, log, amount patterns) are best for PSI. TE/UID aggs have temporal drift risk.

## Discarded Theses
- card1 OOF TE as entity signal (exp_009 showed severe OOT regression)
