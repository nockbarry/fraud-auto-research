# Journal: ieee-cis

## Current Thesis
IEEE-CIS is CNP fraud with 57 raw columns. Identity cluster (id_*/DeviceInfo/R_emaildomain) is dominant signal. Model regularization (max_depth=4→5, gamma+lambda tuned) + stateless deterministic features (amount patterns, anomaly score) are the main levers. TransactionID causes PSI=12.43 val→OOT drift but removing it hurts val-based composite while improving OOT. The val/OOT trade-off is the key tension to resolve.

## Campaign 1 (complete)
- Regularization + TE. Final: composite=0.1357, AUPRC_val=0.3227

## Campaign 2: UID + Velocity (complete)
- UID aggs: train_val_psi dropped 0.132→0.032; velocity: P@R_val 0.083→0.095
- Failed: card1 TE (OOT regression), amount zscore (PSI spike)

## Campaign 3: Interaction Features + Identity Depth (COMPLETE — all steps failed/discarded)
- All steps discarded. addr1_te marginally useful in top10 but not enough to overcome PSI penalty.

## Active Campaign 8: OOT Improvement — Remove TransactionID + Better Model
- Key finding: TransactionID (PSI=12.43) inflates val AUPRC but hurts OOT. Removing it gives OOT=0.287 but val composite drops.
- Strategy: combine TxID removal with features that recover val signal.
- Step 1/3: max_depth=5 + drop TransactionID + additional stateless features (more amt patterns) — status: planned
- Step 2/3: Drop TransactionID + optimize anomaly score features for val recovery — status: planned
- Step 3/3: Ensemble of max_depth=4 (val-optimized) and max_depth=5 (OOT-optimized) — status: planned

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
- exp_025: Anomaly score (Mahalanobis distance 20 numeric features): KEEP composite 0.1479→0.1500 (+1.4%); AUPRC_oot 0.258→0.262; PSI stable.
- exp_026-033: Various TE (id_31/33/30), category deviation, email aggs, identity consistency, entity sharing, addr1 aggs — all discard; interference with existing strong features.
- exp_034: max_depth=5 + gamma=1.5 + reg_lambda=7.0: KEEP composite 0.1500→0.1519 (+1.3%); AUPRC_oot 0.262→0.276! tv_psi=0.054 (higher but OOT improves).
- exp_035: max_depth=5 + drop TransactionID: discard (composite 0.1495); but AUPRC_oot=0.287 (best!); tv_psi=0.002; val-based selection penalizes TxID removal despite OOT gain.
- RULE: Stateless deterministic transforms (cyclic time, log, amount patterns) are best for PSI. TE/UID aggs have temporal drift risk.
- RULE: max_depth=5 with stronger regularization beats max_depth=4 with current 77-feature set. Expressive depth + regularization is the right balance.

## Discarded Theses
- card1 OOF TE as entity signal (exp_009 showed severe OOT regression)
