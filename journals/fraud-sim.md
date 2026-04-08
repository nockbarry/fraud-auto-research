# Journal: fraud-sim

## Current Thesis (max 5 sentences)
The dominant fraud signal in fraud-sim is geographic (haversine distance from card home to merchant) combined with rapid velocity (1h/10min count per card). Behavioral deviation (amount z-score vs card mean) adds secondary signal. Category and merchant TE add risk signal but are partially redundant with freq_enc. Per-card diversity (distinct merchants/categories in training) is a weak but additive signal. Population shift is not causing OOT degradation (PSI < 0.001) because the features are behavioral/structural rather than temporal.

## Active Campaign
- Goal: Reach AUPRC_oot 0.65+ while keeping score PSI < 0.20 — ACHIEVED (0.947)
- Step 1: haversine distance + velocity (1h/1d/7d per card_id) + behavioral deviation + cyclic time — status: done(exp_001)
- Step 2: smoothed merchant_te + category_te — status: done(exp_003)
- Step 3: demographic features (log_city_pop, amt_per_city_pop) — status: done(exp_007)
- Abandon criteria: N/A — campaign completed

## Open Questions
- Why did adding state_te + job_te simultaneously (exp_008) cause regression to 0.649 AUPRC?
- Does per-card distinct merchant count proxy for card age/activity level rather than fraud risk?

## Lessons Learned (append-only, max 20 — prune oldest when full)
- exp_000: baseline AUPRC_oot=0.69 — TransactionAmt + category_freq dominant
- exp_001: haversine + velocity + behavioral → AUPRC_oot 0.69→0.941 — single biggest jump ever seen
- exp_003: merchant_te + category_te replacing freq_enc → marginal improvement 0.941→0.946
- exp_004: behavioral fingerprint (hour deviation, dist_zscore) → ZERO effect — haversine already saturates distance signal
- exp_007: log_city_pop + amt_per_city_pop → AUPRC_oot 0.946→0.950 (kept)
- exp_008: replacing state_freq_enc + job_freq_enc with TE simultaneously → massive regression to 0.649 — unknown cause, avoided by restoring SOTA
- exp_009: state_te additive (keep existing freq_enc) → discarded, hurt slightly
- exp_010: per-card distinct merchant/category count → AUPRC_oot 0.950→0.947 (marginal keep on composite)

## Discarded Theses (graveyard, max 5)
- "Behavioral fingerprint (unusual hour, unusual distance for this card) adds signal" — disproven, haversine absolute distance already captures distance signal
