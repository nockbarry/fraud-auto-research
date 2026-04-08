# Journal: fdh

## Current Thesis (max 5 sentences)
FDH signal is driven by: (1) rolling terminal fraud rate 28-day window capturing scenario 2 terminal compromise; (2) customer behavioral deviation (amt_vs_p90) for scenario 1 high-amount; (3) velocity (1h/6h/1d/7d) per customer and terminal for scenario 3 CNP. Cyclic time (sin/cos hour/dow, is_night/weekend) adds meaningful signal. The AUROC train→val gap (~0.19) persists because velocity features computed per-split still differ slightly even with training-tail prepended.

## Active Campaign
- Goal: Reach AUPRC_val 0.40+ by systematically covering all 3 documented fraud scenarios
- Step 1: velocity stack (1h/6h/24h/7d count+sum per CUSTOMER_ID + TERMINAL_ID) + customer behavioral deviation — status: done(exp_002)
- Step 2: rolling terminal fraud rate 28-day window (Recipe 17) — status: done(exp_003)
- Step 3: cyclic time (sin/cos hour/dow, is_night/weekend) — status: done(exp_008)
- Abandon criteria: 3 consecutive steps each lose >0.005 composite

## Open Questions
- Why does term_static_fraud_rate cause AUROC_train=0.998? Scenario 2 terminal rotation means static rates don't generalize — confirmed.
- Can customer distinct-terminal velocity (count per 24h) capture scenario 3 CNP better than velocity count alone?

## Lessons Learned (append-only, max 20 — prune oldest when full)
- exp_000: baseline AUPRC_val=0.13 — TX_AMOUNT dominates (0.63), raw ordering proxies dominate
- exp_001-002: velocity + behavioral deviation → AUPRC_oot=0.25; training-tail prepend critical to reduce PSI from 0.245→0.183
- exp_003: rolling terminal fraud rate 28d → composite jumps 0.025→0.095; train_val_psi drops to 0.006
- exp_004-005: cust_term_is_new and time_since_last_tx cause PSI=0.79 — these features have totally different distributions in train vs val (train=0, val=non-zero)
- exp_007: amt_vs_p90*term_fraud_rate interaction causes PSI=0.94 — interaction amplifies OOT degradation of rolling feature
- exp_008: cyclic time (no interactions) → composite 0.095→0.120, AUPRC_oot 0.234→0.293 — cyclic features must be computed BEFORE dropping TX_DATETIME
- exp_009: model regularization (depth=4, subsample=0.8) → discarded, reduced AUPRC more than helped gap
- exp_011: term_static_fraud_rate causes AUROC_train=0.998 — scenario 2 terminal compromise rotates, static historical rates memorize training but don't generalize

## Discarded Theses (graveyard, max 5)
- "Customer-terminal novelty (first-visit) improves OOT" — disproven, causes split-leaky feature with PSI=0.79
- "Static terminal fraud rate helps as stable alternative to rolling" — disproven, causes terminal ID memorization (AUROC_train=0.998)
