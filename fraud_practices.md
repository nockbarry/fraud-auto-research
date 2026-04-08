# Fraud Detection: Datasets, Feature Engineering & Model Design Reference

A practical reference for LLM-driven autonomous feature engineering and model development on transaction data.
Covers public datasets, fraud type taxonomy, concrete feature formulas, state-of-the-art model architectures, foundation models, competition-validated techniques, and production deployment patterns.

**Last updated:** April 2026. Incorporates findings from TabPFN-2.5, Stripe's Payments Foundation Model, TALENT/TabArena benchmarks, IEEE-CIS and AmEx competition post-mortems, and the Grinsztajn et al./McElfresh et al. GBDT-vs-DL debate.

---

## Part 1: The Current Landscape — What Actually Works

### 1.1 The GBDT vs. Deep Learning Debate (Resolved, Mostly)

The central question in tabular fraud ML — whether gradient-boosted trees or neural networks perform better — has been extensively benchmarked. The answer is nuanced but actionable.

**The foundational papers:**
- **Grinsztajn, Oyallon, Varoquaux (NeurIPS 2022):** "Why do tree-based models still outperform deep learning on tabular data?" identified three structural NN disadvantages: sensitivity to uninformative features, failure to preserve data orientation, and difficulty learning irregular target functions.
- **McElfresh et al. (NeurIPS 2023):** Tested 19 algorithms across 176 datasets and concluded the NN-vs-GBDT debate is **overemphasized** — on many datasets, the gap is negligible with proper tuning.
- **TALENT benchmark (Ye et al., 2024, revised Nov 2025):** Evaluated 300+ datasets; tree ensembles remain highly competitive, but top DL architectures (TabPFN v2, TabICL, RealMLP, TabR, ModernNCA) now match or exceed GBDTs on significant subsets.
- **TabArena (NeurIPS 2025):** Living benchmark confirming deep learning has closed the gap with sufficient time budgets and ensembling.

**The practical hierarchy for fraud detection:**

For **production real-time scoring** (sub-100ms latency, billions of transactions): **XGBoost/LightGBM/CatBoost** remain dominant. They deliver sub-millisecond inference, support SHAP-based explainability required by GDPR and PCI-DSS, handle class imbalance natively via `scale_pos_weight`, and retrain daily as fraudsters evolve. The performance gap between these three libraries is statistically insignificant per TALENT — library choice is secondary to feature engineering quality.

For **offline enrichment and embedding generation**: Foundation models and transformers excel. Stripe's Payments Foundation Model, Featurespace's NPPR, and graph neural networks generate dense behavioral embeddings that become input features for the GBDT classifier.

For **small-to-medium datasets** (<10K–50K samples): **TabPFN-2.5** achieves a near-100% win rate against default XGBoost on classification datasets under 10K samples, matching a 4-hour AutoGluon ensemble in 2.8 seconds. This is the right starting point when labeled fraud data is scarce.

**Deep learning architectures that underperformed expectations:**
- **TabNet** (Google, 2021): High variability and lack of robustness per TALENT.
- **SAINT**: Inconsistent gains from intersample attention.
- **NODE** (Neural Oblivious Decision Ensembles): Superseded by simpler methods.
- **FT-Transformer**: Competitive on specific datasets but doesn't consistently beat GBDTs.

**What does work in deep learning for tabular:**
- **TabM** (Gorishniy et al., ICLR 2025, Yandex Research): Best average rank across 46 datasets using parameter-efficient MLP ensembling. The key finding: "the complexity of attention- and retrieval-based methods does not pay off."
- **RealMLP**: A "bag of tricks for MLPs" approach that's competitive with far less complexity than transformers.
- **ModernNCA**: Retrieval-augmented approach using nearest-neighbor classification with learned embeddings.

**Agent implication:** Default to XGBoost/LightGBM. Invest effort in feature engineering over model architecture. Consider TabPFN-2.5 for small datasets. Use deep learning for embedding generation, not as the final classifier.

---

### 1.2 Foundation Models for Tabular Data

The most significant 2024–2026 development is foundation models that perform in-context learning without per-dataset training.

**TabPFN (Prior-Fitted Networks) — Prior Labs / Hollmann, Müller, Hutter:**
- **v1** (ICLR 2023): Limited to 1,000 rows, 100 numerical features
- **v2** (Nature, Jan 2025): 10K rows, 500 features, categoricals, missing values, regression — matching 4-hour AutoGluon in 2.8 seconds
- **v2.5** (Nov 2025): 50K rows, 2K features, 20× data capacity increase, 87% win rate vs. tuned XGBoost on datasets up to 100K samples
- **v2.6** (Enterprise): Up to 10M rows via distillation to compact MLPs or trees

H2O.ai testing showed TabPFN achieving **73% recall** vs. LightGBM's 47% on insurance fraud with extreme class imbalance. Prior Labs lists automotive finance fraud as a production use case.

**For fraud detection specifically:** TabPFN's size constraint (50K rows in v2.5) limits direct application to production fraud systems processing billions of transactions. However, the distillation engine in v2.6 compresses TabPFN's knowledge into deployable tree or MLP models. The most promising workflow: use TabPFN for rapid prototyping and feature importance analysis, then distill to a production-ready model.

**Domain-specific foundation models:**
- **Featurespace NPPR** (ACM ICAIF 2023): Pretrained on **5.1 billion transactions from 180 European banks** using a GRU architecture with next-event prediction and past-reconstruction objectives. Generates contextualized transaction embeddings that transfer across institutions.
- **Feedzai RiskFM** (2025/2026): First tabular foundation model purpose-built for financial data. Uses Mixture of Experts with federated learning across institutions processing $8–9 trillion annually. Reportedly matches bespoke supervised models out-of-the-box.
- **Stripe's Payments Foundation Model** (May 2025): Transformer that tokenizes each charge (card BIN, merchant category, amount, IP, device) and learns the "grammar of payments" via self-supervised pretraining. Card-testing detection improved from 59% to 97% overnight with no increase in false positives.

**Other foundation model approaches:**
- **CARTE** (Kim et al., 2024): Graph representations with pretrained text embeddings for cross-table transfer learning — promising for heterogeneous financial data sources.
- **XTab** (ICML 2023): Cross-table pretraining via federated-style transformer decomposition.
- **XTFormer** (2024): Beats XGBoost and CatBoost on 72% of 190 downstream tasks.
- **UniPredict**: LLM as universal tabular predictor, 100%+ improvement over XGBoost in low-resource settings.
- **TabICL** (2025): Scales in-context learning to 500K+ samples.

**Agent implication:** Foundation model embeddings as features into XGBoost is the emerging dominant pattern. When available, use NPPR-style transaction embeddings or TabPFN distilled representations as additional input features alongside hand-engineered features.

---

### 1.3 Competition-Validated Techniques

The most instructive evidence about what works comes from Kaggle competitions where thousands compete on identical datasets.

**IEEE-CIS Fraud Detection (2019, 6,381 teams):**
Winner: FraudSquad (including Chris Deotte). Private LB AUC: **0.9459**. Ensemble of XGBoost, CatBoost, LightGBM. The decisive breakthrough was feature engineering:
- **UID construction**: `card1 + addr1 + D1` → unique client identifier. This single entity-resolution insight boosted local CV from ~0.90 to ~0.948.
- **262 group-aggregation features**: Mean, std, count of transaction amounts and time deltas per client.
- **D-column normalization**: Subtracting D1 from transaction day → "days since card first used."
- **V-feature reduction**: Correlation-based removal of redundant Vesta-engineered features.
- **Post-processing**: Replaced individual predictions with client-average predictions.

**American Express Default Prediction (2022, 4,875 teams):**
Winner: "jxzly" with 7-stage pipeline: denoise → manual features → series features → feature combination → LightGBM → neural network → weighted ensemble. Key: aggregating time-series billing data (190 features × monthly statements) into customer-level features using mean, std, min, max, last-value aggregations. Winning blend: LightGBM 65% + neural networks 25%. Chris Deotte placed 15th with a Transformer + LightGBM knowledge distillation approach.

**Home Credit — Credit Risk Model Stability (2024):**
Introduced Gini stability metric penalizing models that degrade over time. Winners used LightGBM + CatBoost with StratifiedGroupKFold by weekly temporal grouping. Rewarded robustness over peak accuracy.

**Consistent patterns across all major tabular competitions (2024–2025):**
- **GBDTs dominate**: 16 winning solutions used LightGBM in 2024 alone.
- **Ensembling is mandatory**: Winners blend 10–70+ models. Chris Deotte won April 2025 Kaggle Playground with a 3-level stack of 72 models.
- **Feature engineering is the primary differentiator**: Model architecture matters far less than feature quality.
- **AutoGluon** appeared in 2 winning solutions in 2025, sometimes beating manually tuned ensembles.

**Expert practitioners to follow:**
- **Chris Deotte** — 4× Kaggle Grandmaster, NVIDIA. 1st in IEEE-CIS Fraud Detection. Canonical "XGB Fraud with Magic" notebook (0.9600 AUC). Advocates GPU-accelerated brute-force feature search via RAPIDS cuML.
- **Gilberto Titericz Jr. (Giba)** — 13× Kaggle Grandmaster, formerly #1 worldwide. 2nd in IEEE-CIS.
- **Jean-François Puget (CPMP)** — NVIDIA Distinguished Engineer, 2× Grandmaster. 2nd in IEEE-CIS. Leads NVIDIA's KGMoN team.
- **Bojan Tunguz** — 4× Grandmaster, first Top 10 in all four Kaggle categories. Won Home Credit Default Risk (7,198 teams).
- **Kazuki Onodera** — Deep learning expertise for structured data.
- **NVIDIA KGMoN team** published "The Kaggle Grandmasters Playbook: 7 Battle-Tested Modeling Techniques for Tabular Data" (2025): smarter EDA, diverse baselines, massive feature engineering, hill-climbing ensembles, model stacking, pseudo-labeling, domain-specific training tricks.

---

### 1.4 What Production Systems Actually Deploy

**Stripe Radar:** Processes $1.4T+ annually. Neural networks on billions of transactions. Assesses 1,000+ characteristics per transaction. Retrains daily. PFM produces dense behavioral embedding per transaction in <100ms. 92% of cards on network seen before, enabling cross-merchant intelligence.

**Visa VAAI:** Scores 300 billion transactions/year using 500+ attributes. Generative AI trained on 15B+ VisaNet transactions with 6× more features than previous models. 85% FPR reduction. Blocked $40B in fraud in year ending Sept 2023.

**Mastercard Decision Intelligence:** Scores 143 billion transactions/year in 50ms using proprietary RNN + transformer. Up to 300% improvement in fraud detection, 50% reduction in false declines. Uses graph + generative AI to predict full compromised card numbers from partial data.

**JP Morgan Chase:** $9B+ invested in neural network development. TigerGraph for real-time graph analysis of 50M+ daily transactions with sub-80ms response. $50M savings from graph-based fraud detection. Project AIKYA: federated learning for cross-institutional fraud detection.

**Capital One:** XGBoost/TensorFlow/PyTorch on AWS. Announced at KDD 2025 a strategic shift "From Features to Sequences" — transitioning from traditional feature-engineered GBDTs to transformer architectures.

**PayPal:** Hybrid Random Forest + neural networks. Quokka shadow platform reduces model deployment time by 80%.

**Consistent production patterns:**
- Sub-100ms real-time scoring
- 500–1,000+ features per transaction
- Daily or more frequent retraining
- Shadow/canary deployment environments
- Hybrid: rules + ML models + foundation model embeddings
- Class imbalance via massive data volume + cost-sensitive learning, **not** SMOTE (practitioners increasingly view SMOTE as problematic for heterogeneous fraud data)

---

## Part 2: Public Fraud Detection Datasets

### Dataset Selection Criteria

For autonomous feature engineering to be meaningful, a dataset needs:
1. **Raw / interpretable features** — not PCA-anonymized columns
2. **Timestamps** — required for out-of-time (OOT) splits and velocity features
3. **Entity IDs** — customer, card, or account IDs to build behavioral profiles
4. **Reasonable size** — >50K rows for stable model evaluation
5. **Realistic fraud rate** — very low rates (<0.5%) make evaluation fragile without careful stratification

---

### Tier 1 — Recommended (Raw Features + Temporal Structure)

#### 1. IEEE-CIS Fraud Detection (Vesta/Kaggle, 2019)
- **URL:** https://www.kaggle.com/competitions/ieee-fraud-detection
- **Size:** ~590K train transactions (+ 507K test); identity table: ~144K rows; 394 columns total
- **Fraud rate:** 3.5%
- **Feature types:**
  - `TransactionDT`: time delta in seconds from a reference point (not a real timestamp, but monotonically increasing — usable for OOT split)
  - `TransactionAmt`: transaction amount
  - `ProductCD`: product category (W/H/C/S/R)
  - `card1–card6`: card attributes (type, bank, country, etc.)
  - `addr1, addr2`: billing/postal area codes
  - `P_emaildomain, R_emaildomain`: purchaser and recipient email domains
  - `C1–C14`: count features (number of addresses associated with payment card, etc.)
  - `D1–D15`: timedelta features (days between events — e.g., last transaction, account age)
  - `M1–M9`: match features (name-on-card match, address match, etc.) as True/False/NaN
  - `V1–V339`: 339 Vesta-engineered features (ranking, counting, entity relations) — pre-derived but still provide rich signal
  - `id_01–id_11`: numerical identity features (device rating, proxy rating, login failure counts, time-on-page)
  - `id_12–id_38`: categorical identity features (network connection type, browser, OS, device info)
  - `DeviceType`, `DeviceInfo`
- **What makes it interesting:**
  - Real e-commerce CNP transactions from Vesta Corporation
  - Rich identity table joinable via `TransactionID`
  - C and D column families require reverse-engineering to understand semantics — a real agent challenge
  - V-columns are pre-engineered but still have signal worth learning from
  - Top solutions built UIDs from `card1 + addr1 + D1` to create customer proxies, then aggregated over them
- **Raw engineering potential:** HIGH. Raw card/address/email/device attributes support velocity, graph, and behavioral features. D-columns hint at recency structure an agent can exploit.
- **OOT split:** Yes — sort by `TransactionDT`, hold out the last ~20% of time.
- **Competition winning techniques:** UID construction from card1+addr1+D1, 262 group-aggregation features, D-column normalization, V-feature correlation pruning, client-average post-processing. See Part 1.3 for details.

---

#### 2. Sparkov Simulated Credit Card Transactions (kartik2112/Kaggle, 2020)
- **URL:** https://www.kaggle.com/datasets/kartik2112/fraud-detection
- **Size:** ~1.3M train rows, 20K test rows; 17 features (10 categorical, 6 numerical, 1 text)
- **Fraud rate:** 5.7%
- **Feature types:**
  - `trans_date_trans_time`: real datetime — direct OOT splits possible
  - `cc_num`: credit card number (customer proxy)
  - `merchant`, `category`: merchant name and MCC-equivalent category
  - `amt`: transaction amount (USD)
  - `first`, `last`, `gender`, `street`, `city`, `state`, `zip`: cardholder PII
  - `lat`, `long`: cardholder location
  - `city_pop`: city population
  - `job`, `dob`: occupation and date of birth
  - `merch_lat`, `merch_long`: merchant location
  - `unix_time`: unix timestamp
- **What makes it interesting:**
  - Geo coordinates for both cardholder and merchant — distance-from-home is a classic CNP/CP feature
  - Full PII enables email/name-pattern features (though synthetic)
  - Simple enough for rapid iteration, large enough for robust evaluation
  - Generated by Sparkov Data Generation tool, covers 1,000 customers × 800 merchants × 6 months
- **Raw engineering potential:** VERY HIGH. Amount, geo, time, category, and customer identity all available in interpretable form.
- **OOT split:** Yes — real timestamps, easy chronological split.
- **Limitation:** Synthetic; fraud patterns may be simpler than real-world distributions.

---

#### 3. Fraud E-Commerce (fraudecom / Kaggle vbinh002)
- **URL:** https://www.kaggle.com/datasets/vbinh002/fraud-ecommerce
- **Size:** ~151K rows (120K train, 30K test); 6 features + 1 enrichable
- **Fraud rate:** 10.6%
- **Feature types:**
  - `signup_time`, `purchase_time`: dual timestamps — account age is derivable
  - `purchase_value`: transaction amount
  - `device_id`: device identifier
  - `source`: traffic source (ads, SEO, direct)
  - `browser`: browser type
  - `ip_address`: real IP addresses — enrichable with geo/ASN/proxy data
  - `sex`, `age`: demographic features
- **What makes it interesting:**
  - Real IP addresses for geo/ASN enrichment (rare among public datasets)
  - Time-between-signup-and-purchase is a strong fraud signal; the raw data makes this derivable
  - Small feature set means an agent must engineer nearly everything — ideal test of creativity
  - High fraud rate (10.6%) makes evaluation metrics more stable
- **Raw engineering potential:** HIGH relative to feature count. IP enrichment, account-age, device velocity, and browser anomaly features all derivable.
- **Caveat:** No customer ID separate from device — device sharing = customer re-use assumption.

---

### Tier 2 — Useful with Caveats

#### 4. PaySim Synthetic Mobile Money (Lopez-Rojas et al., 2016 / Kaggle ealaxi)
- **URL:** https://www.kaggle.com/datasets/ealaxi/paysim1
- **Size:** ~6.3M rows (1/4 scale of original 24M); ~10 features
- **Fraud rate:** ~0.13% on TRANSFER/CASH_OUT; ~1.3% labeled overall
- **Feature types:**
  - `step`: time step (1 step = 1 hour; 744 steps = 30 days)
  - `type`: CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER
  - `amount`: transaction amount
  - `nameOrig`, `nameDest`: originator and destination account IDs
  - `oldbalanceOrg`, `newbalanceOrig`: balance before/after for sender
  - `oldbalanceDest`, `newbalanceDest`: balance before/after for recipient
  - `isFraud`, `isFlaggedFraud`: labels
- **What makes it interesting:**
  - Balance-based features: `amount - (oldbalanceOrg - newbalanceOrig)` flags balance inconsistencies
  - Transfer + CASH_OUT chains are the fraud pattern; graph features across `nameOrig`/`nameDest` are essential
  - Fraud is restricted to TRANSFER and CASH_OUT types only — agent needs to discover this
- **Raw engineering potential:** MEDIUM. Balance deltas, transaction chains, and step-based velocity are all derivable. Limited feature set constrains diversity.
- **OOT split:** Yes — `step` provides temporal ordering.
- **Limitation:** Fraud rate is very low; class imbalance is extreme.

---

#### 5. BankSim (Kaggle / López-Rojas)
- **URL:** https://www.kaggle.com/datasets/ealaxi/banksim1
- **Size:** ~594K rows (7,200 fraudulent)
- **Fraud rate:** ~1.2%
- **Feature types:** `step` (time), `customer`, `age`, `gender`, `zipcodeOri`, `merchant`, `zipMerchant`, `category`, `amount`
- **What makes it interesting:**
  - Customer + merchant IDs allow behavioral profiling
  - MCC-equivalent category enables spending-pattern features
  - Based on aggregated data from a Spanish bank — somewhat realistic distributions
- **Raw engineering potential:** MEDIUM. Simpler than Sparkov but interpretable.

---

### Tier 3 — Not Recommended for Raw Feature Engineering

#### 6. Credit Card Fraud Detection (ULB/Worldline, mlg-ulb/Kaggle)
- **URL:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Size:** ~284K rows over 2 days
- **Fraud rate:** 0.17%
- **Features:** V1–V28 (PCA components, fully anonymized), `Time`, `Amount`
- **Verdict:** Only `Time` and `Amount` are raw. PCA destroys semantic meaning — an agent cannot engineer interpretable features. Only useful for testing classification techniques on highly imbalanced data.

#### 7. Elliptic Bitcoin Dataset (Elliptic / Kaggle)
- **URL:** https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
- **Size:** 203,769 nodes, 234,355 directed edges; 166 features per node
- **Fraud rate:** ~2% labeled illicit (21% unknown)
- **Features:** 94 local transaction features (inputs/outputs, fees, volumes) + 72 aggregated neighborhood features
- **Verdict:** Useful for GNN/graph research; features are partially interpretable but oriented toward blockchain specifics. The neighborhood features are pre-aggregated. Good for graph feature engineering experiments, not traditional tabular fraud.

---

### Dataset Summary Table

| Dataset | Rows | Fraud Rate | Has Timestamps | Has Entity IDs | Raw Features | Agent Suitability |
|---------|------|-----------|---------------|----------------|--------------|-------------------|
| IEEE-CIS | 590K | 3.5% | Partial (DT delta) | Card IDs | Partial (V-cols pre-eng.) | Excellent |
| Sparkov | 1.3M | 5.7% | Yes (real datetime) | Card number | Yes (geo, MCC, PII) | Excellent |
| fraudecom | 151K | 10.6% | Yes (2 timestamps) | Device ID | Yes (IP, device) | Very Good |
| PaySim | 6.3M | 0.13% | Yes (step) | Account IDs | Yes (balances) | Good |
| BankSim | 594K | 1.2% | Yes (step) | Customer ID | Yes | Good |
| ULB CC | 284K | 0.17% | Yes | None | No (PCA) | Poor |
| Elliptic | 203K nodes | 2% | Yes (time step) | Node IDs | Partial | Niche (graph) |

---

## Part 3: Fraud Detection Feature Engineering Reference

### 3.1 Fraud Type Taxonomy

---

### Type 1: Card Not Present (CNP) / E-Commerce Fraud

**Definition:** Stolen card credentials (PAN, expiry, CVV) used for online purchases where physical card possession is not verified.

**Key behavioral signals:**
- Transaction from a new device not previously associated with the card
- Shipping address differs from billing address
- High-value orders placed immediately after card number change
- Rapid sequential transactions across multiple merchants in a short window
- Mismatched email domain patterns (e.g., random-looking strings, burner domains like `mailinator.com`)
- IP geolocation inconsistent with cardholder's historical location
- Orders to high-resale-value categories (electronics, gift cards, jewelry) from new customers
- Card testing: very small amounts ($0.01–$1.00) before a large transaction

**Most predictive feature categories:**
1. Device novelty (new device + card combination)
2. Velocity of transactions on the card in the last 1h/24h
3. Email domain risk (free webmail, known temp-mail providers)
4. Geo-distance between IP and billing address
5. Time-since-last-transaction on this card
6. Merchant category novelty for this cardholder

**Anti-patterns / features that don't generalize:**
- Raw transaction amount alone — fraudsters adjust to thresholds
- Single-feature device rules — device sharing is common in legitimate households
- Country-level IP blocks — VPNs and proxies trivially bypass them
- Velocity rules without self-normalization — legitimate customers' baseline velocity varies enormously

---

### Type 2: Account Takeover (ATO)

**Definition:** Attacker obtains credentials to a legitimate account (via phishing, credential stuffing, data breach) and conducts fraudulent transactions from it.

**Key behavioral signals:**
- Login from a previously unseen device or IP range
- Password reset followed quickly by high-value transaction
- Change to shipping address, email, or phone followed by transaction
- Login at unusual time-of-day relative to the account's historical pattern
- Failed login attempts just before a successful login
- Sudden shift in transaction velocity, amount, or merchant category after the login event
- Geographic impossibility: login from City A, transaction from City B within minutes
- Multiple accounts logging in from the same IP or device (credential stuffing cluster)

**Most predictive feature categories:**
1. Session/behavioral biometrics at login (typing speed, mouse movement — not always available in datasets)
2. Device-change flag: `first_time_device_for_account`
3. Days since last login (recency of account activity)
4. Geo-velocity: `km_per_hour_between_last_login_and_transaction`
5. Count of failed logins in the 24h preceding success
6. Profile-change-to-transaction lag: days between PII update and next transaction
7. Login hour deviation: `|login_hour - account_median_login_hour|`

**Anti-patterns / features that don't generalize:**
- Hard IP blocklists — stolen accounts are used through victim's ISP to avoid geo triggers
- Single password-reset flag without temporal context
- Device-only signals — attackers use stolen cookies to bypass device fingerprinting

---

### Type 3: First-Party Fraud (Bust-Out, Friendly Fraud)

**Definition:**
- **Bust-out:** Legitimate identity deliberately builds credit, then maxes out all lines and disappears
- **Friendly fraud (chargeback fraud):** Cardholder receives goods/services, then disputes transaction claiming non-receipt

**Key behavioral signals (bust-out):**
- Rapidly increasing credit utilization over 60–90 days
- Multiple new credit lines opened in short succession
- Payments that maintain minimum required to avoid default (keeping account alive during build-up phase)
- Address/phone/email changes preceding the final spend-up
- Final spending burst: multiple large transactions in last 7 days of account activity
- Spending shift toward easily liquidated categories (gift cards, crypto, wire transfers)

**Key behavioral signals (friendly fraud):**
- Dispute rate significantly higher than peer customers
- Disputes concentrated in specific merchant types or recurring merchants
- Long lag between purchase and dispute (near chargeback deadline)
- Customer disputes but continues purchasing from same merchant

**Most predictive feature categories (bust-out):**
1. Utilization trajectory: `avg_utilization_last_30d - avg_utilization_90d_to_60d_ago`
2. New account opening velocity in last 90 days (requires bureau data)
3. Spending category shift: `%_gift_card_spending_last_7d - %_gift_card_spending_lifetime`
4. Minimum payment pattern: binary flag if payments always exactly meet minimum threshold
5. Time-to-default from first account open (survival analysis feature)

**Most predictive feature categories (friendly fraud):**
1. Lifetime dispute rate per customer: `disputes / total_transactions`
2. Dispute-to-purchase ratio by merchant
3. Days-to-dispute: `dispute_date - transaction_date`
4. Delivery confirmation availability (binary)

**Anti-patterns:**
- Credit score alone — bust-out accounts often have pristine scores at the start
- Transaction-level fraud models for bust-out — the pattern is account-lifecycle level, not transaction level
- Short lookback windows — bust-out requires 60–180 day behavioral arcs

---

### Type 4: Third-Party Fraud (Stolen/Compromised Cards)

**Definition:** Attacker uses a stolen physical card or compromised card credentials belonging to an uninvolved victim.

**Key behavioral signals:**
- Transaction far from cardholder's home location
- Transaction category mismatch with cardholder's spending history
- Multiple transactions at different merchants in rapid succession (geographic spread)
- High-value single transaction from a cardholder with a history of small amounts
- Card used at an ATM shortly before fraudulent POS/CNP transactions
- International transaction for domestic-only historical card usage

**Most predictive feature categories:**
1. Geo-distance from home: `haversine(transaction_lat_long, cardholder_home_lat_long)` (available in Sparkov)
2. Amount deviation: `(txn_amount - customer_avg_amount_30d) / customer_std_amount_30d` (z-score)
3. MCC novelty: `is_first_time_category_for_customer` — binary flag if merchant category never seen before
4. Time-since-last-transaction-on-card: hours elapsed
5. Transaction count on card in last 1h, 24h
6. Amount percentile rank within customer's historical distribution

**Concrete feature formulas:**
```
# Amount z-score
amt_zscore = (txn_amt - CUSTOMER_AVG_AMT_30D) / max(CUSTOMER_STD_AMT_30D, 1.0)

# Geo-distance (haversine, km)
from math import radians, sin, cos, sqrt, atan2
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

dist_from_home = haversine(txn_lat, txn_lon, card_home_lat, card_home_lon)

# MCC novelty
mcc_never_seen = 1 if (customer_id, mcc) not in historical_customer_mcc_set else 0
```

**Anti-patterns:**
- Pure geographic rules — travelers trigger them constantly; requires normalization by travel frequency
- MCC matching without time-decay — spending categories evolve over months

---

### Type 5: Synthetic Identity Fraud

**Definition:** Fabricated identity combining real elements (e.g., a real SSN from a child or deceased person) with fake name, address, and contact details. Used to obtain credit, build history, then bust out.

**Key behavioral signals:**
- SSN first-use date inconsistent with claimed age (SSN issued after DOB)
- Thin or non-existent credit file despite claimed age
- Address linked to multiple unrelated identities
- Phone number is VoIP or associated with other fraud-linked identities
- Rapid credit-building from zero: authorized-user additions, secured cards, then escalating unsecured lines
- Application form submitted via automation (high-speed form fill, copy-paste patterns)
- Email address contains patterns common to fraudsters (numeric suffix, disposable domain)
- Same device ID used across multiple different identity applications

**Most predictive feature categories (requires application-level data):**
1. SSN issuance year vs. applicant birth year mismatch (if SSN issued after applicant was 7+, flag)
2. `days_since_first_credit_inquiry` vs. `claimed_years_of_credit_history`
3. Address linkage count: number of distinct identities sharing this address in the last 12 months
4. Device linkage count: distinct application IDs from same device fingerprint
5. Email entropy / domain risk: `is_free_email`, `is_disposable_email`, email character entropy
6. Velocity of credit line increases: `sum_credit_limit_increases_last_90d`
7. First payment behavior: does the account ever make payments, or just charges?

**Anti-patterns:**
- Single-feature thin-file flag — many legitimate young consumers have thin files
- Address-only deduplication — families share addresses legitimately

---

### Type 6: Application Fraud

**Definition:** Fraudulent account applications using stolen identity, manipulated income/employment data, or synthetic identities to obtain credit or financial products.

**Key behavioral signals:**
- Income stated significantly above what is verifiable from bureau data
- Employment claims that cannot be verified or reference numbers fail lookups
- Multiple applications in rapid succession from the same device or IP
- Application data inconsistencies: zip code doesn't match stated city, age doesn't match SSN vintage
- Use of known fraudster phone numbers, emails, or addresses (matching watchlists)
- High application-to-approval ratio at specific broker/channel (channel risk)
- Browser/device fingerprint shows automation indicators (headless browser, scripting patterns)

**Most predictive feature categories:**
1. Stated income vs. bureau-estimated income ratio (if bureau data available)
2. Application velocity from same device: `count_applications_device_24h`
3. Application velocity from same IP: `count_applications_ip_1h`
4. Email domain risk score (free, disposable, newly registered domain)
5. Address link score: number of adverse records at same address
6. Field-completion time (too fast = bot; too slow = copied from another window)

---

## Part 4: Cross-Cutting Feature Engineering Techniques

### 4.1 Velocity / Aggregation Windows

**Core pattern (RFM framework):**
For entity E (card, customer, device, IP, merchant) over window W:
- `COUNT_W`: number of transactions
- `SUM_AMT_W`: total spend
- `AVG_AMT_W`: average amount
- `STD_AMT_W`: standard deviation of amounts
- `DISTINCT_MERCHANT_W`: unique merchants transacted with
- `DISTINCT_MCC_W`: unique merchant categories
- `FRAUD_RATE_W` (terminal-side only, with delay): proportion labeled fraud

**Recommended window sizes:**
| Window | Use Case |
|--------|----------|
| 1 minute | Bot attacks, card testing (Stripe PFM detected card testing at this granularity) |
| 5 minutes | Credential stuffing, scanning attacks |
| 1 hour | Card testing, credential stuffing, rapid sequential fraud |
| 4 hours | ATO spend-up detection |
| 24 hours | Daily velocity limits, single-day bust patterns |
| 7 days | Weekly behavioral baseline |
| 30 days | Monthly spend profile, baseline for z-score |
| 90 days | Long-term behavior profile, bust-out detection |

**Why these windows matter:** The aggregation period accounts for ~88% of the variance in classifier performance across window experiments (Bahnsen et al., 2016). Bahnsen demonstrated that aggregated features improve fraud detection by **200%+** versus raw features alone. Short windows catch acute attacks; long windows establish the behavioral baseline.

**Concrete feature naming convention (from Fraud Detection Handbook):**
```
{ENTITY_TYPE}_{AGGREGATION_FUNCTION}_{FEATURE}_{N}DAY_WINDOW

Examples:
  CARD_COUNT_TXN_1DAY
  CARD_AVG_AMT_7DAY
  CARD_STD_AMT_30DAY
  TERMINAL_FRAUD_RATE_7DAY  # requires delay period
  IP_COUNT_DISTINCT_CARD_1HOUR
  DEVICE_COUNT_DISTINCT_CUSTOMER_24HOUR
```

**Delay period for terminal/merchant risk features:**
Fraudulent labels are only confirmed after investigation (typically 3–7 days). Terminal risk features must be computed with a 7-day lookback offset to prevent leakage:
```
TERMINAL_RISK_7DAY = (
    fraud_count in [day-8 to day-1]
    / max(tx_count in [day-8 to day-1], 1)
)
```

**The ratio trick (competition-validated):**
Raw counts don't normalize for customer frequency. Prefer:
```
# Count in 1-day relative to 30-day baseline
txn_freq_ratio_1d_30d = CARD_COUNT_TXN_1DAY / max(CARD_COUNT_TXN_30DAY / 30, 0.1)
```

This ratio normalization was used extensively in both IEEE-CIS and AmEx winning solutions. The z-score variant is even more robust:
```
# Velocity z-score: how unusual is today's count vs. historical distribution?
vel_zscore_1d = (CARD_COUNT_TXN_1DAY - CARD_AVG_DAILY_COUNT_30D) / max(CARD_STD_DAILY_COUNT_30D, 1.0)
```

---

### 4.2 Behavioral Profiling (Self-Deviation and Peer-Deviation)

**Deviation from self (intra-customer anomaly):**
Compare current transaction to the customer's own history. Catches ATO, stolen card, and bust-out.

```python
# Amount z-score vs self
amt_zscore_30d = (txn_amt - CUSTOMER_AVG_AMT_30D) / max(CUSTOMER_STD_AMT_30D, 1.0)

# Hours since last transaction on this card
hours_since_last_txn = (current_ts - CUSTOMER_LAST_TXN_TS).total_seconds() / 3600

# MCC novelty vs historical profile
is_new_mcc_30d = 1 if mcc not in CUSTOMER_DISTINCT_MCC_SET_30D else 0

# Merchant novelty
is_new_merchant_90d = 1 if merchant_id not in CUSTOMER_DISTINCT_MERCHANT_SET_90D else 0
```

**Deviation from peers (peer group analysis):**
Compare to customers with similar profiles. Catches fraud that's subtle against the individual but abnormal vs. their cohort. Bolton and Hand (2002) formalized this as Peer Group Analysis (PGA).

```python
# Peer group: same age_band, same state, same card_type
peer_avg_amt_30d = COHORT_AVG_AMT_30D
peer_std_amt_30d = COHORT_STD_AMT_30D
amt_zscore_vs_peer = (txn_amt - peer_avg_amt_30d) / max(peer_std_amt_30d, 1.0)

# Customer spend this month vs. peer group monthly spend
spend_vs_peer_ratio = CUSTOMER_SUM_AMT_30D / max(PEER_AVG_SUM_AMT_30D, 1.0)
```

**UID aggregation (IEEE-CIS winning technique):**
When direct customer IDs aren't available, construct a customer proxy UID from stable attributes:
```python
# Proxy customer UID from stable card attributes
uid = str(card1) + '_' + str(addr1)  # card issuer + billing postal area
# or
uid = str(card1) + '_' + str(addr1) + '_' + str(int(D1_normalized))

# Then aggregate over uid for behavioral features
CUSTOMER_PROXY_AVG_AMT_7D = transactions.groupby('uid')['TransactionAmt'].rolling('7D').mean()
```

**Von Mises distribution features (Bahnsen et al.):**
Model the circular statistics of transaction timing to detect purchases at unusual hours. Instead of raw hour-of-day, fit a von Mises distribution per customer to capture their typical transaction-time concentration and flag deviations from it.

---

### 4.3 Entity Resolution / Graph Features

**Graph structure:**
Nodes: customers, cards, devices, emails, IP addresses, merchants, phone numbers
Edges: `HAS_CARD`, `USES_DEVICE`, `TRANSACTS_WITH`, `SHARES_ADDRESS`, `SAME_EMAIL_DOMAIN`

**Key tabular graph features (extractable without a full GNN):**

```python
# Degree features — how many entities share this node
device_n_distinct_cards = df.groupby('device_id')['card_id'].nunique()  # >3 is suspicious
ip_n_distinct_customers = df.groupby('ip_address')['customer_id'].nunique()
email_n_distinct_cards = df.groupby('email')['card_id'].nunique()
card_n_distinct_devices = df.groupby('card_id')['device_id'].nunique()

# Link to known-bad entities
device_is_shared = (device_n_distinct_cards > 3).astype(int)
ip_is_high_velocity = (ip_n_distinct_customers > 10).astype(int)

# Second-order linkage: is this customer's device shared with any known fraudsters?
# Requires a labeled fraud set; compute during training
device_fraud_rate = df.groupby('device_id')['label'].mean()
card_device_fraud_rate = df.merge(device_fraud_rate, on='device_id')['label_device_avg']
```

**Graph-derived features (validated by TigerGraph and academic benchmarks):**

TigerGraph demonstrated that adding PageRank, component size, and betweenness centrality to tabular features improved XGBoost accuracy by **14 percentage points** on Ethereum fraud data. Research shows LINE-based embeddings outperform GNN methods (GraphSAGE, GCN) on large sparse graphs.

```python
# PageRank: devices used by many high-value cards score high
# Compute using networkx or graph database, store as per-entity feature
device_pagerank = nx.pagerank(device_card_graph)[device_id]

# Connected component size: fraud networks separate from legitimate giant component
component_id = union_find.find(entity_id)
component_size = component_sizes[component_id]  # large anomalous component = fraud ring

# Shortest path to known fraudster: graph distance as risk proximity
min_distance_to_fraud = shortest_path_to_nearest_fraud_node(entity_id)

# Community detection via Louvain: fraud rings form distinct clusters
community_id = louvain_communities[entity_id]
community_fraud_rate = fraud_count_in_community / community_size
```

**Node embeddings as features (the emerging hybrid pattern):**
Rather than hand-engineering graph features, compute dense embeddings via FastRP, LINE, or Node2Vec and feed them as features to XGBoost. This is the architecture Stripe, JP Morgan, and Visa use — GNN/graph-generated embeddings consumed by tree-based classifiers.

```python
# FastRP embedding (via graph database or stellargraph)
node_embedding = fastrp_model.get_embedding(entity_id)  # dense vector, e.g., 64-dim
# Add embedding dimensions as features: emb_0, emb_1, ..., emb_63
```

**Community detection feature (simplified):**
Count the number of "suspicious neighbors" within 2 hops — shared-device fraudsters, shared-IP fraudsters.

**Shared attribute flags (IEEE-CIS style):**
```python
# Card seen with multiple email domains in 30 days
CARD_DISTINCT_EMAILS_30D = 1  # flag if > 1

# P_emaildomain used by many different cards (shared burner email domain)
EMAIL_DOMAIN_N_DISTINCT_CARDS_30D  # high value = suspicious domain
```

---

### 4.4 Amount Pattern Analysis

**Round number detection:**
Fraudsters and card testers often use round amounts. Legitimate spending skews to specific price points (e.g., $9.99, $49.95).

```python
# Cents component (split dollars and cents)
txn_cents = txn_amt % 1.0
is_round_amount = (txn_cents < 0.01).astype(int)  # $100.00, $500.00 etc.
is_below_threshold = (txn_amt < 1.00).astype(int)  # card testing

# Log transform for skewed amount distributions
log_amt = np.log1p(txn_amt)

# Amount relative to customer's distribution
amt_percentile_vs_customer_90d = percentile_rank(txn_amt, CUSTOMER_AMT_DISTRIBUTION_90D)
```

**Corridor analysis (below-threshold structuring):**
Fraudsters avoid triggering velocity rules by keeping amounts just below known thresholds (e.g., $9,999 instead of $10,000 for AML thresholds; $99 instead of $100 for fraud velocity rules).

```python
# Transactions just below common threshold values
is_below_100 = (txn_amt >= 90) & (txn_amt < 100)
is_below_500 = (txn_amt >= 450) & (txn_amt < 500)
is_below_1000 = (txn_amt >= 900) & (txn_amt < 1000)
```

**Amount escalation:**
Fraudsters often start small (card test) then escalate. Signal: recent max amount >> 30-day average.

```python
amt_escalation_ratio = CARD_MAX_AMT_24H / max(CARD_AVG_AMT_30D, 1.0)
```

---

### 4.5 Temporal Pattern Features

**Time-of-day and day-of-week:**
```python
# Hour of day (0–23)
txn_hour = pd.Timestamp(txn_ts).hour

# Cyclic encoding (prevents discontinuity between 23 and 0)
hour_sin = np.sin(2 * np.pi * txn_hour / 24)
hour_cos = np.cos(2 * np.pi * txn_hour / 24)

# Day of week (0=Mon, 6=Sun)
txn_dow = pd.Timestamp(txn_ts).dayofweek
dow_sin = np.sin(2 * np.pi * txn_dow / 7)
dow_cos = np.cos(2 * np.pi * txn_dow / 7)

# Weekend flag
is_weekend = (txn_dow >= 5).astype(int)

# Night flag (midnight to 6am)
is_night = (txn_hour <= 6).astype(int)
```

**Cyclical spatiotemporal encoding (per 2025 literature):**
Hourly intra-day frequency: `1/3600`; daily: `1/86400`; weekly: `1/604800`
These frequencies can be used as Fourier basis features for continuous time representation.

**Recency features:**
```python
# Days since account opened
account_age_days = (txn_date - account_open_date).days

# Days since last transaction on this card
days_since_last_txn = (txn_ts - CARD_LAST_TXN_TS).total_seconds() / 86400

# Days since last login (if login events available)
hours_since_last_login = (txn_ts - ACCOUNT_LAST_LOGIN_TS).total_seconds() / 3600

# Normalized: is this transaction soon after account opening?
txn_day_of_account_life = account_age_days  # low values = early account risk
```

**Transaction timing deviation:**
```python
# Deviation from customer's typical transaction hour
hour_deviation = abs(txn_hour - CUSTOMER_MEDIAN_TXN_HOUR_90D)

# Is this transaction at an unusual hour for this customer?
is_unusual_hour = (hour_deviation > 4).astype(int)
```

**Temporal importance:** For offline (brick-and-mortar) transactions, time-of-day + day-of-week features account for ~17% of model importance; for online CNP, ~15%. (Source: MINT/VLDB temporal analysis.)

---

### 4.6 Categorical Feature Encoding

High-cardinality categoricals (merchant, email domain, device type, IP subnet) require special handling.

**Frequency encoding (leakage-free):**
```python
# Replace category with its frequency in the training set
merchant_freq = train_df.groupby('merchant')['merchant'].transform('count') / len(train_df)
```

**Target encoding (with regularization to prevent leakage):**
Introduced by Micci-Barreca (2001) specifically for fraud detection with sparse categorical variables (ZIP, IP, SKU).
```python
# Smooth target encoding: blend category mean with global mean
global_mean = train_df['label'].mean()
category_stats = train_df.groupby('merchant')['label'].agg(['mean', 'count'])
smoothing = 30  # strength of smoothing
category_stats['encoded'] = (
    (category_stats['count'] * category_stats['mean'] + smoothing * global_mean)
    / (category_stats['count'] + smoothing)
)
```

**Weight of Evidence (WoE) encoding:**
Popular in banking/credit fraud; interpretable. Empirically best-performing for imbalanced fraud data alongside James-Stein encoding. Penalizes categories with no fraud exposure.

```python
# WoE = log(P(category | fraud) / P(category | legit))
# Requires careful OOT encoding to avoid leakage
```

**Count encoding for shared-entity signals:**
```python
# How many times has this device been seen in the training window?
device_count_encode = train_df.groupby('device_id').size()
```

**CatBoost native handling (competition-validated):**
CatBoost's ordered target encoding handles high-cardinality categoricals natively and was a key component in many competition-winning solutions. When using CatBoost, pass raw categorical columns rather than pre-encoding — the algorithm's built-in ordered encoding prevents leakage more reliably than manual approaches.

---

### 4.7 Population Stability and OOT Generalization

**The core problem:** Fraud patterns shift faster than most ML problems. A model trained on January data may degrade by March as fraudsters adapt. Features that appear predictive in an IID split may not generalize to a time-shifted test set. The Home Credit 2024 competition formalized this with its Gini stability metric.

**Out-of-time (OOT) split protocol:**
```
Timeline: |----Train (months 1–4)----|--Validation (month 5)--|--OOT Test (month 6+)--|

Rules:
1. NEVER use future data to compute aggregations for past transactions
2. For velocity features: use only transactions strictly prior to the current transaction
3. For merchant/terminal risk features: apply the delay period (see §4.1)
4. Sort all data by timestamp before any train/test split
5. Prefer time-based split over random split even for validation
6. Use StratifiedGroupKFold by weekly temporal grouping for CV (Home Credit 2024 winning approach)
```

**Population Stability Index (PSI):**
Quantifies feature distribution shift between training and OOT populations. PSI is the standard metric in banking for detecting feature drift.

```python
def psi(expected, actual, buckets=10):
    """PSI between training and OOT distributions."""
    breakpoints = np.linspace(0, 100, buckets + 1)
    expected_pcts = np.histogram(expected, bins=np.percentile(expected, breakpoints))[0] / len(expected)
    actual_pcts = np.histogram(actual, bins=np.percentile(expected, breakpoints))[0] / len(actual)
    # Avoid log(0)
    expected_pcts = np.clip(expected_pcts, 1e-6, None)
    actual_pcts = np.clip(actual_pcts, 1e-6, None)
    return np.sum((actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts))

# Interpretation:
# PSI < 0.10: No significant shift — feature is stable
# PSI 0.10–0.25: Moderate shift — monitor, may need recalibration
# PSI > 0.25: Significant shift — feature should be reengineered or dropped
```

**Features that tend to be OOT-stable:**
- Relative behavioral features (z-score vs. self) — robust to scale shifts
- Cyclical temporal features (hour/day encoding) — patterns repeat
- Entity relationship features (shared device, shared IP) — structural signals
- Log-transformed amounts — stable under inflation
- Graph-derived features (component size, PageRank) — structural, not distribution-dependent

**Features that tend to be OOT-unstable:**
- Raw transaction amounts without normalization
- Merchant-specific target encodings — merchant mix changes
- Absolute velocity thresholds — fraud volumes shift seasonally
- Model-score-derived features — feedback loops compound drift

**Recommended anti-drift engineering patterns:**
1. Always compute amount features as ratios or z-scores rather than raw values
2. Use rolling baselines that update dynamically rather than static training-set statistics
3. For merchant/terminal risk features, apply temporal smoothing (EWMA over recent weeks)
4. When creating target encodings, use a lookback window rather than full training history to capture recency

---

## Part 5: Email, Device, and Identity Features (Deep Reference)

*These patterns come from production CNP/ATO pipelines and are underrepresented in academic literature. Most datasets won't have all of these signals, but any subset is worth engineering when available.*

---

### 5.1 Email Feature Engineering (30+ Patterns)

Email is one of the richest single features in online fraud. A legitimate user registers once with a consistent email; fraudsters rotate aliases, use disposable providers, and generate addresses programmatically.

**Local-part structural features:**
```python
local = email.split('@')[0]
domain = email.split('@')[1] if '@' in email else ''

# Entropy of local part (high entropy = random/generated)
from collections import Counter
import math
def entropy(s):
    p = [c/len(s) for c in Counter(s).values()]
    return -sum(x * math.log2(x) for x in p if x > 0)
local_entropy = entropy(local)           # >3.5 = suspicious for a name

# Character composition
local_digit_ratio = sum(c.isdigit() for c in local) / max(len(local), 1)
local_special_ratio = sum(not c.isalnum() for c in local) / max(len(local), 1)
local_upper_ratio = sum(c.isupper() for c in local) / max(len(local), 1)
local_length = len(local)

# Pattern indicators
has_plus_tag = '+' in local          # real users: jsmith+amazon; fraudsters: bulk test aliases
local_starts_with_digit = local[0].isdigit() if local else False
local_all_digits = local.isdigit()   # e.g., 1234567@domain.com
local_hex_like = all(c in '0123456789abcdefABCDEF' for c in local) and len(local) >= 8
local_uuid_like = bool(re.match(r'^[0-9a-f]{8}-?[0-9a-f]{4}', local))  # generated UUIDs
local_sequential = bool(re.search(r'(abc|123|xyz)', local.lower()))
```

**Name ↔ email coherence (strong ATO signal):**
```python
import Levenshtein  # pip install python-Levenshtein

# Edit distance between sender name and email local part
# Legitimate users often have jsmith, john.smith, smithj etc.
def name_email_coherence(full_name, email_local):
    name_parts = full_name.lower().replace('-', ' ').split()
    min_dist = min(
        Levenshtein.distance(email_local.lower(), part)
        for part in name_parts
    ) if name_parts else 999
    return min_dist / max(len(email_local), 1)  # normalized: <0.3 = coherent

name_email_distance = name_email_coherence(sender_name, local)
name_in_email = any(part in local.lower() for part in sender_name.lower().split())
```

**Domain classification:**
```python
# Free consumer email providers
FREEMAIL_DOMAINS = {'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
                    'icloud.com', 'aol.com', 'protonmail.com', 'mail.com'}

# Disposable/temporary email providers (maintain list, ~1000+ domains)
DISPOSABLE_DOMAINS = {'mailinator.com', 'guerrillamail.com', 'tempmail.com',
                      'throwaway.email', 'yopmail.com', '10minutemail.com'}

# High-risk TLDs disproportionately used for fraud accounts
RISKY_TLDS = {'.tk', '.ml', '.ga', '.cf', '.gq',  # free ccTLDs
              '.xyz', '.click', '.link', '.download', '.loan', '.work'}

email_is_freemail = domain in FREEMAIL_DOMAINS
email_is_disposable = domain in DISPOSABLE_DOMAINS or any(domain.endswith(t) for t in RISKY_TLDS)
email_is_custom_domain = not email_is_freemail and not email_is_disposable

# Domain-level signals
domain_has_digits = bool(re.search(r'\d', domain.split('.')[0]))
domain_depth = domain.count('.')  # subdomain depth; abc.mail.com = 2
domain_age_flag = 0  # can integrate with WHOIS if available
```

**Suspicious keyword detection:**
```python
SUSPICIOUS_KEYWORDS = ['temp', 'burner', 'trash', 'spam', 'fake', 'anon',
                       'test', 'noreply', 'dummy', 'void', 'null', 'drop']
local_has_suspicious = any(kw in local.lower() for kw in SUSPICIOUS_KEYWORDS)
```

---

### 5.2 Device Signal Features

**Battery + charging state (emulator detection):**
```python
# Real mobile devices: highly variable battery levels, may or may not be charging
# Emulators: typically report 100% battery and charging=True
battery_level = device.get('battery_level', -1)     # 0-100 or -1 if missing
battery_charging = device.get('battery_charging', None)

battery_is_missing = (battery_level == -1)
battery_is_round = (battery_level % 10 == 0)         # 0, 10, 20...100 - suspicious
battery_is_full_charging = (battery_level == 100 and battery_charging == True)
battery_emulator_score = (battery_is_full_charging or battery_is_missing).astype(int)

# Composite device risk
device_risk_score = (
    0.3 * battery_emulator_score +
    0.3 * debugger_attached +
    0.2 * headless_browser +
    0.2 * (client_server_time_diff_abs > 300)  # >5 min clock skew
)
```

**Device fingerprinting features (production-validated by Stripe):**
Stripe collects behavioral signals and reports that 92% of cards on its network have been seen before, enabling rich cross-merchant device intelligence. Key engineered features:

```python
# Core device fingerprinting features
is_new_device = device_id not in KNOWN_DEVICES_FOR_CARD
num_accounts_per_device = device_distinct_accounts_30d
device_location_mismatch = (device_geo_country != billing_country)
emulator_detected = (battery_emulator_score > 0.5) | headless_browser

# Canvas/WebGL fingerprint hash — detects virtual machines and spoofing
canvas_fingerprint_hash = hash(canvas_data)
canvas_is_known_emulator = canvas_fingerprint_hash in KNOWN_EMULATOR_HASHES

# JA3 TLS fingerprint — identifies client software independent of IP
ja3_hash = compute_ja3(tls_handshake)
ja3_is_suspicious = ja3_hash in KNOWN_BOT_JA3_HASHES

# Behavioral biometrics (when available)
mouse_movement_entropy = entropy(mouse_path_deltas)  # bots have low entropy
typing_cadence_variance = var(keystroke_intervals)    # bots have low variance
copy_paste_ratio = paste_events / total_field_fills   # >0.5 = suspicious
```

**Client-server time differential (clock skew):**
```python
# Bots and emulators often have misconfigured clocks
client_server_time_diff = client_timestamp - server_timestamp  # seconds, signed
client_server_time_diff_abs = abs(client_server_time_diff)

# Bucketed for model interpretability
time_diff_bucket = pd.cut(client_server_time_diff_abs,
    bins=[-1, 30, 120, 300, 900, float('inf')],
    labels=['<30s', '30-120s', '2-5min', '5-15min', '>15min'])
```

**Session behavioral signals:**
```python
# Form completion time: too fast = auto-fill/bot; too slow = looking up stolen data
form_fill_seconds = submit_timestamp - page_load_timestamp
is_suspiciously_fast = (form_fill_seconds < 3)     # <3s = likely programmatic
is_suspiciously_slow = (form_fill_seconds > 600)   # >10min = unusual

# Copy-paste detection (if available from browser events)
fields_pasted = count of form fields where paste event was detected
paste_ratio = fields_pasted / total_form_fields     # >0.5 = suspicious for name/address

# Session depth: did user navigate or go straight to high-risk action?
pages_before_transaction = session_page_count - 1   # 0 = direct deep-link = suspicious
session_duration_seconds = session_end - session_start
actions_per_minute = total_session_actions / max(session_duration_seconds / 60, 1)
```

---

### 5.3 Identity Stability and Entity Resolution (4-Layer Framework)

This is the most powerful and underused technique in production fraud systems. A legitimate user has a coherent, slowly-drifting identity footprint. Fraudsters either show sudden discontinuities (ATO) or never establish a stable footprint (synthetic identity). Mules look individually clean but are connected to fraudsters via shared identity elements.

**Layer 1: Identity Element Tracking**

For each user, maintain the historical set of identity elements they've presented. Features measure stability of that set.

```python
# Per-user history of identity elements (maintained in feature store)
# Updated after each transaction; only past transactions used for current scoring

KNOWN_DEVICES = set of device_ids seen for this user_id historically
KNOWN_IPS = set of ip_addresses
KNOWN_IP_SUBNETS = set of /24 subnets
KNOWN_EMAILS = set of emails
KNOWN_BANK_ACCOUNTS = set of destination bank_accts

# Current transaction element novelty flags
identity_device_is_new = device_id not in KNOWN_DEVICES
identity_ip_is_new = ip_address not in KNOWN_IPS
identity_subnet_is_new = ip_subnet not in KNOWN_IP_SUBNETS
identity_email_is_new = email not in KNOWN_EMAILS
identity_bank_acct_is_new = bank_acct not in KNOWN_BANK_ACCOUNTS

# Known ratio: what fraction of current session's identity elements have been seen before?
elements_presented = [device_id, ip_address, email, ...]
known_count = sum(element in historical_set for element, historical_set in zip(elements_presented, sets))
identity_known_ratio = known_count / len(elements_presented)  # <0.5 = high ATO risk

# Jaccard similarity to modal identity (most common configuration for this user)
modal_identity_vector = {modal_device, modal_ip_subnet, modal_email}
current_vector = {device_id, ip_subnet, email}
identity_jaccard = len(modal_identity_vector & current_vector) / len(modal_identity_vector | current_vector)
```

**Layer 2: Behavioral Profile Deviation**

Count-based windows (last N transactions) rather than time-based — adapts to each user's transaction cadence automatically. A weekly user needs 6 months to build 30-transaction profile; a daily user needs 30 days.

```python
# Compute against last 30 transactions (not last 30 days)
CUSTOMER_AMT_HISTORY_30TXN = sorted list of amounts from last 30 transactions

# Amount percentile rank in user's own history
amt_pctile_vs_self = percentile_rank(txn_amt, CUSTOMER_AMT_HISTORY_30TXN)
is_max_amount_ever = (txn_amt > max(CUSTOMER_AMT_HISTORY_ALL))

# Hour-of-day deviation from user's norm
CUSTOMER_MEDIAN_TXN_HOUR = median(hour for each past transaction)
hour_deviation_from_norm = abs(current_hour - CUSTOMER_MEDIAN_TXN_HOUR)
is_unusual_hour_for_user = (hour_deviation_from_norm > 4)

# Transaction frequency burst
CUSTOMER_MEDIAN_TXN_INTERVAL_DAYS = median(days between consecutive transactions)
days_since_last = (current_ts - CUSTOMER_LAST_TXN_TS).days
burst_ratio = CUSTOMER_MEDIAN_TXN_INTERVAL_DAYS / max(days_since_last, 0.01)
# burst_ratio >> 1 = unusually frequent; = 1 = normal cadence

# First-seen flags (high predictive power, especially × high amount)
is_first_transaction_to_recipient = recipient not in CUSTOMER_KNOWN_RECIPIENTS
is_first_transaction_to_country = destination_country not in CUSTOMER_KNOWN_COUNTRIES
is_first_time_device = identity_device_is_new
new_recipient_x_high_amount = is_first_transaction_to_recipient * amt_pctile_vs_self

# Minimum history guard: suppress behavioral features until N prior transactions
behav_features_valid = (CUSTOMER_TXN_COUNT >= 5)
```

**Layer 3: Entity Resolution (Shared Identity Clustering)**

Link user accounts by shared identity elements. Individual accounts may look clean; the cluster reveals coordinated fraud.

```python
# Shared-entity counts (compute as aggregation features)
# "How many distinct user_ids have been seen on this device in the last 30 days?"
device_distinct_users_30d = df[df.device_id == current_device].user_id.nunique()
ip_subnet_distinct_users_7d = df[df.ip_subnet == current_subnet].user_id.nunique()
dest_bank_distinct_senders_7d = df[df.bank_acct == current_bank_acct].user_id.nunique()

# Flag highly shared entities
device_is_shared = (device_distinct_users_30d > 3)    # >3 users on 1 device = mule network
ip_is_high_density = (ip_subnet_distinct_users_7d > 20)
dest_is_mule_candidate = (dest_bank_distinct_senders_7d > 10)

# Cluster-level risk: what fraction of this user's "neighbors" have been flagged?
# Requires Union-Find or graph component computation over shared elements
cluster_size = count of users in same connected component (shared device/IP/email)
cluster_fraud_rate = (fraud cases in cluster) / cluster_size
cluster_max_risk_score = max(risk scores of all users in cluster)
```

**Layer 4: Cross-Layer Interaction Features**

```python
# Identity instability × behavioral deviation = ATO signature
stability_x_deviation = (1 - identity_known_ratio) * amt_pctile_vs_self

# Triple novelty: everything unprecedented at once
triple_novelty_score = (is_max_amount_ever + is_first_transaction_to_recipient +
                        identity_device_is_new) / 3.0

# New account + high value + shared device = synthetic identity + mule
new_account_x_high_amount = (account_age_days < 30) * txn_amt
new_account_x_shared_device = (account_age_days < 30) * device_is_shared
```

**Design principles:**
- **Causal ordering**: always compute features using only transactions strictly *before* the current one. Update the state *after* feature computation.
- **Count-based windows** adapt to user cadence; time-based windows under-profile low-frequency users.
- **Minimum history guard**: suppress behavioral features until ≥5 prior transactions.
- **Production**: maintain per-user state in a feature store (Redis/DynamoDB/Feast), do point lookups at scoring time.

---

### 5.4 Recipient-Side and Cross-Entity Aggregations

**Most fraud systems over-index on sender-side aggregations.** The receiver is often the most predictive aggregation key because mule accounts receive from many clean-looking senders. A receiving bank account that has received transfers from 15 different senders in one week is almost certainly a mule account, even if each individual sender looks fine.

**Recipient-side aggregations:**
```python
# Aggregate BY the destination, not the sender
dest_account_distinct_senders_7d = count of unique senders to this bank_acct in 7 days
dest_account_total_received_24h = sum of amounts received by this bank_acct in 24h
dest_account_avg_txn_amount = mean amount received historically
dest_account_first_seen_days = days since dest bank_acct was first seen in our system

# Flag patterns
dest_is_new_in_system = (dest_account_first_seen_days < 7)
dest_has_many_senders = (dest_account_distinct_senders_7d > 5)
dest_received_spike = (dest_account_total_received_24h > 10 * dest_account_avg_daily_received)
```

**Cross-entity aggregations (diversity metrics):**
```python
# Diversity of identities on shared infrastructure = credential stuffing / enumeration
ip_distinct_email_domains_1h = count of unique email domains transacting from this IP in 1h
ip_distinct_usernames_1h = count of unique users from this IP in 1h
device_distinct_accounts_24h = count of unique accounts using this device in 24h

# High diversity on single entity = scanning/stuffing attack
ip_is_scanning = (ip_distinct_email_domains_1h > 10)
device_shared_across_accounts = (device_distinct_accounts_24h > 3)
```

**Failure-rate aggregations:**
```python
# Failed transaction attempts by entity in time window
# (declined, authentication failures, OTP failures, 3DS failures)
card_failure_rate_24h = card_failed_attempts_24h / max(card_total_attempts_24h, 1)
ip_failure_rate_1h = ip_failed_attempts_1h / max(ip_total_attempts_1h, 1)
user_otp_failures_last_login = count of OTP failures in session before successful auth

# High failure rate = credential testing, card testing
card_is_testing = (card_failure_rate_24h > 0.5 and card_total_attempts_24h > 3)
```

**Corridor aggregations:**
```python
# Aggregate by (sender_country, destination_country) pair
# Fraud concentrates in specific corridors
corridor = (sender_country, destination_country)
corridor_avg_fraud_rate_30d = historical fraud rate for this corridor
corridor_avg_txn_amount = mean amount for this corridor
corridor_txn_count_24h = volume on this corridor in last 24h

# Current transaction vs corridor norms
txn_amount_vs_corridor_avg = txn_amt / max(corridor_avg_txn_amount, 1.0)
```

**Systematic aggregation dimensions (9 keys × 4 windows):**

The most comprehensive aggregation coverage comes from computing count, sum_amount, mean_amount, std_amount, distinct_recipients across these dimensions × time windows:

| Grouping Key | Signal |
|---|---|
| `user_id` / `card_id` | Individual behavioral baseline |
| `ip_address` | Point source attacks |
| `ip_subnet_24` (/24) | Subnet-level campaigns |
| `ip_subnet_16` (/16) | ASN/datacenter-level campaigns |
| `email_domain` | Domain-level coordinated fraud |
| `bank_institution` | Institutional risk concentration |
| `destination_country` | Corridor risk |
| `device_id` | Device sharing / mule networks |
| `sender_name` | Name reuse across accounts |

| Time Window | Primary Signal |
|---|---|
| 5 min | Card testing, credential stuffing |
| 1 hour | ATO spend-up, burst attack |
| 24 hours | Daily fraud campaigns |
| 7 days | Weekly patterns, bust-out ramp-up |

---

## Part 6: Modeling Methodology for Extreme Imbalance

*Relevant when fraud rate is ≤2%. At 0.05% fraud rate (1 in 2,000 transactions), standard evaluation metrics break down entirely.*

---

### 6.1 Evaluation Hierarchy

**Never use accuracy or ROC AUC as primary metrics at extreme imbalance.**

At 0.05% fraud, a model that flags everything as legitimate achieves 99.95% accuracy and ~0.5 ROC AUC (random). ROC AUC inflates because the FPR denominator (true negatives) is enormous — even a large number of false positives barely moves the FPR.

```
Primary metric:   AUPRC (Precision-Recall AUC)
Operating metric: Recall @ target_FPR (e.g., "fraud capture rate at 1% false positive rate")
Secondary:        Lift curve, calibration curve (Brier score), expected financial cost
Stability:        Gini stability metric (Home Credit 2024 approach — penalizes temporal degradation)
```

**Bootstrapped confidence intervals are mandatory on small OOT sets:**
```python
# At 0.05% fraud with 100K OOT transactions, you have ~50 fraud cases.
# Point estimates are meaningless — always report CIs.
from sklearn.utils import resample
from sklearn.metrics import average_precision_score

def bootstrap_auprc(y_true, y_score, n_boot=1000):
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        if y_true[idx].sum() < 2: continue
        scores.append(average_precision_score(y_true[idx], y_score[idx]))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return np.mean(scores), lo, hi  # (mean, lower_ci, upper_ci)

# Report: AUPRC = 0.423 [95% CI: 0.387, 0.459]
# If two models' CIs overlap, there is no statistical evidence of difference.
```

---

### 6.2 Imbalance Treatment Comparison

| Treatment | Tradeoff | When to Use |
|---|---|---|
| **Threshold-only** (no resampling) | Cleanest theoretically; may underfit minority class with very weak features | Strong features + enough positives (>5K in train) |
| **Class weights** (scale_pos_weight) | Correct: modifies loss function. Set ratio 10–50:1, not 1/fraud_rate (too aggressive) | Default starting point |
| **Random undersampling** | Loses majority class information. Target 5:1–20:1 ratio, not 1:1. Never resample eval data | When training set is large enough to afford it |
| **SMOTE** | **Use with caution.** Interpolates between rare fraud cases — may create nonsensical samples in high-dimensional feature space. Production practitioners increasingly view SMOTE as problematic for heterogeneous fraud data. If used, apply SMOTE-ENN or SMOTE-Tomek, target 10:1–5:1 not 1:1 | Verify carefully on OOT; prefer class weights instead |
| **Focal loss** | Theoretically cleanest: down-weights easy negatives, focuses gradient on hard boundary cases | Best when class weight alone underperforms |
| **Anomaly hybrid** | Train isolation forest on all data → add anomaly score as feature → supervised classifier | When labeled fraud is <500 cases |

**Note on SMOTE skepticism (2025 consensus):** Research (arXiv:2412.07437) and competition practitioners increasingly recommend against SMOTE for fraud data. The core issue is that fraud transactions are heterogeneous — they don't form compact clusters that interpolation can meaningfully expand. Production systems at Stripe, Visa, and Mastercard address class imbalance through massive data volume and cost-sensitive learning (class weights), not synthetic oversampling.

**Focal loss with XGBoost:**
```python
import numpy as np

def focal_loss_gradient(gamma=2.0):
    """XGBoost custom objective for focal loss."""
    def focal_obj(y_pred, dtrain):
        y_true = dtrain.get_label()
        y_pred = 1 / (1 + np.exp(-y_pred))  # sigmoid
        
        # Focal weight
        pt = np.where(y_true == 1, y_pred, 1 - y_pred)
        focal_weight = (1 - pt) ** gamma
        
        # Gradient and hessian
        grad = focal_weight * (y_pred - y_true) + gamma * focal_weight * np.log(pt + 1e-8) * (y_pred * (1 - y_pred))
        hess = focal_weight * y_pred * (1 - y_pred)
        return grad, hess
    return focal_obj

# Usage:
model = xgb.train(
    params={'eval_metric': 'aucpr'},
    dtrain=dtrain,
    obj=focal_loss_gradient(gamma=2.0),
    ...
)
```

---

### 6.3 Recommended Ensemble Strategy

Train 3+ models with diverse imbalance treatments, rank-average their scores. This is the dominant pattern across all major Kaggle fraud competitions — every winning solution uses ensembling.

```python
# Model 1: Focal loss + moderate class weight
model_1 = xgb.XGBClassifier(scale_pos_weight=20, ...)  # or focal loss custom obj

# Model 2: Random undersampling (10:1 neg:pos ratio)
neg_idx = np.where(y_train == 0)[0]
pos_idx = np.where(y_train == 1)[0]
neg_sample = np.random.choice(neg_idx, size=10 * len(pos_idx), replace=False)
idx_under = np.concatenate([pos_idx, neg_sample])
model_2 = xgb.XGBClassifier(scale_pos_weight=1, ...)
model_2.fit(X_train[idx_under], y_train[idx_under], ...)

# Model 3: Anomaly score as additional feature
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=fraud_rate, random_state=42)
iso.fit(X_train)
X_train_aug = X_train.copy()
X_train_aug['anomaly_score'] = -iso.score_samples(X_train)  # higher = more anomalous
model_3 = xgb.XGBClassifier(scale_pos_weight=20, ...)
model_3.fit(X_train_aug, y_train, ...)

# Model 4 (optional): LightGBM with different feature subset (diversity)
model_4 = lgb.LGBMClassifier(is_unbalance=True, ...)

# Rank-average ensemble (more robust than probability averaging at extreme imbalance)
from scipy.stats import rankdata
def rank_avg(*score_arrays):
    n = len(score_arrays[0])
    ranks = [rankdata(s) / n for s in score_arrays]
    return np.mean(ranks, axis=0)

y_oot_pred = rank_avg(
    model_1.predict_proba(X_oot)[:, 1],
    model_2.predict_proba(X_oot)[:, 1],
    model_3.predict_proba(X_oot_aug)[:, 1],
    model_4.predict_proba(X_oot)[:, 1],
)
```

**Why rank-average over probability average:** At extreme imbalance, raw probability estimates are poorly calibrated across models. Rank-averaging preserves the ordinal signal while neutralizing calibration differences.

**Competition-validated stacking (Chris Deotte pattern):**
For maximum performance, use multi-level stacking: Level 1 = diverse base models (XGBoost, LightGBM, CatBoost, NN); Level 2 = meta-learner trained on Level 1 out-of-fold predictions. Chris Deotte's April 2025 Kaggle Playground win used a 3-level stack of 72 models. In production, 3–5 model ensembles are more practical.

---

### 6.4 Practical XGBoost Settings for Fraud

```python
xgb.XGBClassifier(
    tree_method="hist",         # 5–10x faster than "exact" on large datasets
    device="cuda",              # GPU acceleration (auto-detect with get_gpu_info())
    max_bin=256,                # histogram resolution — sweet spot for tabular data
    n_estimators=1000,          # with early stopping, many trees are harmless
    early_stopping_rounds=50,   # stop when val AUPRC doesn't improve
    eval_metric="aucpr",        # AUPRC-optimized early stopping (vs logloss)
    max_depth=6,                # 4–7 typical for fraud; deeper = more overfit
    learning_rate=0.05,         # lower lr + more trees > higher lr + fewer trees
    subsample=0.8,              # row sampling per tree — regularization + diversity
    colsample_bytree=0.7,       # column sampling — critical with 100+ correlated aggregations
    min_child_weight=5,         # minimum samples in leaf — prevents overfitting on rare fraud
    scale_pos_weight=20,        # tune as hyperparameter; 10–50 range; not 1/fraud_rate
    nthread=-1,                 # use all available CPU cores
    random_state=42,
)
```

**Feature selection note:** At 100–400 features, recursive feature elimination or importance-based pruning should happen *inside* the cross-validation loop — not as a pre-processing step on the full dataset. Feature selection on the same data you train on inflates importance of noise features.

**LightGBM alternative settings (equally competitive per TALENT):**
```python
lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=-1,               # unlimited depth; controlled by num_leaves
    num_leaves=63,              # 2^6 - 1; LightGBM grows leaf-wise
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.7,
    is_unbalance=True,          # LightGBM's built-in class weight handling
    metric='average_precision',
    early_stopping_rounds=50,
    verbose=-1,
)
```

---

## Part 7: Advanced Sequence Modeling & Lifecycle Detection

Advanced techniques that move beyond per-transaction features to capture the **shape and trajectory** of user behavior over time. These are Tier 2–4 signals — most powerful when stacked on top of Tier 1 (RFM/behavioral) features in an XGBoost ensemble.

---

### 7.1 RFM and Fraud Extensions

RFM (Recency, Frequency, Monetary) is the right starting point for the same reason it works in retail — it compresses complex transaction history into three numbers that capture the most important dimensions of behavior. The fraud signal isn't the RFM values themselves, but the **rate of change**.

#### How RFM maps to each fraud type

**Transaction monitoring (card-present / card-not-present):**
The RFM baseline represents normal consumer spending. Fraud signal = deviation from that baseline.
- Recency shift: suddenly transacting after a dormancy period → account takeover or credential resale
- Frequency spike: burst of transactions in a short window → card testing or exploitation window
- Monetary jump: first transaction above historical max → amount corridor breach

The z-scores we build in behavioral profiling (`behav_amt_zscore`, `behav_hour_deviation`) are essentially tracking per-card RFM drift in real time.

**Account Takeover (ATO):**
The legitimate account owner has an established RFM profile built over months. The attacker's behavior creates a discontinuity in all three dimensions simultaneously.
- `triple_novelty_score` = (fastest-ever inter-transaction interval) × (new recipient flag) × (max-ever amount) — directly captures simultaneous RFM regime change
- A soft-day-1 ATO may probe with a low-amount test transaction (anomalous Recency, normal M), then escalate (anomalous Frequency + Monetary in hour 2)

**Synthetic identity fraud (bust-out):**
The fraudster deliberately builds a legitimate-looking RFM trajectory. Small purchases, on-time payments, gradual credit limit increases. The bust-out is the terminal event.
- Raw RFM can't catch this — the account looks legitimate until it doesn't
- RFM compared to **legitimate cluster centroids** can: the synthetic identity's trajectory is a little too smooth, too textbook, with suspiciously absent the noise and irregularity of real human behavior
- Feature: `rfm_cluster_distance` = Euclidean distance from the card's RFM vector to the nearest legitimate cluster centroid (computed in fit() via K-means on fraud-negative training cards)

**First-party fraud (friendly fraud / chargeback abuse):**
Normal purchasing behavior punctuated by periodic disputes.
- `dispute_frequency_ratio` = disputes_per_month / transactions_per_month
- `disputed_amount_ratio` = mean disputed amount / mean non-disputed amount
- `dispute_recency` = days since last dispute

**Synthetic features to compute in fit():**
```python
# Per-card RFM vectors (stored in state for cluster distance computation)
if card_col and amt_col and time_col:
    last_time = df_train[time_col].max()
    rfm = df_train.groupby(card_col).agg(
        recency=(time_col, lambda x: float(last_time - x.max())),
        frequency=(card_col, "count"),
        monetary=(amt_col, "mean"),
    ).reset_index()
    
    # K-means on fraud-negative cards to define "legitimate" clusters
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    neg_rfm = rfm[rfm[card_col].isin(
        df_train[y_train == 0][card_col].unique()
    )][["recency", "frequency", "monetary"]].fillna(0)
    
    scaler = StandardScaler()
    neg_scaled = scaler.fit_transform(neg_rfm)
    
    km = KMeans(n_clusters=8, random_state=42, n_init=10)
    km.fit(neg_scaled)
    
    state["rfm_scaler_means"] = scaler.mean_.tolist()
    state["rfm_scaler_stds"] = scaler.scale_.tolist()
    state["rfm_centroids"] = km.cluster_centers_.tolist()
    state["rfm_per_card"] = {
        str(row[card_col]): {
            "recency": float(row["recency"]),
            "frequency": float(row["frequency"]),
            "monetary": float(row["monetary"]),
        }
        for _, row in rfm.iterrows()
    }
```

---

### 7.2 HMM Fraud Lifecycle Modeling

Hidden Markov Models are the most naturally suited framework for fraud lifecycle detection. The hidden states correspond to fraud lifecycle stages; the observable emissions are transaction features. This is Tier 2 in the architecture: sequence-level signal that point-in-time features miss.

Research has shown that HMM-based features — specifically the likelihood of a transaction sequence given the learned model — significantly improve fraud detection by quantifying how "normal" or "fraudulent" the sequence looks as a whole. Lucas et al. (2019) extended this with "multiple perspectives HMMs" generating 8 features from both cardholder and merchant viewpoints.

#### Hidden state design per fraud type

**Account Takeover (4 states):**
```
legitimate_owner → credential_compromise → active_exploitation → cashout
```
- Observable signals: amount deviation, recipient novelty, device familiarity, time-of-day anomaly
- Key pattern: the credential_compromise → active_exploitation transition often happens within 24 hours
- `active_exploitation` state emits: high amount deviation, novel recipient, unfamiliar device, unusual hour

**Synthetic Identity / Bust-out (4 states):**
```
identity_building → credit_nurturing → limit_testing → bust_out
```
- Observable signals: amount level, amount growth rate, frequency, time since account opening
- Key pattern: legitimate customers' dwell time in each state is irregular. Synthetic identities are mechanically regular — driven by a playbook with suspiciously uniform state transitions.

**Money Laundering / Mule Accounts (4 states):**
```
dormant → receiving → layering → extraction
```
- Observable signals: transaction direction (in vs. out), counterparty diversity, amount pattern
- Key pattern: `receiving` state = high incoming from diverse senders; `extraction` = rapid outgoing to concentrated endpoints

#### Practical implementation in the fit/transform API

The HMM is trained in `fit()` and serialized as JSON-serializable numpy arrays. In `transform()`, it is reconstructed to compute posterior state probabilities per transaction.

The **posterior state probability** for each transaction is the core feature: `P(state=exploitation | transaction sequence up to this point)`. These become columns in the feature matrix fed to XGBoost.

**Critical detail:** HMMs are trained on **sequences**, not individual transactions. Group by card, sort by time, pass the sequence to hmmlearn. The per-transaction Viterbi-decoded state and posterior probabilities are joined back to the transaction DataFrame.

```python
# Fitting: requires hmmlearn (pip install hmmlearn)
# Transforms that depend on sequence order: sort by [card_col, time_col] in both fit and transform
```

See **Recipe 8** in `recipes.md` for full fit/transform implementation.

#### When to use HMMs

| Fraud type | Benefit | When it helps |
|---|---|---|
| ATO | Detect exploitation state even before large transaction | After ≥3 transactions on the account |
| Bust-out | Detect bust-out trajectory vs. legitimate maturation curve | Long account history (6+ months) |
| Mule accounts | Detect receiving→layering→extraction transition | Accounts with recurring inbound/outbound patterns |
| General CNP | Identify "unusual sequence" even if each transaction looks normal | Any card with 5+ transactions in train |

**Limitation:** HMMs work best when cards have sufficient transaction history (≥5 in training). Cards with 1–2 transactions get noisy posterior estimates — clip features and use `hmm_n_obs` (count of training transactions) as a reliability indicator.

---

### 7.3 Transformer and Attention-Based Sequence Models

**Capital One's "From Features to Sequences" transition (KDD 2025)** signals where the industry is heading: treating the raw transaction sequence as input to a transformer, rather than pre-computing features from it. This approach has several advantages: it discovers interaction patterns that manual feature engineering misses, handles variable-length histories naturally, and can be pretrained on massive unlabeled transaction corpora.

**Practical transformer approaches for fraud:**

**FraudGT** (IBM Research, ICAIF 2024): A graph transformer that uses multi-dimensional edge attributes for financial fraud detection.

**Graph-Temporal Contrastive Transformer** (December 2025): Combines graph and temporal encoders with contrastive learning, achieving AUC 0.982.

**Mastercard's Decision Intelligence Pro**: Uses a proprietary RNN that treats merchant visit history as a "prompt" — analogous to how LLMs process text — to assess whether a transaction fits cardholder behavior.

**AmEx competition approach (Chris Deotte, 15th place)**: Transformer with LightGBM knowledge distillation. Train a transformer on raw time-series billing data, then distill its learned representations into features consumed by LightGBM.

**Implementation guidance for the agent:**
```python
# Transaction sequence transformer (simplified)
# Each transaction = token with features: [amount, hour, day, merchant_emb, device_emb, ...]
# Positional encoding = time delta from first transaction (continuous, not discrete)

# For the fit/transform API:
# 1. fit(): Train a small transformer on training sequences, extract embeddings
# 2. transform(): Forward-pass each card's sequence, extract the final hidden state
# 3. Add hidden state dimensions as features to the XGBoost input

# This is the "foundation model embeddings as features" pattern from Part 1.2
```

**When to use transformers over HMMs:** When you have >100K labeled sequences and GPU access. HMMs are more sample-efficient and interpretable; transformers are more expressive but require more data and compute.

---

### 7.4 State-Space Models (Mamba/S4) — Emerging Frontier

State-space models represent the newest class of sequence models being applied to tabular and financial data.

**MambaTab** (January 2024): First application of Mamba SSM blocks to tabular data. Matches TransTab with less than 1% of the parameters.

**Mambular** (August 2024): Treats each feature as a token in a sequence processed through Mamba layers. Best average rank across UCI benchmarks versus FT-Transformer, TabTransformer, XGBoost, and MLP.

**Why SSMs are theoretically attractive for fraud:** Linear-time processing (vs. quadratic for transformers) enables efficient scoring of very long transaction histories. Selective state spaces can model both gradual behavioral drift (bust-out ramp-up) and sudden regime changes (ATO exploitation burst).

**Current status:** Direct fraud applications have not yet been published. The agent should monitor this space but default to HMMs or transformers for sequence features in the near term.

---

### 7.5 Feature-Group Autoencoder Anomaly Detection

Train autoencoders on **specific feature groups** — not all features at once. Anomaly signal for "this device profile is unusual" and "this amount/time pattern is unusual" are different signals. Blending them into one autoencoder muddies the signal.

**Feature group decomposition:**
- **Amount group**: log_amt, amt_cents, amt_round_flag, amt_per_merchant_zscore, amt_per_category_zscore
- **Time group**: hour_sin/cos, dow_sin/cos, is_night, time_since_last_txn
- **Identity group** (IEEE-CIS): email match, device match, addr match, n_cards_per_device
- **Behavioral group**: velocity features, behavioral z-scores from behavioral profiling

**Why group-level autoencoders work:**
A transaction where the amount is normal but the device pattern is highly anomalous has `ae_device_error = HIGH`, `ae_amount_error = LOW`. This is more informative than a single blended reconstruction error. XGBoost can learn "device error combined with new recipient = high fraud signal."

**Implementation:** A 3-layer MLP autoencoder (input → bottleneck → input) with ReLU activations and a linear output layer. Trained in `fit()` on fraud-negative transactions. Weights stored as nested lists (JSON-serializable). Forward pass implemented in numpy in `transform()`.

```python
def _relu(x):
    return np.maximum(0, x)

def _ae_forward(X, coefs, intercepts):
    """Forward pass: [input → hidden → bottleneck → hidden → output]."""
    a = X
    for i, (W, b) in enumerate(zip(coefs, intercepts)):
        z = a @ np.array(W) + np.array(b)
        a = _relu(z) if i < len(coefs) - 1 else z
    return a
```

Reconstruction error = mean squared error between input and output. High error = anomalous.

See **Recipe 9** in `recipes.md` for full per-feature-group implementation.

**When to use:** Both datasets. Best when combined with HMM state features — an anomalous autoencoder score + an anomalous HMM state is a very strong combined signal.

---

### 7.6 Dynamic Time Warping (DTW) Trajectory Analysis

DTW compares behavioral trajectories where the **shape** matters but the **timing** varies. In retail: "active → dormant → explosive" maps directly to bust-out fraud pattern matching.

**Use cases:**

**Bust-out fraud:** The trajectory is "small purchases → gradual increase → sudden massive spend → disappear." DTW can match this pattern even when the time axis varies — one fraudster executes in 6 months, another in 18. The shape is the signal, not the clock time.

**Mule account recruitment:** Normal consumer behavior for months/years, then sudden activation with high incoming diversity + rapid outgoing to concentrated endpoints. DTW aligns this "before and after" shape across mules even when the inflection happens at different calendar dates.

**Practical approach (without external DTW library):**
Rather than full pairwise DTW, use a simplified approach:
1. Compute each card's **amount trajectory** — rolling N-transaction window of log-amounts
2. Compare the trajectory's shape (monotone increase? sudden jump? flat then spike?) via simple statistics:
   - `traj_slope`: linear regression slope of log_amt over last N transactions
   - `traj_variance`: variance of log_amt over last N transactions
   - `traj_max_jump`: maximum single-step increase in log_amt
   - `traj_bust_score`: `max(log_amt[-5:]) - mean(log_amt[:-5])` — did the end spike relative to the beginning?

```python
# In fit(): Per-card trajectory statistics computed on training data
# Store: slope, variance, max_jump, bust_score per card
# In transform(): Map each transaction to its card's trajectory stats
# + add "how many standard deviations above the training trajectory slope is this card?"
```

For a full DTW-based implementation using `dtaidistance` (if available), compute pairwise distances to a set of **prototype fraud trajectories** extracted from confirmed fraud cases, and use the distance to the nearest fraud prototype as a feature.

---

### 7.7 CUSUM Behavioral Shift Detection

CUSUM (Cumulative Sum) is a sequential change detection algorithm that flags when a process has shifted to a new regime. It's ideal for detecting behavioral shifts that accumulate gradually over multiple transactions.

**How it works:** For each card, maintain a running cumulative sum of standardized deviations from the card's baseline. When the CUSUM exceeds a threshold (typically 5σ), a shift is detected.

```
CUSUM[t] = max(0, CUSUM[t-1] + (x[t] - μ) / σ - k)
```

where `k` is a slack parameter (typically 0.5) that prevents the CUSUM from accumulating on minor fluctuations.

**Why this matters for fraud:**
- A card that gradually increases amounts each transaction stays below per-transaction thresholds indefinitely
- The CUSUM accumulates the "excess" over multiple transactions and fires when the total deviation is large
- This is exactly the pattern of limit-testing in synthetic identity fraud and the early-stage escalation in ATO

**Features generated:**
- `cusum_score`: the current CUSUM value — how much accumulated deviation has occurred
- `cusum_is_triggered`: binary flag when CUSUM exceeds 5σ threshold
- `cusum_slope`: rate of change of CUSUM — how fast the deviation is accumulating
- `cusum_reset_count`: number of times CUSUM has reset (reached 0 after a period of high values)

See **Recipe 10** in `recipes.md` for the fit/transform implementation.

---

### 7.8 PU Learning and Adversarial Adaptation

#### The label problem: Positive and Unlabeled (PU) learning

Confirmed fraud labels are a **biased sample** — they represent fraudsters already caught. Undetected fraud looks like legitimate behavior in the training data. This is the PU (positive and unlabeled) learning problem:
- Confirmed fraud = labeled positive
- Everything else = unlabeled (may include uncaught fraud)

**Practical implications:**
1. Class weight tuning is critical — treating all non-fraud as negatives overfits to currently-detectable patterns
2. Cluster analysis on the "legitimate" population may reveal small tight clusters with near-identical behavior — potential undetected fraud rings
3. Semi-supervised approaches: propagate fraud signals through shared identity clusters (the entity resolution work in Part 5.3 already does this partially via `dest_is_mule_candidate`)
4. Calibrate your AUPRC interpretation: if your model catches 60% of known fraud, it may still be missing a different 30% of total fraud that was never labeled

**Agent implication:** When you find a cluster of high-confidence fraud predictions on "legitimate" accounts, that's a signal to examine — not a false positive rate problem. These may be true positives that slipped through the labeling process.

#### Adversarial adaptation and concept drift

Fraudsters probe the decision boundary and adapt. Your model's behavioral clusters drift not just from natural population changes but from active evasion.

**Practical rules:**
1. PSI monitoring (already built into the harness) catches **passive** population drift. Active adversarial adaptation appears as sudden PSI spikes on specific feature subsets.
2. Feature-level PSI in `harness/feature_analysis.py` — run it periodically to find which features are drifting fastest.
3. High-importance features that drift = evasion target. Low-importance features that drift = natural demographic shift.
4. Any sequence model (HMM, autoencoder) needs a retraining cadence. For commodity fraud (CNP): weeks. For sophisticated synthetic identity rings: months (they can afford patience during the bust-out lifecycle).
5. **Feature importance stability is a deployment signal**: if top features change dramatically between train and OOT without PSI spike, the model is learning unstable patterns.
6. **Federated learning** (JP Morgan's Project AIKYA, Feedzai's RiskFM) enables cross-institutional intelligence without sharing raw data — making it harder for fraudsters to exploit blind spots at individual institutions.

---

### 7.9 Five-Tier Architecture Synthesis (Updated 2026)

The full production architecture stacks tiers, with XGBoost consuming features from all of them. The 2026 update adds a foundation model embedding tier.

**Tier 1 — Fast features (per-transaction, real-time):**
- RFM-based: behavioral z-scores, amount corridors, velocity counts
- Identity stability: email/device/address match flags, entity sharing counts
- Amount patterns: round numbers, cents distribution, corridor analysis
- These run on every transaction. Catches obvious ATO with dramatic behavior changes, new accounts hitting high-value transactions.

**Tier 2 — Sequence models (per-card, daily/weekly):**
- HMM state posteriors: `P(active_exploitation | sequence)`, `P(bust_out | sequence)`
- CUSUM behavioral shift scores
- RFM cluster distances (fraud-negative centroid distances)
- Transformer hidden states (if compute budget allows)
- Run Viterbi decoding on each user's recent history. Catches trajectory patterns that point-in-time features miss.

**Tier 3 — Anomaly detection (batch, nightly):**
- Feature-group autoencoder reconstruction errors (amount group, time group, identity group)
- Mahalanobis distance from training centroid (already in Recipe 6)
- K-means cluster outlier scores (distance to nearest centroid)
- This is the "unknown unknowns" detector — finds behavioral patterns that don't match any known archetype.

**Tier 4 — Entity clustering (batch, nightly):**
- Union-Find entity resolution counts (shared device/email/address clusters from Part 5.3)
- `device_distinct_users_30d`, `dest_is_mule_candidate`, `cross_entity_velocity`
- Graph-derived features: PageRank, connected component size, community fraud rate
- Node embeddings via FastRP/LINE as dense feature vectors
- Catches coordinated fraud and synthetic identity rings.
- (Full GNN-based Tier 4: GraphSAGE, GCN, or FraudGT if graph infrastructure is available.)

**Tier 5 — Foundation model embeddings (batch or real-time, new in 2025–2026):**
- Transaction sequence embeddings from pretrained models (Featurespace NPPR, Stripe PFM-style)
- TabPFN distilled representations for small-data scenarios
- Cross-table transfer embeddings (CARTE, XTab) when multiple data sources available
- This tier replaces some manual feature engineering with learned representations.

**XGBoost on top:** Consumes features from Tiers 1–5. Learns optimal combination — an anomalous Tier 3 score alone may be a false positive, but combined with a Tier 2 exploitation state and a Tier 4 shared-device cluster flag is almost certainly fraud.

**Agent implication:** Start with Tier 1 (already in baseline). Add Tier 2 features (HMM, CUSUM) in subsequent iterations. Add Tier 3 (autoencoders) once Tier 1 has plateaued. Tier 4 entity resolution is already partially in Recipe 5 — extend with graph features. Tier 5 foundation model embeddings should be explored when dataset size is small or when pretrained models for the domain are available.

---

### 7.10 Input Data Architecture: Multi-Source Joins

The feature engineering framework is designed for raw transaction data joined to multiple vendor signals before features are engineered. The typical join pattern:

```
transactions (base)
  LEFT JOIN device_signals ON device_fingerprint_id        -- battery, clock skew, session
  LEFT JOIN ip_intelligence ON ip_address                   -- proxy score, ISP, geo
  LEFT JOIN email_reputation ON email_domain                -- disposable, risky TLD, domain age
  LEFT JOIN vendor_enrichment ON merchant_id                -- MCC category, merchant risk tier
  LEFT JOIN historical_aggregates ON (card_id, window_key) -- pre-computed velocity counts
  LEFT JOIN graph_embeddings ON entity_id                   -- GNN/FastRP node embeddings
  LEFT JOIN foundation_model_scores ON transaction_id       -- pretrained model embeddings
```

In the fit/transform API:
- **fit()** sees this joined DataFrame and computes statistics over all available columns
- **transform()** applies the fitted state — the joins should already be applied upstream in the data pipeline before the DataFrame arrives at fit()/transform()

**Feature engineering at multiple aggregation states:**
The 9×4 matrix from Part 5.4 applies to the joined data:
- Per-card aggregations over joined device/IP signals (e.g., `card_distinct_device_fingerprints_7d`)
- Per-merchant aggregations over joined email signals (e.g., `merchant_disposable_email_rate_30d`)
- Per-IP aggregations over joined email signals (e.g., `ip_distinct_email_domains_1h`)

The agent should add vendor/device/IP join-dependent features progressively, starting from whichever join columns are present in the dataset (check `config.dataset_profile` for `has_geo`, `has_identity`, `has_device`).

---

## Part 8: Feature Engineering Checklist for LLM Agent

When generating features for a new fraud dataset, work through these categories systematically:

### A. Baseline Transformations (always start here)
- [ ] Parse timestamps: extract `hour`, `dow`, `is_weekend`, `is_night`
- [ ] Apply cyclic encoding: `sin/cos(hour)`, `sin/cos(dow)`
- [ ] Log-transform `amount` / `txn_amt` (handle zeros with `log1p`)
- [ ] Split amount into integer and fractional parts
- [ ] Compute `is_round_amount`, `is_micro_amount` (<$1)

### B. Customer/Card Behavioral Features (requires entity ID + sorted timestamps)
- [ ] Transaction count: 1min, 5min, 1h, 4h, 24h, 7d, 30d
- [ ] Sum amount: 24h, 7d, 30d
- [ ] Avg amount: 7d, 30d, 90d
- [ ] Std amount: 30d (for z-score normalization)
- [ ] Days/hours since last transaction
- [ ] `amount_zscore_30d = (amt - avg_30d) / max(std_30d, 1.0)`
- [ ] Velocity ratio: `count_1d / max(count_30d / 30, 0.1)`
- [ ] Velocity z-score: `(count_1d - avg_daily_30d) / max(std_daily_30d, 1.0)`
- [ ] Distinct merchant count: 7d, 30d
- [ ] Distinct MCC count: 30d
- [ ] `is_new_merchant`, `is_new_mcc` (first time for this customer)
- [ ] `amount_percentile_rank_30d`

### C. Terminal / Merchant Risk Features (delayed label, training only)
- [ ] Transaction count: 7d (with 7-day delay)
- [ ] Fraud rate: 7d, 30d (with 7-day delay)
- [ ] Distinct customer count: 7d

### D. Entity Sharing / Graph Features
- [ ] Device → distinct card count (shared device signal)
- [ ] IP → distinct customer count
- [ ] Email domain → distinct card count
- [ ] Card → distinct device count
- [ ] `device_is_shared` (threshold >3)
- [ ] `email_domain_is_high_risk` (target-encode domain)
- [ ] `device_fraud_rate` (proportion of prior transactions on this device that were fraud)
- [ ] Connected component size (Union-Find over shared entities)
- [ ] PageRank score (if graph infrastructure available)
- [ ] Node embedding dimensions (FastRP/LINE, if graph infrastructure available)

### E. Geo / Distance Features (if lat/lon available)
- [ ] `dist_from_home_km` = haversine(txn_lat/lon, cardholder_lat/lon)
- [ ] `dist_from_last_txn_km` (geo velocity)
- [ ] `implied_speed_kmh` = `dist_from_last_txn_km / hours_since_last_txn`
- [ ] `is_international` (different country from home)

### F. Account Age and Recency
- [ ] `account_age_days` = `txn_date - account_open_date`
- [ ] `days_since_signup_to_first_purchase` (for e-commerce)
- [ ] `is_new_account` (<30 days old)
- [ ] `pct_of_account_life_with_activity`

### G. Amount Patterns
- [ ] `amount_vs_limit_ratio` (utilization, if credit limit available)
- [ ] `amount_relative_to_7d_max`
- [ ] `amt_escalation_ratio = max_24h / avg_30d`
- [ ] `is_below_threshold_100`, `is_below_threshold_500`

### H. Categorical Encoding
- [ ] Frequency-encode all high-cardinality categoricals (merchant, device, email domain)
- [ ] Target-encode (with smoothing) merchant and email domain on training set only
- [ ] WoE encode if interpretability is required
- [ ] Use CatBoost native handling when CatBoost is in the ensemble

### I. Foundation Model / Embedding Features (when available)
- [ ] TabPFN distilled predictions (for small datasets, <50K rows)
- [ ] Transaction sequence embeddings (if pretrained model available)
- [ ] Graph node embeddings (FastRP, LINE, or GNN-derived)
- [ ] Cross-table transfer embeddings (CARTE-style, if multiple data sources)

---

## References and Sources

### Datasets
- IEEE-CIS Fraud Detection: https://www.kaggle.com/competitions/ieee-fraud-detection
- Sparkov Simulated Transactions: https://www.kaggle.com/datasets/kartik2112/fraud-detection
- Fraud E-Commerce (fraudecom): https://www.kaggle.com/datasets/vbinh002/fraud-ecommerce
- PaySim: https://www.kaggle.com/datasets/ealaxi/paysim1
- BankSim: https://www.kaggle.com/datasets/ealaxi/banksim1
- Credit Card Fraud (ULB/PCA): https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Elliptic Bitcoin: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

### Key Papers — Tabular ML and Benchmarks
- **Why do tree-based models still outperform deep learning on tabular data?** Grinsztajn et al. (NeurIPS 2022). https://arxiv.org/abs/2207.08815
- **When Do Neural Nets Outperform Boosted Trees on Tabular Data?** McElfresh et al. (NeurIPS 2023). https://arxiv.org/abs/2305.02997
- **TALENT: A Closer Look at Deep Learning Methods on Tabular Datasets.** Ye et al. (2024, revised Nov 2025). https://arxiv.org/abs/2407.00956
- **TabArena: A Living Benchmark for Machine Learning on Tabular Data.** (NeurIPS 2025). https://arxiv.org/abs/2506.16791
- **TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling.** Gorishniy et al. (ICLR 2025). https://arxiv.org/abs/2410.24210
- **Tabular data: Deep learning is not all you need.** Shwartz-Ziv and Armon (2021). https://arxiv.org/abs/2106.03253

### Key Papers — Foundation Models for Tabular Data
- **Accurate predictions on small data with a tabular foundation model.** Hollmann et al. (Nature 2025). https://www.nature.com/articles/s41586-024-08328-6
- **TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models.** (Nov 2025). https://arxiv.org/abs/2511.08667
- **CARTE: Pretraining and Transfer for Tabular Learning.** Kim et al. (2024). https://arxiv.org/abs/2402.16785
- **XTab: Cross-table Pretraining for Tabular Transformers.** (ICML 2023). https://arxiv.org/abs/2305.06090
- **Towards a Foundation Purchasing Model: Pretrained Generative Autoregression on Transaction Sequences.** (2024). https://arxiv.org/abs/2401.01641

### Key Papers — Fraud Detection Methods
- **Feature Engineering Strategies for Credit Card Fraud Detection:** Bahnsen et al. (2016). Expert Systems with Applications. https://albahnsen.github.io/files/Feature%20Engineering%20Strategies%20for%20Credit%20Card%20Fraud%20Detection_published.pdf
- **Multiple perspectives HMM-based feature engineering for credit card fraud detection.** Lucas et al. (2019). https://arxiv.org/abs/1905.06247
- **FraudGT: A Simple, Effective, and Efficient Graph Transformer for Financial Fraud Detection.** IBM Research (ICAIF 2024). https://jshun.csail.mit.edu/FraudGT.pdf
- **Graph-Temporal Contrastive Transformer for Financial Fraud Detection.** (Dec 2025). https://www.mdpi.com/1999-4893/18/12/770
- **An Introduction to Machine Learning Methods for Fraud Detection.** MDPI (2025). https://www.mdpi.com/2076-3417/15/21/11787
- **MambaTab: A Simple Yet Effective Approach for Handling Tabular Data.** (2024). https://arxiv.org/abs/2401.08867
- **Mambular: A Sequential Model for Tabular Deep Learning.** (2024). https://arxiv.org/abs/2408.06291

### Competition Writeups and Practitioner Resources
- **NVIDIA IEEE-CIS Top Solution Summary:** https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution/
- **The Kaggle Grandmasters Playbook: 7 Battle-Tested Modeling Techniques:** https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/
- **AmEx Default Prediction 1st Place Solution:** https://deepwiki.com/jxzly/Kaggle-American-Express-Default-Prediction-1st-solution
- **State of ML Competitions 2024:** https://mlcontests.com/state-of-machine-learning-competitions-2024/
- **State of ML Competitions 2025:** https://mlcontests.com/state-of-machine-learning-competitions-2025/
- **Feature Engineering for Fraud Detection (AI Infrastructure Alliance):** https://ai-infrastructure.org/feature-engineering-for-fraud-detection/
- **From XGBoost to Foundation Models (ML Frontiers):** https://mlfrontiers.substack.com/p/from-xgboost-to-foundation-models

### Industry References
- **Stripe Payments Foundation Model:** https://stripe.com/newsroom/news/sessions-2025
- **Stripe Radar:** https://stripe.com/radar
- **Visa VAAI:** https://investor.visa.com/news/news-details/2024/Visa-Announces-Generative-AI-Powered-Fraud-Solution-to-Combat-Account-Attacks/
- **Mastercard Decision Intelligence:** https://www.mastercard.com/us/en/news-and-trends/press/2024/may/mastercard-accelerates-card-fraud-detection-with-generative-ai-technology.html
- **JP Morgan Project AIKYA:** https://www.jpmorgan.com/kinexys/content-hub/project-aikya
- **Feedzai RiskFM:** https://www.zenml.io/llmops-database/ai-powered-fraud-detection-using-mixture-of-experts-and-federated-learning
- **TabPFN GitHub:** https://github.com/PriorLabs/TabPFN

### Other Technical References
- **Fraud Dataset Benchmark (FDB):** Grover et al. (2022), Amazon Science. https://arxiv.org/abs/2208.14417
- **Transaction Aggregation as a Strategy for Fraud Detection:** Van Vlasselaer et al. ECML 2008. https://euro.ecom.cmu.edu/resources/elibrary/epay/s10618-008-0116-z.pdf
- **Unsupervised Profiling Methods for Fraud Detection (Peer Group Analysis):** Bolton and Hand (2002). Statistical Science.
- **Towards Automated Feature Engineering for Credit Card Fraud (HMM-based):** arXiv:1909.01185. https://arxiv.org/abs/1909.01185
- **Reproducible ML for Credit Card Fraud Detection (Practical Handbook):** Le Borgne et al. https://fraud-detection-handbook.github.io/fraud-detection-handbook/
- **Real-time Feature Engineering for Fraud (Feldera):** https://docs.feldera.com/use_cases/fraud_detection/
- **Dynamic Feature Engineering for Adaptive Fraud Detection (2024):** MDPI. https://www.mdpi.com/2673-4591/107/1/68
- **Population Stability Index (PSI):** NannyML guide. https://www.nannyml.com/blog/population-stability-index-psi
- **High-Cardinality Categorical Attributes in Fraud:** MDPI Mathematics (2022). https://www.mdpi.com/2227-7390/10/20/3808
- **PaySim Paper:** Lopez-Rojas et al. (2016). EMSS. https://www.msc-les.org/proceedings/emss/2016/EMSS2016_249.pdf
