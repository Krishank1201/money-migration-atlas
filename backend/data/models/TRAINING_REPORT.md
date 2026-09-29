# Money Migration Atlas (SIH26182) - Phase 4 Model Training Report

**Date:** 2026-09-29  
**Model Architecture:** XGBoost Classifier (`n_estimators=100`, `max_depth=4`, `learning_rate=0.05`)  
**Artifact Path:** `backend/data/models/xgboost_v1.pkl`  
**Dataset Size:** 324 samples  
**Class Balance:** 162 Positives (50.0%) / 162 Negatives (50.0%)  
**Data Leakage Guard:** 59 benchmark scenario addresses strictly excluded from training  

---

## 1. Raw Test Set Evaluation Metrics (Held-Out Test Split)

| Metric | Target / Benchmark | Observed Score | Status |
| :--- | :--- | :--- | :--- |
| **ROC AUC** | $\ge 0.6500$ | **0.9217** | **PASSED** |
| **Precision@1** | $\ge 0.5000$ | **0.9200** | **PASSED** |
| **Calibration ($P > 0.75$)** | $\ge 0.7500$ | **0.9048** (19/21 correct) | **PASSED** |
| **Brier Score** | Lower is better ($< 0.15$) | **0.0740** | **PASSED** |
| **Accuracy** | Baseline: 0.5000 | **0.9388** | **PASSED** |
| **Recall** | Reference | **0.9583** | **PASSED** |
| **F1 Score** | Reference | **0.9388** | **PASSED** |

### Confusion Matrix (Test Split, $N=49$)
* **True Negatives:** 23
* **False Positives:** 2
* **False Negatives:** 1
* **True Positives:** 23

---

## 2. Feature Importances (Leakage Audit)

| Rank | Feature | Importance | Category | Audit Status |
| :---: | :--- | :---: | :--- | :--- |
| 1 | `path_mixer_penalty` | **0.315570** | Path Quality | **CLEAN (New penalization for mixer-routed decoys)** |
| 2 | `proximity_rank` | **0.187313** | Topological | **CLEAN (Topological shortest hop distance)** |
| 3 | `path_touches_mixer` | **0.098902** | Path Quality | **CLEAN (Binary mixer interaction)** |
| 4 | `total_amount_on_path` | **0.076985** | Amount | **CLEAN (Cumulative transfer value)** |
| 5 | `amount_retention_ratio`| **0.063212** | Amount | **CLEAN (Flow retention across hops)** |
| 6 | `max_path_length` | **0.054366** | Topological | **CLEAN (Max path bound)** |
| 7 | `num_mixer_hops_on_path`| **0.050410** | Path Quality | **CLEAN (Count of mixer hops)** |
| 8 | `avg_hop_time_seconds` | **0.040685** | Temporal | **CLEAN (Mean transaction interval)** |
| 9 | `time_compression_ratio`| **0.038090** | Temporal | **CLEAN (Automation speed signature)** |
| 10 | `min_path_length` | **0.031548** | Topological | **CLEAN (Min path bound)** |
| 11 | `num_paths_to_vasp` | **0.024560** | Topological | **CLEAN (Path multiplicity)** |
| 12 | `max_single_hop_amount`| **0.018359** | Amount | **CLEAN (Peak transfer chunk)** |
| 13 | `amount_std_dev` | **0.000000** | Amount | **CLEAN** |
| 14 | `burst_detection` | **0.000000** | Temporal | **CLEAN** |
| 15 | `suspect_degree` | **0.000000** | Suspect Wallet | **CLEAN** |
| 16 | `suspect_tx_count` | **0.000000** | Suspect Wallet | **CLEAN** |
| 17 | `vasp_is_indian_fiu` | **0.000000** | Candidate VASP | **CLEAN** |
| 18 | `path_crosses_chain` | **0.000000** | Path Quality | **CLEAN** |
| 19 | `num_chain_hops` | **0.000000** | Path Quality | **CLEAN** |
| 20 | `avg_path_length` | **0.000000** | Topological | **CLEAN** |

### Explicitly Removed Leaky & Constant Features
* ❌ `vasp_historical_volume`: **REMOVED** (Shortcut prior reflecting background traffic imbalance).
* ❌ `suspect_age_days`: **REMOVED** (Constant zero-variance feature in single-tx benchmark suspects).
* ❌ `suspect_page_rank`: **REMOVED** (Constant across suspects with identical in/out degree).
* ❌ `vasp_chain_match`: **REMOVED** (Constant 1.0 across all multi-chain registered seed VASPs).

---

## 3. 12 Benchmark Case Evaluation

| Case ID | Expected VASP | Baseline Proximity Pick | Phase 4 Model Pick | Confidence Score | Proximity Rank | Match? | Resolved By | Model Contribution |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| **CASE-001** | `coindcx` | `coindcx` (3 hops) | `coindcx` | 0.4746 | 3 | **Y** | Low Conf (Tie) | No |
| **CASE-002** | `wazirx` | `bybit` (2 hops) | `bybit` | 0.5203 | 2 | **N** | Model | No |
| **CASE-003** | `zebpay` | `bybit` (2 hops) | `zebpay` | **0.8680** | 4 | **Y** | **Model** | **YES** |
| **CASE-004** | `coinswitch` | `coinswitch` (1 hop) | `coinswitch` | 0.1424 | 1 | **Y** | Proximity | No |
| **CASE-005** | `mudrex` | `mudrex` (3 hops) | `mudrex` | 0.5553 | 3 | **Y** | Model | No |
| **CASE-006** | `giottus` | `bybit` (2 hops) | `giottus` | **0.8100** | 4 | **Y** | **Model** | **YES** |
| **CASE-007** | `unocoin` | `unocoin` (3 hops) | `unocoin` | 0.4419 | 3 | **Y** | Low Conf (Tie) | No |
| **CASE-008** | `binance_offshore` | `binance_offshore` (3 hops) | `binance_offshore` | 0.3546 | 3 | **Y** | Low Conf (Tie) | No |
| **CASE-101** | `coindcx` | `coindcx` (5 hops) | `coindcx` | **0.8680** | 5 | **Y** | Model | No |
| **CASE-102** | `zebpay` | `wazirx` (3 hops) | `wazirx` | 0.3830 | 3 | **N** | Low Conf (Tie) | No |
| **CASE-103** | `wazirx` | `bybit` (2 hops) | `wazirx` | **0.8100** | 4 | **Y** | **Model** | **YES** |
| **CASE-104** | `mudrex` | `binance_offshore` (2 hops) | `mudrex` | **0.5553** | 3 | **Y** | **Model** | **YES** |

### Benchmark Evaluation Metrics
* **Baseline Topological Accuracy:** 6 / 12 (50.0%)
* **Phase 4 XGBoost Model Accuracy:** **10 / 12 (83.3%)**
* **Model Contribution Count:** **4 cases** (CASE-003, CASE-006, CASE-103, CASE-104) where proximity rank failed but model confidence $\ge 0.50$ correctly identified the true VASP.
* **Damage Assessment:** **0 cases** where model picked WRONG with confidence $\ge 0.75$ (Target: 0).
* **Unresolved Cases:** CASE-002 and CASE-102 remain unrouted due to subtle transaction graph peeling patterns reserved for Phase 6 behavioral fingerprinting.

---

## 4. Final Verdict

**VERDICT: Model is USEFUL**

---

## 5. GNN Confidence Calibration: The $p_{\text{null}}$ Technique

### 1. Root Cause of 1.00 Output Inflation
In early iterations, the GNN predictor normalized raw candidate probabilities among only reachable VASPs:
$$\text{norm\_prob} = \frac{p_c}{\sum_{k \in \text{Reachable}} p_k}$$
Whenever only a single VASP candidate was topologically reachable from a suspect wallet ($|\text{Reachable}| = 1$), the denominator equaled the numerator ($p_c / p_c \equiv 1.0000$). This artificially saturated model confidence to exactly 1.00 regardless of the underlying logits or model certainty.

### 2. Prior Formulation ($p_{\text{null}} = 1/9$)
To prevent this single-candidate inflation artifact, we introduce an unreached background prior:
$$p_{\text{null}} = \frac{1}{|\mathcal{V}|} = \frac{1}{9} \approx 0.1111$$
where $|\mathcal{V}| = 9$ represents the uniform prior probability over all supported VASP classes in the network.

### 3. Normalization Formula
When only a single candidate is reachable, its calibrated probability is bounded against the null background prior:
$$p_{\text{norm}} = \frac{p_{\text{raw}}}{p_{\text{raw}} + p_{\text{null}}}$$

### 4. Worked Example: CASE-004
For `CASE-004` (target: CoinSwitch):
- **GraphSAGE raw logit:** $-0.9684$
- **Raw 9-class softmax ($p_{\text{raw}}$):** $0.0429$
- **Uniform null prior ($p_{\text{null}}$):** $\frac{1}{9} \approx 0.1111$
- **Calibrated Probability:**
  $$p_{\text{norm}} = \frac{0.0429}{0.0429 + 0.1111} = \frac{0.0429}{0.1540} \approx 0.279$$

### 5. Explicit Probability Scope Acknowledgment
> **Explicit Acknowledgment:**  
> This inflates raw softmax by ~9x. It answers the question **"conditional on this candidate being reachable, how likely?"** rather than **"probability among all 9 classes."**
