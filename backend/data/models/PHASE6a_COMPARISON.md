# Phase 6a Attribution Benchmark: 6-Way Comparison Report

**System:** Money Migration Atlas (SIH26182)  
**Philosophy:** XGBoost (Tabular) + GNN (Structural) + Behavioral Fingerprint (Habitual Signatures)  
**Constraint Check:** `proximity_rank`, `confidence_score` (XGB), `gnn_confidence_score` (GNN), and `behavioral_confidence_score` are strictly independent and **NEVER blended**.

---

## 1. Overall Performance Summary

| Approach | Accuracy | Score | Description |
| :--- | :---: | :---: | :--- |
| **1. Proximity Only** (Phase 3 Baseline) | 6/12 | 50.0% | Shortest graph hops (Dijkstra) |
| **2. XGBoost Only** (Phase 4 Tabular) | 10/12 | 83.3% | Path, mixer, and temporal feature vectors |
| **3. GNN Only** (Phase 5 Structural) | 10/12 | 83.3% | GraphSAGE + GATv2 message-passing embeddings |
| **4. Behavioral Fingerprint Only** (Phase 6a) | **N/A\*** | **INSUFFICIENT_DATA** | All 12 benchmark suspects have <3 txs (Bucket A: 1-2 txs) |
| **5. Consensus (Max Ensemble)** | **11/12** | **91.7%** | `consensus_score = max(xgb, gnn, behavioral)` |

\* *Note: If computed naively on 1–2 transactions via path aggregation or amount matching, behavioral scored 10/12 (83.3%). However, honesty auditing proved that 100% of benchmark suspect wallets possess only 1–2 outgoing transactions. Under strict wallet maturity enforcement (Fix 2: $\ge 3$ outgoing txs required), behavioral returns `None` (`INSUFFICIENT_DATA`) for all benchmark cases.*

---

## 2. Honest 6-Way Per-Case Breakdown

| Case | Target | Proximity | XGB | GNN | Behavioral (Maturity Check) | Consensus (max) | Consensus Tier | Match |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| `CASE-001` | **coindcx** | coindcx (3h) | coindcx (0.47) | coindcx (0.46) | `INSUFFICIENT_DATA` (None) | coindcx (0.47) | `UNCERTAIN` | YES |
| `CASE-002` | **wazirx** | bybit (2h) | bybit (0.52) | wazirx (0.86) | `INSUFFICIENT_DATA` (None) | wazirx (0.86) | `AMBIGUOUS` | YES |
| `CASE-003` | **zebpay** | bybit (2h) | zebpay (0.87) | zebpay (0.94) | `INSUFFICIENT_DATA` (None) | zebpay (0.94) | `CONFIRMED` | YES |
| `CASE-004` | **coinswitch** | coinswitch (1h) | coinswitch (0.14) | coinswitch (0.39) | `INSUFFICIENT_DATA` (None) | coinswitch (0.39) | `UNCERTAIN` | YES |
| `CASE-005` | **mudrex** | mudrex (3h) | mudrex (0.56) | mudrex (0.43) | `INSUFFICIENT_DATA` (None) | mudrex (0.56) | `UNCERTAIN` | YES |
| `CASE-006` | **giottus** | bybit (2h) | giottus (0.81) | giottus (0.67) | `INSUFFICIENT_DATA` (None) | giottus (0.81) | `CONFIRMED` | YES |
| `CASE-007` | **unocoin** | unocoin (3h) | unocoin (0.44) | unocoin (0.67) | `INSUFFICIENT_DATA` (None) | unocoin (0.67) | `SINGLE_MODEL` | YES |
| `CASE-008` | **binance_offshore** | binance_offshore (3h) | binance_offshore (0.35) | binance_offshore (0.46) | `INSUFFICIENT_DATA` (None) | binance_offshore (0.46) | `UNCERTAIN` | YES |
| `CASE-101` | **coindcx** | coindcx (5h) | coindcx (0.87) | coindcx (0.46) | `INSUFFICIENT_DATA` (None) | coindcx (0.87) | `AMBIGUOUS` | YES |
| `CASE-102` | **zebpay** | wazirx (3h) | wazirx (0.38) | zebpay (0.63) | `INSUFFICIENT_DATA` (None) | zebpay (0.63) | `SINGLE_MODEL` | YES |
| `CASE-103` | **wazirx** | bybit (2h) | wazirx (0.81) | bybit (0.53) | `INSUFFICIENT_DATA` (None) | wazirx (0.81) | `SINGLE_MODEL` | YES |
| `CASE-104` | **mudrex** | binance_offshore (2h) | mudrex (0.56) | binance_offshore (0.61) | `INSUFFICIENT_DATA` (None) | binance_offshore (0.61) | `SINGLE_MODEL` | NO |

---

## 3. Honesty Audit & Behavioral Signal Diagnostic

### Audit Findings:
1. **Fingerprint Variance (Audit 1):**
   - Out of 64 dimensions across all 550 graph wallets, 49 dimensions (76.6%) have $\text{std} \ge 0.10$, and 60 dimensions (93.8%) have $\text{std} \ge 0.01$. 4 dimensions (indices 9, 21, 22, 31) are near-constant ($\text{std} < 0.01$).
2. **Wallet Maturity Breakdown (Audit 2):**
   - **Bucket A (1–2 transactions):** 12 / 12 benchmark suspect cases (100%).
   - **Bucket B (3–5 transactions):** 0 / 12 cases.
   - **Bucket C (6+ transactions):** 0 / 12 cases.
   - A fingerprint derived from 1–2 transactions is a static template, not a true habit fingerprint.
3. **Randomized Label Control (Audit 3):**
   - Shuffling ground-truth VASP labels dropped unconstrained behavioral accuracy from 83.3% (10/12) to **8.3% (1/12)**, consistent with 1/9 random baseline (~11.1%).
4. **Feature Ablation — Timing/Gas Only (Audit 4):**
   - Removing all amount structuring features and retaining only 32 timing + gas fee dimensions dropped accuracy from 10/12 to **6/12 (50.0%)**.
   - 40% of the behavioral signal on synthetic cases was disguised amount-matching (which XGBoost already models).

---

## 4. Damage Assessment & Post-Fix Validation

- **Cases where consensus picked WRONG with tier=CONFIRMED:** **0 (Target: 0)**.
  - In `CASE-104`, the previous `CONFIRMED` false-positive bug was eliminated. Top candidate Binance Offshore is now assigned `SINGLE_MODEL` (XGB=0.28, GNN=0.61, Behavioral=None).
- **Cases where consensus changed answer vs XGB:** 3 (`CASE-002`, `CASE-102`, `CASE-104`).
  - **Net improvement:** +2 (`CASE-002`, `CASE-102` resolved by GNN).
  - **Net regression:** -1 (`CASE-104`, adversarial 2-hop decoy).
- **Consensus Accuracy Stability:** Phase 5 (11/12) $\rightarrow$ **Phase 6a (11/12)**. Zero accuracy regression.
- **Architectural Status:** Behavioral habit attribution downgraded to a **supporting signal active exclusively on wallets with sufficient outgoing transaction history ($\ge 3$ txs)**.
