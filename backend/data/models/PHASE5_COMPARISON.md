# Phase 5 Attribution Benchmark: 4-Way Comparison Report

**System:** Money Migration Atlas (SIH26182)
**Philosophy:** XGBoost (Tabular) + GNN (Structural) Dual-Score Multi-Model Agreement
**Constraint Check:** `proximity_rank`, `confidence_score`, and `gnn_confidence_score` are strictly NEVER blended.

---

## Overall Performance Summary

| Approach | Accuracy | Score | Solved Cases |
| :--- | :--- | :--- | :--- |
| **1. Proximity Only** (Phase 3 Baseline) | 6/12 | 50.0% | Shortest graph hops |
| **2. XGBoost Only** (Phase 4 Tabular) | 10/12 | 83.3% | Path, mixer, and temporal features |
| **3. GNN Only** (Phase 5 Structural) | 10/12 | 83.3% | GraphSAGE + GATv2 message passing |
| **4. Consensus (by consensus_score)** | **11/12** | **91.7%** | Max-probability ensemble ranking |

---

## 4-Way Per-Case Breakdown

| Case | Target | Proximity | XGB | GNN | Consensus (by consensus_score) | Consensus Tier | XGB right? | GNN right? | Consensus right? | GNN changed answer vs XGB? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `CASE-001` | **coindcx** | coindcx (3h) | coindcx (0.47) | coindcx (0.46) | coindcx (0.47) | `UNCERTAIN` | YES | YES | YES | NO |
| `CASE-002` | **wazirx** | bybit (2h) | bybit (0.52) | wazirx (0.86) | wazirx (0.86) | `AMBIGUOUS` | NO | YES | YES | YES |
| `CASE-003` | **zebpay** | bybit (2h) | zebpay (0.87) | zebpay (0.94) | zebpay (0.94) | `CONFIRMED` | YES | YES | YES | NO |
| `CASE-004` | **coinswitch** | coinswitch (1h) | coinswitch (0.14) | coinswitch (0.39) | coinswitch (0.39) | `UNCERTAIN` | YES | YES | YES | NO |
| `CASE-005` | **mudrex** | mudrex (3h) | mudrex (0.56) | mudrex (0.43) | mudrex (0.56) | `UNCERTAIN` | YES | YES | YES | NO |
| `CASE-006` | **giottus** | bybit (2h) | giottus (0.81) | giottus (0.67) | giottus (0.81) | `CONFIRMED` | YES | YES | YES | NO |
| `CASE-007` | **unocoin** | unocoin (3h) | unocoin (0.44) | unocoin (0.67) | unocoin (0.67) | `SINGLE_MODEL` | YES | YES | YES | NO |
| `CASE-008` | **binance_offshore** | binance_offshore (3h) | binance_offshore (0.35) | binance_offshore (0.46) | binance_offshore (0.46) | `UNCERTAIN` | YES | YES | YES | NO |
| `CASE-101` | **coindcx** | coindcx (5h) | coindcx (0.87) | coindcx (0.46) | coindcx (0.87) | `AMBIGUOUS` | YES | YES | YES | NO |
| `CASE-102` | **zebpay** | wazirx (3h) | wazirx (0.38) | zebpay (0.63) | zebpay (0.63) | `SINGLE_MODEL` | NO | YES | YES | YES |
| `CASE-103` | **wazirx** | bybit (2h) | wazirx (0.81) | bybit (0.53) | wazirx (0.81) | `SINGLE_MODEL` | YES | NO | YES | NO |
| `CASE-104` | **mudrex** | binance_offshore (2h) | mudrex (0.56) | binance_offshore (0.61) | binance_offshore (0.61) | `SINGLE_MODEL` | YES | NO | NO | YES |

---

## Model Accuracy Summary

- **XGB alone:** 10/12
- **GNN alone:** 10/12
- **Consensus (max):** 11/12

### GNN Contribution Cases (GNN changed answer from wrong to right):

- **`CASE-002` (wazirx)**: XGBoost wrongly predicted `bybit` (0.52). GNN correctly predicted `wazirx` (0.86). Consensus picked `wazirx`.
- **`CASE-102` (zebpay)**: XGBoost wrongly predicted `wazirx` (0.38). GNN correctly predicted `zebpay` (0.63). Consensus picked `zebpay`.

### Consensus Tier Distribution across 12 Cases:
- **CONFIRMED (2):** `CASE-003`, `CASE-006`
- **AMBIGUOUS (2):** `CASE-002`, `CASE-101`
- **SINGLE_MODEL (4):** `CASE-007`, `CASE-102`, `CASE-103`, `CASE-104`
- **UNCERTAIN (4):** `CASE-001`, `CASE-004`, `CASE-005`, `CASE-008`

---

## Damage Assessment

- **Cases where consensus picked WRONG with tier=CONFIRMED:** 0 (target: 0)
- **Cases where consensus picked WRONG with tier=SINGLE_MODEL:** 1 (`CASE-104`)
- **Cases where consensus changed the answer vs XGB:** 3
  - **Net improvement:** +2 (`CASE-002`, `CASE-102`)
  - **Net regression:** -1 (`CASE-104`, but flagged for human review)
- **Cases where GNN contributed to the correct answer:** 2

---

**Conclusion:** Consensus scoring via max-probability ensemble enables graph-structural signals to correct tabular false assignments while maintaining absolute separation of underlying model scores.