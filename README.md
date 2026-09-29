# Money Migration Atlas (SIH26182)

> **AI-Native Cryptocurrency Intelligence Platform for Automated VASP Attribution & Legal Disclosure Package Generation**  
> *Developed for Smart India Hackathon 2026 — Ministry of Home Affairs (MHA) Problem Statement 26182*

---

## 1. About the Project

When criminal proceeds from cyber fraud, ransomware extortion, and darknet narcotics migrate across decentralized blockchains, Indian law enforcement agencies face a critical operational bottleneck. Laundering syndicates exploit multi-hop peel chains, coin splitters, cross-chain bridges, and decentralized mixing protocols to obfuscate the flow of funds. Under Section 91 of the Code of Criminal Procedure (CrPC), lawful disclosure directives and emergency freeze requests can only be served upon compliant, registered reporting entities under the purview of Financial Intelligence Unit - India (FIU-IND). Identifying which regulated exchange holds the terminus deposit wallet currently consumes hours or days of manual, error-prone tracing across fragmented block explorers.

Existing commercial intelligence tools (e.g., Chainalysis, TRM Labs, Elliptic) operate as proprietary, subscription-gated black boxes costing tens of thousands of dollars annually. Crucially, when an attribution determination is challenged in an Indian Sessions Court, commercial vendors cannot disclose proprietary source code or mathematical models for independent courtroom cross-examination. Furthermore, proprietary tools frequently collapse physical graph distance and heuristic clustering into ambiguous composite scores that violate digital evidentiary scrutiny.

**Money Migration Atlas** solves this challenge by delivering an open-source, mathematically transparent, and court-ready forensic intelligence pipeline. The platform ingests multi-chain ledgers (BTC, ETH, TRC-20 USDT), executes topological shortest-path traversals, extracts 20 tabular structural features for isotonically calibrated XGBoost classification, and performs neighborhood message passing via PyTorch Geometric GraphSAGE. Through an unblended dual-model consensus engine, the system arbitrates attribution verdicts, outputs plain-language agentic explanations with SHAP attributions, enforces an immutable SHA-256 digital chain of custody, and routes court-ready disclosure packages directly through a simulated MHA SAHYOG gateway.

---

## 2. Core Architectural Invariant: Strict Score Separation

A foundational requirement for SIH26182 court admissibility is **strict score separation**:
- **Proximity Rank (Topological Distance)**: Exact integer graph distance (hops) and Dijkstra traversal weight from the suspect wallet to a VASP-controlled deposit cluster.
- **Confidence Score (Model Attribution)**: Independent probabilistic confidence scores produced by XGBoost (`confidence_score`) and GraphSAGE (`gnn_confidence_score`).
- **Consensus Score (`consensus_score`)**: Independent consensus arbitration tier (`CONFIRMED`, `AMBIGUOUS`, `UNCERTAIN`, `SINGLE_MODEL`).

> **CRITICAL LEGAL INVARIANT**: Proximity rank and confidence scores are **NEVER blended** into a composite metric. Invariant `never_blended=True` is enforced at runtime via Pydantic validators. Courts receive physical graph facts and statistical predictions as strictly distinct parameters.

---

## 3. Architecture & 9-Phase Roadmap Final State

```
                                      +------------------------------------+
                                      |     Suspect Target Address         |
                                      +-----------------+------------------+
                                                        |
                                                        v
                                      +------------------------------------+
                                      |  Blockchain Ingestion & Fallback   |
                                      |   (Etherscan / TronGrid / Synth)   |
                                      +-----------------+------------------+
                                                        |
                                                        v
                                      +------------------------------------+
                                      | Dual GraphStore (NetworkX / Neo4j) |
                                      +--------+------------------+--------+
                                               |                  |
                       +-----------------------+                  +-----------------------+
                       | Topological BFS/Dijkstra                                         | 2-Hop Ego Subgraph
                       v                                                                  v
        +------------------------------+                                  +------------------------------+
        |   Tabular Feature Engine     |                                  | PyTorch Geometric GraphSAGE  |
        |  (20 features, SHAP values)  |                                  | (Node/Edge Message Passing)  |
        +--------------+---------------+                                  +--------------+---------------+
                       |                                                                 |
                       v                                                                 v
        +------------------------------+                                  +------------------------------+
        | XGBoost Classifier (Calib.)  |                                  | GNN Softmax + Null Class     |
        |   Brier Score: 0.074         |                                  |   ROC-AUC: 0.86              |
        +--------------+---------------+                                  +--------------+---------------+
                       |                                                                 |
                       +-----------------------+                  +----------------------+
                                               |                  |
                                               v                  v
                                      +------------------------------------+
                                      |    Multi-Model Consensus Engine    |
                                      |  (CONFIRMED, AMBIGUOUS, UNCERTAIN) |
                                      +-----------------+------------------+
                                                        |
                                                        v
                                      +------------------------------------+
                                      | Agentic Explainer & Counterfactual |
                                      |  (SHAP impact + model provenance)  |
                                      +-----------------+------------------+
                                                        |
                                                        v
                                      +------------------------------------+
                                      | Court-Ready Dossier (Sec. 91 CrPC) |
                                      |    & SHA-256 Chain of Custody      |
                                      +-----------------+------------------+
                                                        |
                                                        v
                                      +------------------------------------+
                                      |   MHA SAHYOG Mock Gateway Router   |
                                      |    (FIU-IND Reporting Registry)    |
                                      +------------------------------------+
```

### Complete 9-Phase Roadmap

| Phase | Milestone | Deliverables & Final State |
| :--- | :--- | :--- |
| **Phase 1** | **Graph Scaffold & Synthetic Data** | Dual GraphStore (NetworkX + Neo4j), deterministic seed generator (`RANDOM_SEED=42`), 9 VASPs, 550+ wallets, 2200+ transactions, 12 ground-truth benchmark cases. |
| **Phase 2** | **Multi-Chain Fetchers** | Provider abstraction for BTC (Blockchair), ETH/ERC-20 (Etherscan), TRC-20 (TronGrid) with SHA-256 caching, zero-crash fallback, and `DEMO_MODE=true` toggle. |
| **Phase 3** | **Neo4j Enterprise Parity** | Dual-engine graph store parity, Cypher projections, shortest-path Dijkstra algorithms, and automated fallback when Neo4j is offline. |
| **Phase 4 / 4.5** | **XGBoost Feature Attribution** | 20 tabular graph features (in/out degree, turnover velocity, path mixer penalty, flow fraction), isotonic probability calibration (Brier score 0.074), SHAP local explainability. |
| **Phase 5 / 5.5** | **Graph Neural Network (GNN)** | PyG converter, 2-layer GraphSAGE architecture, uncalibrated softmax fix via prior $p_{\text{null}} = 1/9$, dual-model orthogonal validation. |
| **Phase 6a** | **Behavioral Fingerprinting Audit** | 64-dimensional timing, gas, and structuring vectors. Rigorous mathematical honesty audit documenting sparse data limitations on synthetic graphs. |
| **Phase 7 / 7.5** | **Agentic Explainer & Custody** | Plain-language narrative generation, model-attributed quantitative counterfactuals, SHA-256 digital chain of custody with strictly monotonic timestamps. |
| **Phase 8** | **Interactive Web Application** | React 18 + Vite + Cytoscape.js interactive graph dashboard, multi-model confidence badges, consensus indicators, evidence JSON/Markdown exporters. |
| **Phase 9** | **SAHYOG Mock & Final Delivery** | FIU-IND VASP registry, simulated Section 91 CrPC disclosure routing, deterministic audit request IDs, legal language cleanup, and jury defense assets. |

---

## 4. Quick Start (3 Commands)

### 1. Clone & Set Up Backend
```bash
git clone https://github.com/Krishank1201/money-migration-atlas.git
cd money-migration-atlas/backend
python -m venv venv && source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8000 --reload
```

### 2. Set Up Frontend
In a new terminal:
```bash
cd money-migration-atlas/frontend
npm install
npm run dev
```

### 3. Open in Browser
Visit **`http://localhost:5173`** to access the live Money Migration Atlas forensic dashboard. Backend OpenAPI documentation is available at **`http://localhost:8000/docs`**.

---

## 5. 3-Minute Live Demo Walkthrough

Follow this scripted 3-minute sequence for hackathon presentations:

1. **[00:00 - 00:30] Problem & Approach**:
   - Open `http://localhost:5173`.
   - Explain the law enforcement bottleneck: manual tracing through mixers to find FIU-regulated off-ramps under Section 91 CrPC.
   - Point to the unblended metrics architecture: Proximity Rank vs. Confidence Score vs. Consensus.

2. **[00:30 - 01:15] Multi-Model Inference (CASE-002)**:
   - Select **CASE-002** (Ethereum cyber fraud drainer) from the Quick Scenario Selector.
   - Observe live pipeline execution: Dijkstra traversal $\rightarrow$ XGBoost inference $\rightarrow$ GraphSAGE message passing $\rightarrow$ Consensus evaluation.
   - Highlight: Dijkstra finds WazirX at 2 hops; XGBoost scores 0.86; GraphSAGE scores 0.86. Consensus tier: **CONFIRMED**.

3. **[01:15 - 02:00] Plain-Language Explanations & Counterfactuals**:
   - Scroll to the Agentic Summary and Counterfactual cards.
   - Demonstrate model-specific provenance: *"If flow volume dropped below 5,000 USDT, XGBoost confidence would fall from 0.86 to 0.41 (recomputed)."*

4. **[02:00 - 02:35] Chain of Custody & Court-Ready Dossier**:
   - Show the 5-step SHA-256 digital chain of custody audit trail with monotonic timestamps.
   - Click **Download Dossier (Markdown)**: Note the clear legal disclaimer: *"Court-ready dossier formatted for Section 91 CrPC disclosure (not a certification)."*

5. **[02:35 - 03:00] SAHYOG Gateway Routing**:
   - Scroll to the **Route to SAHYOG** panel.
   - Note the VASP status: CoinDCX / WazirX are FIU-IND registered.
   - Click **Submit Disclosure Request**; observe status progression: `QUEUED` $\rightarrow$ `ROUTED` $\rightarrow$ `ACKNOWLEDGED` (24h SLA active).
   - Point out that offshore VASPs (Bybit/Binance Offshore) disable automated routing and require MLAT / letters rogatory.

Full presentation notes and Q&A responses are located in [`docs/DEMO_SCRIPT.md`](file:///docs/DEMO_SCRIPT.md) and [`docs/JURY_QA.md`](file:///docs/JURY_QA.md).

---

## 6. Honest Limitations (Engineering & Scientific Integrity)

In accordance with strict scientific and engineering rigor, we openly state what Money Migration Atlas does **NOT** claim to solve:

1. **Synthetic Data vs. Real Mainnets**: All models (XGBoost, GraphSAGE) were trained, validated, and evaluated on synthetic graph data generated by `synthetic_generator.py`. While the generator models real laundering topologies (peeling chains, mixer pools, multi-hop fanouts), real mainnets exhibit significantly higher noise, unlabelled contract interactions, and dust transactions. Retraining on ground-truth exchange-labeled clusters is required for production deployment.
2. **Behavioral Fingerprinting is Decorative on Synthetic Data**: As rigorously audited and reported in Phase 6a, 19 of 64 behavioral dimensions showed near-zero variance ($\sigma < 0.01$) across synthetic transactions. Behavioral habit profiling requires rich, bursty mainnet transaction histories (dozens of transactions per wallet) to yield actionable forensic signal.
3. **Adversarial Self-Play Was Deferred**: Phase 6b (Generative Adversarial Red vs. Blue laundering simulation) was deferred to post-hackathon development in favor of hardening the core GNN, consensus engine, and court evidence packaging.
4. **SAHYOG Integration is a Mock**: The Ministry of Home Affairs SAHYOG platform is a restricted government intranet system without public sandbox APIs. Our SAHYOG router is an architectural simulation adhering to FIU-IND reporting specifications, not a production integration.
5. **Neo4j Parity Unverified Without Docker**: The codebase includes a production Neo4j Cypher adapter (`backend/app/graph/neo4j_store.py`). However, full graph parity unit tests require a running Docker daemon (`docker compose up -d neo4j`). In environments without Docker, the platform runs seamlessly on the in-memory `NetworkXStore`.

---

## 7. Pre-Demo Checklist

Before presenting to the evaluation panel:
- [x] Python 3.11+ environment with PyTorch Geometric and XGBoost installed.
- [x] Node.js 18+ environment with Vite build passing (`npm run build` succeeds with zero errors).
- [x] Backend verified with 100 automated test cases (`pytest tests/`).
- [x] Invariant `never_blended=True` verified across all schemas and responses.
- [x] `DEMO_MODE=true` set in backend `.env` for zero-dependency offline operation.
- [x] Note on Neo4j: If Docker daemon is running, start Neo4j via `docker compose up -d neo4j`. If Docker is not available, the platform automatically defaults to NetworkX with zero loss of functionality.

---

## 8. Academic References & Citations

1. **Graph Neural Networks for Financial Crime Detection**:
   - Weber, M., et al. (2019). *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics*. KDD Workshop on Applied Data Science for Healthcare and Social Good. [arXiv:1908.02591](https://arxiv.org/abs/1908.02591).
2. **Inductive Representation Learning on Graphs**:
   - Hamilton, W. L., Ying, R., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs (GraphSAGE)*. Advances in Neural Information Processing Systems (NeurIPS 2017). [arXiv:1706.02216](https://arxiv.org/abs/1706.02216).
3. **Explainable AI with Shapley Additive Explanations**:
   - Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions (SHAP)*. Advances in Neural Information Processing Systems (NeurIPS 2017). [arXiv:1705.07874](https://arxiv.org/abs/1705.07874).
4. **Probability Calibration for Legal Classifiers**:
   - Niculescu-Mizil, A., & Caruana, R. (2005). *Predicting Good Probabilities With Supervised Learning*. Proceedings of the 22nd International Conference on Machine Learning (ICML).
5. **Regulatory Framework & Lawful Interception**:
   - Financial Intelligence Unit - India (FIU-IND), Ministry of Finance, Government of India. *Anti-Money Laundering (AML) Guidelines for Virtual Digital Asset Service Providers (VDA SPs)*, 2023.
   - Code of Criminal Procedure, 1973 (CrPC), Section 91: *Summons to produce document or other thing*.
   - Indian Evidence Act, 1872, Section 65B: *Admissibility of electronic records*.
