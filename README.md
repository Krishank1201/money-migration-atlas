# Money Migration Atlas (SIH26182)

> **AI-Native Cryptocurrency Intelligence Platform for Automated VASP Attribution & Asset Freezing**  
> *Developed for Smart India Hackathon 2026 — Ministry of Home Affairs (MHA)*

---

## 1. Problem Statement & Mission

When criminal proceeds (cyber fraud, ransomware, narcotics, extortion) migrate across public blockchains, investigators must identify which regulated Indian Virtual Asset Service Provider (VASP) holds custody of the destination wallet to issue Section 91 CrPC lawful disclosure directives and emergency freeze requests.

Currently, this investigative process takes **hours to days** of manual clustering and tracing across fragmented block explorers.

**Money Migration Atlas** automates this workflow down to seconds using:
1. Multi-chain transaction graph ingestion (BTC, ETH, TRC-20, BSC, Solana, Polygon).
2. Topological proximity ranking combined with Graph Neural Networks (GraphSAGE / GATv2).
3. Behavioral fingerprinting to de-anonymize mixer passes and peeling chains.
4. Generative adversarial red/blue laundering simulations.
5. Plain-language agentic explanations with complete cryptographic proof chains.
6. Direct mock routing into the **MHA SAHYOG** disclosure portal.

---

## 2. Core Architectural Invariant: Strict Score Separation

A foundational requirement for SIH26182 court admissibility is **strict score separation**:
- **Proximity Rank (Topological Distance)**: Exact integer graph distance (hops) and weighted Dijkstra traversal cost from the suspect wallet to a VASP-controlled hot wallet or deposit sweeper.
- **Confidence Score (Model Attribution)**: Independent probabilistic classification (0.0 to 1.0) produced by the GNN and behavioral classifier.

> **CRITICAL RULE**: These two metrics are **NEVER blended** into a composite score. Investigators and judges require distinct, unadulterated evidence.

---

## 3. Phase Roadmap (Phases 1–9)

| Phase | Description | Status |
| :--- | :--- | :--- |
| **Phase 1** | **Project Scaffold + Synthetic Data Generator**: Dual-engine GraphStore (Neo4j + NetworkX), deterministic seed generator, 9 VASPs, 550+ wallets, 2200+ transactions, 8 ground-truth benchmarks, health check. | **Completed** |
| **Phase 2** | **Multi-Chain Blockchain Fetcher**: Pluggable provider architecture for live on-chain ingestion (BTC, ETH, TRC-20, BSC, SOL, POLYGON) with rate limiting and local caching. | *Upcoming* |
| **Phase 3** | **Graph Construction & Neo4j Integration**: Full Cypher synchronization, APOC-accelerated graph projections, and multi-hop neighborhood expansion. | *Upcoming* |
| **Phase 4** | **Baseline ML & Feature Engineering**: Tabular graph feature extraction (in/out degree, turnover velocity, peeling indicators) and XGBoost baseline. | *Upcoming* |
| **Phase 5** | **Graph Neural Network (GNN)**: PyTorch Geometric GraphSAGE / GATv2 architecture predicting wallet-to-VASP attribution probabilities. | *Upcoming* |
| **Phase 6** | **Behavioral Fingerprinting & Adversarial Self-Play**: Temporal timing analysis, gas price profiling, and Red AI (laundering generator) vs. Blue AI (detection). | *Upcoming* |
| **Phase 7** | **Agentic AI Co-Investigator & API**: Autonomous LLM agent reasoning over the graph, generating plain-language reports with cited tx hashes. | *Upcoming* |
| **Phase 8** | **Frontend UI (Next.js / React + Cytoscape.js)**: Dark-mode dashboard, interactive transaction graph explorer, live proximity inspection, and telemetry. | *Upcoming* |
| **Phase 9** | **Court-Admissible Evidence Chain & SAHYOG Mock**: Automated PDF/JSON evidence package generator with hash verification and mock Section 91 freeze requests. | *Upcoming* |

---

## 4. Phase 1 Deliverables & Architecture

- **Dual Graph Engine (`app.graph`)**:
  - `GraphStore`: Abstract base class for clean swappability.
  - `NetworkXStore`: High-performance in-memory graph store with multi-hop BFS and path extraction. Ideal for zero-dependency hackathon laptop demos.
  - `Neo4jStore`: Enterprise graph database driver with automatic reachability fallback.
- **Deterministic Synthetic Generator (`app.data.synthetic_generator`)**:
  - Controlled by `RANDOM_SEED=42` for 100% reproducible results across machines.
  - Generates 9 VASPs (CoinDCX, WazirX, ZebPay, CoinSwitch, Mudrex, Giottus, Unocoin, Binance Offshore, Bybit).
  - 550+ wallets and 2200+ transactions across BTC, ETH, and TRON-TRC20.
  - Models 4 distinct laundering topologies: peeling chains, mixer/CoinJoin pools, nested transit hops, and deposit sweeper aggregation.
- **8 Ground-Truth Benchmark Cases**:
  - Pre-packaged investigative scenarios (`CASE-001` through `CASE-008`) with known ground-truth targets for automatic accuracy validation.

---

## 5. Quickstart

### Prerequisites
- Python 3.11+
- Virtual environment tool (`venv`)

### 1. Setup Environment
```bash
# Clone and enter directory
cd money-migration-atlas

# Create and activate Python virtual environment
python -m venv venv
# On Windows Powershell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 2. Configure Environment Variables
```bash
cp .env.example .env
```
Default `.env` runs with `DEMO_MODE=true` and `RANDOM_SEED=42`.

### 3. Run Automated Tests
```bash
cd backend
python -m pytest tests/ -v
```

### 4. Start Backend Server
```bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

### 5. Verify Health & Demo Endpoints
- **Health Check**: `GET http://localhost:8000/health`
- **List Benchmark Cases**: `GET http://localhost:8000/api/v1/demo/test-cases`
- **Evaluate Proximity**: `GET http://localhost:8000/api/v1/demo/benchmark/CASE-001/proximity`
- **Interactive OpenAPI Docs**: `http://localhost:8000/docs`
