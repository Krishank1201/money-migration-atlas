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
| **Phase 1** | **Project Scaffold + Synthetic Data Generator**: Dual-engine GraphStore (Neo4j + NetworkX), deterministic seed generator, 9 VASPs, 550+ wallets, 2200+ transactions, 12 ground-truth benchmarks, health check. | **Completed** |
| **Phase 2** | **Multi-Chain Blockchain Fetcher**: Provider abstraction for BTC (Blockchair), ETH/ERC-20 (Etherscan), TRC-20 (TronGrid) with SHA-256 file caching, resilient fallback, and DEMO_MODE toggle. | **Completed** |
| **Phase 3** | **Graph Construction & Neo4j Integration**: Full Cypher synchronization, APOC-accelerated graph projections, UNWIND bulk ingestion, multi-hop live ingestion, and topological analytics. | **Completed** |
| **Phase 4** | **Baseline ML & Feature Engineering**: Tabular graph feature extraction (in/out degree, turnover velocity, peeling indicators) and XGBoost baseline. | *Upcoming* |
| **Phase 5** | **Graph Neural Network (GNN)**: PyTorch Geometric GraphSAGE / GATv2 architecture predicting wallet-to-VASP attribution probabilities. | *Upcoming* |
| **Phase 6** | **Behavioral Fingerprinting & Adversarial Self-Play**: Temporal timing analysis, gas price profiling, and Red AI (laundering generator) vs. Blue AI (detection). | *Upcoming* |
| **Phase 7** | **Agentic AI Co-Investigator & API**: Autonomous LLM agent reasoning over the graph, generating plain-language reports with cited tx hashes. | *Upcoming* |
| **Phase 8** | **Frontend UI (Next.js / React + Cytoscape.js)**: Dark-mode dashboard, interactive transaction graph explorer, live proximity inspection, and telemetry. | *Upcoming* |
| **Phase 9** | **Court-Admissible Evidence Chain & SAHYOG Mock**: Automated PDF/JSON evidence package generator with hash verification and mock Section 91 freeze requests. | *Upcoming* |

---

## 4. Phase 2 Architecture: Multi-Chain Ingestion & Resilient Fallback

The ingestion layer (`backend/app/fetchers/`) unifies public blockchain explorers under a shared `BlockchainProvider` contract:
- **Bitcoin (`BitcoinProvider`)**: Blockchair API (`/bitcoin/dashboards/address/{address}`).
- **Ethereum (`EthereumProvider`)**: Etherscan API v2 (native ETH `txlist` + ERC-20 `tokentx`).
- **Tron (`TronProvider`)**: TronGrid API (native TRX + TRC-20 USDT contract transfers).

### Fallback Hierarchy & Zero-Crash Guarantee
1. **`DEMO_MODE=true` (Default for Hackathons)**: Bypasses all live network calls, immediately serving deterministic synthetic topologies. Guaranteed to function offline without API keys or internet.
2. **Local SHA-256 File Cache (`backend/data/cache/`)**: Checks cached JSON files by `hash(chain:address)`. Avoids burning third-party rate limits during active investigation.
3. **Live On-Chain API**: Queries Blockchair / Etherscan / TronGrid with timeout and exponential backoff.
4. **Resilient Fallback**: If an API is rate-limited, unreachable, or returns an error, the orchestrator logs a warning and generates synthetic history for that wallet. **The platform never crashes.**

### Toggling Between Live and Demo Modes
In `.env`:
```bash
# Offline demo mode (zero external API calls)
DEMO_MODE=true

# Live blockchain mode
DEMO_MODE=false
ETHERSCAN_API_KEY=your_etherscan_api_key
BLOCKCHAIR_API_KEY=your_blockchair_api_key  # Optional
TRONGRID_API_KEY=your_trongrid_api_key      # Optional
```

---

## 5. Phase 1 Architecture: Dual-Engine Graph & Hardened Benchmarks

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
- **Graph Statistics & Degree Distribution**: `GET http://localhost:8000/api/v1/graph/stats`
- **List Benchmark Cases**: `GET http://localhost:8000/api/v1/demo/test-cases`
- **Evaluate Proximity**: `GET http://localhost:8000/api/v1/demo/benchmark/CASE-001/proximity`
- **Extract Ego Subgraph**: `GET http://localhost:8000/api/v1/graph/subgraph/ETH/0x...`
- **Interactive OpenAPI Docs**: `http://localhost:8000/docs`

---

## 6. Graph Backend Selection (NetworkX & Neo4j Dual Engine)

Money Migration Atlas implements a resilient **Dual-Engine Graph Architecture**:
- **NetworkX (Default)**: In-memory multi-directed graph store. Zero external dependencies, starts in < 1 second, and powers fully reproducible offline hackathon presentations.
- **Neo4j (Production / Scale)**: Enterprise graph database with persistent Bolt connection, Cypher queries, UNWIND bulk ingestion, and APOC / Graph Data Science plugin support.

Both engines implement the identical `GraphStore` interface and produce **exact parity** across all 12 benchmark cases.

### Starting Neo4j (Optional)
```bash
# Start Neo4j 5.x container with APOC and GDS
docker compose up -d neo4j
```

### Dynamic Backend Switching Without Restart
Switch live between NetworkX and Neo4j at runtime via the Admin API:
```bash
# Inspect currently active store
curl http://localhost:8000/api/v1/admin/active-store

# Switch to Neo4j
curl -X POST "http://localhost:8000/api/v1/admin/switch-store?backend=neo4j"

# Switch back to NetworkX
curl -X POST "http://localhost:8000/api/v1/admin/switch-store?backend=networkx"
```
If Neo4j is offline or unreachable, the system gracefully falls back to NetworkX without crashing.

---

## 7. The Three Behavioral Signals (Phase 6a)

Traditional attribution follows money: addresses, hops, and transaction graph paths.
**Money Migration Atlas ALSO follows human behavior.**

Criminal syndicates frequently discard addresses, swap intermediate wallets, and peel transactions through coin mixers (e.g., Tornado Cash, ChipMixer) to break graph linkages. However, **perpetrators cannot easily change their operational habits**.

Phase 6a extracts a 64-dimensional behavioral fingerprint that captures three enduring habitual dimensions:

### 1. Timing: Inter-Hop Delay Patterns (16 Dimensions)
- **What it captures**: Log-spaced histograms of inter-transaction latency, median transfer intervals, standard deviations, min/max wait times, and rapid-hop burst ratios ($\le 60$ seconds).
- **Why it persists across wallet changes**: Automated scripts (bot syndicates) execute hops with predictable sub-second or fixed-interval intervals (e.g., exactly 120s or 1800s batch sweeps). Human operators display circadian sleep-wake cycles, business-hour clustering, and typical manual reaction delays. Even when transferring funds to a completely fresh address, automated or manual operational rhythms remain constant.

### 2. Gas Fee: Quantization Preferences (16 Dimensions)
- **What it captures**: Mean and median gas fees, gas fee quantization score (fraction aligned with round Gwei or round decimal increments), outlier ratios, skewness, kurtosis, and zero-fee transactions.
- **Why it persists across wallet changes**: Wallet software, custom RPC nodes, and developer scripts possess distinct default gas setting configurations (e.g., "fast", "standard", or hardcoded gas prices like 20 Gwei). Laundering operations operating from consistent hardware, bots, or script frameworks continue to inject their default fee profiles across every newly generated intermediary wallet.

### 3. Amount Structuring: Round-Number Splitting (16 Dimensions)
- **What it captures**: Log-scaled amount moments, round-number fraction (divisible by 10, 100, 1000 or clean decimals), 5-tier denomination ratios ($<0.1$, $0.1–1$, $1–10$, $10–100$, $>100$), and output-to-input split ratios.
- **Why it persists across wallet changes**: Peeling chains and structured smurfing follow specific syndicate liquidation policies (e.g., peeling exactly 5 ETH or 10,000 USDT increments). Smurfers systematically avoid irregular fractional outputs or adhere to predefined capital tranche quotas that persist across every peeling step before deposit.

### Strict Score Separation Invariant:
The resulting `behavioral_confidence_score` operates as a **strictly independent third signal**:
- `proximity_rank`: Topological graph hops (how close?)
- `confidence_score`: XGBoost tabular signals (how sure based on path structure?)
- `gnn_confidence_score`: GNN relational embeddings (how sure based on graph topology?)
- `behavioral_confidence_score`: Fingerprint cosine match (how sure based on habit signatures?)
- `consensus_score = max(xgb, gnn, behavioral)`: Legitimate max-probability ensemble for final ranking.
- All underlying raw scores remain visible and **NEVER blended**.


