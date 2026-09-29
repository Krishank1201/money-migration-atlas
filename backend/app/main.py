import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.graph.networkx_store import NetworkXStore
from app.graph.neo4j_store import Neo4jStore
from app.graph.store import GraphStore
from app.data.synthetic_generator import generate_synthetic_data
from app.api.v1.demo import router as demo_router, set_demo_context
from app.api.v1.fetcher import router as fetcher_router, set_fetch_orchestrator
from app.api.v1.graph import router as graph_router, set_graph_context
from app.api.v1.admin import router as admin_router, set_admin_context
from app.api.v1.ml import router as ml_router, set_ml_context
from app.api.v1.gnn import gnn_router, consensus_router, set_gnn_context
from app.api.v1.behavioral import behavioral_router, set_behavioral_context
from app.api.v1.agentic import agentic_router, set_agentic_context
from app.api.v1.evidence import evidence_router, set_evidence_context
from app.fetchers.orchestrator import FetchOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("mma")

# Global active graph store, backends, and orchestrator
active_graph_store: GraphStore = None
nx_store: NetworkXStore = None
neo4j_store: Neo4jStore = None
fetch_orchestrator: FetchOrchestrator = None
benchmark_cases = []


def on_store_switched(new_store: GraphStore):
    """Callback triggered by admin router when graph store is switched."""
    global active_graph_store, fetch_orchestrator, benchmark_cases
    active_graph_store = new_store
    set_demo_context(new_store, benchmark_cases)
    set_graph_context(new_store, fetch_orchestrator)
    set_ml_context(new_store, benchmark_cases)
    set_gnn_context(new_store, benchmark_cases)
    set_behavioral_context(new_store)
    set_agentic_context(new_store)
    set_evidence_context(new_store)
    if fetch_orchestrator:
        fetch_orchestrator.graph_store = new_store
    logger.info("Active store propagated to all modules: %s", new_store.__class__.__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global active_graph_store, nx_store, neo4j_store, fetch_orchestrator, benchmark_cases
    settings = get_settings()
    logger.info("Initializing %s (DEMO_MODE=%s, SEED=%s, BACKEND=%s)", 
                settings.APP_NAME, settings.DEMO_MODE, settings.RANDOM_SEED, settings.GRAPH_BACKEND)

    # 1. Initialize both graph store backends
    nx_store = NetworkXStore()
    neo4j_store = Neo4jStore(
        uri=settings.NEO4J_URI,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DATABASE,
        timeout=settings.NEO4J_TIMEOUT_SECONDS
    )

    # Determine initial active store based on configuration and availability
    if settings.GRAPH_BACKEND.lower() == "neo4j" and neo4j_store.is_available():
        logger.info("Using Neo4j graph store backend as configured.")
        active_graph_store = neo4j_store
    else:
        logger.info("Using high-performance in-memory NetworkX graph store backend.")
        active_graph_store = nx_store

    # 2. Populate synthetic data in DEMO_MODE
    if settings.DEMO_MODE:
        logger.info("Generating deterministic synthetic crypto graph...")
        # Always populate NetworkX store as reliable baseline
        vasps, wallets, transactions, benchmark_cases = generate_synthetic_data(
            seed=settings.RANDOM_SEED,
            store=nx_store
        )
        # If Neo4j is active and reachable, populate it as well
        if active_graph_store is neo4j_store and neo4j_store.is_available():
            for v in vasps:
                neo4j_store.add_vasp(v)
            neo4j_store.bulk_ingest(wallets, transactions)

        logger.info(
            "Synthetic graph ready: %d VASPs, %d wallets, %d transactions, %d benchmark cases loaded.",
            len(vasps), len(wallets), len(transactions), len(benchmark_cases)
        )

    # 3. Initialize FetchOrchestrator
    fetch_orchestrator = FetchOrchestrator(graph_store=active_graph_store)

    # 4. Bind contexts
    set_demo_context(active_graph_store, benchmark_cases)
    set_fetch_orchestrator(fetch_orchestrator)
    set_graph_context(active_graph_store, fetch_orchestrator)
    set_admin_context(nx_store, neo4j_store, active_graph_store, on_store_switched)
    set_ml_context(active_graph_store, benchmark_cases)
    set_gnn_context(active_graph_store, benchmark_cases)
    set_behavioral_context(active_graph_store)
    set_agentic_context(active_graph_store)
    set_evidence_context(active_graph_store)

    yield

    # Teardown
    if neo4j_store:
        neo4j_store.close()


app = FastAPI(
    title="Money Migration Atlas API",
    description="Automated AI-native crypto investigation platform for VASP attribution (SIH26182).",
    version="0.1.0",
    lifespan=lifespan
)

# Enable CORS for development frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routers
app.include_router(demo_router, prefix="/api/v1")
app.include_router(fetcher_router, prefix="/api/v1")
app.include_router(graph_router, prefix="/api/v1")
app.include_router(admin_router, prefix="/api/v1")
app.include_router(ml_router, prefix="/api/v1")
app.include_router(gnn_router, prefix="/api/v1")
app.include_router(consensus_router, prefix="/api/v1")
app.include_router(behavioral_router, prefix="/api/v1")
app.include_router(agentic_router, prefix="/api/v1")
app.include_router(evidence_router, prefix="/api/v1")


@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint exposing system state, active backend, and dataset scale."""
    settings = get_settings()
    stats = active_graph_store.stats() if active_graph_store else {}
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "demo_mode": settings.DEMO_MODE,
        "random_seed": settings.RANDOM_SEED,
        "graph_backend": stats.get("backend", "uninitialized"),
        "node_count": stats.get("node_count", 0),
        "edge_count": stats.get("edge_count", 0),
        "vasp_count": stats.get("vasp_count", 0),
        "wallet_count": stats.get("wallet_count", 0),
        "transaction_count": stats.get("transaction_count", 0),
        "benchmark_cases_count": len(benchmark_cases)
    }


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
