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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("mma")

# Global active graph store
active_graph_store: GraphStore = None
benchmark_cases = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global active_graph_store, benchmark_cases
    settings = get_settings()
    logger.info("Initializing %s (DEMO_MODE=%s, SEED=%s)", settings.APP_NAME, settings.DEMO_MODE, settings.RANDOM_SEED)

    # 1. Attempt Neo4j connection reachability check
    neo4j_store = Neo4jStore(
        uri=settings.NEO4J_URI,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DATABASE,
        timeout=settings.NEO4J_TIMEOUT_SECONDS
    )

    if not settings.DEMO_MODE and neo4j_store.is_available():
        logger.info("Using Neo4j graph store backend")
        active_graph_store = neo4j_store
    else:
        logger.info("Using high-performance in-memory NetworkX graph store backend")
        active_graph_store = NetworkXStore()

    # 2. Populate with synthetic data in DEMO_MODE
    if settings.DEMO_MODE:
        logger.info("Generating deterministic synthetic crypto graph...")
        vasps, wallets, transactions, benchmark_cases = generate_synthetic_data(
            seed=settings.RANDOM_SEED,
            store=active_graph_store
        )
        set_demo_context(active_graph_store, benchmark_cases)
        logger.info(
            "Synthetic graph ready: %d VASPs, %d wallets, %d transactions, %d benchmark cases loaded.",
            len(vasps), len(wallets), len(transactions), len(benchmark_cases)
        )

    yield

    # Teardown
    if isinstance(active_graph_store, Neo4jStore):
        active_graph_store.close()


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
