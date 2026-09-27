from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query

from app.core.schemas import (
    Chain,
    SubgraphResponse,
    IngestionReport,
    NearestVASPCandidate,
    ConfidenceTier
)
from app.graph.store import GraphStore
from app.graph.ingestion import GraphIngestionPipeline
from app.graph.analytics import (
    degree_distribution,
    betweenness_centrality,
    detect_mixer_candidates,
    vasp_reachability_map
)

router = APIRouter(prefix="/graph", tags=["Graph & Analytics"])

_current_graph_store: Optional[GraphStore] = None
_fetch_orchestrator: Any = None
_pipeline = GraphIngestionPipeline()


def set_graph_context(store: GraphStore, orchestrator: Any = None):
    global _current_graph_store, _fetch_orchestrator
    _current_graph_store = store
    if orchestrator is not None:
        _fetch_orchestrator = orchestrator


def get_store() -> GraphStore:
    if _current_graph_store is None:
        raise HTTPException(status_code=503, detail="GraphStore not initialized")
    return _current_graph_store


@router.get("/stats")
async def get_graph_stats(store: GraphStore = Depends(get_store)) -> Dict[str, Any]:
    """Return graph metrics including node/edge counts and topological degree statistics."""
    stats = store.stats()
    deg_stats = degree_distribution(store)
    return {
        **stats,
        "analytics": {
            "degree_distribution": deg_stats
        }
    }


@router.get("/subgraph/{chain}/{address}", response_model=SubgraphResponse)
async def get_ego_subgraph(
    chain: Chain,
    address: str,
    hops: int = Query(default=3, ge=1, le=8),
    store: GraphStore = Depends(get_store)
) -> SubgraphResponse:
    """Extract ego-subgraph centered around an address for visualization."""
    sub = store.get_subgraph(address, hops=hops)
    return SubgraphResponse(
        center_address=address,
        chain=chain,
        hops=hops,
        nodes=sub.get("nodes", []),
        edges=sub.get("edges", [])
    )


@router.post("/ingest/{chain}/{address}", response_model=IngestionReport)
async def ingest_live_wallet(
    chain: Chain,
    address: str,
    depth: int = Query(default=2, ge=1, le=4),
    store: GraphStore = Depends(get_store)
) -> IngestionReport:
    """Fetch a wallet live and recursively ingest its multi-hop neighborhood into the graph."""
    if _fetch_orchestrator is None:
        raise HTTPException(status_code=503, detail="Fetch orchestrator not initialized")
    
    report = _pipeline.ingest_from_live_fetch(
        store=store,
        address=address,
        chain=chain,
        orchestrator=_fetch_orchestrator,
        depth=depth
    )
    return report


@router.get("/nearest-vasp/{chain}/{address}", response_model=List[NearestVASPCandidate])
async def get_nearest_vasps(
    chain: Chain,
    address: str,
    max_hops: int = Query(default=6, ge=1, le=10),
    store: GraphStore = Depends(get_store)
) -> List[NearestVASPCandidate]:
    """
    Find nearest VASP candidates sorted by proximity rank (graph hops).
    Confidence score is strictly independent and remains UNKNOWN until Phase 5 GNN evaluation.
    """
    raw_candidates = store.find_nearest_vasp(address, max_hops=max_hops)
    candidates: List[NearestVASPCandidate] = []
    
    for c in raw_candidates:
        candidates.append(
            NearestVASPCandidate(
                vasp_id=c["vasp_id"],
                vasp_name=c["vasp_name"],
                proximity_rank=c["proximity_rank"],
                distance=c["distance"],
                target_wallet=c.get("target_wallet"),
                path=c.get("path", []),
                tx_hashes=c.get("tx_hashes", []),
                fiu_ind_registered=c.get("fiu_ind_registered", False),
                confidence_score=None,
                confidence_tier=ConfidenceTier.UNKNOWN
            )
        )
    return candidates


@router.get("/analytics/mixers")
async def get_mixer_candidates(store: GraphStore = Depends(get_store)) -> List[Dict[str, Any]]:
    """Identify potential cryptocurrency mixer nodes based on fan-in, fan-out, and flow ratios."""
    return detect_mixer_candidates(store)


@router.get("/analytics/reachability/{chain}/{address}")
async def get_reachability_map(
    chain: Chain,
    address: str,
    max_hops: int = Query(default=6, ge=1, le=10),
    store: GraphStore = Depends(get_store)
) -> Dict[str, int]:
    """Return map of reachable VASPs and their minimum hop distances from the specified address."""
    return vasp_reachability_map(store, address, max_hops=max_hops)
