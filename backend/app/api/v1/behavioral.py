"""
Behavioral Fingerprint API Endpoints (Phase 6a).
Money Migration Atlas (SIH26182).

Endpoints:
- GET /api/v1/behavioral/fingerprint/{chain}/{address} → 64-dim fingerprint + human-readable summary
- GET /api/v1/behavioral/similar/{chain}/{address}?top_k=5 → top-k behaviorally similar wallets
- POST /api/v1/behavioral/predict/{chain}/{address}?max_hops=6 → attribution with behavioral_confidence_score
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Path

from app.core.schemas import (
    Chain,
    MLPredictResponse,
    NearestVASPCandidate,
    ConfidenceTier,
    BehavioralFingerprint
)
from app.graph.store import GraphStore
from app.behavioral.fingerprint import extract_fingerprint, extract_fingerprint_safe
from app.behavioral.similarity import find_similar_wallets
from app.behavioral.attributor import BehavioralAttributor

logger = logging.getLogger("mma.api.behavioral")

behavioral_router = APIRouter(prefix="/behavioral", tags=["Behavioral Fingerprinting"])

# Injected by lifespan
_current_graph_store: Optional[GraphStore] = None
_attributor: Optional[BehavioralAttributor] = None


def set_behavioral_context(store: GraphStore):
    global _current_graph_store, _attributor
    _current_graph_store = store
    _attributor = BehavioralAttributor(store)
    logger.info("Behavioral API context initialized.")


def get_store() -> GraphStore:
    if _current_graph_store is None:
        raise HTTPException(status_code=503, detail="GraphStore not initialized")
    return _current_graph_store


def get_attributor() -> BehavioralAttributor:
    global _attributor
    if _attributor is None:
        store = get_store()
        _attributor = BehavioralAttributor(store)
    return _attributor


@behavioral_router.get(
    "/fingerprint/{chain}/{address}",
    response_model=BehavioralFingerprint,
    summary="Extract 64-dimensional behavioral habit fingerprint"
)
async def get_fingerprint_endpoint(
    chain: Chain = Path(..., description="Target blockchain network"),
    address: str = Path(..., description="Target wallet address"),
    store: GraphStore = Depends(get_store)
) -> BehavioralFingerprint:
    """
    Extracts the 64-dimensional behavioral fingerprint capturing timing intervals,
    gas fee preferences, amount structuring, and temporal/chain habits.
    """
    try:
        return extract_fingerprint(store, address, min_txs=3)
    except ValueError:
        try:
            return extract_fingerprint(store, address, min_txs=1, direction="both")
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))


@behavioral_router.get(
    "/similar/{chain}/{address}",
    summary="Find behaviorally similar wallets across the network"
)
async def get_similar_wallets_endpoint(
    chain: Chain = Path(..., description="Target blockchain network"),
    address: str = Path(..., description="Target wallet address"),
    top_k: int = Query(5, ge=1, le=50, description="Number of nearest neighbors to return"),
    store: GraphStore = Depends(get_store)
) -> Dict[str, Any]:
    """
    Scans the graph population for wallets displaying matching transaction habits.
    Returns ranked similar addresses with normalized cosine similarity scores.
    """
    similar = find_similar_wallets(address, store, top_k=top_k)
    return {
        "address": address,
        "chain": chain.value,
        "top_k": top_k,
        "similar_wallets": [
            {"address": item[0], "similarity_score": round(item[1], 4)}
            for item in similar
        ]
    }


@behavioral_router.post(
    "/predict/{chain}/{address}",
    response_model=MLPredictResponse,
    summary="Independent behavioral VASP attribution"
)
async def predict_behavioral_endpoint(
    chain: Chain = Path(..., description="Suspect wallet chain"),
    address: str = Path(..., description="Suspect wallet address"),
    max_hops: int = Query(6, ge=1, le=10, description="Maximum graph exploration radius"),
    store: GraphStore = Depends(get_store),
    attributor: BehavioralAttributor = Depends(get_attributor)
) -> MLPredictResponse:
    """
    Evaluates reachable VASP deposit pathways and scores candidates strictly via
    behavioral fingerprint matching against VASP deposit sweeper infrastructures.
    """
    # 1. Topological reachability
    raw_candidates = store.find_nearest_vasp(address, max_hops=max_hops)
    if not raw_candidates:
        return MLPredictResponse(suspect_wallet=address, chain=chain, candidates=[])

    candidates: List[NearestVASPCandidate] = []
    for c in raw_candidates:
        cand = NearestVASPCandidate(
            vasp_id=c["vasp_id"],
            vasp_name=c["vasp_name"],
            proximity_rank=c["proximity_rank"],
            distance=c["distance"],
            target_wallet=c.get("target_wallet"),
            path=c.get("path", []),
            tx_hashes=c.get("tx_hashes", []),
            chain_path=c.get("chain_path", []),
            fiu_ind_registered=c.get("fiu_ind_registered", False),
            confidence_score=0.0,
            never_blended=True
        )
        candidates.append(cand)

    # 2. Enrich with behavioral confidence scores
    attributor.attribute_candidates(address, candidates)

    # Rank by behavioral_confidence_score descending, proximity_rank ascending
    candidates.sort(key=lambda c: (c.behavioral_confidence_score or 0.0, -c.proximity_rank), reverse=True)

    return MLPredictResponse(
        suspect_wallet=address,
        chain=chain,
        candidates=candidates,
        never_blended=True
    )
