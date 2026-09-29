"""
SAHYOG Mock Router API Endpoints (Phase 9).
Money Migration Atlas (SIH26182).

Provides simulated integration endpoints for the Indian Ministry of Home Affairs (MHA)
SAHYOG cyber-fraud repository and legal disclosure routing portal.
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Body
from pydantic import BaseModel

from app.core.schemas import Chain, SahyogRequest, SahyogAction
from app.graph.store import GraphStore
from app.integrations.sahyog_mock import sahyog_router
from app.ml.consensus import ConsensusScorer
from app.evidence.chain import EvidenceChainBuilder

logger = logging.getLogger("mma.api.sahyog")

sahyog_api_router = APIRouter(prefix="/sahyog", tags=["SAHYOG Mock Integration"])

_current_graph_store: Optional[GraphStore] = None


class SubmitRequestBody(BaseModel):
    requested_action: SahyogAction = SahyogAction.DISCLOSURE


def set_sahyog_context(store: GraphStore):
    global _current_graph_store
    _current_graph_store = store
    logger.info("SAHYOG API context initialized.")


def get_store() -> GraphStore:
    if _current_graph_store is None:
        raise HTTPException(status_code=503, detail="GraphStore not initialized")
    return _current_graph_store


@sahyog_api_router.post("/request/{chain}/{address}", response_model=SahyogRequest)
def submit_sahyog_request(
    chain: Chain,
    address: str,
    body: Optional[SubmitRequestBody] = Body(default=SubmitRequestBody()),
    store: GraphStore = Depends(get_store)
):
    """
    Submits a mock Section 91 CrPC legal disclosure or asset freeze request
    to the top-ranked candidate VASP via the MHA SAHYOG portal.
    """
    scorer = ConsensusScorer()
    candidates = scorer.predict(address, store, max_hops=6, include_behavioral=True)

    if not candidates:
        raise HTTPException(
            status_code=404,
            detail=f"No candidate Virtual Asset Service Providers reachable from {address}"
        )

    top_candidate = candidates[0]
    builder = EvidenceChainBuilder()
    evidence_package = builder.build(top_candidate, store, address, chain=chain.value)

    action = body.requested_action if body else SahyogAction.DISCLOSURE

    try:
        req = sahyog_router.submit_disclosure_request(
            candidate=top_candidate,
            evidence_package=evidence_package,
            requested_action=action,
            chain=chain.value
        )
        return req
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@sahyog_api_router.get("/status/{request_id}", response_model=SahyogRequest)
def get_request_status(request_id: str):
    """
    Polls the real-time processing status of a submitted SAHYOG request (QUEUED -> ROUTED -> ACKNOWLEDGED).
    """
    req = sahyog_router.get_request_status(request_id)
    if not req:
        raise HTTPException(status_code=404, detail=f"SAHYOG request '{request_id}' not found")
    return req


@sahyog_api_router.get("/requests", response_model=List[SahyogRequest])
def list_submitted_requests():
    """
    Lists all mock legal disclosure and freeze requests submitted during the active session.
    """
    return sahyog_router.list_requests()


@sahyog_api_router.get("/vasp-registry")
def get_vasp_registry() -> Dict[str, Any]:
    """
    Returns the mock FIU-IND registry of registered and overseas cryptocurrency VASPs.
    """
    return {
        "disclaimer": sahyog_router.registry.get("disclaimer"),
        "authority": sahyog_router.registry.get("authority"),
        "registry_version": sahyog_router.registry.get("registry_version"),
        "vasps": sahyog_router.get_all_registered_vasps()
    }
