"""
Agentic Co-Investigator API Endpoints (Phase 7).
Money Migration Atlas (SIH26182).

Endpoints:
- POST /api/v1/agentic/investigate/{chain}/{address}?max_hops=6
  Executes comprehensive forensic investigation returning plain-language narrative,
  counterfactuals, court evidence dossier, and recommended legal triage actions.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Path

from app.core.schemas import Chain, InvestigationReport
from app.graph.store import GraphStore
from app.agentic.agent import ForensicAgent

logger = logging.getLogger("mma.api.agentic")

agentic_router = APIRouter(prefix="/agentic", tags=["Agentic Co-Investigator"])

_current_graph_store: Optional[GraphStore] = None
_forensic_agent: Optional[ForensicAgent] = None


def set_agentic_context(store: GraphStore):
    global _current_graph_store, _forensic_agent
    _current_graph_store = store
    _forensic_agent = ForensicAgent()
    logger.info("Agentic API context initialized.")


def get_agent() -> ForensicAgent:
    global _forensic_agent
    if _forensic_agent is None:
        _forensic_agent = ForensicAgent()
    return _forensic_agent


def get_store() -> GraphStore:
    if _current_graph_store is None:
        raise HTTPException(status_code=503, detail="GraphStore not initialized")
    return _current_graph_store


@agentic_router.post(
    "/investigate/{chain}/{address}",
    response_model=InvestigationReport,
    summary="Generate full agentic investigation report with plain-language narrative"
)
async def investigate_wallet(
    chain: Chain = Path(..., description="Target blockchain ecosystem"),
    address: str = Path(..., description="Suspect on-chain wallet address"),
    max_hops: int = Query(6, ge=1, le=8, description="Maximum graph traversal depth")
) -> InvestigationReport:
    """
    Executes automated forensic co-investigation for suspect wallet.
    Produces plain-language reasoning, multi-candidate explanations, counterfactuals,
    and court-ready evidence package.
    """
    store = get_store()
    agent = get_agent()

    try:
        report = agent.investigate(
            suspect_wallet=address,
            store=store,
            chain=chain,
            max_hops=max_hops
        )
        return report
    except Exception as e:
        logger.error("Agentic investigation failed for %s: %s", address, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Forensic investigation failed: {str(e)}")
