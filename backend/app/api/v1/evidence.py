"""
Evidence Chain & Court Package API Endpoints (Phase 7).
Money Migration Atlas (SIH26182).

Endpoints:
- GET /api/v1/evidence/{chain}/{address}?format=json|markdown
  Exports complete forensic evidence package in JSON or court-admissible markdown brief.
- GET /api/v1/evidence/{chain}/{address}/chain-of-custody
  Returns immutable digital chain-of-custody audit sequence.
"""

import logging
from typing import List, Optional, Union
from fastapi import APIRouter, HTTPException, Query, Path, Response
from fastapi.responses import PlainTextResponse

from app.core.schemas import Chain, EvidencePackage, CustodyStep
from app.graph.store import GraphStore
from app.ml.consensus import ConsensusScorer
from app.evidence.chain import EvidenceChainBuilder

logger = logging.getLogger("mma.api.evidence")

evidence_router = APIRouter(prefix="/evidence", tags=["Forensic Evidence Chain"])

_current_graph_store: Optional[GraphStore] = None
_consensus_scorer: Optional[ConsensusScorer] = None
_evidence_builder: Optional[EvidenceChainBuilder] = None


def set_evidence_context(store: GraphStore):
    global _current_graph_store, _consensus_scorer, _evidence_builder
    _current_graph_store = store
    _consensus_scorer = ConsensusScorer()
    _evidence_builder = EvidenceChainBuilder()
    logger.info("Evidence API context initialized.")


def get_store() -> GraphStore:
    if _current_graph_store is None:
        raise HTTPException(status_code=503, detail="GraphStore not initialized")
    return _current_graph_store


def get_components():
    global _consensus_scorer, _evidence_builder
    if _consensus_scorer is None:
        _consensus_scorer = ConsensusScorer()
    if _evidence_builder is None:
        _evidence_builder = EvidenceChainBuilder()
    return _consensus_scorer, _evidence_builder


@evidence_router.get(
    "/{chain}/{address}",
    summary="Export court-admissible forensic evidence package (JSON or Markdown)",
    response_model=Optional[EvidencePackage]
)
async def get_evidence_package(
    chain: Chain = Path(..., description="Target blockchain ecosystem"),
    address: str = Path(..., description="Suspect on-chain wallet address"),
    format: str = Query("json", pattern="^(json|markdown)$", description="Export format: 'json' or 'markdown'"),
    max_hops: int = Query(6, ge=1, le=8, description="Maximum graph traversal depth")
):
    """
    Assembles and exports a court-ready evidence dossier formatted for Section 91 CrPC disclosure (not a certification).
    Returns JSON object or Markdown legal brief.
    """
    store = get_store()
    scorer, builder = get_components()

    candidates = scorer.predict(
        suspect_wallet=address,
        store=store,
        max_hops=max_hops,
        include_behavioral=True
    )
    if not candidates:
        raise HTTPException(status_code=404, detail=f"No candidate VASP reachable from suspect wallet {address}")

    top_candidate = candidates[0]
    package = builder.build(
        candidate=top_candidate,
        store=store,
        suspect_wallet=address,
        chain=chain.value
    )

    if format == "markdown":
        md_text = builder.to_markdown(package)
        return PlainTextResponse(content=md_text, media_type="text/markdown")

    return package


@evidence_router.get(
    "/{chain}/{address}/chain-of-custody",
    response_model=List[CustodyStep],
    summary="Retrieve immutable digital chain of custody audit sequence"
)
async def get_chain_of_custody(
    chain: Chain = Path(..., description="Target blockchain ecosystem"),
    address: str = Path(..., description="Suspect on-chain wallet address"),
    max_hops: int = Query(6, ge=1, le=8, description="Maximum graph traversal depth")
) -> List[CustodyStep]:
    """
    Returns sequential audit trail documenting evidence ingestion, traversal,
    probabilistic inference, and consensus verification steps.
    """
    store = get_store()
    scorer, builder = get_components()

    candidates = scorer.predict(
        suspect_wallet=address,
        store=store,
        max_hops=max_hops,
        include_behavioral=True
    )
    if not candidates:
        raise HTTPException(status_code=404, detail=f"No candidate VASP reachable from suspect wallet {address}")

    package = builder.build(
        candidate=candidates[0],
        store=store,
        suspect_wallet=address,
        chain=chain.value
    )
    return package.chain_of_custody
