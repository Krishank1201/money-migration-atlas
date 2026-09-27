from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends

from app.core.schemas import GroundTruthTestCase, VASP, Attribution
from app.graph.store import GraphStore

router = APIRouter(prefix="/demo", tags=["Demo & Benchmarks"])

# Injected by main app lifespan
_current_graph_store: GraphStore = None
_benchmark_cases: List[GroundTruthTestCase] = []


def set_demo_context(store: GraphStore, test_cases: List[GroundTruthTestCase]):
    global _current_graph_store, _benchmark_cases
    _current_graph_store = store
    _benchmark_cases = test_cases


def get_store() -> GraphStore:
    if _current_graph_store is None:
        raise HTTPException(status_code=503, detail="GraphStore not initialized")
    return _current_graph_store


@router.get("/test-cases", response_model=List[GroundTruthTestCase])
async def list_benchmark_test_cases():
    """Retrieve all 8 benchmark test cases with known ground truth for jury evaluation."""
    return _benchmark_cases


@router.get("/vasps", response_model=List[VASP])
async def list_registered_vasps(store: GraphStore = Depends(get_store)):
    """Retrieve all FIU-IND registered and offshore VASPs in the active graph."""
    return store.get_all_vasps()


@router.get("/benchmark/{case_id}/proximity")
async def evaluate_case_proximity(case_id: str, store: GraphStore = Depends(get_store)) -> Dict[str, Any]:
    """
    Evaluates topological proximity rank from suspect wallet to nearest VASP.
    Demonstrates graph hop isolation without synthetic or blended confidence scoring.
    """
    case = next((c for c in _benchmark_cases if c.case_id == case_id), None)
    if not case:
        raise HTTPException(status_code=404, detail=f"Benchmark case {case_id} not found")

    candidates = store.find_nearest_vasp(case.suspect_wallet, max_hops=6)
    
    # Check if expected VASP is found and at what rank
    top_candidate = candidates[0] if candidates else None
    matches_expected = top_candidate["vasp_id"] == case.expected_vasp if top_candidate else False

    return {
        "case_id": case.case_id,
        "description": case.case_description,
        "suspect_wallet": case.suspect_wallet,
        "chain": case.chain,
        "laundering_pattern": case.laundering_pattern,
        "ground_truth_expected_vasp": case.expected_vasp,
        "ground_truth_expected_distance": case.expected_graph_distance,
        "topological_attribution": {
            "top_match_vasp": top_candidate["vasp_id"] if top_candidate else None,
            "top_match_vasp_name": top_candidate["vasp_name"] if top_candidate else None,
            "proximity_rank_hops": top_candidate["proximity_rank"] if top_candidate else None,
            "proximity_distance": top_candidate["distance"] if top_candidate else None,
            "evidence_path": top_candidate["path"] if top_candidate else [],
            "evidence_tx_hashes": top_candidate["tx_hashes"] if top_candidate else [],
            "ground_truth_validated": matches_expected
        },
        "score_separation_guarantee": {
            "never_blended": True,
            "confidence_score": None,  # Strictly populated by GNN in Phase 5
            "note": "Proximity rank reflects deterministic graph distance. Model confidence is computed by a separate GNN pipeline."
        },
        "all_candidate_vasps": candidates
    }
