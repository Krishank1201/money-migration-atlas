"""
Graph Neural Network (GraphSAGE & GATv2) and Multi-Model Consensus Endpoints (Phase 5).
Money Migration Atlas (SIH26182).

Endpoints:
- POST /api/v1/gnn/train → Retrains GraphSAGE and GATv2 on candidate subgraphs
- GET /api/v1/gnn/model-info → Returns GNN model versions, parameters, and loss curves
- POST /api/v1/gnn/predict/{chain}/{address} → Independent GNN structural attribution
- POST /api/v1/ml/predict-consensus/{chain}/{address} → Dual-model consensus (XGBoost + GNN)
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Path

from app.core.schemas import (
    Chain,
    MLPredictResponse,
    NearestVASPCandidate,
    ConfidenceTier,
    ConsensusTier,
    GroundTruthTestCase
)
from app.graph.store import GraphStore
from app.ml.gnn.predictor import GNNVASPConfidencePredictor, _resolve_model_path
from app.ml.gnn.trainer import train_gnn_models
from app.ml.consensus import ConsensusScorer

logger = logging.getLogger("mma.api.gnn")

gnn_router = APIRouter(prefix="/gnn", tags=["Graph Neural Networks (GNN)"])
consensus_router = APIRouter(prefix="/ml", tags=["Multi-Model Consensus"])

# Context injected by main lifespan
_current_graph_store: Optional[GraphStore] = None
_benchmark_cases: List[GroundTruthTestCase] = []
_gnn_predictor: Optional[GNNVASPConfidencePredictor] = None
_consensus_scorer: Optional[ConsensusScorer] = None


def set_gnn_context(store: GraphStore, test_cases: List[GroundTruthTestCase]):
    """Inject active graph store and benchmark cases from application lifespan."""
    global _current_graph_store, _benchmark_cases, _gnn_predictor, _consensus_scorer
    _current_graph_store = store
    _benchmark_cases = test_cases
    _gnn_predictor = GNNVASPConfidencePredictor()
    _consensus_scorer = ConsensusScorer(gnn_predictor=_gnn_predictor)
    logger.info("GNN API context initialized: ready=%s", _gnn_predictor.is_ready())


def get_store() -> GraphStore:
    if _current_graph_store is None:
        raise HTTPException(status_code=503, detail="GraphStore not initialized")
    return _current_graph_store


def get_gnn_predictor() -> GNNVASPConfidencePredictor:
    global _gnn_predictor
    if _gnn_predictor is None:
        _gnn_predictor = GNNVASPConfidencePredictor()
    return _gnn_predictor


def get_consensus_scorer() -> ConsensusScorer:
    global _consensus_scorer
    if _consensus_scorer is None:
        _consensus_scorer = ConsensusScorer(gnn_predictor=get_gnn_predictor())
    return _consensus_scorer


@gnn_router.post("/train", response_model=Dict[str, Any])
async def retrain_gnn(
    seed: int = Query(default=42, description="Random seed for reproducibility"),
    epochs: int = Query(default=100, ge=10, le=200, description="Max training epochs"),
    store: GraphStore = Depends(get_store)
) -> Dict[str, Any]:
    """
    Retrains both GraphSAGE and GATv2 models on structural subgraphs extracted from the graph.
    Data leakage guard: strictly excludes all 12 benchmark cases.
    """
    global _gnn_predictor, _consensus_scorer
    logger.info("Retraining GNN models with seed=%d, epochs=%d...", seed, epochs)
    results = train_gnn_models(
        seed=seed,
        store=store,
        benchmark_cases=_benchmark_cases,
        max_epochs=epochs
    )
    # Reload predictors with new weights
    _gnn_predictor = GNNVASPConfidencePredictor()
    _consensus_scorer = ConsensusScorer(gnn_predictor=_gnn_predictor)
    return results


@gnn_router.get("/model-info", response_model=Dict[str, Any])
async def get_gnn_model_info(
    predictor: GNNVASPConfidencePredictor = Depends(get_gnn_predictor)
) -> Dict[str, Any]:
    """
    Retrieves metadata, training timestamp, architecture details,
    evaluation metrics, and training loss curves for GraphSAGE and GATv2.
    """
    curves_path = _resolve_model_path("gnn_training_curves.json")
    curves_data = {}
    if os.path.exists(curves_path):
        try:
            with open(curves_path, "r", encoding="utf-8") as f:
                curves_data = json.load(f)
        except Exception as e:
            logger.warning("Could not read gnn_training_curves.json: %s", e)

    meta = getattr(predictor, "metadata", {})

    return {
        "model_version": predictor.model_version,
        "is_ready": predictor.is_ready(),
        "architectures": {
            "graphsage": {
                "layers": 3,
                "hidden_channels": 64,
                "out_channels": 32,
                "dropout": 0.3,
                "skip_connections": True
            },
            "gatv2": {
                "layers": 2,
                "heads": 4,
                "hidden_channels": 64,
                "out_channels": 32,
                "dropout": 0.3
            }
        },
        "metadata": meta,
        "training_curves": curves_data
    }


@gnn_router.post("/predict/{chain}/{address}", response_model=MLPredictResponse)
async def predict_gnn_attribution(
    chain: Chain = Path(..., description="Blockchain network"),
    address: str = Path(..., description="Suspect wallet address"),
    max_hops: int = Query(default=6, ge=1, le=8, description="Maximum graph search radius"),
    model_preference: str = Query(default="ensemble", pattern="^(graphsage|gatv2|ensemble)$"),
    store: GraphStore = Depends(get_store),
    predictor: GNNVASPConfidencePredictor = Depends(get_gnn_predictor)
) -> MLPredictResponse:
    """
    Structural GNN attribution endpoint returning independent graph-neural scores:
    - Score 1: Proximity Rank (topological hops)
    - Score 3: GNN Confidence Score (GraphSAGE / GATv2 probability)
    - Subgraph edge attention/explanation
    
    CRITICAL INVARIANT: The scores are strictly unblended (never_blended=True).
    """
    candidates = predictor.predict(
        suspect_wallet=address,
        store=store,
        max_hops=max_hops,
        model_preference=model_preference
    )

    return MLPredictResponse(
        suspect_wallet=address,
        chain=chain,
        candidates=candidates,
        never_blended=True
    )


@consensus_router.post("/predict-consensus/{chain}/{address}", response_model=MLPredictResponse)
async def predict_consensus_attribution(
    chain: Chain = Path(..., description="Blockchain network"),
    address: str = Path(..., description="Suspect wallet address"),
    max_hops: int = Query(default=6, ge=1, le=8, description="Maximum graph search radius"),
    model_preference: str = Query(default="ensemble", pattern="^(graphsage|gatv2|ensemble)$"),
    store: GraphStore = Depends(get_store),
    consensus_scorer: ConsensusScorer = Depends(get_consensus_scorer)
) -> MLPredictResponse:
    """
    Dual-model consensus attribution endpoint combining XGBoost tabular signals
    and GNN graph-structural signals into an agreement layer.

    Produces:
    - Proximity Rank (topological hops)
    - XGBoost Confidence Score (tabular patterns) + SHAP explanation
    - GNN Confidence Score (structural neighborhoods) + GNN edge explanation
    - Agreement Score (1 - |xgb - gnn|)
    - Consensus Tier (CONFIRMED / AMBIGUOUS / UNCERTAIN / SINGLE_MODEL)

    CRITICAL INVARIANT: Scores are NEVER blended (never_blended=True).
    The consensus tier is a third dimension, not an arithmetic blend.
    """
    candidates = consensus_scorer.predict(
        suspect_wallet=address,
        store=store,
        max_hops=max_hops,
        gnn_model_preference=model_preference
    )

    return MLPredictResponse(
        suspect_wallet=address,
        chain=chain,
        candidates=candidates,
        never_blended=True
    )
