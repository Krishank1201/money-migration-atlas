"""
ML attribution and explainability endpoints (Phase 4).
Money Migration Atlas (SIH26182).

Enforces strictly independent scores:
1. Proximity Rank (topological graph distance)
2. Confidence Score (XGBoost classification probability + SHAP)
NEVER BLENDED.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Path

from app.core.schemas import (
    Chain,
    MLPredictResponse,
    MLModelInfoResponse,
    NearestVASPCandidate,
    ConfidenceTier,
    GroundTruthTestCase
)
from app.graph.store import GraphStore
from app.ml.predictor import VASPConfidencePredictor, _resolve_artifact_path
from app.ml.trainer import train_model

logger = logging.getLogger("mma.api.ml")

router = APIRouter(prefix="/ml", tags=["Machine Learning & Attribution"])

# Injected by main app lifespan
_current_graph_store: Optional[GraphStore] = None
_benchmark_cases: List[GroundTruthTestCase] = []
_predictor: Optional[VASPConfidencePredictor] = None


def set_ml_context(store: GraphStore, test_cases: List[GroundTruthTestCase]):
    """Inject active graph store and benchmark cases from application lifespan."""
    global _current_graph_store, _benchmark_cases, _predictor
    _current_graph_store = store
    _benchmark_cases = test_cases
    _predictor = VASPConfidencePredictor()
    logger.info("ML API context initialized with predictor: ready=%s", _predictor.is_ready())


def get_store() -> GraphStore:
    if _current_graph_store is None:
        raise HTTPException(status_code=503, detail="GraphStore not initialized")
    return _current_graph_store


def get_predictor() -> VASPConfidencePredictor:
    global _predictor
    if _predictor is None:
        _predictor = VASPConfidencePredictor()
    return _predictor


@router.post("/train", response_model=Dict[str, Any])
async def retrain_model(
    seed: int = Query(default=42, description="Random seed for reproducibility"),
    store: GraphStore = Depends(get_store)
) -> Dict[str, Any]:
    """
    Retrains the XGBoost attribution model on the synthetic graph.
    Data leakage guard: strictly excludes all 12 benchmark cases.
    """
    global _predictor
    logger.info("Initiating model retraining with seed=%d...", seed)
    results = train_model(
        seed=seed,
        store=store,
        benchmark_cases=_benchmark_cases
    )
    # Reload predictor with fresh weights
    _predictor = VASPConfidencePredictor()
    return results


@router.get("/model-info", response_model=MLModelInfoResponse)
async def get_model_info(
    predictor: VASPConfidencePredictor = Depends(get_predictor)
) -> MLModelInfoResponse:
    """
    Retrieves metadata, training timestamp, evaluation metrics,
    hyperparameters, and global feature importance from the loaded XGBoost model.
    """
    if not predictor.is_ready():
        return MLModelInfoResponse(
            model_version="uninitialized",
            trained_at=None,
            metrics={"status": "model_not_trained"},
            feature_importance=[],
            hyperparameters={},
            training_samples=0
        )

    meta = getattr(predictor, "metadata", {})
    metrics = meta.get("metrics", {})
    feat_imp = []

    # Try reading feature importance from disk or metadata
    importance_path = None
    if hasattr(predictor, "model_path") and predictor.model_path:
        cand = os.path.join(os.path.dirname(predictor.model_path), "feature_importance.json")
        if os.path.exists(cand):
            importance_path = cand
    if not importance_path:
        cand = _resolve_artifact_path("backend/data/models/feature_importance.json")
        if os.path.exists(cand):
            importance_path = cand

    if importance_path and os.path.exists(importance_path):
        try:
            with open(importance_path, "r", encoding="utf-8") as f:
                feat_imp = json.load(f)
        except Exception as e:
            logger.warning("Could not read feature_importance.json: %s", e)
    elif "feature_importance" in meta:
        feat_imp = meta["feature_importance"]

    return MLModelInfoResponse(
        model_version=predictor.model_version,
        trained_at=meta.get("trained_at"),
        metrics=metrics,
        feature_importance=feat_imp,
        hyperparameters=meta.get("hyperparameters", {}),
        training_samples=metrics.get("train_samples", 0)
    )


@router.post("/predict/{chain}/{address}", response_model=MLPredictResponse)
async def predict_attribution(
    chain: Chain = Path(..., description="Blockchain network"),
    address: str = Path(..., description="Suspect wallet address"),
    max_hops: int = Query(default=6, ge=1, le=8, description="Maximum graph search radius"),
    store: GraphStore = Depends(get_store),
    predictor: VASPConfidencePredictor = Depends(get_predictor)
) -> MLPredictResponse:
    """
    Full ML attribution endpoint returning dual independent scores:
    - Score 1: Proximity Rank (topological hops)
    - Score 2: Confidence Score (XGBoost probability [0.0 - 1.0])
    - SHAP local explanation for the prediction
    
    CRITICAL INVARIANT: The two scores are strictly unblended (never_blended=True).
    """
    candidates = predictor.predict(
        suspect_wallet=address,
        store=store,
        max_hops=max_hops
    )

    return MLPredictResponse(
        suspect_wallet=address,
        chain=chain,
        candidates=candidates,
        never_blended=True
    )
