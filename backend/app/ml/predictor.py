"""
Prediction Orchestrator for Independent Dual-Score VASP Attribution (Phase 4).
Money Migration Atlas (SIH26182).

CRITICAL INVARIANT:
- Proximity Rank (Topological hops) and Confidence Score (ML probability)
  MUST REMAIN INDEPENDENT AND NEVER BE BLENDED.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import joblib
import pandas as pd

from app.config import get_settings
from app.graph.store import GraphStore
from app.core.schemas import NearestVASPCandidate, ConfidenceTier, ShapFeature
from app.ml.features import extract_features, FEATURE_NAMES
from app.ml.explain import SHAPExplainer

logger = logging.getLogger("mma.ml.predictor")


def _resolve_artifact_path(relative_or_abs: str) -> str:
    """Helper to locate files whether executing from repo root or backend subdir."""
    if os.path.isabs(relative_or_abs) and os.path.exists(relative_or_abs):
        return relative_or_abs
    if os.path.exists(relative_or_abs):
        return os.path.abspath(relative_or_abs)
    
    # Try relative to repo root
    current = Path(__file__).resolve()
    # current: backend/app/ml/predictor.py -> repo_root is 4 parents up
    repo_root = current.parent.parent.parent.parent
    c1 = repo_root / relative_or_abs
    if c1.exists():
        return str(c1)
    
    # Try backend/data/models/<filename>
    c2 = repo_root / "backend" / "data" / "models" / Path(relative_or_abs).name
    if c2.exists():
        return str(c2)

    return relative_or_abs


class VASPConfidencePredictor:
    """
    Orchestrates candidate retrieval from Phase 3 proximity search,
    feature extraction, XGBoost inference for confidence scores,
    and SHAP local explainability.
    """

    def __init__(self, model_path: Optional[str] = None):
        settings = get_settings()
        raw_path = model_path or settings.ML_MODEL_PATH
        resolved_path = _resolve_artifact_path(raw_path)

        if not os.path.exists(resolved_path):
            logger.warning("XGBoost model file not found at %s. Predictor will initialize lazily or require training.", resolved_path)
            self.model = None
            self.explainer = None
            self.model_version = "xgboost_v1_uninitialized"
            self.metadata: Dict[str, Any] = {}
            return

        self.model_path = resolved_path
        self.model = joblib.load(resolved_path)
        self.explainer = SHAPExplainer(model=self.model)

        # Load metadata if present
        meta_path = os.path.join(os.path.dirname(resolved_path), "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            self.model_version = self.metadata.get("model_version", "xgboost_v1")
        else:
            self.metadata = {}
            self.model_version = "xgboost_v1"

        logger.info("Loaded VASPConfidencePredictor with model version: %s", self.model_version)

    def is_ready(self) -> bool:
        """Returns True if the XGBoost model is loaded and ready for inference."""
        return self.model is not None and self.explainer is not None

    def _determine_tier(self, score: float) -> ConfidenceTier:
        """Assigns categorical confidence tier based on configured thresholds."""
        settings = get_settings()
        if score > settings.ML_CONFIDENCE_HIGH_THRESHOLD:
            return ConfidenceTier.HIGH
        elif score >= settings.ML_CONFIDENCE_MEDIUM_THRESHOLD:
            return ConfidenceTier.MEDIUM
        elif score >= settings.ML_CONFIDENCE_LOW_THRESHOLD:
            return ConfidenceTier.LOW
        else:
            return ConfidenceTier.UNKNOWN

    def predict(
        self,
        suspect_wallet: str,
        store: GraphStore,
        max_hops: int = 6
    ) -> List[NearestVASPCandidate]:
        """
        Calculates independent dual scores for candidate VASPs reachable from suspect_wallet:
        1. Proximity Rank (topological hops)
        2. Confidence Score (XGBoost probability [0.0 - 1.0])
        
        Strictly preserves the 'never_blended' invariant.
        """
        raw_candidates = store.find_nearest_vasp(suspect_wallet, max_hops=max_hops)
        if not raw_candidates:
            return []

        results: List[NearestVASPCandidate] = []

        for c in raw_candidates:
            cand_vasp_id = c["vasp_id"]
            path = c.get("path", [])
            tx_hashes = c.get("tx_hashes", [])
            prox_rank = c["proximity_rank"]

            # 1. Extract feature vector
            feat = extract_features(
                suspect_wallet=suspect_wallet,
                candidate_vasp_id=cand_vasp_id,
                store=store,
                candidate_path=path,
                candidate_tx_hashes=tx_hashes,
                max_hops=max_hops
            )

            # 2. Run inference if model is ready
            if self.is_ready():
                X = pd.DataFrame([[feat[fn] for fn in FEATURE_NAMES]], columns=FEATURE_NAMES)
                probs = self.model.predict_proba(X)[0]
                conf_score = float(probs[1]) if len(probs) > 1 else float(probs[0])
                conf_score = round(conf_score, 4)
                tier = self._determine_tier(conf_score)
                shap_explanation = self.explainer.explain(feat, top_k=5)
            else:
                conf_score = None
                tier = ConfidenceTier.UNKNOWN
                shap_explanation = None

            # 3. Create candidate with strictly separated scores
            candidate = NearestVASPCandidate(
                vasp_id=c["vasp_id"],
                vasp_name=c["vasp_name"],
                proximity_rank=prox_rank,           # INDEPENDENT SCORE #1
                distance=c.get("distance", 0.0),
                target_wallet=c.get("target_wallet"),
                path=path,
                tx_hashes=tx_hashes,
                chain_path=c.get("chain_path", []),
                fiu_ind_registered=c.get("fiu_ind_registered", False),
                confidence_score=conf_score,         # INDEPENDENT SCORE #2
                confidence_tier=tier,
                shap_explanation=shap_explanation,
                model_version=self.model_version,
                never_blended=True                  # ENFORCED INVARIANT
            )

            # Runtime assertion of never_blended invariant
            assert candidate.never_blended is True, "CRITICAL ERROR: never_blended invariant was modified!"
            results.append(candidate)

        # FIX 6 Sanity Guard 1: Leaky feature detection in high-confidence predictions
        for cand in results:
            if cand.confidence_score is not None and cand.confidence_score >= 0.75 and cand.shap_explanation:
                for sf in cand.shap_explanation:
                    if sf.feature == "vasp_historical_volume":
                        raise ValueError("Leaky feature detected: vasp_historical_volume in SHAP top contributors")

        # FIX 6 Sanity Guard 2: Multi-VASP high-confidence invariant
        high_conf_cands = [c for c in results if c.confidence_score is not None and c.confidence_score >= 0.75]
        if len(high_conf_cands) > 1:
            raise ValueError(
                f"Confidence invariant violated: {len(high_conf_cands)} VASPs assigned HIGH confidence (>= 0.75) "
                f"for suspect {suspect_wallet}: {[c.vasp_id for c in high_conf_cands]}"
            )

        # Sort primarily by confidence score descending, with proximity rank as tiebreaker
        # (Both scores remain strictly independent fields in the object)
        results.sort(
            key=lambda item: (
                item.confidence_score if item.confidence_score is not None else -1.0,
                -item.proximity_rank
            ),
            reverse=True
        )

        return results
