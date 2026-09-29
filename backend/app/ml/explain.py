"""
SHAP Explainability Module (Phase 4).
Generates local feature attributions for individual VASP predictions.
Formatted for investigator consumption and Phase 7 LLM reasoning.
"""

import os
import logging
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import joblib
import shap

from app.config import get_settings
from app.core.schemas import ShapFeature
from app.ml.features import FEATURE_NAMES

logger = logging.getLogger("mma.ml.explain")


class SHAPExplainer:
    """
    Computes TreeSHAP local attributions for the XGBoost attribution model.
    """

    def __init__(self, model: Optional[Any] = None, model_path: Optional[str] = None):
        settings = get_settings()
        if model is not None:
            self.model = model
        else:
            path = model_path or settings.ML_MODEL_PATH
            # Resolve relative paths
            if not os.path.isabs(path) and not os.path.exists(path):
                # Try finding in backend/data/models
                repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                cand1 = os.path.join(repo_root, path)
                cand2 = os.path.join(repo_root, "backend", "data", "models", os.path.basename(path))
                if os.path.exists(cand1):
                    path = cand1
                elif os.path.exists(cand2):
                    path = cand2
            
            if not os.path.exists(path):
                raise FileNotFoundError(f"Trained XGBoost model not found at: {path}")
            
            self.model = joblib.load(path)
        
        self.explainer = shap.TreeExplainer(self.model)

    def explain(self, features: Dict[str, float], top_k: int = 5) -> List[ShapFeature]:
        """
        Compute SHAP values for an extracted feature dictionary and return top-k features.
        
        Args:
            features: Dictionary containing all 23 feature values.
            top_k: Number of top contributing features to return (default 5).
            
        Returns:
            List of ShapFeature objects sorted by absolute impact descending.
        """
        # Ensure row aligns with FEATURE_NAMES
        row = [float(features.get(name, 0.0)) for name in FEATURE_NAMES]
        X = pd.DataFrame([row], columns=FEATURE_NAMES)

        raw_shap = self.explainer.shap_values(X)
        
        # Handle SHAP output shapes across library versions
        if isinstance(raw_shap, list):
            # Binary classification list: [class_0_shap, class_1_shap]
            values = raw_shap[1][0] if len(raw_shap) > 1 else raw_shap[0][0]
        elif hasattr(raw_shap, "values"):
            # Explanation object
            values = raw_shap.values[0]
            if values.ndim > 1:
                values = values[:, 1]
        else:
            values = raw_shap[0] if raw_shap.ndim > 1 else raw_shap

        # Pair features with SHAP contributions
        pairs = []
        for name, val, s in zip(FEATURE_NAMES, row, values):
            s_val = float(s)
            pairs.append({
                "feature": name,
                "value": round(val, 4),
                "shap": round(s_val, 4),
                "abs_shap": abs(s_val),
                "direction": "positive" if s_val >= 0 else "negative"
            })

        # Sort by absolute contribution magnitude
        pairs.sort(key=lambda item: item["abs_shap"], reverse=True)

        # Slice top_k and convert to schema
        top_pairs = pairs[:top_k]
        return [
            ShapFeature(
                feature=p["feature"],
                value=p["value"],
                shap=p["shap"],
                direction=p["direction"]
            )
            for p in top_pairs
        ]

    def explain_as_dict(self, features: Dict[str, float], top_k: int = 5) -> Dict[str, Any]:
        """Returns structured JSON-compatible dictionary for Phase 7 LLM consumption."""
        exps = self.explain(features, top_k=top_k)
        return {
            "top_features": [sf.model_dump() for sf in exps]
        }
