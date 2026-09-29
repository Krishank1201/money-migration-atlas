"""
Machine Learning Attribution & Explainability Package (Phase 4).
Money Migration Atlas (SIH26182).
"""

from app.ml.features import extract_features, FEATURE_NAMES
from app.ml.trainer import train_model, build_training_dataset
from app.ml.explain import SHAPExplainer
from app.ml.predictor import VASPConfidencePredictor
from app.ml.evaluate import evaluate_benchmark_cases

__all__ = [
    "extract_features",
    "FEATURE_NAMES",
    "train_model",
    "build_training_dataset",
    "SHAPExplainer",
    "VASPConfidencePredictor",
    "evaluate_benchmark_cases"
]
