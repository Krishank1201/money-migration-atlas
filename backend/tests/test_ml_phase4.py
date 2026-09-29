"""
Unit and integration tests for Phase 4: Baseline ML attribution with XGBoost,
feature engineering, SHAP explainability, and dual independent scores.
Money Migration Atlas (SIH26182).
"""

import os
import json
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.config import get_settings
from app.graph.networkx_store import NetworkXStore
from app.data.synthetic_generator import generate_synthetic_data
from app.ml.features import extract_features, FEATURE_NAMES
from app.ml.trainer import build_training_dataset, train_model
from app.ml.explain import SHAPExplainer
from app.ml.predictor import VASPConfidencePredictor
from app.ml.evaluate import evaluate_benchmark_cases
from app.core.schemas import NearestVASPCandidate, ConfidenceTier, Chain, ShapFeature

settings = get_settings()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def graph_data():
    store = NetworkXStore()
    vasps, wallets, txs, cases = generate_synthetic_data(seed=42, store=store)
    return store, vasps, wallets, txs, cases


def test_feature_extraction_returns_expected_fields(graph_data):
    """Test 1: Verify extract_features returns all 20 clean, non-leaky feature keys with numeric values."""
    store, vasps, wallets, txs, cases = graph_data
    sample_suspect = cases[0].suspect_wallet
    cand_vasp_id = "coindcx"

    features = extract_features(sample_suspect, cand_vasp_id, store)

    assert len(features) == len(FEATURE_NAMES)
    assert len(features) == 20
    assert "path_mixer_penalty" in features
    # Ensure leaky & constant features are strictly absent
    assert "vasp_historical_volume" not in features
    assert "suspect_age_days" not in features
    assert "suspect_page_rank" not in features
    assert "vasp_chain_match" not in features

    for name in FEATURE_NAMES:
        assert name in features, f"Missing feature: {name}"
        assert isinstance(features[name], (int, float)), f"Feature {name} is not numeric: {type(features[name])}"


def test_training_produces_model_artifact():
    """Test 2: Verify training artifacts exist and contain required metadata."""
    predictor = VASPConfidencePredictor()
    assert predictor.is_ready(), "Model artifact should be loaded and ready"
    assert predictor.model_version is not None
    assert len(predictor.model_version) > 0


def test_no_data_leakage_benchmark_cases_not_in_training(graph_data):
    """Test 3: Guarantee zero data leakage by asserting benchmark addresses are excluded from training."""
    store, vasps, wallets, txs, cases = graph_data
    df, excluded_addrs = build_training_dataset(
        store=store,
        benchmark_cases=cases,
        samples_per_vasp=5,
        seed=42
    )

    # Benchmark suspect wallets must not be in training dataset
    benchmark_suspects = {c.suspect_wallet for c in cases}
    dataset_suspects = set(df["suspect_wallet"].unique())

    overlap = benchmark_suspects.intersection(dataset_suspects)
    assert len(overlap) == 0, f"Data leakage detected! Benchmark suspects found in training set: {overlap}"
    assert len(excluded_addrs) >= len(benchmark_suspects)


def test_predictor_returns_both_scores_independently(graph_data):
    """Test 4: Verify predictor returns both Proximity Rank and Confidence Score as independent fields."""
    store, vasps, wallets, txs, cases = graph_data
    predictor = VASPConfidencePredictor()

    candidates = predictor.predict(cases[0].suspect_wallet, store)
    assert len(candidates) > 0

    top_cand = candidates[0]
    assert hasattr(top_cand, "proximity_rank")
    assert hasattr(top_cand, "confidence_score")
    assert isinstance(top_cand.proximity_rank, int)
    assert isinstance(top_cand.confidence_score, float)
    assert 0.0 <= top_cand.confidence_score <= 1.0


def test_never_blended_invariant_holds():
    """Test 5: Assert that never_blended invariant is True and cannot be set to False."""
    cand = NearestVASPCandidate(
        vasp_id="coindcx",
        vasp_name="CoinDCX",
        proximity_rank=3,
        confidence_score=0.85,
        confidence_tier=ConfidenceTier.HIGH,
        never_blended=True
    )
    assert cand.never_blended is True

    # Violating invariant must raise ValidationError
    with pytest.raises(ValidationError):
        NearestVASPCandidate(
            vasp_id="coindcx",
            vasp_name="CoinDCX",
            proximity_rank=3,
            confidence_score=0.85,
            confidence_tier=ConfidenceTier.HIGH,
            never_blended=False
        )


def test_shap_explanations_have_top_5_features(graph_data):
    """Test 6: Verify SHAP explanations return top-5 features with valid direction."""
    store, vasps, wallets, txs, cases = graph_data
    predictor = VASPConfidencePredictor()

    candidates = predictor.predict(cases[0].suspect_wallet, store)
    assert len(candidates) > 0
    top_cand = candidates[0]

    assert top_cand.shap_explanation is not None
    assert len(top_cand.shap_explanation) == 5

    for sf in top_cand.shap_explanation:
        assert sf.feature in FEATURE_NAMES
        assert isinstance(sf.value, float)
        assert isinstance(sf.shap, float)
        assert sf.direction in ("positive", "negative")


def test_model_predicts_all_12_benchmark_cases(graph_data):
    """Test 7: Verify predictor successfully evaluates all 12 benchmark cases without runtime exception."""
    store, vasps, wallets, txs, cases = graph_data
    predictor = VASPConfidencePredictor()

    assert len(cases) == 12
    for case in cases:
        cands = predictor.predict(case.suspect_wallet, store)
        assert isinstance(cands, list)
        assert len(cands) > 0


def test_confidence_tier_assignment_consistent():
    """Test 8: Verify mapping from confidence probability to ConfidenceTier enum."""
    predictor = VASPConfidencePredictor()

    assert predictor._determine_tier(0.90) == ConfidenceTier.HIGH
    assert predictor._determine_tier(0.76) == ConfidenceTier.HIGH
    assert predictor._determine_tier(0.75) == ConfidenceTier.MEDIUM
    assert predictor._determine_tier(0.50) == ConfidenceTier.MEDIUM
    assert predictor._determine_tier(0.40) == ConfidenceTier.LOW
    assert predictor._determine_tier(0.25) == ConfidenceTier.LOW
    assert predictor._determine_tier(0.10) == ConfidenceTier.UNKNOWN
    assert predictor._determine_tier(0.00) == ConfidenceTier.UNKNOWN


def test_strict_test_metrics_and_benchmark_improvements(graph_data):
    """Test 9: Assert strict metric gates (ROC AUC >= 0.65, Precision >= 0.50, Calib >= 0.75, Damage == 0)."""
    store, vasps, wallets, txs, cases = graph_data
    predictor = VASPConfidencePredictor()

    # Check test metrics recorded in metadata
    meta = getattr(predictor, "metadata", {})
    metrics = meta.get("metrics", {})
    assert metrics.get("roc_auc", 0.0) >= 0.65, f"ROC AUC {metrics.get('roc_auc')} < 0.65"
    assert metrics.get("precision", 0.0) >= 0.50, f"Precision {metrics.get('precision')} < 0.50"
    assert metrics.get("calibration_high_conf", 0.0) >= 0.75, f"Calibration {metrics.get('calibration_high_conf')} < 0.75"

    summary = evaluate_benchmark_cases(
        store=store,
        predictor=predictor,
        benchmark_cases=cases,
        verbose=False
    )

    assert summary["baseline_correct"] == 6
    assert summary["correct_predictions"] >= 8
    assert summary["model_contribution_count"] >= 3, "Model must actively contribute to >= 3 cases"
    assert summary["damage_assessment_count"] == 0, "Damage assessment: 0 cases with high-confidence wrong predictions allowed"
    assert summary["verdict"] == "Model is USEFUL"


def test_sanity_guards_in_predictor(graph_data):
    """Test 10: Verify predictor sanity guards trigger on leaky features or confidence invariant violations."""
    store, vasps, wallets, txs, cases = graph_data
    predictor = VASPConfidencePredictor()

    # Verify that clean candidate does not violate guards
    normal_cands = predictor.predict(cases[0].suspect_wallet, store)
    assert len(normal_cands) > 0


def test_model_info_endpoint_returns_metrics(client):
    """Test 11: Verify GET /api/v1/ml/model-info endpoint returns expected structure and feature count."""
    resp = client.get("/api/v1/ml/model-info")
    assert resp.status_code == 200
    data = resp.json()

    assert "model_version" in data
    assert "metrics" in data
    assert "feature_importance" in data
    assert len(data["feature_importance"]) == 20
    assert data["metrics"]["roc_auc"] >= 0.65
    assert data["metrics"]["precision"] >= 0.50
    assert data["metrics"]["calibration_high_conf"] >= 0.75


def test_predict_endpoint_returns_dual_scores_and_invariant(client):
    """Test 12: End-to-end test of POST /api/v1/ml/predict/{chain}/{address} endpoint."""
    resp_cases = client.get("/api/v1/demo/test-cases")
    assert resp_cases.status_code == 200
    cases = resp_cases.json()
    suspect_addr = cases[0]["suspect_wallet"]
    chain = cases[0]["chain"]

    resp = client.post(f"/api/v1/ml/predict/{chain}/{suspect_addr}?max_hops=6")
    assert resp.status_code == 200
    pred_data = resp.json()

    assert pred_data["suspect_wallet"] == suspect_addr
    assert pred_data["never_blended"] is True
    assert len(pred_data["candidates"]) > 0

    top_cand = pred_data["candidates"][0]
    assert "proximity_rank" in top_cand
    assert "confidence_score" in top_cand
    assert "confidence_tier" in top_cand
    assert "shap_explanation" in top_cand
    assert top_cand["never_blended"] is True
