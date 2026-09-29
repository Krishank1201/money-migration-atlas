"""
Benchmark evaluation module for VASP attribution (Phase 4).
Money Migration Atlas (SIH26182).

Evaluates the trained XGBoost model against the 12 ground-truth benchmark cases.
Compares ML dual-score attribution against baseline pure topological proximity.
Tracks honest metrics: Model Contribution and Damage Assessment.
"""

import sys
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Add backend directory to sys.path if invoked directly
current_file = Path(__file__).resolve()
backend_dir = current_file.parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.graph.networkx_store import NetworkXStore
from app.graph.store import GraphStore
from app.data.synthetic_generator import generate_synthetic_data
from app.ml.predictor import VASPConfidencePredictor
from app.core.schemas import GroundTruthTestCase

logger = logging.getLogger("mma.ml.evaluate")


def evaluate_benchmark_cases(
    store: Optional[GraphStore] = None,
    predictor: Optional[VASPConfidencePredictor] = None,
    benchmark_cases: Optional[List[GroundTruthTestCase]] = None,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluates attribution performance on the 12 benchmark cases.
    
    Reports:
    - Expected vs Predicted VASP
    - Confidence score & Proximity rank (strictly independent)
    - Resolved by Model vs Proximity (Model if conf >= 0.50, Proximity if conf < 0.25)
    - Model Contribution (cases where model conf >= 0.50 AND model pick != proximity pick)
    - Damage Assessment (cases where model picked WRONG with confidence >= 0.75 - must be 0)
    
    Returns:
        Dict with per-case details, accuracy metrics, and honest model verdict.
    """
    if store is None or benchmark_cases is None:
        store = NetworkXStore()
        vasps, wallets, txs, benchmark_cases = generate_synthetic_data(seed=seed, store=store)

    if predictor is None:
        predictor = VASPConfidencePredictor()

    if not predictor.is_ready():
        raise RuntimeError("Predictor model is not loaded. Train the model first.")

    case_results: List[Dict[str, Any]] = []
    correct_count = 0
    baseline_correct_count = 0
    model_contribution_count = 0
    damage_assessment_count = 0

    y_true: List[str] = []
    y_pred: List[str] = []
    y_baseline: List[str] = []

    for case in benchmark_cases:
        suspect = case.suspect_wallet
        expected = case.expected_vasp

        # 1. Baseline Proximity Rank (topological shortest path)
        raw_candidates = store.find_nearest_vasp(suspect, max_hops=6)
        baseline_top1 = raw_candidates[0]["vasp_id"] if raw_candidates else "NONE"
        baseline_match = (baseline_top1 == expected)
        if baseline_match:
            baseline_correct_count += 1
        y_baseline.append(baseline_top1)

        # 2. ML Dual-Score Prediction (sorted by confidence_score, proximity_rank preserved)
        ml_candidates = predictor.predict(suspect, store, max_hops=6)
        if ml_candidates:
            top_cand = ml_candidates[0]
            pred_vasp = top_cand.vasp_id
            conf_score = top_cand.confidence_score if top_cand.confidence_score is not None else 0.0
            conf_tier = top_cand.confidence_tier.value
            prox_rank = top_cand.proximity_rank
        else:
            pred_vasp = "NONE"
            conf_score = 0.0
            conf_tier = "UNKNOWN"
            prox_rank = 99

        match = (pred_vasp == expected)
        if match:
            correct_count += 1

        # FIX 5: Resolved by Model vs Proximity
        if conf_score >= 0.50:
            resolved_by = "Model"
        elif conf_score < 0.25:
            resolved_by = "Proximity"
        else:
            resolved_by = "Low Conf (Tie)"

        # Model contribution: model confidence >= 0.50 AND model pick != proximity pick
        is_model_contrib = bool(conf_score >= 0.50 and pred_vasp != baseline_top1)
        if is_model_contrib and match:
            model_contribution_count += 1

        # Damage assessment: model picked WRONG with confidence >= 0.75
        is_high_conf_wrong = bool(not match and conf_score >= 0.75)
        if is_high_conf_wrong:
            damage_assessment_count += 1

        y_true.append(expected)
        y_pred.append(pred_vasp)

        case_results.append({
            "case_id": case.case_id,
            "description": case.case_description,
            "chain": case.chain.value if hasattr(case.chain, "value") else str(case.chain),
            "expected_vasp": expected,
            "baseline_vasp": baseline_top1,
            "baseline_correct": baseline_match,
            "predicted_vasp": pred_vasp,
            "confidence_score": conf_score,
            "confidence_tier": conf_tier,
            "proximity_rank": prox_rank,
            "is_correct": match,
            "was_correct": "Y" if match else "N",
            "resolved_by": resolved_by,
            "model_contribution": is_model_contrib,
            "high_conf_wrong": is_high_conf_wrong,
            "never_blended": True
        })

    total_cases = len(benchmark_cases)
    accuracy = correct_count / total_cases if total_cases > 0 else 0.0
    baseline_accuracy = baseline_correct_count / total_cases if total_cases > 0 else 0.0

    # Macro Precision and Recall across unique expected classes
    unique_classes = sorted(list(set(y_true)))
    precisions = []
    recalls = []

    for cls in unique_classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp == cls)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != cls and yp == cls)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp != cls)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)

    macro_precision = float(sum(precisions) / len(precisions)) if precisions else 0.0
    macro_recall = float(sum(recalls) / len(recalls)) if recalls else 0.0
    macro_f1 = (
        float(2 * (macro_precision * macro_recall) / (macro_precision + macro_recall))
        if (macro_precision + macro_recall) > 0
        else 0.0
    )

    # Determine honest verdict
    if damage_assessment_count > 0:
        verdict = "Model is HARMFUL"
    elif model_contribution_count >= 2 and accuracy >= 0.65:
        verdict = "Model is USEFUL"
    else:
        verdict = "Model is NOISE"

    summary = {
        "total_cases": total_cases,
        "correct_predictions": correct_count,
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "baseline_correct": baseline_correct_count,
        "baseline_accuracy": round(baseline_accuracy, 4),
        "model_contribution_count": model_contribution_count,
        "damage_assessment_count": damage_assessment_count,
        "verdict": verdict,
        "target_met": (8 <= correct_count <= 10) or correct_count >= 8,
        "target_range": "8-10 / 12",
        "case_results": case_results
    }

    if verbose:
        print("\n" + "=" * 115)
        print("MONEY MIGRATION ATLAS - 12 BENCHMARK CASE HONEST EVALUATION REPORT")
        print("Phase 4: XGBoost Attribution + Independent Confidence Scoring")
        print("=" * 115)
        print(f"{'Case ID':<10} | {'Expected VASP':<16} | {'Prox Pick':<16} | {'Model Pick':<16} | {'Conf':<7} | {'Prox':<4} | {'Match':<5} | {'Resolved By':<12} | {'Model Contrib'}")
        print("-" * 115)
        for r in case_results:
            contrib_str = "YES" if r["model_contribution"] else "no"
            print(f"{r['case_id']:<10} | {r['expected_vasp']:<16} | {r['baseline_vasp']:<16} | {r['predicted_vasp']:<16} | {r['confidence_score']:<7.4f} | {r['proximity_rank']:<4} | {r['was_correct']:<5} | {r['resolved_by']:<12} | {contrib_str}")
        print("-" * 115)
        print(f"Baseline Topological Accuracy : {baseline_correct_count}/{total_cases} ({baseline_accuracy:.1%})")
        print(f"Phase 4 XGBoost ML Accuracy   : {correct_count}/{total_cases} ({accuracy:.1%})")
        print(f"Model Contribution Count      : {model_contribution_count} cases resolved where proximity failed")
        print(f"Damage Assessment Count       : {damage_assessment_count} cases wrong with conf >= 0.75 (Target: 0)")
        print(f"Macro Precision               : {macro_precision:.4f}")
        print(f"Macro Recall                  : {macro_recall:.4f}")
        print(f"Macro F1 Score                : {macro_f1:.4f}")
        print(f"Verdict                       : {verdict}")
        print("=" * 115 + "\n")

    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluate_benchmark_cases(verbose=True)
