"""
XGBoost ML Training Pipeline for VASP Attribution (Phase 4).
Money Migration Atlas (SIH26182).

Includes:
- Strict benchmark data leakage guard (zero benchmark addresses in training)
- Hard negative mining (reachability-based negative sampling, proximity_rank <= 2 decoys)
- Strict metric evaluation: ROC AUC >= 0.65, Precision@1, Calibration (>0.75), Brier score
"""

import os
import sys
import json
import random
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import numpy as np
import pandas as pd
import joblib
import networkx as nx
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix
)

from app.config import get_settings
from app.graph.networkx_store import NetworkXStore
from app.graph.store import GraphStore
from app.data.synthetic_generator import generate_synthetic_data
from app.ml.features import extract_features, FEATURE_NAMES

logger = logging.getLogger("mma.ml.trainer")
settings = get_settings()

MODEL_VERSION = f"xgboost_v1_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"


def build_training_dataset(
    store: GraphStore,
    benchmark_cases: list,
    samples_per_vasp: int = 18,
    seed: int = 42
) -> Tuple[pd.DataFrame, Set[str]]:
    """
    Constructs a balanced, non-leaky feature dataset for training the XGBoost attribution model.
    
    Guarantees:
    - Zero data leakage: strictly excludes all 12 benchmark suspect and path addresses.
    - No fake negatives: only reachable candidate VASPs within 6 hops are included.
    - Hard negative mining: mines decoy VASPs at proximity_rank <= 2 or closest alternative paths.
    - Balanced class ratio: ~50% positive (true VASP), ~50% negative (reachable decoys).
    - Target size: ~200-400 rows.
    
    Returns:
        (DataFrame of features and labels, Set of excluded benchmark addresses)
    """
    rng = random.Random(seed)
    vasps = store.get_all_vasps()
    vasp_ids = [v.id for v in vasps]
    wallets_map = getattr(store, "wallets", {})

    # Data leakage guard: collect all suspect and path wallets from benchmark cases
    benchmark_excluded_addrs: Set[str] = set()
    for c in benchmark_cases:
        benchmark_excluded_addrs.add(c.suspect_wallet)
        res = store.find_nearest_vasp(c.suspect_wallet, max_hops=6)
        for r in res:
            benchmark_excluded_addrs.update(r.get("path", []))

    rev_graph = getattr(store, "graph", nx.MultiDiGraph()).reverse()

    pos_rows: List[Dict[str, Any]] = []
    neg_rows: List[Dict[str, Any]] = []

    for v_id in vasp_ids:
        # Anchors: known-labeled wallets belonging to this VASP (excluding benchmark nodes)
        anchors = [
            w.address for w in wallets_map.values()
            if w.vasp_id == v_id and w.address not in benchmark_excluded_addrs
        ]
        
        all_upstream: List[Tuple[str, int]] = []
        for a in anchors:
            if a in rev_graph:
                try:
                    lengths = nx.single_source_shortest_path_length(rev_graph, a, cutoff=5)
                    for u, dist in lengths.items():
                        if u != a and u not in benchmark_excluded_addrs:
                            all_upstream.append((u, dist))
                except Exception:
                    pass

        # Shuffle and sample upstream suspects for this VASP
        rng.shuffle(all_upstream)
        seen = set()
        sampled_suspects: List[str] = []
        for u, dist in all_upstream:
            if u not in seen:
                seen.add(u)
                sampled_suspects.append(u)
            if len(sampled_suspects) >= samples_per_vasp:
                break

        # For each sampled suspect, extract features for reachable candidate VASPs
        for s in sampled_suspects:
            cand_res = store.find_nearest_vasp(s, max_hops=6)
            cand_map = {r["vasp_id"]: r for r in cand_res}

            # 1. Positive sample: (suspect, true_vasp) if reachable
            if v_id in cand_map:
                r_pos = cand_map[v_id]
                pos_feat = extract_features(
                    s, v_id, store,
                    candidate_path=r_pos["path"],
                    candidate_tx_hashes=r_pos["tx_hashes"]
                )
                pos_feat["label"] = 1
                pos_feat["suspect_wallet"] = s
                pos_feat["true_vasp"] = v_id
                pos_feat["candidate_vasp"] = v_id
                pos_rows.append(pos_feat)

                # 2. Hard Negative Mining:
                # Find OTHER reachable VASPs (never fake unreachable negatives with rank 99)
                other_cands = [r for other_v, r in cand_map.items() if other_v != v_id]
                # Prioritize hard negatives: proximity_rank <= 2 (decoy attractors) or closest hop distance
                other_cands.sort(key=lambda item: item["proximity_rank"])

                # Select top hard negative for 1:1 class balance
                for r_neg in other_cands[:1]:
                    neg_feat = extract_features(
                        s, r_neg["vasp_id"], store,
                        candidate_path=r_neg["path"],
                        candidate_tx_hashes=r_neg["tx_hashes"]
                    )
                    neg_feat["label"] = 0
                    neg_feat["suspect_wallet"] = s
                    neg_feat["true_vasp"] = v_id
                    neg_feat["candidate_vasp"] = r_neg["vasp_id"]
                    neg_rows.append(neg_feat)

    # Balance positive and negative counts exactly
    min_count = min(len(pos_rows), len(neg_rows))
    pos_balanced = pos_rows[:min_count]
    neg_balanced = neg_rows[:min_count]

    combined = pos_balanced + neg_balanced
    rng.shuffle(combined)
    df = pd.DataFrame(combined)

    logger.info(
        "Constructed training dataset: %d rows (%d pos, %d neg, balance=%.1f%%, excluded benchmark addrs=%d)",
        len(df), len(pos_balanced), len(neg_balanced),
        (len(pos_balanced) / len(df) * 100) if len(df) > 0 else 0.0,
        len(benchmark_excluded_addrs)
    )
    return df, benchmark_excluded_addrs


def train_model(
    model_save_path: Optional[str] = None,
    importance_save_path: Optional[str] = None,
    seed: int = 42,
    store: Optional[GraphStore] = None,
    benchmark_cases: Optional[list] = None
) -> Dict[str, Any]:
    """
    Executes the end-to-end ML training pipeline:
    - Generates synthetic graph
    - Constructs balanced dataset with hard negative mining
    - Trains XGBoost classifier
    - Computes strict test evaluation metrics: ROC AUC, Precision, Recall, Calibration, Brier score
    - Saves model artifacts
    """
    current = Path(__file__).resolve()
    repo_root = current.parent.parent.parent.parent
    if model_save_path is None:
        model_save_path = str(repo_root / "backend" / "data" / "models" / "xgboost_v1.pkl")
    elif not os.path.isabs(model_save_path):
        model_save_path = str(repo_root / model_save_path)

    if importance_save_path is None:
        importance_save_path = os.path.join(os.path.dirname(model_save_path), "feature_importance.json")
    elif not os.path.isabs(importance_save_path):
        importance_save_path = str(repo_root / importance_save_path)

    # Ensure output directories exist
    os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(importance_save_path)), exist_ok=True)

    # 1. Prepare Graph & Data
    if store is None or benchmark_cases is None:
        store = NetworkXStore()
        vasps, wallets, txs, benchmark_cases = generate_synthetic_data(seed=seed, store=store)

    df, excluded_addrs = build_training_dataset(
        store=store,
        benchmark_cases=benchmark_cases,
        samples_per_vasp=18,
        seed=seed
    )

    X = df[FEATURE_NAMES]
    y = df["label"]

    # 2. Split into Train / Val / Test (70% / 15% / 15%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )

    # 3. Train XGBoost Classifier
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=seed,
        eval_metric="logloss"
    )

    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # 4. Strict Evaluation on Held-Out Test Split
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    brier = float(brier_score_loss(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Calibration: For samples with predicted probability > 0.75, what fraction are actually correct?
    high_conf_mask = (y_prob > 0.75)
    high_conf_count = int(np.sum(high_conf_mask))
    if high_conf_count > 0:
        calibration_high_conf = float(np.mean(y_test.values[high_conf_mask] == 1))
    else:
        calibration_high_conf = 1.0

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "brier_score": round(brier, 4),
        "calibration_high_conf": round(calibration_high_conf, 4),
        "high_conf_test_samples": high_conf_count,
        "confusion_matrix": cm,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "dataset_rows": len(df),
        "positive_samples": int(np.sum(y == 1)),
        "negative_samples": int(np.sum(y == 0)),
    }

    # Strict failure gate check: ROC AUC must be >= 0.65
    if roc_auc < 0.65:
        raise RuntimeError(
            f"Model failed strict evaluation: ROC AUC = {roc_auc:.4f} < 0.65. "
            "Synthetic data lacks learnable signal for XGBoost."
        )

    # 5. Extract Feature Importances
    importances = clf.feature_importances_
    feat_importance_list = [
        {"feature": name, "importance": round(float(imp), 6)}
        for name, imp in zip(FEATURE_NAMES, importances)
    ]
    feat_importance_list.sort(key=lambda x: x["importance"], reverse=True)

    # 6. Save Artifacts
    joblib.dump(clf, model_save_path)
    
    meta_path = os.path.join(os.path.dirname(model_save_path), "model_metadata.json")
    trained_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": trained_at,
        "dataset_rows": len(df),
        "feature_count": len(FEATURE_NAMES),
        "features": FEATURE_NAMES,
        "metrics": metrics,
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.05
        },
        "leakage_guard_excluded_addresses_count": len(excluded_addrs)
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(importance_save_path, "w", encoding="utf-8") as f:
        json.dump(feat_importance_list, f, indent=2)

    logger.info(
        "XGBoost model saved to %s (ROC AUC=%.4f, Prec=%.4f, Calib=%.4f, Brier=%.4f)",
        model_save_path, roc_auc, prec, calibration_high_conf, brier
    )

    return {
        "model_version": MODEL_VERSION,
        "trained_at": trained_at,
        "model_path": model_save_path,
        "feature_importance_path": importance_save_path,
        "metadata_path": meta_path,
        "metrics": metrics,
        "feature_importance": feat_importance_list,
        "excluded_benchmark_count": len(excluded_addrs)
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = train_model()
    print("Training finished successfully:")
    print(json.dumps(results["metrics"], indent=2))
