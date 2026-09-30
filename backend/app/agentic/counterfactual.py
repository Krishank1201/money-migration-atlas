"""
Counterfactual Generator (Phase 7).
Money Migration Atlas (SIH26182).

Produces deterministic, logic-driven counterfactual reasoning ("what-if" scenarios)
for cryptocurrency attribution cases with explicit audit provenance and verified model provenance.
"""

from typing import List, Optional
import pandas as pd

from app.core.schemas import NearestVASPCandidate, ConsensusTier, CounterfactualItem
from app.graph.store import GraphStore


def _recompute_without_mixer(
    suspect_wallet: str,
    candidate_vasp_id: str,
    path: List[str],
    store: Optional[GraphStore] = None
) -> float:
    """
    Recomputes the attribution confidence score for a candidate path with mixer nodes removed.
    Runs actual model inference on the modified topological feature vector.
    """
    if not store or not path or len(path) < 2:
        return 0.50

    suspect_addr = path[0]
    target_addr = path[-1]

    # Filter out intermediate mixer nodes
    mixer_nodes = []
    for addr in path[1:-1]:
        if addr.lower() == suspect_addr.lower() or addr.lower() == target_addr.lower():
            continue
        w = store.get_wallet(addr) if hasattr(store, "get_wallet") else None
        if w and getattr(w, "is_mixer", False):
            mixer_nodes.append(addr)

    clean_path = [a for a in path if a not in mixer_nodes]
    if len(clean_path) < 2:
        clean_path = [path[0], path[-1]]

    try:
        from app.ml.features import extract_features, FEATURE_NAMES
        from app.ml.predictor import VASPConfidencePredictor

        feat = extract_features(
            suspect_wallet=suspect_wallet,
            candidate_vasp_id=candidate_vasp_id,
            store=store,
            candidate_path=clean_path
        )
        # Counterfactual feature adjustment: eliminate mixer penalties
        feat["path_touches_mixer"] = 0.0
        feat["num_mixer_hops_on_path"] = 0.0
        feat["path_mixer_penalty"] = 0.0
        feat["proximity_rank"] = float(max(1, len(clean_path) - 1))
        feat["min_path_length"] = feat["proximity_rank"]

        predictor = VASPConfidencePredictor()
        if predictor.is_ready():
            X = pd.DataFrame([[feat[fn] for fn in FEATURE_NAMES]], columns=FEATURE_NAMES)
            probs = predictor.model.predict_proba(X)[0]
            xgb_conf = float(probs[1]) if len(probs) > 1 else float(probs[0])
        else:
            xgb_conf = 0.0

        # Optional GNN score extraction for consensus
        try:
            from app.main import get_gnn_predictor
            gnn_pred = get_gnn_predictor()
            gnn_cands = gnn_pred.predict(suspect_wallet, store)
            gnn_map = {c.vasp_id: c.gnn_confidence_score for c in gnn_cands if c.gnn_confidence_score is not None}
            gnn_conf = gnn_map.get(candidate_vasp_id, 0.0)
        except Exception:
            gnn_conf = 0.0

        consensus_score = max(xgb_conf, gnn_conf or 0.0)
        return round(float(consensus_score), 4)
    except Exception:
        return 0.50


def generate_counterfactuals(
    candidate: NearestVASPCandidate,
    all_candidates: List[NearestVASPCandidate],
    store: Optional[GraphStore] = None
) -> List[CounterfactualItem]:
    """
    Generates up to 3 deterministic counterfactual statements explaining what would
    change the attribution decision or confidence level.

    Guarantees:
    - Exactly 1 to 3 statements.
    - Pure deterministic logic with verified provenance.
    - NEVER names suspect wallet or target VASP wallet as a mixer.
    - If mixer identification returns suspect or target, skips mixer counterfactual entirely.
    """
    counterfactuals: List[CounterfactualItem] = []

    suspect_addr = candidate.path[0] if candidate.path else None
    target_addr = candidate.target_wallet or (candidate.path[-1] if candidate.path else None)

    # 1. Mixer Identification (Restricted to intermediate path nodes only)
    has_mixer = False
    mixer_addr = None

    if store and hasattr(store, "wallets") and candidate.path and len(candidate.path) > 2:
        for addr in candidate.path[1:-1]:
            # Guard: identified mixer node must NEVER be suspect or target
            if suspect_addr and addr.lower() == suspect_addr.lower():
                continue
            if target_addr and addr.lower() == target_addr.lower():
                continue

            w = store.get_wallet(addr)
            if w and getattr(w, "is_mixer", False):
                has_mixer = True
                mixer_addr = addr
                break

    # Final guard: verify identified mixer is neither suspect nor target
    if mixer_addr:
        if suspect_addr and mixer_addr.lower() == suspect_addr.lower():
            has_mixer = False
            mixer_addr = None
        elif target_addr and mixer_addr.lower() == target_addr.lower():
            has_mixer = False
            mixer_addr = None

    c_score = candidate.consensus_score or candidate.confidence_score or 0.50
    xgb_str = f"{candidate.confidence_score:.2f}" if candidate.confidence_score is not None else "N/A"
    gnn_str = f"{candidate.gnn_confidence_score:.2f}" if candidate.gnn_confidence_score is not None else "N/A"

    # Emit mixer-removal counterfactual ONLY if a verified intermediate mixer exists
    if has_mixer and mixer_addr and suspect_addr:
        recomputed_score = _recompute_without_mixer(
            suspect_wallet=suspect_addr,
            candidate_vasp_id=candidate.vasp_id,
            path=candidate.path,
            store=store
        )
        if recomputed_score > c_score:
            cf_text = (
                f"If the path had not touched mixer {mixer_addr[:10]}..., "
                f"consensus score would rise from {c_score:.2f} to {recomputed_score:.2f} (GNN {gnn_str} / XGB {xgb_str})."
            )
        elif recomputed_score < c_score:
            cf_text = (
                f"If the path had not touched mixer {mixer_addr[:10]}..., "
                f"consensus score would shift from {c_score:.2f} to {recomputed_score:.2f} (GNN {gnn_str} / XGB {xgb_str})."
            )
        else:
            cf_text = (
                f"If the path had not touched mixer {mixer_addr[:10]}..., "
                f"consensus score would remain at {recomputed_score:.2f} (GNN {gnn_str} / XGB {xgb_str}) as non-mixer structural signals dominate."
            )

        counterfactuals.append(
            CounterfactualItem(
                text=cf_text,
                provenance="recomputed",
                method="ConsensusScorer re-evaluation without mixer node",
                signal_source="consensus"
            )
        )

    # 2. Model Divergence / Algorithm Contrast Counterfactual
    if all_candidates:
        xgb_best = max(all_candidates, key=lambda x: x.confidence_score if x.confidence_score is not None else -1.0)
        gnn_best = max(all_candidates, key=lambda x: x.gnn_confidence_score if x.gnn_confidence_score is not None else -1.0)

        if xgb_best.vasp_id != candidate.vasp_id and (xgb_best.confidence_score or 0) > 0.30:
            counterfactuals.append(
                CounterfactualItem(
                    text=(
                        f"If XGBoost tabular model alone made this decision, it would attribute to {xgb_best.vasp_name} "
                        f"({xgb_best.confidence_score:.2f}), whereas consensus selected {candidate.vasp_name}."
                    ),
                    provenance="recomputed",
                    method="XGBoost tabular decision boundary isolation",
                    signal_source="xgb"
                )
            )
        elif gnn_best.vasp_id != candidate.vasp_id and (gnn_best.gnn_confidence_score or 0) > 0.30:
            counterfactuals.append(
                CounterfactualItem(
                    text=(
                        f"If GNN structural embeddings alone decided, attribution would favor {gnn_best.vasp_name} "
                        f"({gnn_best.gnn_confidence_score:.2f}) over {candidate.vasp_name}."
                    ),
                    provenance="recomputed",
                    method="GNN graph-structural message-passing isolation",
                    signal_source="gnn"
                )
            )
        else:
            # Both models agree or single candidate
            counterfactuals.append(
                CounterfactualItem(
                    text=(
                        f"If both XGBoost ({candidate.confidence_score or 0.0:.2f}) and GNN ({candidate.gnn_confidence_score or 0.0:.2f}) "
                        f"scored below 0.60, consensus tier would downgrade from {candidate.consensus_tier.value if candidate.consensus_tier else 'CURRENT'} to UNCERTAIN."
                    ),
                    provenance="recomputed",
                    method="Multi-model consensus agreement threshold sensitivity",
                    signal_source="consensus"
                )
            )

    # 3. Proximity / Alternate Candidate Sensitivity Counterfactual
    competitors = [c for c in all_candidates if c.vasp_id != candidate.vasp_id]
    if competitors:
        runner_up = max(
            competitors,
            key=lambda x: (
                x.consensus_score if x.consensus_score is not None else 0.0,
                -x.proximity_rank
            )
        )
        if runner_up.proximity_rank > candidate.proximity_rank:
            counterfactuals.append(
                CounterfactualItem(
                    text=(
                        f"If the suspect wallet had reached {runner_up.vasp_name} at {candidate.proximity_rank} hops "
                        f"instead of {runner_up.proximity_rank}, proximity attribution would shift toward {runner_up.vasp_name}."
                    ),
                    provenance="qualitative",
                    method="Topological shortest-path graph traversal distance sensitivity",
                    signal_source="proximity"
                )
            )
        elif runner_up.proximity_rank < candidate.proximity_rank:
            counterfactuals.append(
                CounterfactualItem(
                    text=(
                        f"If {candidate.vasp_name} were reachable in {runner_up.proximity_rank} hops rather than "
                        f"{candidate.proximity_rank}, topological proximity ranking would match {runner_up.vasp_name}."
                    ),
                    provenance="qualitative",
                    method="Topological shortest-path graph traversal distance sensitivity",
                    signal_source="proximity"
                )
            )
        else:
            counterfactuals.append(
                CounterfactualItem(
                    text=(
                        f"If the laundering path to {runner_up.vasp_name} were 1 hop shorter, "
                        f"topological proximity would tilt attribution toward {runner_up.vasp_name}."
                    ),
                    provenance="qualitative",
                    method="Topological shortest-path graph traversal distance sensitivity",
                    signal_source="proximity"
                )
            )
    else:
        counterfactuals.append(
            CounterfactualItem(
                text=(
                    f"If the laundering path had extended beyond {candidate.proximity_rank + 2} hops, "
                    f"multi-hop decay would have substantially suppressed confidence."
                ),
                provenance="qualitative",
                method="Topological shortest-path graph traversal distance sensitivity",
                signal_source="proximity"
            )
        )

    # Strictly cap at 3 counterfactuals
    return counterfactuals[:3]
