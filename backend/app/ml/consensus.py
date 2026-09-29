"""
XGBoost + GNN + Behavioral Consensus and Agreement Layer (Phase 6a).
Money Migration Atlas (SIH26182).

DESIGN PHILOSOPHY:
1. XGBoost sees tabular features (proximity, amount patterns, temporal signals).
2. GNN sees graph structure directly (node neighborhoods, message passing, motifs).
3. Behavioral Attributor sees human habitual fingerprints (timing delays, gas quantization, structuring).
They are complementary. None replaces the other.
Final system: All three models produce independent confidence scores.
Where they agree, confidence is CONFIRMED. Where they disagree, the system flags for human review.

CRITICAL INVARIANT:
- proximity_rank, confidence_score (XGBoost), gnn_confidence_score (GNN),
  and behavioral_confidence_score (Behavioral) are strictly independent and NEVER blended.
- consensus_score is max(xgb_conf, gnn_conf, behavioral_conf) used for ordering only.
- The consensus tier is a categorical agreement state (multi-dimensional agreement).
"""

import logging
from typing import List, Dict, Any, Optional

from app.graph.store import GraphStore
from app.core.schemas import NearestVASPCandidate, ConfidenceTier, ConsensusTier
from app.ml.predictor import VASPConfidencePredictor
from app.ml.gnn.predictor import GNNVASPConfidencePredictor
from app.behavioral.attributor import BehavioralAttributor

logger = logging.getLogger("mma.ml.consensus")


class ConsensusScorer:
    """
    Evaluates multi-model consensus across tabular (XGBoost), structural (GNN),
    and behavioral fingerprint attribution signals without blending scores.
    """

    def __init__(
        self,
        xgb_predictor: Optional[VASPConfidencePredictor] = None,
        gnn_predictor: Optional[GNNVASPConfidencePredictor] = None,
        behavioral_attributor: Optional[BehavioralAttributor] = None
    ):
        self.xgb_predictor = xgb_predictor or VASPConfidencePredictor()
        self.gnn_predictor = gnn_predictor or GNNVASPConfidencePredictor()
        self.behavioral_attributor = behavioral_attributor

    @staticmethod
    def determine_consensus_tier(
        xgb_conf: Optional[float],
        gnn_conf: Optional[float],
        behavioral_conf: Optional[float] = None,
        other_candidates_max_conf: float = 0.0
    ) -> ConsensusTier:
        """
        Determines consensus tier across available signals.
        - CONFIRMED requires:
            a) at least 2 active signals >= 0.60 on the SAME candidate
            b) pairwise diff of those high signals < 0.20
            c) candidate is not contradicted by any other signal >= 0.60 on a different candidate (other_candidates_max_conf < 0.60)
            d) no active signal on this candidate has severe disagreement (pairwise diff >= 0.35)
        - UNCERTAIN: all active signals below 0.30
        - BEHAVIORAL_ONLY: only behavioral >= 0.60 (and behavioral is not None)
        - AMBIGUOUS: contradicted by another candidate (>= 0.60) or any pair disagrees by >= 0.35
        - SINGLE_MODEL: exactly one signal >= 0.60
        - UNCERTAIN: default for weak-agreement cases
        """
        active_signals = []
        if xgb_conf is not None:
            active_signals.append(float(xgb_conf))
        if gnn_conf is not None:
            active_signals.append(float(gnn_conf))
        if behavioral_conf is not None:
            active_signals.append(float(behavioral_conf))

        if not active_signals:
            return ConsensusTier.UNCERTAIN

        x = float(xgb_conf) if xgb_conf is not None else 0.0
        g = float(gnn_conf) if gnn_conf is not None else 0.0
        b = float(behavioral_conf) if behavioral_conf is not None else 0.0

        high_signals = [s for s in active_signals if s >= 0.60]

        # Contradiction checks:
        # 1. Contradicted by another candidate scoring >= 0.60
        # 2. Contradicted internally: any pair of active signals has difference >= 0.35
        contradicted_by_other = (other_candidates_max_conf >= 0.60)
        has_internal_disagreement = any(
            abs(s1 - s2) >= 0.35 for i, s1 in enumerate(active_signals) for s2 in active_signals[i+1:]
        )

        # 1. CONFIRMED: at least 2 signals >= 0.60 AND spread < 0.20 AND no contradictions
        if len(high_signals) >= 2:
            if (max(high_signals) - min(high_signals)) < 0.20:
                if not contradicted_by_other and not has_internal_disagreement:
                    return ConsensusTier.CONFIRMED
                else:
                    return ConsensusTier.AMBIGUOUS

        # 2. UNCERTAIN: all below 0.30
        if all(s < 0.30 for s in active_signals):
            return ConsensusTier.UNCERTAIN

        # 3. BEHAVIORAL_ONLY: only behavioral >= 0.60
        if behavioral_conf is not None and b >= 0.60 and x < 0.60 and g < 0.60:
            return ConsensusTier.BEHAVIORAL_ONLY

        # 4. AMBIGUOUS: contradicted by other candidate >= 0.60 or internal disagreement >= 0.35
        if contradicted_by_other or has_internal_disagreement:
            return ConsensusTier.AMBIGUOUS

        # 5. SINGLE_MODEL: exactly one signal >= 0.60
        if len(high_signals) == 1:
            return ConsensusTier.SINGLE_MODEL

        return ConsensusTier.UNCERTAIN

    def predict(
        self,
        suspect_wallet: str,
        store: GraphStore,
        max_hops: int = 6,
        gnn_model_preference: str = "ensemble",
        include_behavioral: Optional[bool] = None
    ) -> List[NearestVASPCandidate]:
        """
        Runs XGBoost, GNN, and Behavioral Attributor on candidate VASPs.
        
        Strictly preserves the 'never_blended' invariant:
        - proximity_rank: topological distance (unblended)
        - confidence_score: XGBoost tabular confidence (visible & unblended)
        - gnn_confidence_score: GNN graph-structural confidence (visible & unblended)
        - behavioral_confidence_score: Behavioral habit confidence (visible & unblended)
        - consensus_score: max(xgb_conf, gnn_conf, behavioral_conf) — used for ranking
        - consensus_tier: categorical agreement state (multi-dimensional)
        """
        # Determine whether behavioral attribution should be computed
        use_behavioral = include_behavioral if include_behavioral is not None else (self.behavioral_attributor is not None)
        # Run XGBoost inference
        xgb_candidates = self.xgb_predictor.predict(
            suspect_wallet=suspect_wallet,
            store=store,
            max_hops=max_hops
        )
        xgb_map: Dict[str, NearestVASPCandidate] = {c.vasp_id: c for c in xgb_candidates}

        # Run GNN inference
        gnn_candidates = self.gnn_predictor.predict(
            suspect_wallet=suspect_wallet,
            store=store,
            max_hops=max_hops,
            model_preference=gnn_model_preference
        )
        gnn_map: Dict[str, NearestVASPCandidate] = {c.vasp_id: c for c in gnn_candidates}

        # Union of candidate VASP IDs
        all_vasp_ids = list(dict.fromkeys(list(xgb_map.keys()) + list(gnn_map.keys())))
        if not all_vasp_ids:
            return []

        # Initialize or retrieve behavioral attributor
        if use_behavioral:
            if self.behavioral_attributor is None and hasattr(store, "graph"):
                self.behavioral_attributor = BehavioralAttributor(store)

        intermediate_results = []

        for vasp_id in all_vasp_ids:
            xc = xgb_map.get(vasp_id)
            gc = gnn_map.get(vasp_id)

            base = xc if xc is not None else gc
            if base is None:
                continue

            x_conf = xc.confidence_score if xc is not None else None
            g_conf = gc.gnn_confidence_score if gc is not None else None

            # Agreement score: 1 - |xgb_conf - gnn_conf|
            if x_conf is not None and g_conf is not None:
                agreement = round(1.0 - abs(x_conf - g_conf), 4)
            else:
                agreement = 0.0

            # Behavioral attribution
            b_conf: Optional[float] = None
            b_wallets: Optional[List[str]] = None
            b_scores: Optional[List[float]] = None

            if use_behavioral and self.behavioral_attributor:
                score, wallets, sim_scores = self.behavioral_attributor.compute_behavioral_score(
                    suspect_wallet=suspect_wallet,
                    candidate_vasp_id=vasp_id,
                    path=base.path
                )
                b_conf = score
                b_wallets = wallets
                b_scores = sim_scores

            signals_to_max = [s for s in [x_conf, g_conf, b_conf] if s is not None]
            c_score = round(max(signals_to_max), 4) if signals_to_max else 0.0

            intermediate_results.append({
                "vasp_id": vasp_id,
                "base": base,
                "xc": xc,
                "gc": gc,
                "x_conf": x_conf,
                "g_conf": g_conf,
                "b_conf": b_conf,
                "b_wallets": b_wallets,
                "b_scores": b_scores,
                "agreement": agreement,
                "c_score": c_score,
            })

        merged_candidates: List[NearestVASPCandidate] = []

        # Pass 2: assign tier considering cross-candidate contradiction signals
        for item in intermediate_results:
            vid = item["vasp_id"]
            other_max = max(
                [other["c_score"] for other in intermediate_results if other["vasp_id"] != vid],
                default=0.0
            )
            cons_tier = self.determine_consensus_tier(
                xgb_conf=item["x_conf"],
                gnn_conf=item["g_conf"],
                behavioral_conf=item["b_conf"],
                other_candidates_max_conf=other_max
            )

            candidate = NearestVASPCandidate(
                vasp_id=item["base"].vasp_id,
                vasp_name=item["base"].vasp_name,
                proximity_rank=item["base"].proximity_rank,
                distance=item["base"].distance,
                target_wallet=item["base"].target_wallet,
                path=item["base"].path,
                tx_hashes=item["base"].tx_hashes,
                chain_path=item["base"].chain_path,
                fiu_ind_registered=item["base"].fiu_ind_registered,
                confidence_score=item["x_conf"],
                confidence_tier=item["xc"].confidence_tier if item["xc"] is not None else ConfidenceTier.UNKNOWN,
                shap_explanation=item["xc"].shap_explanation if item["xc"] is not None else None,
                model_version=item["xc"].model_version if item["xc"] is not None else None,
                gnn_confidence_score=item["g_conf"],
                gnn_confidence_tier=item["gc"].gnn_confidence_tier if item["gc"] is not None else ConfidenceTier.UNKNOWN,
                gnn_model_used=item["gc"].gnn_model_used if item["gc"] is not None else None,
                consensus_tier=cons_tier,
                xgb_gnn_agreement=item["agreement"],
                consensus_score=item["c_score"],
                gnn_subgraph_explanation=item["gc"].gnn_subgraph_explanation if item["gc"] is not None else None,
                behavioral_confidence_score=item["b_conf"],
                behavioral_similar_wallets=item["b_wallets"],
                behavioral_similarity_scores=item["b_scores"],
                never_blended=True
            )
            assert candidate.never_blended is True, "Invariant violated: never_blended must be True"
            merged_candidates.append(candidate)

        # RANKING: Ranked by consensus_score descending, proximity_rank ascending (tiebreaker)
        merged_candidates.sort(
            key=lambda c: (
                c.consensus_score if c.consensus_score is not None else 0.0,
                -c.proximity_rank
            ),
            reverse=True
        )

        return merged_candidates
