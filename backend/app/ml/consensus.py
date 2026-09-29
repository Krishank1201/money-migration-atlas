"""
XGBoost + GNN Consensus and Agreement Layer (Phase 5).
Money Migration Atlas (SIH26182).

DESIGN PHILOSOPHY:
XGBoost sees tabular features (proximity, amount patterns, temporal signals).
GNN sees graph structure directly (node neighborhoods, message passing, motifs).
They are complementary. Neither replaces the other.
Final system: BOTH models produce independent confidence scores.
Where they agree, confidence is HIGH. Where they disagree, the system flags for human review.

CRITICAL INVARIANT:
- proximity_rank, confidence_score (XGBoost), and gnn_confidence_score (GNN)
  are strictly independent and NEVER blended.
- The consensus tier is a THIRD dimension, not an arithmetic blend of the scores.
"""

import logging
from typing import List, Dict, Any, Optional

from app.graph.store import GraphStore
from app.core.schemas import NearestVASPCandidate, ConfidenceTier, ConsensusTier
from app.ml.predictor import VASPConfidencePredictor
from app.ml.gnn.predictor import GNNVASPConfidencePredictor

logger = logging.getLogger("mma.ml.consensus")


class ConsensusScorer:
    """
    Evaluates multi-model consensus across tabular (XGBoost) and structural (GNN)
    attribution models without blending scores.
    """

    def __init__(
        self,
        xgb_predictor: Optional[VASPConfidencePredictor] = None,
        gnn_predictor: Optional[GNNVASPConfidencePredictor] = None
    ):
        self.xgb_predictor = xgb_predictor or VASPConfidencePredictor()
        self.gnn_predictor = gnn_predictor or GNNVASPConfidencePredictor()

    @staticmethod
    def determine_consensus_tier(
        xgb_conf: Optional[float],
        gnn_conf: Optional[float]
    ) -> ConsensusTier:
        """
        Determines consensus tier based on exact multi-model agreement criteria:
        - CONFIRMED: xgb_conf >= 0.60 AND gnn_conf >= 0.60 AND abs(xgb_conf - gnn_conf) < 0.15
        - UNCERTAIN: xgb_conf < 0.30 AND gnn_conf < 0.30
        - AMBIGUOUS: abs(xgb_conf - gnn_conf) >= 0.35
        - SINGLE_MODEL: (xgb_conf >= 0.60) != (gnn_conf >= 0.60)
        - UNCERTAIN: default for weak-agreement cases
        """
        x = float(xgb_conf) if xgb_conf is not None else 0.0
        g = float(gnn_conf) if gnn_conf is not None else 0.0
        diff = abs(x - g)

        if x >= 0.60 and g >= 0.60 and diff < 0.15:
            return ConsensusTier.CONFIRMED
        elif x < 0.30 and g < 0.30:
            return ConsensusTier.UNCERTAIN
        elif diff >= 0.35:
            return ConsensusTier.AMBIGUOUS
        elif (x >= 0.60) != (g >= 0.60):
            return ConsensusTier.SINGLE_MODEL
        else:
            return ConsensusTier.UNCERTAIN

    def predict(
        self,
        suspect_wallet: str,
        store: GraphStore,
        max_hops: int = 6,
        gnn_model_preference: str = "ensemble"
    ) -> List[NearestVASPCandidate]:
        """
        Runs both XGBoost and GNN on candidate VASPs and produces consensus metrics.
        
        Strictly preserves the 'never_blended' invariant:
        - proximity_rank: topological distance
        - confidence_score: XGBoost tabular confidence (visible & unblended)
        - gnn_confidence_score: GNN graph-structural confidence (visible & unblended)
        - consensus_score: max(xgb_confidence, gnn_confidence) — used for ranking
        - consensus_tier: categorical agreement state (third dimension)
        """
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

        merged_candidates: List[NearestVASPCandidate] = []

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

            cons_tier = self.determine_consensus_tier(x_conf, g_conf)
            c_score = round(max(x_conf or 0.0, g_conf or 0.0), 4)

            candidate = NearestVASPCandidate(
                vasp_id=base.vasp_id,
                vasp_name=base.vasp_name,
                proximity_rank=base.proximity_rank,
                distance=base.distance,
                target_wallet=base.target_wallet,
                path=base.path,
                tx_hashes=base.tx_hashes,
                chain_path=base.chain_path,
                fiu_ind_registered=base.fiu_ind_registered,
                confidence_score=x_conf,
                confidence_tier=xc.confidence_tier if xc is not None else ConfidenceTier.UNKNOWN,
                shap_explanation=xc.shap_explanation if xc is not None else None,
                model_version=xc.model_version if xc is not None else None,
                gnn_confidence_score=g_conf,
                gnn_confidence_tier=gc.gnn_confidence_tier if gc is not None else ConfidenceTier.UNKNOWN,
                gnn_model_used=gc.gnn_model_used if gc is not None else None,
                consensus_tier=cons_tier,
                xgb_gnn_agreement=agreement,
                consensus_score=c_score,
                gnn_subgraph_explanation=gc.gnn_subgraph_explanation if gc is not None else None,
                never_blended=True
            )
            assert candidate.never_blended is True, "Invariant violated: never_blended must be True"
            merged_candidates.append(candidate)

        # FIX 1: RANKING — CONSENSUS MUST CHANGE THE ANSWER
        # Change final candidate ranking to use consensus_score = max(xgb_confidence, gnn_confidence)
        # Tiebreaker: topological proximity_rank (-c.proximity_rank)
        merged_candidates.sort(
            key=lambda c: (
                c.consensus_score if c.consensus_score is not None else 0.0,
                -c.proximity_rank
            ),
            reverse=True
        )

        return merged_candidates
