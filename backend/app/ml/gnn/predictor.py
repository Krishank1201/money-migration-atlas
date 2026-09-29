"""
GNN Prediction Orchestrator (Phase 5).
Money Migration Atlas (SIH26182).

Orchestrates GraphSAGE and GATv2 neural inference for structural VASP attribution.
Produces independent GNN confidence scores without blending with XGBoost or Proximity.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import torch
import torch.nn.functional as F

from app.config import get_settings
from app.graph.store import GraphStore
from app.core.schemas import NearestVASPCandidate, ConfidenceTier
from app.ml.gnn.converter import VASP_TO_IDX, IDX_TO_VASP
from app.ml.gnn.graphsage import VASPGraphSAGE
from app.ml.gnn.gat import VASPGATv2
from app.ml.gnn.trainer import extract_candidate_subgraph
from app.ml.gnn.explain import GNNExplainerModule

logger = logging.getLogger("mma.ml.gnn.predictor")


def _resolve_model_path(filename: str) -> str:
    """Helper to locate GNN model files regardless of working directory."""
    current = Path(__file__).resolve()
    backend_root = current.parent.parent.parent.parent
    if backend_root.name == "backend":
        cand = backend_root / "data" / "models" / filename
    else:
        cand = backend_root / "backend" / "data" / "models" / filename

    if cand.exists():
        return str(cand)
    # Direct fallback
    return str(Path("backend/data/models") / filename)


class GNNVASPConfidencePredictor:
    """
    Evaluates graph-structural VASP attribution using GraphSAGE and GATv2.
    Produces independent GNN confidence scores and subgraph edge explanations.
    """

    def __init__(
        self,
        sage_path: Optional[str] = None,
        gat_path: Optional[str] = None
    ):
        settings = get_settings()
        self.sage_path = sage_path or _resolve_model_path("graphsage_v1.pt")
        self.gat_path = gat_path or _resolve_model_path("gatv2_v1.pt")
        self.meta_path = _resolve_model_path("gnn_model_metadata.json")

        self.sage_model: Optional[VASPGraphSAGE] = None
        self.gat_model: Optional[VASPGATv2] = None
        self.explainer: Optional[GNNExplainerModule] = None
        self.model_version = "gnn_v1_uninitialized"
        self.metadata: Dict[str, Any] = {}

        self._load_models()

    def _load_models(self):
        try:
            if os.path.exists(self.sage_path):
                self.sage_model = VASPGraphSAGE(in_channels=8, hidden_channels=64, out_channels=32, num_classes=9)
                self.sage_model.load_state_dict(torch.load(self.sage_path, map_location="cpu", weights_only=True))
                self.sage_model.eval()

            if os.path.exists(self.gat_path):
                self.gat_model = VASPGATv2(in_channels=8, hidden_channels=16, heads=4, out_channels=32, num_classes=9)
                self.gat_model.load_state_dict(torch.load(self.gat_path, map_location="cpu", weights_only=True))
                self.gat_model.eval()

            if self.gat_model is not None:
                self.explainer = GNNExplainerModule(model=self.gat_model)
            elif self.sage_model is not None:
                self.explainer = GNNExplainerModule(model=self.sage_model)

            if os.path.exists(self.meta_path):
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                self.model_version = self.metadata.get("model_version", "gnn_v1")

            logger.info("Loaded GNNVASPConfidencePredictor: SAGE=%s, GAT=%s",
                        self.sage_model is not None, self.gat_model is not None)
        except Exception as e:
            logger.warning("Error loading GNN model weights: %s", e)

    def is_ready(self) -> bool:
        """Returns True if at least one GNN model is loaded and ready for inference."""
        return self.sage_model is not None or self.gat_model is not None

    def _determine_tier(self, score: float) -> ConfidenceTier:
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
        max_hops: int = 6,
        model_preference: str = "ensemble"
    ) -> List[NearestVASPCandidate]:
        """
        Calculates GNN confidence score for candidate VASPs reachable from suspect_wallet.
        
        Strictly preserves the 'never_blended' invariant:
        - proximity_rank: topological distance
        - gnn_confidence_score: structural neural probability
        """
        raw_candidates = store.find_nearest_vasp(suspect_wallet, max_hops=max_hops)
        if not raw_candidates:
            return []

        if not self.is_ready():
            logger.warning("GNN models not loaded. Returning unpopulated GNN confidence.")
            results = []
            for c in raw_candidates:
                results.append(NearestVASPCandidate(
                    vasp_id=c["vasp_id"],
                    vasp_name=c["vasp_name"],
                    proximity_rank=c["proximity_rank"],
                    distance=c.get("distance", 0.0),
                    target_wallet=c.get("target_wallet"),
                    path=c.get("path", []),
                    tx_hashes=c.get("tx_hashes", []),
                    chain_path=c.get("chain_path", []),
                    fiu_ind_registered=c.get("fiu_ind_registered", False),
                    gnn_confidence_score=None,
                    gnn_confidence_tier=ConfidenceTier.UNKNOWN,
                    gnn_model_used=model_preference,
                    never_blended=True
                ))
            return results

        # 1. Extract subgraphs and raw logits for each candidate VASP
        cand_probs_sage: Dict[str, float] = {}
        cand_probs_gat: Dict[str, float] = {}
        cand_subgraphs: Dict[str, Any] = {}

        for c in raw_candidates:
            cand_v = c["vasp_id"]
            if cand_v not in VASP_TO_IDX:
                continue

            v_idx = VASP_TO_IDX[cand_v]
            d = extract_candidate_subgraph(
                suspect_wallet=suspect_wallet,
                candidate_vasp_id=cand_v,
                store=store,
                path=c.get("path")
            )
            cand_subgraphs[cand_v] = d

            with torch.no_grad():
                if self.sage_model is not None:
                    out_s = self.sage_model(d.x, d.edge_index)
                    p_s = float(F.softmax(out_s[d.suspect_idx], dim=0)[v_idx])
                    cand_probs_sage[cand_v] = p_s

                if self.gat_model is not None:
                    out_g = self.gat_model(d.x, d.edge_index)
                    p_g = float(F.softmax(out_g[d.suspect_idx], dim=0)[v_idx])
                    cand_probs_gat[cand_v] = p_g

        # 2. Normalize probabilities across reachable candidate set.
        # Include non-candidate background prior (p_null = 1/9) when candidate set has <= 1 unique VASP,
        # preventing artificial 1.00 saturation when only a single candidate is reachable.
        p_null = 1.0 / 9.0
        unique_cand_count = len(cand_probs_sage) if cand_probs_sage else len(raw_candidates)
        null_offset = p_null if unique_cand_count <= 1 else 0.0

        sum_sage = (sum(cand_probs_sage.values()) + null_offset) if cand_probs_sage else 1.0
        sum_gat = (sum(cand_probs_gat.values()) + null_offset) if cand_probs_gat else 1.0

        results: List[NearestVASPCandidate] = []

        for c in raw_candidates:
            cand_v = c["vasp_id"]
            raw_s = cand_probs_sage.get(cand_v, 0.0)
            raw_g = cand_probs_gat.get(cand_v, 0.0)

            calib_s = (raw_s / sum_sage) if sum_sage > 0 else 0.0
            calib_g = (raw_g / sum_gat) if sum_gat > 0 else 0.0

            if model_preference == "graphsage":
                conf = calib_s
                model_used = "graphsage"
            elif model_preference == "gatv2":
                conf = calib_g
                model_used = "gatv2"
            else:  # ensemble
                conf = max(calib_s, calib_g)
                model_used = "ensemble"

            conf = round(float(conf), 4)
            tier = self._determine_tier(conf)

            # Extract subgraph explanation if available
            d = cand_subgraphs.get(cand_v)
            if d is not None and self.explainer is not None:
                subgraph_exp = self.explainer.explain(d, target_idx=VASP_TO_IDX.get(cand_v), top_k=5)
            else:
                subgraph_exp = None

            cand = NearestVASPCandidate(
                vasp_id=c["vasp_id"],
                vasp_name=c["vasp_name"],
                proximity_rank=c["proximity_rank"],      # INDEPENDENT SCORE #1
                distance=c.get("distance", 0.0),
                target_wallet=c.get("target_wallet"),
                path=c.get("path", []),
                tx_hashes=c.get("tx_hashes", []),
                chain_path=c.get("chain_path", []),
                fiu_ind_registered=c.get("fiu_ind_registered", False),
                confidence_score=None,                   # Reserved for XGBoost
                gnn_confidence_score=conf,               # INDEPENDENT SCORE #3
                gnn_confidence_tier=tier,
                gnn_model_used=model_used,
                gnn_subgraph_explanation=subgraph_exp,
                never_blended=True                       # ENFORCED INVARIANT
            )
            assert cand.never_blended is True, "Invariant violated: never_blended must be True"
            results.append(cand)

        # Sort primarily by GNN confidence score descending, with proximity rank as tiebreaker
        results.sort(
            key=lambda item: (
                item.gnn_confidence_score if item.gnn_confidence_score is not None else -1.0,
                -item.proximity_rank
            ),
            reverse=True
        )

        return results
