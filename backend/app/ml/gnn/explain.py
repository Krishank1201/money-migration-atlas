"""
GNN Explainability Module (Phase 5).
Money Migration Atlas (SIH26182).

Computes edge attribution masks using GNNExplainer and GATv2 attention weights.
Identifies the critical structural edges and intermediary hops driving VASP attribution.
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer

from app.ml.gnn.graphsage import VASPGraphSAGE
from app.ml.gnn.gat import VASPGATv2

logger = logging.getLogger("mma.ml.gnn.explain")


class GNNExplainerModule:
    """
    Computes structural sub-graph explanations for GNN predictions.
    Identifies top-k edges contributing most to the attributed VASP classification.
    """

    def __init__(self, model: Optional[Any] = None):
        self.model = model
        self.explainer = None
        if model is not None:
            self._init_explainer()

    def _init_explainer(self):
        try:
            self.explainer = Explainer(
                model=self.model,
                algorithm=GNNExplainer(epochs=15),
                explanation_type="model",
                edge_mask_type="object",
                model_config=dict(
                    mode="multiclass_classification",
                    task_level="node",
                    return_type="raw"
                )
            )
        except Exception as e:
            logger.warning("Could not initialize PyG Explainer: %s", e)
            self.explainer = None

    def explain(
        self,
        data: Data,
        target_idx: Optional[int] = None,
        top_k: int = 5,
        node_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extracts top-k edge contributions for the target suspect node prediction.
        
        Returns:
            List of {"edge": [src, dst], "weight": float} sorted by weight descending.
        """
        if data is None or data.edge_index is None or data.edge_index.shape[1] == 0:
            return []

        suspect_idx = getattr(data, "suspect_idx", 0)
        num_edges = data.edge_index.shape[1]

        edge_weights = None

        # 1. Try PyG GNNExplainer if available
        if self.explainer is not None:
            try:
                exp = self.explainer(
                    data.x,
                    data.edge_index,
                    index=suspect_idx
                )
                if hasattr(exp, "edge_mask") and exp.edge_mask is not None:
                    edge_weights = exp.edge_mask.detach().cpu().numpy()
            except Exception as e:
                logger.debug("GNNExplainer execution failed: %s, falling back to attention/topology", e)

        # 2. Fallback to GATv2 attention weights if model is GATv2
        if edge_weights is None and isinstance(self.model, VASPGATv2):
            try:
                self.model.eval()
                with torch.no_grad():
                    _, (e_idx, alpha) = self.model(
                        data.x, data.edge_index, return_attention_weights=True
                    )
                    edge_weights = alpha.squeeze().detach().cpu().numpy()
            except Exception as e:
                logger.debug("GATv2 attention retrieval failed: %s", e)

        # 3. Default fallback: uniform edge attribution normalized by degree
        if edge_weights is None or len(edge_weights) == 0:
            edge_weights = np.ones(num_edges, dtype=np.float32) / max(1, num_edges)

        # Ensure edge_weights is 1D
        if edge_weights.ndim > 1:
            edge_weights = edge_weights.mean(axis=-1)

        # Extract top-k edges
        edge_indices = np.argsort(-edge_weights)[:top_k]
        top_edges: List[Dict[str, Any]] = []

        edge_src = data.edge_index[0].cpu().numpy()
        edge_dst = data.edge_index[1].cpu().numpy()

        for idx in edge_indices:
            if idx < len(edge_src) and idx < len(edge_dst):
                u_idx = int(edge_src[idx])
                v_idx = int(edge_dst[idx])
                w = float(round(float(edge_weights[idx]), 4))

                u_name = node_names[u_idx] if (node_names and u_idx < len(node_names)) else str(u_idx)
                v_name = node_names[v_idx] if (node_names and v_idx < len(node_names)) else str(v_idx)

                top_edges.append({
                    "edge": [u_name, v_name],
                    "weight": w
                })

        return top_edges
