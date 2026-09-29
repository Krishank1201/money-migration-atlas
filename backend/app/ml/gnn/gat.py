"""
GATv2 (Graph Attention Network v2) for VASP Attribution (Phase 5).
Money Migration Atlas (SIH26182).

Architecture:
- 2 GATv2Conv layers with 4 multi-head dynamic attention mechanisms
- Returns edge attention weights for structural forensic explainability
- Multi-layer perceptron classification head into 9 VASP classes
"""

from typing import Union, Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv


class VASPGATv2(nn.Module):
    """
    2-Layer GATv2 model with multi-head attention and attention weight extraction
    for explainable VASP graph attribution.
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden_channels: int = 16,
        heads: int = 4,
        out_channels: int = 32,
        num_classes: int = 9,
        dropout: float = 0.3
    ):
        super().__init__()
        self.dropout = dropout
        self.heads = heads

        # Layer 1: Multi-head attention (in_channels -> hidden_channels * heads = 16 * 4 = 64)
        self.gat1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout
        )

        # Layer 2: Aggregated attention (64 -> out_channels = 32)
        self.gat2 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=1,
            concat=False,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(hidden_channels * heads)
        self.norm2 = nn.LayerNorm(out_channels)

        # Classifier Head (identical dimension to GraphSAGE for parity)
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(
        self,
        data_or_x: Union[Data, torch.Tensor],
        edge_index: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with optional attention weight retrieval.
        
        Returns:
            logits: (num_nodes, 9)
            If return_attention_weights=True: (logits, (edge_index, attention_weights))
        """
        if isinstance(data_or_x, Data):
            x = data_or_x.x
            edge_index = data_or_x.edge_index
        else:
            x = data_or_x
            if edge_index is None:
                raise ValueError("edge_index must be provided when passing Tensor x")

        # Layer 1
        h1 = self.gat1(x, edge_index)
        h1 = self.norm1(h1)
        h1 = F.elu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        # Layer 2
        if return_attention_weights:
            h2, (edge_idx_attn, alpha) = self.gat2(h1, edge_index, return_attention_weights=True)
        else:
            h2 = self.gat2(h1, edge_index)
            edge_idx_attn, alpha = None, None

        h2 = self.norm2(h2)
        h2 = F.elu(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)

        # Output logits
        logits = self.mlp(h2)

        if return_attention_weights:
            return logits, (edge_idx_attn, alpha)
        return logits
