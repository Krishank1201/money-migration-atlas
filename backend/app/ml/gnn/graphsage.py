"""
GraphSAGE Neural Network for VASP Attribution (Phase 5).
Money Migration Atlas (SIH26182).

Architecture:
- 3 GraphSAGE convolution layers (in=8 -> 64 -> 64 -> 32)
- Residual / skip connections
- Dropout 0.3
- Multi-layer perceptron classification head into 9 VASP classes
"""

from typing import Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


class VASPGraphSAGE(nn.Module):
    """
    3-Layer GraphSAGE model with skip connections and dropout
    for node-level and subgraph-level VASP attribution.
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden_channels: int = 64,
        out_channels: int = 32,
        num_classes: int = 9,
        dropout: float = 0.3
    ):
        super().__init__()
        self.dropout = dropout

        # 3 GraphSAGE Convolutions
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)

        # Skip projection for dimension matching
        self.skip1 = nn.Linear(in_channels, hidden_channels)

        # Batch / Layer Normalization for stable training
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.norm3 = nn.LayerNorm(out_channels)

        # MLP Classifier Head
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(
        self,
        data_or_x: Union[Data, torch.Tensor],
        edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass accepting either PyG Data object or (x, edge_index).
        
        Returns:
            Logits of shape (num_nodes, 9)
        """
        if isinstance(data_or_x, Data):
            x = data_or_x.x
            edge_index = data_or_x.edge_index
        else:
            x = data_or_x
            if edge_index is None:
                raise ValueError("edge_index must be provided when passing Tensor x")

        # Layer 1
        residual = self.skip1(x)
        h1 = self.conv1(x, edge_index)
        h1 = self.norm1(h1)
        h1 = F.relu(h1 + residual)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        # Layer 2 with residual connection
        h2 = self.conv2(h1, edge_index)
        h2 = self.norm2(h2)
        h2 = F.relu(h2 + h1)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)

        # Layer 3
        h3 = self.conv3(h2, edge_index)
        h3 = self.norm3(h3)
        h3 = F.relu(h3)
        h3 = F.dropout(h3, p=self.dropout, training=self.training)

        # Classifier head
        logits = self.mlp(h3)
        return logits
