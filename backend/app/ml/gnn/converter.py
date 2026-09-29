"""
PyTorch Geometric Data Converter (Phase 5).
Money Migration Atlas (SIH26182).

Converts GraphStore (NetworkX graph) into PyG Data format with rich
node and edge feature vectors, class masks, and ego-subgraph extraction.
"""

from typing import Dict, List, Tuple, Any, Optional, Set
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data

from app.graph.store import GraphStore
from app.core.schemas import Wallet, Transaction, VASP, Chain

VASP_TO_IDX: Dict[str, int] = {
    "coindcx": 0,
    "wazirx": 1,
    "zebpay": 2,
    "coinswitch": 3,
    "mudrex": 4,
    "unocoin": 5,
    "giottus": 6,
    "binance_offshore": 7,
    "bybit": 8,
}

IDX_TO_VASP: Dict[int, str] = {v: k for k, v in VASP_TO_IDX.items()}

CHAIN_TO_IDX: Dict[str, int] = {
    Chain.BTC.value: 0,
    Chain.ETH.value: 1,
    Chain.TRON_TRC20.value: 2,
    Chain.BSC.value: 3,
    Chain.SOL.value: 4,
    Chain.POLYGON.value: 5,
}


def networkx_to_pyg(
    store: GraphStore,
    suspect_wallets: Optional[List[str]] = None,
    benchmark_excluded_addrs: Optional[Set[str]] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    return_mappings: bool = False
) -> Data:
    """
    Converts full NetworkX cryptocurrency transaction graph into a PyTorch Geometric Data object.
    
    Node features (8-dim per wallet):
    [degree_in, degree_out, tx_count, total_in_amt, total_out_amt, avg_fee, is_mixer, is_vasp_wallet]
    
    Edge features (4-dim per tx):
    [amount_normalized, timestamp_delta, chain_id, is_cross_chain]
    
    Node labels:
    vasp_id index (0-8) for labeled wallets, -1 for unlabeled
    
    Returns:
        PyG Data object (with train_mask, val_mask, test_mask)
    """
    graph: nx.MultiDiGraph = getattr(store, "graph", nx.MultiDiGraph())
    wallets_map: Dict[str, Wallet] = getattr(store, "wallets", {})
    txs_map: Dict[str, Transaction] = getattr(store, "transactions", {})

    excluded = set(benchmark_excluded_addrs) if benchmark_excluded_addrs else set()
    if suspect_wallets:
        excluded.update(suspect_wallets)
    benchmark_excluded_addrs = excluded

    if benchmark_excluded_addrs is None:
        benchmark_excluded_addrs = set()

    # Index all nodes in graph
    nodes = list(graph.nodes())
    node_to_idx: Dict[str, int] = {node: i for i, node in enumerate(nodes)}
    idx_to_node: Dict[int, str] = {i: node for i, node in enumerate(nodes)}

    # Precompute transaction aggregations per wallet
    in_amounts: Dict[str, float] = {}
    out_amounts: Dict[str, float] = {}
    fees_sum: Dict[str, float] = {}
    tx_counts: Dict[str, int] = {}
    all_timestamps: List[int] = []

    for tx in txs_map.values():
        all_timestamps.append(tx.timestamp)
        out_amounts[tx.from_address] = out_amounts.get(tx.from_address, 0.0) + tx.amount
        in_amounts[tx.to_address] = in_amounts.get(tx.to_address, 0.0) + tx.amount
        fees_sum[tx.from_address] = fees_sum.get(tx.from_address, 0.0) + tx.fee
        tx_counts[tx.from_address] = tx_counts.get(tx.from_address, 0) + 1
        tx_counts[tx.to_address] = tx_counts.get(tx.to_address, 0) + 1

    min_ts = min(all_timestamps) if all_timestamps else 1700000000
    max_ts = max(all_timestamps) if all_timestamps else 1715000000
    time_span = max(1.0, float(max_ts - min_ts))

    # 1. Build Node Features (N, 8) and Labels (N,)
    x_list: List[List[float]] = []
    y_list: List[int] = []

    for node in nodes:
        w = wallets_map.get(node)
        in_deg = float(graph.in_degree(node))
        out_deg = float(graph.out_degree(node))
        cnt = float(tx_counts.get(node, 0))
        tot_in = float(np.log1p(in_amounts.get(node, 0.0)))
        tot_out = float(np.log1p(out_amounts.get(node, 0.0)))
        avg_fee = float(fees_sum.get(node, 0.0) / max(1, tx_counts.get(node, 1)))
        is_mixer = 1.0 if (w and w.is_mixer) else 0.0
        is_vasp = 1.0 if (w and (w.is_vasp_hot_wallet or w.is_vasp_deposit_sweeper or w.vasp_id)) else 0.0

        feat = [
            in_deg,
            out_deg,
            cnt,
            tot_in,
            tot_out,
            avg_fee,
            is_mixer,
            is_vasp
        ]
        x_list.append(feat)

        if w and w.vasp_id in VASP_TO_IDX and node not in benchmark_excluded_addrs:
            y_list.append(VASP_TO_IDX[w.vasp_id])
        else:
            y_list.append(-1)

    # 2. Build Edge Index (2, E) and Edge Features (E, 4)
    edge_index_src: List[int] = []
    edge_index_dst: List[int] = []
    edge_attr_list: List[List[float]] = []

    for u, v, k, d in graph.edges(keys=True, data=True):
        if u in node_to_idx and v in node_to_idx:
            edge_index_src.append(node_to_idx[u])
            edge_index_dst.append(node_to_idx[v])

            tx_hash = d.get("tx_hash")
            tx = txs_map.get(tx_hash) if tx_hash else None

            if tx:
                amt_norm = float(np.log1p(tx.amount))
                ts_delta = float((tx.timestamp - min_ts) / time_span)
                ch_val = tx.chain.value if hasattr(tx.chain, "value") else str(tx.chain)
                ch_id = float(CHAIN_TO_IDX.get(ch_val, 0))
                w_u = wallets_map.get(u)
                w_v = wallets_map.get(v)
                cross_chain = 1.0 if (w_u and w_v and w_u.chain != w_v.chain) else 0.0
            else:
                amt_norm = 0.0
                ts_delta = 0.0
                ch_id = 0.0
                cross_chain = 0.0

            edge_attr_list.append([amt_norm, ts_delta, ch_id, cross_chain])

    # Convert to PyTorch Tensors
    x = torch.tensor(x_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.long)
    edge_index = torch.tensor([edge_index_src, edge_index_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)

    # 3. Create Balanced Train / Val / Test Masks over Labeled Anchor Nodes
    num_nodes = len(nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    labeled_indices = [i for i, label in enumerate(y_list) if label != -1]
    rng = np.random.default_rng(seed)
    rng.shuffle(labeled_indices)

    n_labeled = len(labeled_indices)
    n_train = int(train_ratio * n_labeled)
    n_val = int(val_ratio * n_labeled)

    train_idx = labeled_indices[:n_train]
    val_idx = labeled_indices[n_train:n_train + n_val]
    test_idx = labeled_indices[n_train + n_val:]

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    data.node_to_idx = node_to_idx
    data.idx_to_node = idx_to_node

    if return_mappings:
        return data, node_to_idx, idx_to_node
    return data


def extract_ego_subgraph(
    center_wallet: str,
    store: GraphStore,
    k_hops: int = 3
) -> Optional[Data]:
    """
    Extracts a k-hop directed ego-subgraph centered around center_wallet
    and converts it into a localized PyG Data object for fast GNN inference.
    """
    graph: nx.MultiDiGraph = getattr(store, "graph", None)
    if graph is None or center_wallet not in graph:
        return None

    # Collect nodes within k_hops in both directions (neighborhood flow)
    forward_lengths = nx.single_source_shortest_path_length(graph, center_wallet, cutoff=k_hops)
    subgraph_nodes = set(forward_lengths.keys())

    try:
        rev_graph = graph.reverse()
        rev_lengths = nx.single_source_shortest_path_length(rev_graph, center_wallet, cutoff=k_hops)
        subgraph_nodes.update(rev_lengths.keys())
    except Exception:
        pass

    sub_g = graph.subgraph(subgraph_nodes)
    sub_nodes = list(sub_g.nodes())
    node_to_idx = {node: i for i, node in enumerate(sub_nodes)}

    wallets_map = getattr(store, "wallets", {})
    txs_map = getattr(store, "transactions", {})

    x_list = []
    y_list = []

    for node in sub_nodes:
        w = wallets_map.get(node)
        in_deg = float(sub_g.in_degree(node))
        out_deg = float(sub_g.out_degree(node))
        is_mixer = 1.0 if (w and w.is_mixer) else 0.0
        is_vasp = 1.0 if (w and (w.is_vasp_hot_wallet or w.is_vasp_deposit_sweeper or w.vasp_id)) else 0.0

        feat = [
            in_deg,
            out_deg,
            in_deg + out_deg,
            0.0,
            0.0,
            0.001,
            is_mixer,
            is_vasp
        ]
        x_list.append(feat)

        if w and w.vasp_id in VASP_TO_IDX:
            y_list.append(VASP_TO_IDX[w.vasp_id])
        else:
            y_list.append(-1)

    edge_index_src = []
    edge_index_dst = []
    edge_attr_list = []

    for u, v, k, d in sub_g.edges(keys=True, data=True):
        edge_index_src.append(node_to_idx[u])
        edge_index_dst.append(node_to_idx[v])

        tx_hash = d.get("tx_hash")
        tx = txs_map.get(tx_hash) if tx_hash else None
        amt = float(np.log1p(tx.amount)) if tx else 0.0
        edge_attr_list.append([amt, 0.5, 0.0, 0.0])

    if not edge_index_src:
        # Isolated node self-loop fallback
        edge_index_src = [0]
        edge_index_dst = [0]
        edge_attr_list = [[0.0, 0.0, 0.0, 0.0]]

    x = torch.tensor(x_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.long)
    edge_index = torch.tensor([edge_index_src, edge_index_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)

    center_idx = node_to_idx.get(center_wallet, 0)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        center_node_idx=center_idx
    )

    return data
