"""
Feature engineering module for (suspect_wallet, candidate_vasp) pairs.
Extracts 20 clean, non-leaky topological, path quality, amount, temporal, suspect, and candidate VASP features.
Money Migration Atlas (SIH26182).
"""

from typing import Dict, List, Optional, Any, Set
import numpy as np
import networkx as nx

from app.graph.store import GraphStore
from app.core.schemas import Wallet, Transaction, VASP, Chain

FEATURE_NAMES: List[str] = [
    # Topological features
    "proximity_rank",
    "num_paths_to_vasp",
    "avg_path_length",
    "min_path_length",
    "max_path_length",
    # Path quality features
    "path_touches_mixer",
    "num_mixer_hops_on_path",
    "path_mixer_penalty",
    "path_crosses_chain",
    "num_chain_hops",
    # Amount features
    "total_amount_on_path",
    "amount_retention_ratio",
    "max_single_hop_amount",
    "amount_std_dev",
    # Temporal features
    "avg_hop_time_seconds",
    "time_compression_ratio",
    "burst_detection",
    # Suspect-wallet features
    "suspect_degree",
    "suspect_tx_count",
    # Candidate-VASP features
    "vasp_is_indian_fiu",
]


def extract_features(
    suspect_wallet: str,
    candidate_vasp_id: str,
    store: GraphStore,
    candidate_path: Optional[List[str]] = None,
    candidate_tx_hashes: Optional[List[str]] = None,
    max_hops: int = 6
) -> Dict[str, float]:
    """
    Extract a 20-dimensional feature vector for a (suspect_wallet, candidate_vasp) pair.
    
    Guarantees:
    - Zero data leakage: no VASP synthetic volume shortcuts.
    - Zero dead features: removes zero-variance suspect_age_days and vasp_chain_match.
    - Path mixer penalty: captures ratio of mixer-flagged nodes relative to total hops.
    """
    graph: nx.MultiDiGraph = getattr(store, "graph", None)
    wallets_map: Dict[str, Wallet] = getattr(store, "wallets", {})
    txs_map: Dict[str, Transaction] = getattr(store, "transactions", {})

    # 1. Resolve candidate VASP targets
    vasp = store.get_vasp(candidate_vasp_id)
    vasp_targets: Set[str] = set()
    if vasp:
        vasp_targets.update(vasp.hot_wallets)
        vasp_targets.update(vasp.deposit_sweepers)
    for addr, w in wallets_map.items():
        if w.vasp_id == candidate_vasp_id:
            vasp_targets.add(addr)

    # 2. Path Finding & Topological Exploration
    paths: List[List[str]] = []
    if candidate_path and len(candidate_path) >= 2 and candidate_path[0] == suspect_wallet:
        paths.append(candidate_path)

    if graph is not None and suspect_wallet in graph:
        # Search paths to candidate targets
        for target in vasp_targets:
            if target in graph and target != suspect_wallet:
                try:
                    if nx.has_path(graph, suspect_wallet, target):
                        shortest = nx.shortest_path(graph, suspect_wallet, target)
                        if len(shortest) - 1 <= max_hops:
                            if shortest not in paths:
                                paths.append(shortest)
                            # Extract all tied shortest paths up to limit
                            for p in nx.all_shortest_paths(graph, suspect_wallet, target):
                                if p not in paths:
                                    paths.append(p)
                                if len(paths) >= 1000:
                                    break
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
            if len(paths) >= 1000:
                break

    is_reachable = len(paths) > 0
    if is_reachable:
        path_lengths = [len(p) - 1 for p in paths]
        proximity_rank = float(min(path_lengths))
        num_paths_to_vasp = float(min(len(paths), 1000))
        avg_path_length = float(np.mean(path_lengths))
        min_path_length = float(min(path_lengths))
        max_path_length = float(max(path_lengths))
        
        # Primary path: use candidate_path if valid, else shortest path
        if candidate_path and len(candidate_path) >= 2:
            primary_path = candidate_path
        else:
            primary_path = min(paths, key=len)
    else:
        proximity_rank = 99.0
        num_paths_to_vasp = 0.0
        avg_path_length = 0.0
        min_path_length = 0.0
        max_path_length = 0.0
        primary_path = []

    # 3. Path Transactions Extraction
    path_txs: List[Transaction] = []
    if is_reachable and len(primary_path) >= 2:
        if candidate_tx_hashes and len(candidate_tx_hashes) == len(primary_path) - 1:
            for th in candidate_tx_hashes:
                tx = txs_map.get(th)
                if tx:
                    path_txs.append(tx)
        
        # If txs not fully resolved, inspect graph edges
        if len(path_txs) != len(primary_path) - 1 and graph is not None:
            path_txs.clear()
            for i in range(len(primary_path) - 1):
                u, v = primary_path[i], primary_path[i + 1]
                edge_data = graph.get_edge_data(u, v)
                tx_obj = None
                if edge_data:
                    first_k = next(iter(edge_data))
                    tx_hash = edge_data[first_k].get("tx_hash")
                    if tx_hash:
                        tx_obj = txs_map.get(tx_hash)
                path_txs.append(tx_obj)

    valid_txs = [t for t in path_txs if t is not None]

    # 4. Path Quality Features
    if is_reachable and len(primary_path) >= 2:
        mixer_nodes_count = sum(
            1 for addr in primary_path if wallets_map.get(addr) and wallets_map[addr].is_mixer
        )
        mixer_txs_count = sum(1 for t in valid_txs if t.is_mixer_tx)
        path_touches_mixer = 1.0 if (mixer_nodes_count > 0 or mixer_txs_count > 0) else 0.0
        num_mixer_hops_on_path = float(mixer_nodes_count + mixer_txs_count)
        
        # FIX 1: path_mixer_penalty = number of mixer-flagged wallets on path / total hops
        total_hops = max(1, len(primary_path) - 1)
        path_mixer_penalty = float(mixer_nodes_count / total_hops)

        path_chains = []
        for addr in primary_path:
            w = wallets_map.get(addr)
            if w:
                val = w.chain.value if hasattr(w.chain, "value") else str(w.chain)
                path_chains.append(val)
        for t in valid_txs:
            val = t.chain.value if hasattr(t.chain, "value") else str(t.chain)
            path_chains.append(val)

        unique_chains = set(path_chains)
        path_crosses_chain = 1.0 if len(unique_chains) > 1 else 0.0

        chain_hops = 0
        for i in range(len(primary_path) - 1):
            w1 = wallets_map.get(primary_path[i])
            w2 = wallets_map.get(primary_path[i + 1])
            if w1 and w2 and w1.chain != w2.chain:
                chain_hops += 1
        num_chain_hops = float(chain_hops)
    else:
        path_touches_mixer = 0.0
        num_mixer_hops_on_path = 0.0
        path_mixer_penalty = 0.0
        path_crosses_chain = 0.0
        num_chain_hops = 0.0

    # 5. Amount Features
    if valid_txs:
        amounts = [t.amount for t in valid_txs]
        total_amount_on_path = float(sum(amounts))
        max_single_hop_amount = float(max(amounts))
        amount_std_dev = float(np.std(amounts)) if len(amounts) > 1 else 0.0

        retention_ratios = []
        for i in range(len(valid_txs) - 1):
            t_in, t_out = valid_txs[i], valid_txs[i + 1]
            if t_in.token_symbol == t_out.token_symbol and t_in.amount > 0 and t_out.amount > 0:
                retention_ratios.append(min(t_out.amount / t_in.amount, t_in.amount / t_out.amount))
            elif t_in.token_symbol != t_out.token_symbol and t_in.amount > 0 and t_out.amount > 0:
                retention_ratios.append(1.0)
        
        amount_retention_ratio = float(np.mean(retention_ratios)) if retention_ratios else 1.0
    else:
        total_amount_on_path = 0.0
        amount_retention_ratio = 0.0
        max_single_hop_amount = 0.0
        amount_std_dev = 0.0

    # 6. Temporal Features
    if valid_txs:
        timestamps = [t.timestamp for t in valid_txs]
        time_diffs = [
            abs(timestamps[i + 1] - timestamps[i]) for i in range(len(timestamps) - 1)
        ]
        avg_hop_time_seconds = float(np.mean(time_diffs)) if time_diffs else 0.0
        time_compression_ratio = float(1.0 / (1.0 + avg_hop_time_seconds / 3600.0))
        time_span = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0
        burst_detection = 1.0 if (len(timestamps) > 1 and time_span <= 7200) else 0.0
    else:
        avg_hop_time_seconds = 0.0
        time_compression_ratio = 0.0
        burst_detection = 0.0

    # 7. Suspect Wallet Features
    if graph is not None and suspect_wallet in graph:
        in_deg = graph.in_degree(suspect_wallet)
        out_deg = graph.out_degree(suspect_wallet)
        suspect_degree = float(in_deg + out_deg)
    else:
        suspect_degree = 0.0

    suspect_tx_cnt = 0
    for tx in txs_map.values():
        if tx.from_address == suspect_wallet or tx.to_address == suspect_wallet:
            suspect_tx_cnt += 1

    suspect_tx_count = float(suspect_tx_cnt)

    # 8. Candidate VASP Features
    vasp_is_indian_fiu = 1.0 if (vasp and vasp.fiu_ind_registered) else 0.0

    # Assemble clean feature vector dictionary
    features: Dict[str, float] = {
        "proximity_rank": proximity_rank,
        "num_paths_to_vasp": num_paths_to_vasp,
        "avg_path_length": avg_path_length,
        "min_path_length": min_path_length,
        "max_path_length": max_path_length,
        "path_touches_mixer": path_touches_mixer,
        "num_mixer_hops_on_path": num_mixer_hops_on_path,
        "path_mixer_penalty": path_mixer_penalty,
        "path_crosses_chain": path_crosses_chain,
        "num_chain_hops": num_chain_hops,
        "total_amount_on_path": total_amount_on_path,
        "amount_retention_ratio": amount_retention_ratio,
        "max_single_hop_amount": max_single_hop_amount,
        "amount_std_dev": amount_std_dev,
        "avg_hop_time_seconds": avg_hop_time_seconds,
        "time_compression_ratio": time_compression_ratio,
        "burst_detection": burst_detection,
        "suspect_degree": suspect_degree,
        "suspect_tx_count": suspect_tx_count,
        "vasp_is_indian_fiu": vasp_is_indian_fiu,
    }

    return features
