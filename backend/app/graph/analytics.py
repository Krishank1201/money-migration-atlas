import logging
from typing import Dict, Any, List, Optional
import networkx as nx

from app.graph.store import GraphStore
from app.graph.networkx_store import NetworkXStore

logger = logging.getLogger(__name__)


def _get_nx_graph(store: GraphStore) -> nx.MultiDiGraph:
    """Helper to obtain or construct a NetworkX MultiDiGraph from any GraphStore."""
    if isinstance(store, NetworkXStore):
        return store.graph
    
    # Reconstruct from generic GraphStore via stats and subgraphs if needed
    G = nx.MultiDiGraph()
    # For ego queries or full traversal fallback
    vasps = store.get_all_vasps()
    for v in vasps:
        for hw in v.hot_wallets:
            sub = store.get_subgraph(hw, hops=4)
            for node in sub.get("nodes", []):
                G.add_node(node["address"], **node)
            for edge in sub.get("edges", []):
                G.add_edge(edge["from_address"], edge["to_address"], **edge)
    return G


def degree_distribution(store: GraphStore) -> Dict[str, Any]:
    """
    Computes degree distribution metrics across the graph:
    - Min, max, average degree
    - Degree bucket histogram
    """
    G = _get_nx_graph(store)
    if G.number_of_nodes() == 0:
        return {"total_wallets": 0, "avg_degree": 0.0, "max_degree": 0, "histogram": {}}

    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    total_degrees = {n: in_degrees.get(n, 0) + out_degrees.get(n, 0) for n in G.nodes()}

    degrees = list(total_degrees.values())
    avg_degree = round(sum(degrees) / len(degrees), 2)
    max_degree = max(degrees)

    # Bucketing
    buckets = {"1": 0, "2-5": 0, "6-10": 0, "11-20": 0, "20+": 0}
    for d in degrees:
        if d == 1:
            buckets["1"] += 1
        elif 2 <= d <= 5:
            buckets["2-5"] += 1
        elif 6 <= d <= 10:
            buckets["6-10"] += 1
        elif 11 <= d <= 20:
            buckets["11-20"] += 1
        else:
            buckets["20+"] += 1

    return {
        "total_wallets": G.number_of_nodes(),
        "total_transactions": G.number_of_edges(),
        "avg_degree": avg_degree,
        "max_degree": max_degree,
        "histogram": buckets
    }


def betweenness_centrality(store: GraphStore, top_n: int = 10) -> List[Dict[str, Any]]:
    """
    Calculates betweenness centrality to identify critical bridging wallets.
    Uses sample-based approximation for large graphs to ensure < 200ms latency.
    """
    G = _get_nx_graph(store)
    if G.number_of_nodes() == 0:
        return []

    # Use k sample for high performance if graph > 200 nodes
    k = min(150, G.number_of_nodes()) if G.number_of_nodes() > 200 else None
    centrality_map = nx.betweenness_centrality(G, k=k, normalized=True)

    sorted_nodes = sorted(centrality_map.items(), key=lambda x: x[1], reverse=True)[:top_n]

    results = []
    for addr, score in sorted_nodes:
        wallet = store.get_wallet(addr)
        results.append({
            "address": addr,
            "centrality_score": round(score, 6),
            "label": wallet.label if wallet else None,
            "vasp_id": wallet.vasp_id if wallet else None,
            "is_mixer": wallet.is_mixer if wallet else False
        })
    return results


def detect_mixer_candidates(
    store: GraphStore,
    min_degree: int = 1,
    flow_ratio_threshold: float = 0.25
) -> List[Dict[str, Any]]:
    """
    Topological mixer detection without ML:
    Identifies nodes with high fan-in + high fan-out + low net flow (topological signature),
    as well as known mixer pools and mixer transaction counterparties.
    """
    G = _get_nx_graph(store)
    candidates: List[Dict[str, Any]] = []

    for node in G.nodes():
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)

        if in_degree >= min_degree and out_degree >= min_degree:
            # Sum inbound and outbound transfer amounts
            in_amount = sum(float(d.get("amount", 0.0)) for _, _, d in G.in_edges(node, data=True))
            out_amount = sum(float(d.get("amount", 0.0)) for _, _, d in G.out_edges(node, data=True))

            total_flow = max(in_amount, out_amount, 0.0001)
            net_flow = abs(in_amount - out_amount)
            flow_ratio = net_flow / total_flow

            wallet = store.get_wallet(node)
            is_known = wallet.is_mixer if wallet else False
            has_mixer_tx = any(d.get("is_mixer_tx", False) for _, _, d in G.in_edges(node, data=True)) or \
                           any(d.get("is_mixer_tx", False) for _, _, d in G.out_edges(node, data=True))

            # Flag if labeled mixer, contains mixer transaction tags, or exhibits balanced passthrough flow
            if is_known or has_mixer_tx or (flow_ratio <= flow_ratio_threshold and (in_degree >= 2 or out_degree >= 2)):
                candidates.append({
                    "address": node,
                    "in_degree": in_degree,
                    "out_degree": out_degree,
                    "total_in_volume": round(in_amount, 4),
                    "total_out_volume": round(out_amount, 4),
                    "net_flow_ratio": round(flow_ratio, 4),
                    "is_known_mixer": is_known,
                    "label": wallet.label if wallet else None
                })

    candidates.sort(key=lambda c: (c["is_known_mixer"], c["in_degree"] + c["out_degree"]), reverse=True)
    return candidates


def vasp_reachability_map(
    store: GraphStore,
    suspect_address: str,
    max_hops: int = 6
) -> Dict[str, int]:
    """
    Maps each reachable VASP to its minimum graph distance (hops) from the suspect wallet.
    """
    candidates = store.find_nearest_vasp(suspect_address, max_hops=max_hops)
    reachability: Dict[str, int] = {}

    for c in candidates:
        vasp_id = c["vasp_id"]
        hops = c["proximity_rank"]
        if vasp_id not in reachability or hops < reachability[vasp_id]:
            reachability[vasp_id] = hops

    return reachability


def path_diversity(
    store: GraphStore,
    from_addr: str,
    to_addr: str,
    max_paths: int = 1000
) -> int:
    """
    Calculates number of distinct simple paths between two addresses (capped at max_paths).
    """
    G = _get_nx_graph(store)
    if from_addr not in G or to_addr not in G:
        return 0

    count = 0
    try:
        for _ in nx.all_simple_paths(G, source=from_addr, target=to_addr, cutoff=6):
            count += 1
            if count >= max_paths:
                break
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return 0

    return count
