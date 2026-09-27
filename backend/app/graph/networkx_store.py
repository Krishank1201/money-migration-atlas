import networkx as nx
from typing import List, Optional, Tuple, Dict, Any, Set
from app.core.schemas import Wallet, Transaction, VASP, Chain
from app.graph.store import GraphStore


class NetworkXStore(GraphStore):
    """
    High-performance in-memory graph store backed by NetworkX MultiDiGraph.
    Acts as the primary offline demo store and fast topological query engine.
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.wallets: Dict[str, Wallet] = {}
        self.transactions: Dict[str, Transaction] = {}
        self.vasps: Dict[str, VASP] = {}

    def add_vasp(self, vasp: VASP) -> None:
        self.vasps[vasp.id] = vasp

    def add_wallet(self, wallet: Wallet) -> None:
        self.wallets[wallet.address] = wallet
        self.graph.add_node(
            wallet.address,
            chain=wallet.chain.value if hasattr(wallet.chain, "value") else str(wallet.chain),
            vasp_id=wallet.vasp_id,
            is_vasp_hot_wallet=wallet.is_vasp_hot_wallet,
            is_vasp_deposit_sweeper=wallet.is_vasp_deposit_sweeper,
            is_mixer=wallet.is_mixer,
            risk_score=wallet.risk_score,
            label=wallet.label or ""
        )

    def add_transaction(self, tx: Transaction) -> None:
        self.transactions[tx.tx_hash] = tx
        
        # Ensure node presence
        if tx.from_address not in self.graph:
            self.graph.add_node(tx.from_address, chain=tx.chain.value)
        if tx.to_address not in self.graph:
            self.graph.add_node(tx.to_address, chain=tx.chain.value)

        # Invert amount slightly for Dijkstra weight so high-value paths have realistic traversal costs
        weight = 1.0 / (1.0 + float(tx.amount)) if tx.amount > 0 else 1.0

        self.graph.add_edge(
            tx.from_address,
            tx.to_address,
            key=tx.tx_hash,
            tx_hash=tx.tx_hash,
            amount=float(tx.amount),
            token_symbol=tx.token_symbol,
            timestamp=tx.timestamp,
            chain=tx.chain.value if hasattr(tx.chain, "value") else str(tx.chain),
            weight=weight,
            is_peeling_chain=tx.is_peeling_chain,
            is_sweeper_tx=tx.is_sweeper_tx,
            is_mixer_tx=tx.is_mixer_tx
        )

    def get_wallet(self, address: str) -> Optional[Wallet]:
        return self.wallets.get(address)

    def get_vasp(self, vasp_id: str) -> Optional[VASP]:
        return self.vasps.get(vasp_id)

    def get_all_vasps(self) -> List[VASP]:
        return list(self.vasps.values())

    def get_neighbors(
        self, address: str, direction: str = "both"
    ) -> List[Tuple[Wallet, Transaction]]:
        results: List[Tuple[Wallet, Transaction]] = []
        if address not in self.graph:
            return results

        if direction in ("out", "both"):
            for _, neighbor, edge_data in self.graph.out_edges(address, data=True):
                w = self.wallets.get(neighbor)
                tx_hash = edge_data.get("tx_hash")
                tx = self.transactions.get(tx_hash)
                if w and tx:
                    results.append((w, tx))

        if direction in ("in", "both"):
            for predecessor, _, edge_data in self.graph.in_edges(address, data=True):
                w = self.wallets.get(predecessor)
                tx_hash = edge_data.get("tx_hash")
                tx = self.transactions.get(tx_hash)
                if w and tx:
                    results.append((w, tx))

        return results

    def shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        if source not in self.graph or target not in self.graph:
            return None
        try:
            return nx.shortest_path(self.graph, source=source, target=target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def find_nearest_vasp(
        self, suspect_address: str, max_hops: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Calculates graph proximity from suspect address to any VASP-controlled address.
        Returns a sorted ranking by hop count (proximity rank).
        """
        if suspect_address not in self.graph:
            return []

        # Collect all target VASP wallet addresses
        vasp_address_map: Dict[str, str] = {}  # address -> vasp_id
        for vasp_id, vasp in self.vasps.items():
            for hw in vasp.hot_wallets:
                vasp_address_map[hw] = vasp_id
            for sw in vasp.deposit_sweepers:
                vasp_address_map[sw] = vasp_id

        # Also collect from labeled wallet nodes
        for addr, wallet in self.wallets.items():
            if wallet.vasp_id and (wallet.is_vasp_hot_wallet or wallet.is_vasp_deposit_sweeper):
                vasp_address_map[addr] = wallet.vasp_id

        # Single-source shortest paths up to max_hops
        candidates: List[Dict[str, Any]] = []
        visited_vasps: Set[str] = set()

        try:
            # BFS tree gives shortest path in terms of hop count
            lengths = nx.single_source_shortest_path_length(self.graph, suspect_address, cutoff=max_hops)
            all_paths = nx.single_source_shortest_path(self.graph, suspect_address, cutoff=max_hops)

            for target_addr, hops in sorted(lengths.items(), key=lambda item: item[1]):
                if hops == 0:
                    continue  # skip the suspect wallet itself
                
                if target_addr in vasp_address_map:
                    vasp_id = vasp_address_map[target_addr]
                    vasp = self.vasps.get(vasp_id)
                    vasp_name = vasp.name if vasp else vasp_id
                    
                    # Extract path, transaction hashes, and cumulative edge weight
                    node_path = all_paths[target_addr]
                    tx_hashes: List[str] = []
                    cum_weight = 0.0
                    for i in range(len(node_path) - 1):
                        u, v = node_path[i], node_path[i + 1]
                        edge_data = self.graph.get_edge_data(u, v)
                        if edge_data:
                            first_key = next(iter(edge_data))
                            cum_weight += edge_data[first_key].get("weight", 1.0)
                            tx_hashes.append(edge_data[first_key].get("tx_hash", ""))

                    candidates.append({
                        "vasp_id": vasp_id,
                        "vasp_name": vasp_name,
                        "target_wallet": target_addr,
                        "proximity_rank": hops,
                        "distance": round(cum_weight, 4),
                        "path": node_path,
                        "tx_hashes": tx_hashes,
                        "fiu_ind_registered": vasp.fiu_ind_registered if vasp else False
                    })
                    visited_vasps.add(vasp_id)

        except (nx.NetworkXError, KeyError):
            pass

        # Sort primarily by proximity rank (hops ascending)
        candidates.sort(key=lambda c: (c["proximity_rank"], c["distance"]))
        return candidates

    def get_subgraph(
        self, center_address: str, hops: int = 2
    ) -> Dict[str, Any]:
        if center_address not in self.graph:
            return {"nodes": [], "edges": []}

        # Ego graph treating edges as undirected for complete context
        undirected = self.graph.to_undirected(as_view=True)
        sub_nodes = nx.single_source_shortest_path_length(undirected, center_address, cutoff=hops).keys()
        sub = self.graph.subgraph(sub_nodes)

        nodes_list = []
        for n in sub.nodes():
            w = self.wallets.get(n)
            if w:
                nodes_list.append(w.model_dump())
            else:
                nodes_list.append({"address": n, "chain": "UNKNOWN"})

        edges_list = []
        for u, v, k, d in sub.edges(keys=True, data=True):
            edges_list.append(d)

        return {"nodes": nodes_list, "edges": edges_list}

    def stats(self) -> Dict[str, Any]:
        chain_dist: Dict[str, int] = {}
        for w in self.wallets.values():
            c = w.chain.value if hasattr(w.chain, "value") else str(w.chain)
            chain_dist[c] = chain_dist.get(c, 0) + 1
            
        vasp_dist: Dict[str, int] = {}
        for w in self.wallets.values():
            if w.vasp_id:
                vasp_dist[w.vasp_id] = vasp_dist.get(w.vasp_id, 0) + 1

        return {
            "backend": "networkx",
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "vasp_count": len(self.vasps),
            "wallet_count": len(self.wallets),
            "transaction_count": len(self.transactions),
            "chain_distribution": chain_dist,
            "vasp_distribution": vasp_dist
        }

    def clear(self) -> None:
        self.graph.clear()
        self.wallets.clear()
        self.transactions.clear()
        self.vasps.clear()
