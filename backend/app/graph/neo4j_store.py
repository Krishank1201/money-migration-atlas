import logging
from typing import List, Optional, Tuple, Dict, Any
from app.core.schemas import Wallet, Transaction, VASP, Chain, VASPCategory
from app.graph.store import GraphStore

logger = logging.getLogger(__name__)


class Neo4jStore(GraphStore):
    """
    Neo4j Graph Store driver implementing GraphStore interface with Cypher.
    Provides scalable graph traversal, schema indexing, and UNWIND bulk ingestion.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        timeout: float = 3.0,
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.timeout = timeout
        self._driver = None
        self._is_connected = False
        self._vasps_cache: Dict[str, VASP] = {}

    def is_available(self) -> bool:
        """Reachability check to test if Neo4j is online with 3s connection timeout."""
        try:
            from neo4j import GraphDatabase
            if self._driver is None:
                self._driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password),
                    connection_timeout=self.timeout,
                    max_connection_lifetime=300,
                    max_connection_pool_size=50
                )
            self._driver.verify_connectivity()
            self._is_connected = True
            logger.info("Successfully connected to Neo4j at %s", self.uri)
            self._init_schema()
            return True
        except Exception as e:
            logger.warning("Neo4j unreachable at %s (%s). Falling back to NetworkX store.", self.uri, e)
            self._is_connected = False
            return False

    def _init_schema(self) -> None:
        """Create indexes on Wallet address, vasp_id, and VASP id."""
        if not self._is_connected or not self._driver:
            return
        schema_queries = [
            "CREATE INDEX wallet_address_idx IF NOT EXISTS FOR (w:Wallet) ON (w.address)",
            "CREATE INDEX wallet_vasp_id_idx IF NOT EXISTS FOR (w:Wallet) ON (w.vasp_id)",
            "CREATE INDEX vasp_id_idx IF NOT EXISTS FOR (v:VASP) ON (v.id)"
        ]
        try:
            with self._driver.session(database=self.database) as session:
                for q in schema_queries:
                    session.run(q)
            logger.info("Neo4j schema indexes verified/created.")
        except Exception as e:
            logger.error("Failed to initialize Neo4j schema: %s", e)

    def add_vasp(self, vasp: VASP) -> None:
        self._vasps_cache[vasp.id] = vasp
        if not self._is_connected or not self._driver:
            return
        query = """
        MERGE (v:VASP {id: $id})
        SET v.name = $name,
            v.category = $category,
            v.jurisdiction = $jurisdiction,
            v.fiu_ind_registered = $fiu_ind_registered,
            v.fiu_registration_number = $fiu_registration_number,
            v.hot_wallets = $hot_wallets,
            v.deposit_sweepers = $deposit_sweepers
        """
        with self._driver.session(database=self.database) as session:
            session.run(
                query,
                id=vasp.id,
                name=vasp.name,
                category=vasp.category.value if hasattr(vasp.category, "value") else str(vasp.category),
                jurisdiction=vasp.jurisdiction,
                fiu_ind_registered=vasp.fiu_ind_registered,
                fiu_registration_number=vasp.fiu_registration_number,
                hot_wallets=vasp.hot_wallets,
                deposit_sweepers=vasp.deposit_sweepers
            )

    def add_wallet(self, wallet: Wallet) -> None:
        if not self._is_connected or not self._driver:
            return
        query = """
        MERGE (w:Wallet {address: $address})
        SET w.chain = $chain,
            w.label = $label,
            w.vasp_id = $vasp_id,
            w.is_vasp_hot_wallet = $is_hot,
            w.is_vasp_deposit_sweeper = $is_sweeper,
            w.is_mixer = $is_mixer,
            w.risk_score = $risk,
            w.first_seen = $first_seen,
            w.last_seen = $last_seen
        """
        chain_val = wallet.chain.value if hasattr(wallet.chain, "value") else str(wallet.chain)
        with self._driver.session(database=self.database) as session:
            session.run(
                query,
                address=wallet.address,
                chain=chain_val,
                label=wallet.label or "",
                vasp_id=wallet.vasp_id,
                is_hot=wallet.is_vasp_hot_wallet,
                is_sweeper=wallet.is_vasp_deposit_sweeper,
                is_mixer=wallet.is_mixer,
                risk=float(wallet.risk_score),
                first_seen=wallet.first_seen,
                last_seen=wallet.last_seen
            )

    def add_transaction(self, tx: Transaction) -> None:
        if not self._is_connected or not self._driver:
            return
        weight = 1.0 / (1.0 + float(tx.amount)) if tx.amount > 0 else 1.0
        chain_val = tx.chain.value if hasattr(tx.chain, "value") else str(tx.chain)
        query = """
        MERGE (src:Wallet {address: $from_addr})
          ON CREATE SET src.chain = $chain
        MERGE (dst:Wallet {address: $to_addr})
          ON CREATE SET dst.chain = $chain
        MERGE (src)-[r:TRANSFERRED {tx_hash: $tx_hash}]->(dst)
        SET r.amount = $amount,
            r.token = $token,
            r.timestamp = $timestamp,
            r.chain = $chain,
            r.fee = $fee,
            r.gas_price = $gas_price,
            r.weight = $weight,
            r.is_peeling_chain = $is_peeling_chain,
            r.is_sweeper_tx = $is_sweeper_tx,
            r.is_mixer_tx = $is_mixer_tx
        """
        with self._driver.session(database=self.database) as session:
            session.run(
                query,
                from_addr=tx.from_address,
                to_addr=tx.to_address,
                tx_hash=tx.tx_hash,
                amount=float(tx.amount),
                token=tx.token_symbol,
                timestamp=tx.timestamp,
                chain=chain_val,
                fee=float(tx.fee),
                gas_price=tx.gas_price,
                weight=weight,
                is_peeling_chain=tx.is_peeling_chain,
                is_sweeper_tx=tx.is_sweeper_tx,
                is_mixer_tx=tx.is_mixer_tx
            )

    def bulk_ingest(self, wallets: List[Wallet], transactions: List[Transaction]) -> None:
        """Optimized bulk insertion method using UNWIND for fast multi-thousand row loads."""
        if not self._is_connected or not self._driver:
            return

        wallet_records = [
            {
                "address": w.address,
                "chain": w.chain.value if hasattr(w.chain, "value") else str(w.chain),
                "label": w.label or "",
                "vasp_id": w.vasp_id,
                "is_hot": w.is_vasp_hot_wallet,
                "is_sweeper": w.is_vasp_deposit_sweeper,
                "is_mixer": w.is_mixer,
                "risk": float(w.risk_score),
                "first_seen": w.first_seen,
                "last_seen": w.last_seen
            }
            for w in wallets
        ]

        tx_records = [
            {
                "from_addr": tx.from_address,
                "to_addr": tx.to_address,
                "tx_hash": tx.tx_hash,
                "amount": float(tx.amount),
                "token": tx.token_symbol,
                "timestamp": tx.timestamp,
                "chain": tx.chain.value if hasattr(tx.chain, "value") else str(tx.chain),
                "fee": float(tx.fee),
                "gas_price": tx.gas_price,
                "weight": 1.0 / (1.0 + float(tx.amount)) if tx.amount > 0 else 1.0,
                "is_peeling_chain": tx.is_peeling_chain,
                "is_sweeper_tx": tx.is_sweeper_tx,
                "is_mixer_tx": tx.is_mixer_tx
            }
            for tx in transactions
        ]

        wallet_query = """
        UNWIND $batch AS w
        MERGE (node:Wallet {address: w.address})
        SET node.chain = w.chain,
            node.label = w.label,
            node.vasp_id = w.vasp_id,
            node.is_vasp_hot_wallet = w.is_hot,
            node.is_vasp_deposit_sweeper = w.is_sweeper,
            node.is_mixer = w.is_mixer,
            node.risk_score = w.risk,
            node.first_seen = w.first_seen,
            node.last_seen = w.last_seen
        """

        tx_query = """
        UNWIND $batch AS tx
        MERGE (src:Wallet {address: tx.from_addr})
          ON CREATE SET src.chain = tx.chain
        MERGE (dst:Wallet {address: tx.to_addr})
          ON CREATE SET dst.chain = tx.chain
        MERGE (src)-[r:TRANSFERRED {tx_hash: tx.tx_hash}]->(dst)
        SET r.amount = tx.amount,
            r.token = tx.token,
            r.timestamp = tx.timestamp,
            r.chain = tx.chain,
            r.fee = tx.fee,
            r.gas_price = tx.gas_price,
            r.weight = tx.weight,
            r.is_peeling_chain = tx.is_peeling_chain,
            r.is_sweeper_tx = tx.is_sweeper_tx,
            r.is_mixer_tx = tx.is_mixer_tx
        """

        with self._driver.session(database=self.database) as session:
            # Batch in chunks of 500 for optimal transaction sizes
            chunk_size = 500
            for i in range(0, len(wallet_records), chunk_size):
                session.run(wallet_query, batch=wallet_records[i:i + chunk_size])

            for i in range(0, len(tx_records), chunk_size):
                session.run(tx_query, batch=tx_records[i:i + chunk_size])

        logger.info("Neo4j bulk_ingest complete: %d wallets, %d txs", len(wallets), len(transactions))

    def get_wallet(self, address: str) -> Optional[Wallet]:
        if not self._is_connected or not self._driver:
            return None
        query = "MATCH (w:Wallet {address: $address}) RETURN w"
        with self._driver.session(database=self.database) as session:
            result = session.run(query, address=address).single()
            if not result:
                return None
            props = dict(result["w"])
            return Wallet(
                address=props["address"],
                chain=Chain(props.get("chain", "ETH")),
                label=props.get("label"),
                vasp_id=props.get("vasp_id"),
                is_vasp_hot_wallet=props.get("is_vasp_hot_wallet", False),
                is_vasp_deposit_sweeper=props.get("is_vasp_deposit_sweeper", False),
                is_mixer=props.get("is_mixer", False),
                risk_score=props.get("risk_score", 0.0),
                first_seen=props.get("first_seen"),
                last_seen=props.get("last_seen"),
            )

    def get_vasp(self, vasp_id: str) -> Optional[VASP]:
        if vasp_id in self._vasps_cache:
            return self._vasps_cache[vasp_id]
        if not self._is_connected or not self._driver:
            return None
        query = "MATCH (v:VASP {id: $id}) RETURN v"
        with self._driver.session(database=self.database) as session:
            result = session.run(query, id=vasp_id).single()
            if not result:
                return None
            props = dict(result["v"])
            vasp = VASP(
                id=props["id"],
                name=props.get("name", props["id"]),
                category=VASPCategory(props.get("category", "INDIAN_FIU")),
                jurisdiction=props.get("jurisdiction", "IN"),
                fiu_ind_registered=props.get("fiu_ind_registered", False),
                fiu_registration_number=props.get("fiu_registration_number"),
                hot_wallets=props.get("hot_wallets", []),
                deposit_sweepers=props.get("deposit_sweepers", [])
            )
            self._vasps_cache[vasp_id] = vasp
            return vasp

    def get_all_vasps(self) -> List[VASP]:
        if self._vasps_cache:
            return list(self._vasps_cache.values())
        if not self._is_connected or not self._driver:
            return []
        query = "MATCH (v:VASP) RETURN v"
        vasps: List[VASP] = []
        with self._driver.session(database=self.database) as session:
            for record in session.run(query):
                props = dict(record["v"])
                vasps.append(VASP(
                    id=props["id"],
                    name=props.get("name", props["id"]),
                    category=VASPCategory(props.get("category", "INDIAN_FIU")),
                    jurisdiction=props.get("jurisdiction", "IN"),
                    fiu_ind_registered=props.get("fiu_ind_registered", False),
                    fiu_registration_number=props.get("fiu_registration_number"),
                    hot_wallets=props.get("hot_wallets", []),
                    deposit_sweepers=props.get("deposit_sweepers", [])
                ))
        return vasps

    def get_neighbors(
        self, address: str, direction: str = "both"
    ) -> List[Tuple[Wallet, Transaction]]:
        if not self._is_connected or not self._driver:
            return []

        rel_pattern = "-[r:TRANSFERRED]->" if direction == "out" else ("<-[r:TRANSFERRED]-" if direction == "in" else "-[r:TRANSFERRED]-")
        query = f"""
        MATCH (w:Wallet {{address: $address}}){rel_pattern}(n:Wallet)
        RETURN n, r
        """
        results: List[Tuple[Wallet, Transaction]] = []
        with self._driver.session(database=self.database) as session:
            for record in session.run(query, address=address):
                n_props = dict(record["n"])
                r_props = dict(record["r"])
                
                wallet = Wallet(
                    address=n_props["address"],
                    chain=Chain(n_props.get("chain", "ETH")),
                    label=n_props.get("label"),
                    vasp_id=n_props.get("vasp_id"),
                    is_vasp_hot_wallet=n_props.get("is_vasp_hot_wallet", False),
                    is_vasp_deposit_sweeper=n_props.get("is_vasp_deposit_sweeper", False),
                    is_mixer=n_props.get("is_mixer", False),
                    risk_score=n_props.get("risk_score", 0.0),
                    first_seen=n_props.get("first_seen"),
                    last_seen=n_props.get("last_seen"),
                )
                start_addr = record["r"].start_node["address"] if hasattr(record["r"], "start_node") else address
                end_addr = record["r"].end_node["address"] if hasattr(record["r"], "end_node") else n_props["address"]
                
                tx = Transaction(
                    tx_hash=r_props["tx_hash"],
                    chain=Chain(r_props.get("chain", "ETH")),
                    from_address=start_addr,
                    to_address=end_addr,
                    amount=float(r_props.get("amount", 0.0)),
                    token_symbol=r_props.get("token", "ETH"),
                    timestamp=r_props.get("timestamp", 0),
                    fee=float(r_props.get("fee", 0.0)),
                    gas_price=r_props.get("gas_price"),
                    is_peeling_chain=r_props.get("is_peeling_chain", False),
                    is_sweeper_tx=r_props.get("is_sweeper_tx", False),
                    is_mixer_tx=r_props.get("is_mixer_tx", False),
                )
                results.append((wallet, tx))
        return results

    def shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        if not self._is_connected or not self._driver:
            return None
        query = """
        MATCH (src:Wallet {address: $source}), (dst:Wallet {address: $target})
        MATCH p = shortestPath((src)-[:TRANSFERRED*]->(dst))
        RETURN [node in nodes(p) | node.address] AS path
        """
        with self._driver.session(database=self.database) as session:
            record = session.run(query, source=source, target=target).single()
            if record and record["path"]:
                return list(record["path"])
        return None

    def find_nearest_vasp(
        self, suspect_address: str, max_hops: int = 6
    ) -> List[Dict[str, Any]]:
        """
        BFS traversal in Cypher to find closest VASP-controlled addresses.
        Returns ranked list identical in structure to NetworkXStore.
        """
        if not self._is_connected or not self._driver:
            return []

        # Find target VASP addresses
        query = f"""
        MATCH (src:Wallet {{address: $suspect_address}})
        MATCH (dst:Wallet)
        WHERE (dst.is_vasp_hot_wallet = true OR dst.is_vasp_deposit_sweeper = true OR dst.vasp_id IS NOT NULL)
          AND dst.address <> $suspect_address
        MATCH p = shortestPath((src)-[:TRANSFERRED*1..{max_hops}]->(dst))
        RETURN dst.address AS target_wallet,
               dst.vasp_id AS vasp_id,
               length(p) AS hops,
               [node in nodes(p) | node.address] AS path,
               [rel in relationships(p) | rel.tx_hash] AS tx_hashes,
               reduce(w = 0.0, rel in relationships(p) | w + coalesce(rel.weight, 1.0 / (1.0 + coalesce(rel.amount, 0.0)))) AS distance
        ORDER BY hops ASC, distance ASC
        """
        candidates: List[Dict[str, Any]] = []
        with self._driver.session(database=self.database) as session:
            for record in session.run(query, suspect_address=suspect_address):
                vasp_id = record["vasp_id"]
                vasp = self.get_vasp(vasp_id) if vasp_id else None
                vasp_name = vasp.name if vasp else (vasp_id or "Unknown VASP")
                fiu_registered = vasp.fiu_ind_registered if vasp else False

                candidates.append({
                    "vasp_id": vasp_id or "unknown",
                    "vasp_name": vasp_name,
                    "target_wallet": record["target_wallet"],
                    "proximity_rank": record["hops"],
                    "distance": round(float(record["distance"]), 4),
                    "path": list(record["path"]),
                    "tx_hashes": list(record["tx_hashes"]),
                    "fiu_ind_registered": fiu_registered
                })

        candidates.sort(key=lambda c: (c["proximity_rank"], c["distance"]))
        return candidates

    def get_subgraph(
        self, center_address: str, hops: int = 2
    ) -> Dict[str, Any]:
        """Extract ego-subgraph within hops of center address for visualization."""
        if not self._is_connected or not self._driver:
            return {"nodes": [], "edges": []}

        query = f"""
        MATCH path = (c:Wallet {{address: $center_address}})-[*0..{hops}]-(n:Wallet)
        UNWIND nodes(path) AS node
        WITH collect(DISTINCT node) AS distinct_nodes
        UNWIND distinct_nodes AS u
        MATCH (u)-[r:TRANSFERRED]->(v:Wallet)
        WHERE v IN distinct_nodes
        RETURN distinct_nodes, collect(DISTINCT r) AS distinct_edges
        """
        nodes_list: List[Dict[str, Any]] = []
        edges_list: List[Dict[str, Any]] = []

        with self._driver.session(database=self.database) as session:
            result = session.run(query, center_address=center_address).single()
            if result:
                for node in result["distinct_nodes"]:
                    props = dict(node)
                    nodes_list.append(props)
                for edge in result["distinct_edges"]:
                    edge_dict = dict(edge)
                    edge_dict["from_address"] = edge.start_node["address"] if hasattr(edge, "start_node") else ""
                    edge_dict["to_address"] = edge.end_node["address"] if hasattr(edge, "end_node") else ""
                    edges_list.append(edge_dict)

        return {"nodes": nodes_list, "edges": edges_list}

    def stats(self) -> Dict[str, Any]:
        if not self._is_connected or not self._driver:
            return {
                "backend": "neo4j",
                "connected": False,
                "node_count": 0,
                "edge_count": 0,
                "vasp_count": 0,
                "wallet_count": 0,
                "transaction_count": 0,
                "chain_distribution": {},
                "vasp_distribution": {}
            }

        with self._driver.session(database=self.database) as session:
            n_count = session.run("MATCH (w:Wallet) RETURN count(w) AS cnt").single()["cnt"]
            e_count = session.run("MATCH ()-[r:TRANSFERRED]->() RETURN count(r) AS cnt").single()["cnt"]
            v_count = session.run("MATCH (v:VASP) RETURN count(v) AS cnt").single()["cnt"]
            
            chain_dist: Dict[str, int] = {}
            for rec in session.run("MATCH (w:Wallet) WHERE w.chain IS NOT NULL RETURN w.chain AS chain, count(w) AS cnt"):
                chain_dist[str(rec["chain"])] = int(rec["cnt"])

            vasp_dist: Dict[str, int] = {}
            for rec in session.run("MATCH (w:Wallet) WHERE w.vasp_id IS NOT NULL RETURN w.vasp_id AS vasp_id, count(w) AS cnt"):
                vasp_dist[str(rec["vasp_id"])] = int(rec["cnt"])

        return {
            "backend": "neo4j",
            "connected": True,
            "node_count": n_count,
            "edge_count": e_count,
            "vasp_count": v_count,
            "wallet_count": n_count,
            "transaction_count": e_count,
            "chain_distribution": chain_dist,
            "vasp_distribution": vasp_dist
        }

    def clear(self) -> None:
        self._vasps_cache.clear()
        if self._is_connected and self._driver:
            with self._driver.session(database=self.database) as session:
                session.run("MATCH (n) DETACH DELETE n")

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._is_connected = False
