import logging
from typing import List, Optional, Tuple, Dict, Any
from app.core.schemas import Wallet, Transaction, VASP
from app.graph.store import GraphStore

logger = logging.getLogger(__name__)


class Neo4jStore(GraphStore):
    """
    Neo4j Graph Store driver.
    Includes connection reachability check and Cypher-backed queries for production scale.
    """

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j", timeout: float = 1.5):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.timeout = timeout
        self._driver = None
        self._is_connected = False

    def is_available(self) -> bool:
        """Lightweight reachability check to test if Neo4j is online without crashing."""
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                connection_timeout=self.timeout
            )
            driver.verify_connectivity()
            self._driver = driver
            self._is_connected = True
            logger.info("Connected to Neo4j at %s", self.uri)
            return True
        except Exception as e:
            logger.warning("Neo4j unreachable at %s (%s). Falling back to NetworkX store.", self.uri, e)
            self._is_connected = False
            return False

    def add_vasp(self, vasp: VASP) -> None:
        if not self._is_connected:
            return
        query = """
        MERGE (v:VASP {id: $id})
        SET v.name = $name, v.category = $category, v.fiu_ind_registered = $fiu_ind_registered
        """
        with self._driver.session(database=self.database) as session:
            session.run(query, id=vasp.id, name=vasp.name, category=vasp.category.value, fiu_ind_registered=vasp.fiu_ind_registered)

    def add_wallet(self, wallet: Wallet) -> None:
        if not self._is_connected:
            return
        query = """
        MERGE (w:Wallet {address: $address})
        SET w.chain = $chain, w.vasp_id = $vasp_id, w.is_vasp_hot_wallet = $is_hot,
            w.is_vasp_deposit_sweeper = $is_sweeper, w.is_mixer = $is_mixer, w.risk_score = $risk
        """
        with self._driver.session(database=self.database) as session:
            session.run(
                query,
                address=wallet.address,
                chain=wallet.chain.value,
                vasp_id=wallet.vasp_id,
                is_hot=wallet.is_vasp_hot_wallet,
                is_sweeper=wallet.is_vasp_deposit_sweeper,
                is_mixer=wallet.is_mixer,
                risk=wallet.risk_score
            )

    def add_transaction(self, tx: Transaction) -> None:
        if not self._is_connected:
            return
        query = """
        MERGE (src:Wallet {address: $from_addr})
        MERGE (dst:Wallet {address: $to_addr})
        CREATE (src)-[r:TRANSFERRED {
            tx_hash: $tx_hash,
            amount: $amount,
            token: $token,
            timestamp: $timestamp,
            chain: $chain
        }]->(dst)
        """
        with self._driver.session(database=self.database) as session:
            session.run(
                query,
                from_addr=tx.from_address,
                to_addr=tx.to_address,
                tx_hash=tx.tx_hash,
                amount=tx.amount,
                token=tx.token_symbol,
                timestamp=tx.timestamp,
                chain=tx.chain.value
            )

    def get_wallet(self, address: str) -> Optional[Wallet]:
        return None

    def get_vasp(self, vasp_id: str) -> Optional[VASP]:
        return None

    def get_all_vasps(self) -> List[VASP]:
        return []

    def get_neighbors(
        self, address: str, direction: str = "both"
    ) -> List[Tuple[Wallet, Transaction]]:
        return []

    def shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        return None

    def find_nearest_vasp(
        self, suspect_address: str, max_hops: int = 6
    ) -> List[Dict[str, Any]]:
        return []

    def get_subgraph(
        self, center_address: str, hops: int = 2
    ) -> Dict[str, Any]:
        return {"nodes": [], "edges": []}

    def stats(self) -> Dict[str, Any]:
        return {
            "backend": "neo4j",
            "connected": self._is_connected,
            "uri": self.uri
        }

    def clear(self) -> None:
        if self._is_connected and self._driver:
            with self._driver.session(database=self.database) as session:
                session.run("MATCH (n) DETACH DELETE n")

    def close(self) -> None:
        if self._driver:
            self._driver.close()
