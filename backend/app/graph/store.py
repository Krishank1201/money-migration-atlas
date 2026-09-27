from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any
from app.core.schemas import Wallet, Transaction, VASP


class GraphStore(ABC):
    """Abstract interface for blockchain transaction graph stores."""

    @abstractmethod
    def add_vasp(self, vasp: VASP) -> None:
        """Register a Virtual Asset Service Provider entity."""
        pass

    @abstractmethod
    def add_wallet(self, wallet: Wallet) -> None:
        """Add or update a wallet node."""
        pass

    @abstractmethod
    def add_transaction(self, tx: Transaction) -> None:
        """Add a directed transaction edge."""
        pass

    def bulk_ingest(self, wallets: List[Wallet], transactions: List[Transaction]) -> None:
        """Bulk ingest wallets and transactions."""
        for w in wallets:
            self.add_wallet(w)
        for tx in transactions:
            self.add_transaction(tx)


    @abstractmethod
    def get_wallet(self, address: str) -> Optional[Wallet]:
        """Fetch wallet metadata by address."""
        pass

    @abstractmethod
    def get_vasp(self, vasp_id: str) -> Optional[VASP]:
        """Fetch VASP metadata by identifier."""
        pass

    @abstractmethod
    def get_all_vasps(self) -> List[VASP]:
        """Retrieve all registered VASPs."""
        pass

    @abstractmethod
    def get_neighbors(
        self, address: str, direction: str = "both"
    ) -> List[Tuple[Wallet, Transaction]]:
        """
        Get adjacent wallets and corresponding connecting transactions.
        direction: 'out' (outgoing), 'in' (incoming), or 'both'
        """
        pass

    @abstractmethod
    def shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Return the sequence of wallet addresses connecting source to target."""
        pass

    @abstractmethod
    def find_nearest_vasp(
        self, suspect_address: str, max_hops: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Identify the closest VASP-controlled addresses from a suspect wallet.
        Returns ranked list of candidate matches with:
        - vasp_id
        - vasp_name
        - target_wallet
        - hops (proximity rank)
        - distance
        - path (list of addresses)
        - tx_hashes (list of edge transaction hashes)
        """
        pass

    @abstractmethod
    def get_subgraph(
        self, center_address: str, hops: int = 2
    ) -> Dict[str, Any]:
        """
        Extract an ego-subgraph centered around an address for visualization.
        Returns {'nodes': [...], 'edges': [...]}
        """
        pass

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Return summary metrics (node count, edge count, VASP count)."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all nodes and edges from the store."""
        pass
