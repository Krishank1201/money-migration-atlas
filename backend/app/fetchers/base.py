from abc import ABC, abstractmethod
from typing import List
from app.core.schemas import Chain, Transaction


class BlockchainProvider(ABC):
    """Abstract interface for blockchain network transaction and balance providers."""

    @abstractmethod
    async def get_wallet_transactions(self, address: str, limit: int = 100) -> List[Transaction]:
        """Fetch on-chain transaction history for a given address."""
        pass

    @abstractmethod
    async def get_wallet_balance(self, address: str) -> float:
        """Fetch current confirmed on-chain native balance for a given address."""
        pass

    @abstractmethod
    def get_chain(self) -> Chain:
        """Return the blockchain network handled by this provider."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider configuration / credentials are valid and reachable."""
        pass
