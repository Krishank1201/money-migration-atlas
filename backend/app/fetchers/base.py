from abc import ABC, abstractmethod
from typing import List
from app.core.schemas import Chain, Transaction


class ProviderError(Exception):
    """Raised when a live provider cannot fulfill a request. Triggers synthetic fallback."""
    def __init__(self, provider: str, reason: str):
        self.provider = provider
        self.reason = reason
        super().__init__(f"[{provider}] {reason}")


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
