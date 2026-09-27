import logging
import time
from typing import Dict, Optional, List
import hashlib

from app.core.schemas import Chain, Transaction, FetchResult, DataSource
from app.config import get_settings
from app.fetchers.base import BlockchainProvider, ProviderError
from app.fetchers.bitcoin_provider import BitcoinProvider
from app.fetchers.ethereum_provider import EthereumProvider
from app.fetchers.tron_provider import TronProvider
from app.fetchers.cache import FileCache
from app.graph.store import GraphStore

logger = logging.getLogger(__name__)


class FetchOrchestrator:
    """
    Coordinates multi-chain blockchain ingestion with caching,
    resilient fallback to synthetic generation, and DEMO_MODE isolation.
    """

    def __init__(self, graph_store: Optional[GraphStore] = None, cache: Optional[FileCache] = None):
        self.settings = get_settings()
        self.graph_store = graph_store
        self.cache = cache or FileCache()

        # Register providers by chain
        self.providers: Dict[Chain, BlockchainProvider] = {
            Chain.BTC: BitcoinProvider(),
            Chain.ETH: EthereumProvider(),
            Chain.TRON_TRC20: TronProvider(),
        }

    def set_graph_store(self, store: GraphStore) -> None:
        self.graph_store = store

    def _generate_synthetic_fallback(
        self, address: str, chain: Chain, data_source: str = DataSource.SYNTHETIC.value
    ) -> FetchResult:
        """Constructs synthetic transactions if address is in graph store or dynamically synthesizes plausible history."""
        txs: List[Transaction] = []
        balance = 1.25

        # Check if graph store has recorded transactions for this wallet
        if self.graph_store:
            wallet_obj = self.graph_store.get_wallet(address)
            if wallet_obj:
                neighbors = self.graph_store.get_neighbors(address, direction="both")
                for _, tx in neighbors:
                    txs.append(tx)
                balance = round(float(sum(t.amount for t in txs[:5])) * 0.15, 4)

        # If not present in graph store, generate deterministic plausible transactions
        if not txs:
            token_map = {Chain.BTC: "BTC", Chain.ETH: "ETH", Chain.TRON_TRC20: "USDT"}
            tok = token_map.get(chain, "CRYPTO")
            h = hashlib.sha256(address.encode()).hexdigest()
            base_time = int(time.time()) - 86400

            for i in range(3):
                tx_hash = f"0x{h[i*10:i*10+64]}" if chain == Chain.ETH else h[i*10:i*10+64]
                txs.append(Transaction(
                    tx_hash=tx_hash,
                    chain=chain,
                    from_address=address if i % 2 == 0 else f"counterparty_{i}_{h[:8]}",
                    to_address=f"counterparty_{i}_{h[:8]}" if i % 2 == 0 else address,
                    amount=round(float((i + 1) * 2.5), 2),
                    token_symbol=tok,
                    timestamp=base_time + (i * 3600),
                    fee=0.001
                ))

        msg = (
            "Offline demo mode active"
            if data_source == DataSource.SYNTHETIC.value
            else "Live provider failed or returned empty; fell back to synthetic data"
        )

        return FetchResult(
            address=address,
            chain=chain,
            data_source=data_source,
            transactions=txs,
            balance=balance,
            error_message=msg
        )

    async def fetch_wallet(
        self, address: str, chain: Chain, limit: int = 100, bypass_cache: bool = False
    ) -> FetchResult:
        """
        Fetches wallet transactions & balance with tiered resolution:
        1. DEMO_MODE=true -> data_source="synthetic"
        2. Cache Check -> data_source="cache"
        3. Live Provider -> data_source="live" (must have >=1 tx OR non-zero balance)
        4. Provider Failure/Empty -> data_source="synthetic_fallback"
        """
        # Tier 1: DEMO_MODE bypasses all external network calls
        if self.settings.DEMO_MODE:
            logger.info("DEMO_MODE=true active. Using synthetic provider for %s:%s", chain.value, address)
            return self._generate_synthetic_fallback(address, chain, data_source=DataSource.SYNTHETIC.value)

        # Tier 2: Check Cache
        if not bypass_cache:
            cached = self.cache.get(address, chain)
            if cached:
                logger.info("Cache hit for %s:%s", chain.value, address)
                return cached

        # Tier 3: Call Live Blockchain Provider
        provider = self.providers.get(chain)
        if not provider:
            logger.warning("No provider registered for chain %s; falling back to synthetic", chain.value)
            return self._generate_synthetic_fallback(
                address, chain, data_source=DataSource.SYNTHETIC_FALLBACK.value
            )

        try:
            logger.info("Attempting live on-chain fetch for %s via %s", address, provider.__class__.__name__)
            txs = await provider.get_wallet_transactions(address, limit=limit)
            try:
                bal = await provider.get_wallet_balance(address)
            except Exception:
                bal = 0.0

            # Post-fetch sanity check: empty transactions and zero balance means failure/no key
            if len(txs) == 0 and bal == 0.0:
                logger.warning("Provider %s returned empty result for %s, treating as failure", provider.__class__.__name__, address)
                raise ProviderError(provider.__class__.__name__, "Provider returned empty result (0 transactions and 0 balance)")

            result = FetchResult(
                address=address,
                chain=chain,
                data_source=DataSource.LIVE.value,
                transactions=txs,
                balance=bal,
                cached_at=int(time.time())
            )

            # Store in cache
            self.cache.set(address, chain, result)
            return result

        except Exception as e:
            # Tier 4: Graceful fallback on API failure, rate limit, or empty result
            logger.warning(
                "Live provider %s failed for %s (%s). Gracefully falling back to synthetic data.",
                provider.__class__.__name__, address, e
            )
            fallback = self._generate_synthetic_fallback(
                address, chain, data_source=DataSource.SYNTHETIC_FALLBACK.value
            )
            fallback.error_message = f"Live provider error: {str(e)}. Fallback to synthetic."
            return fallback

    async def get_balance(self, address: str, chain: Chain) -> float:
        res = await self.fetch_wallet(address, chain)
        return res.balance
