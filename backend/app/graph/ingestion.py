import time
import logging
from collections import deque
from typing import List, Optional, Set, Any
from app.core.schemas import Wallet, Transaction, VASP, Chain, IngestionReport
from app.graph.store import GraphStore
from app.graph.networkx_store import NetworkXStore
from app.config import get_settings

logger = logging.getLogger(__name__)


class GraphIngestionPipeline:
    """
    Ingestion pipeline supporting:
    - Bulk synthetic generation ingestion with deduplication.
    - Live blockchain multi-hop recursive neighborhood ingestion.
    - Fast benchmark scenarios loading for hackathon presentations.
    """

    def __init__(self):
        pass

    def ingest_from_synthetic(
        self, store: GraphStore, generator_output: Any = None
    ) -> IngestionReport:
        """Bulk load synthetic dataset into the target GraphStore with deduplication."""
        start_time = time.perf_counter()
        
        if generator_output is None:
            from app.data.synthetic_generator import SyntheticGenerator
            settings = get_settings()
            gen = SyntheticGenerator(seed=settings.RANDOM_SEED)
            generator_output = gen.generate()

        vasps: List[VASP] = generator_output.vasps
        wallets: List[Wallet] = generator_output.wallets
        transactions: List[Transaction] = generator_output.transactions

        # Register VASPs
        for vasp in vasps:
            store.add_vasp(vasp)

        # Deduplication
        wallets_to_add: List[Wallet] = []
        txs_to_add: List[Transaction] = []

        if isinstance(store, NetworkXStore):
            wallets_to_add = [w for w in wallets if w.address not in store.wallets]
            txs_to_add = [tx for tx in transactions if tx.tx_hash not in store.transactions]
        else:
            # Query-based deduplication for Neo4j or generic store
            for w in wallets:
                if not store.get_wallet(w.address):
                    wallets_to_add.append(w)
            # For transactions, add without duplicating
            txs_to_add = transactions

        logger.info(
            "Ingesting synthetic data: %d new wallets (%d skipped), %d new transactions (%d skipped)",
            len(wallets_to_add),
            len(wallets) - len(wallets_to_add),
            len(txs_to_add),
            len(transactions) - len(txs_to_add)
        )

        store.bulk_ingest(wallets_to_add, txs_to_add)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info("Synthetic ingestion completed in %.2f ms", duration_ms)

        return IngestionReport(
            wallets_added=len(wallets_to_add),
            txs_added=len(txs_to_add),
            duration_ms=duration_ms,
            source="synthetic"
        )

    def ingest_from_live_fetch(
        self,
        store: GraphStore,
        address: str,
        chain: Chain,
        orchestrator: Any,
        depth: int = 2
    ) -> IngestionReport:
        """
        Fetch a wallet live, add to graph, and recursively add 1-hop neighbors
        up to the specified depth with cycle prevention and deduplication.
        """
        start_time = time.perf_counter()
        visited: Set[str] = set()
        queue = deque([(address, 1)])
        
        wallets_added_count = 0
        txs_added_count = 0

        logger.info("Initiating live multi-hop graph ingestion for %s (chain=%s, depth=%d)", address, chain.value, depth)

        while queue:
            curr_addr, curr_depth = queue.popleft()
            if curr_addr in visited:
                continue
            visited.add(curr_addr)

            # Check if wallet already exists in store
            existing_wallet = store.get_wallet(curr_addr)
            if not existing_wallet:
                wallet_node = Wallet(
                    address=curr_addr,
                    chain=chain,
                    label="Live Ingested",
                    risk_score=0.0
                )
                store.add_wallet(wallet_node)
                wallets_added_count += 1

            # Fetch transactions via orchestrator
            fetch_result = orchestrator.fetch_wallet(curr_addr, chain)
            for tx in fetch_result.transactions:
                # Add edge to graph
                store.add_transaction(tx)
                txs_added_count += 1

                # If below depth limit, enqueue connected counterparty
                if curr_depth < depth:
                    counterparty = (
                        tx.to_address
                        if tx.from_address.lower() == curr_addr.lower()
                        else tx.from_address
                    )
                    if counterparty not in visited:
                        queue.append((counterparty, curr_depth + 1))

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "Live ingestion complete: %d wallets added, %d txs added in %.2f ms (visited %d nodes)",
            wallets_added_count, txs_added_count, duration_ms, len(visited)
        )

        return IngestionReport(
            wallets_added=wallets_added_count,
            txs_added=txs_added_count,
            duration_ms=duration_ms,
            source="live"
        )

    def ingest_benchmark_cases(self, store: GraphStore) -> IngestionReport:
        """Ingest the 12 ground truth benchmark scenarios for demonstration."""
        start_time = time.perf_counter()
        from app.data.synthetic_generator import SyntheticGenerator
        settings = get_settings()
        gen = SyntheticGenerator(seed=settings.RANDOM_SEED)
        gen_output = gen.generate()

        # Ingest benchmark cases
        report = self.ingest_from_synthetic(store, gen_output)
        report.source = "benchmark"
        return report
