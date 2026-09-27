import logging
import time
from typing import List, Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.schemas import Chain, Transaction
from app.fetchers.base import BlockchainProvider
from app.config import get_settings

logger = logging.getLogger(__name__)


class BitcoinProvider(BlockchainProvider):
    """
    Bitcoin provider backed by Blockchair API.
    Free tier supports 1,440 requests/day without an API key.
    """

    BASE_URL = "https://api.blockchair.com/bitcoin/dashboards/address"

    def __init__(self, api_key: Optional[str] = None, timeout: Optional[int] = None):
        settings = get_settings()
        self.api_key = api_key or settings.BLOCKCHAIR_API_KEY
        self.timeout = timeout or settings.FETCH_TIMEOUT_SECONDS

    def get_chain(self) -> Chain:
        return Chain.BTC

    def is_available(self) -> bool:
        return True

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
        reraise=True
    )
    async def _fetch_dashboard(self, address: str) -> dict:
        params = {}
        if self.api_key:
            params["key"] = self.api_key

        url = f"{self.BASE_URL}/{address}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(url, params=params)
            if resp.status_code == 429:
                logger.warning("Blockchair API rate limit hit for %s", address)
                resp.raise_for_status()
            resp.raise_for_status()
            return resp.json()

    async def get_wallet_balance(self, address: str) -> float:
        try:
            data = await self._fetch_dashboard(address)
            addr_data = data.get("data", {}).get(address, {}).get("address", {})
            satoshis = addr_data.get("balance", 0)
            return round(satoshis / 1e8, 8)
        except Exception as e:
            logger.warning("Failed to fetch BTC balance for %s: %s", address, e)
            raise

    async def get_wallet_transactions(self, address: str, limit: int = 100) -> List[Transaction]:
        try:
            data = await self._fetch_dashboard(address)
            raw_txs = data.get("data", {}).get(address, {}).get("transactions", [])
            parsed: List[Transaction] = []

            for item in raw_txs[:limit]:
                if isinstance(item, str):
                    # Blockchair returns array of hashes by default
                    tx_hash = item
                    parsed.append(Transaction(
                        tx_hash=tx_hash,
                        chain=Chain.BTC,
                        from_address=address,
                        to_address="unknown_counterparty",
                        amount=0.0,
                        token_symbol="BTC",
                        timestamp=int(time.time()),
                        fee=0.0001
                    ))
                elif isinstance(item, dict):
                    # Rich payload (mock or detailed dashboard)
                    tx_hash = item.get("hash") or item.get("tx_hash") or "unknown_tx"
                    amt = float(item.get("amount") or item.get("balance_change", 0))
                    parsed.append(Transaction(
                        tx_hash=tx_hash,
                        chain=Chain.BTC,
                        from_address=item.get("from_address") or address,
                        to_address=item.get("to_address") or "unknown_counterparty",
                        amount=abs(amt),
                        token_symbol=item.get("token_symbol", "BTC"),
                        timestamp=int(item.get("time") or item.get("timestamp") or time.time()),
                        fee=float(item.get("fee", 0.0001))
                    ))

            return parsed
        except Exception as e:
            logger.warning("Failed to fetch BTC transactions for %s: %s", address, e)
            raise
