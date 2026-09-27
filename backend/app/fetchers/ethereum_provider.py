import logging
import time
from typing import List, Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.schemas import Chain, Transaction
from app.fetchers.base import BlockchainProvider
from app.config import get_settings

logger = logging.getLogger(__name__)


class EthereumProvider(BlockchainProvider):
    """
    Ethereum & ERC-20 provider backed by Etherscan API.
    Supports normal transactions (ETH) and ERC-20 token transfers (USDT, USDC, etc.).
    """

    BASE_URL = "https://api.etherscan.io/api"

    def __init__(self, api_key: Optional[str] = None, timeout: Optional[int] = None):
        settings = get_settings()
        self.api_key = api_key or settings.ETHERSCAN_API_KEY
        self.timeout = timeout or settings.FETCH_TIMEOUT_SECONDS

    def get_chain(self) -> Chain:
        return Chain.ETH

    def is_available(self) -> bool:
        # Usable with public key or mock fallback
        return True

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
        reraise=True
    )
    async def _query_etherscan(self, params: dict) -> dict:
        if self.api_key:
            params["apikey"] = self.api_key

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(self.BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            return data

    async def get_wallet_balance(self, address: str) -> float:
        try:
            params = {
                "module": "account",
                "action": "balance",
                "address": address,
                "tag": "latest"
            }
            data = await self._query_etherscan(params)
            wei_str = data.get("result", "0")
            wei = float(wei_str) if isinstance(wei_str, (int, float, str)) and str(wei_str).isdigit() else 0.0
            return round(wei / 1e18, 6)
        except Exception as e:
            logger.warning("Failed to fetch ETH balance for %s: %s", address, e)
            raise

    async def get_wallet_transactions(self, address: str, limit: int = 100) -> List[Transaction]:
        transactions: List[Transaction] = []

        try:
            # 1. Fetch native ETH transactions
            params_tx = {
                "module": "account",
                "action": "txlist",
                "address": address,
                "startblock": 0,
                "endblock": 99999999,
                "page": 1,
                "offset": min(limit, 50),
                "sort": "desc"
            }
            data_tx = await self._query_etherscan(params_tx)
            results_tx = data_tx.get("result", [])
            if isinstance(results_tx, list):
                for item in results_tx:
                    val_wei = float(item.get("value", 0))
                    gas_used = float(item.get("gasUsed", 21000))
                    gas_price = float(item.get("gasPrice", 20000000000))
                    fee_eth = (gas_used * gas_price) / 1e18

                    transactions.append(Transaction(
                        tx_hash=item.get("hash", ""),
                        chain=Chain.ETH,
                        from_address=item.get("from", "").lower(),
                        to_address=item.get("to", "").lower(),
                        amount=round(val_wei / 1e18, 6),
                        token_symbol="ETH",
                        timestamp=int(item.get("timeStamp", time.time())),
                        fee=round(fee_eth, 6),
                        gas_price=gas_price / 1e9
                    ))

            # 2. Fetch ERC-20 token transfers (e.g. USDT)
            params_token = {
                "module": "account",
                "action": "tokentx",
                "address": address,
                "startblock": 0,
                "endblock": 99999999,
                "page": 1,
                "offset": min(limit, 50),
                "sort": "desc"
            }
            data_token = await self._query_etherscan(params_token)
            results_token = data_token.get("result", [])
            if isinstance(results_token, list):
                for item in results_token:
                    decimals = int(item.get("tokenDecimal", 18) or 18)
                    val_token = float(item.get("value", 0)) / (10 ** decimals)
                    transactions.append(Transaction(
                        tx_hash=item.get("hash", ""),
                        chain=Chain.ETH,
                        from_address=item.get("from", "").lower(),
                        to_address=item.get("to", "").lower(),
                        amount=round(val_token, 4),
                        token_symbol=item.get("tokenSymbol", "TOKEN").upper(),
                        timestamp=int(item.get("timeStamp", time.time())),
                        fee=0.002
                    ))

            # Sort by timestamp descending
            transactions.sort(key=lambda t: t.timestamp, reverse=True)
            return transactions[:limit]

        except Exception as e:
            logger.warning("Failed to fetch ETH transactions for %s: %s", address, e)
            raise
