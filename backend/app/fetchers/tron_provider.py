import logging
import time
from typing import List, Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.schemas import Chain, Transaction
from app.fetchers.base import BlockchainProvider, ProviderError
from app.config import get_settings

logger = logging.getLogger(__name__)


class TronProvider(BlockchainProvider):
    """
    Tron provider backed by TronGrid API.
    Supports native TRX and TRC-20 (Tether USDT) transfers.
    """

    BASE_URL = "https://api.trongrid.io/v1"

    def __init__(self, api_key: Optional[str] = None, timeout: Optional[int] = None):
        settings = get_settings()
        self.api_key = api_key or settings.TRONGRID_API_KEY
        self.timeout = timeout or settings.FETCH_TIMEOUT_SECONDS

    def get_chain(self) -> Chain:
        return Chain.TRON_TRC20

    def is_available(self) -> bool:
        return True

    def _get_headers(self) -> dict:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["TRON-PRO-API-KEY"] = self.api_key
        return headers

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
        reraise=True
    )
    async def get_wallet_balance(self, address: str) -> float:
        url = f"{self.BASE_URL}/accounts/{address}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(url, headers=self._get_headers())
                resp.raise_for_status()
                data = resp.json()

            if data.get("success") is False:
                raise ProviderError("TronProvider", f"TronGrid query failed: {data.get('error', 'success=false')}")

            acc_list = data.get("data")
            if not isinstance(acc_list, list):
                raise ProviderError("TronProvider", f"Invalid TronGrid accounts response: expected list, got {type(acc_list).__name__}")

            if acc_list:
                balance_sun = acc_list[0].get("balance", 0)
                return round(balance_sun / 1e6, 4)
            return 0.0
        except ProviderError:
            raise
        except Exception as e:
            logger.warning("Failed to fetch Tron balance for %s: %s", address, e)
            raise ProviderError("TronProvider", str(e))

    async def get_wallet_transactions(self, address: str, limit: int = 100) -> List[Transaction]:
        transactions: List[Transaction] = []

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # 1. Fetch TRC-20 transfers (primary for Tether USDT laundering)
                url_trc20 = f"{self.BASE_URL}/accounts/{address}/transactions/trc20"
                params = {"limit": min(limit, 50)}
                resp_trc20 = await client.get(url_trc20, params=params, headers=self._get_headers())
                resp_trc20.raise_for_status()
                json_trc20 = resp_trc20.json()

                if json_trc20.get("success") is False:
                    raise ProviderError("TronProvider", f"TRC20 query failed: {json_trc20.get('error', 'success=false')}")

                data_trc20 = json_trc20.get("data")
                if not isinstance(data_trc20, list):
                    raise ProviderError("TronProvider", f"Invalid TRC20 response: expected list, got {type(data_trc20).__name__}")

                for item in data_trc20:
                    symbol = item.get("token_info", {}).get("symbol", "USDT")
                    decimals = int(item.get("token_info", {}).get("decimals", 6))
                    val = float(item.get("value", 0)) / (10 ** decimals)
                    ts = int(item.get("block_timestamp", time.time() * 1000)) // 1000

                    transactions.append(Transaction(
                        tx_hash=item.get("transaction_id", ""),
                        chain=Chain.TRON_TRC20,
                        from_address=item.get("from", ""),
                        to_address=item.get("to", ""),
                        amount=round(val, 2),
                        token_symbol=symbol.upper(),
                        timestamp=ts,
                        fee=1.5
                    ))

                # 2. Fetch Native Tron Transactions if needed
                if len(transactions) < limit:
                    url_tx = f"{self.BASE_URL}/accounts/{address}/transactions"
                    resp_tx = await client.get(url_tx, params=params, headers=self._get_headers())
                    resp_tx.raise_for_status()
                    json_tx = resp_tx.json()

                    if json_tx.get("success") is False:
                        raise ProviderError("TronProvider", f"Tron transactions query failed: {json_tx.get('error', 'success=false')}")

                    data_tx = json_tx.get("data")
                    if not isinstance(data_tx, list):
                        raise ProviderError("TronProvider", f"Invalid transactions response: expected list, got {type(data_tx).__name__}")

                    for item in data_tx:
                        tx_id = item.get("txID", "")
                        raw_data = item.get("raw_data", {})
                        ts = int(raw_data.get("timestamp", time.time() * 1000)) // 1000
                        contracts = raw_data.get("contract", [])
                        val_trx = 0.0
                        owner_addr = address
                        to_addr = "contract"

                        if contracts:
                            c_param = contracts[0].get("parameter", {}).get("value", {})
                            val_trx = float(c_param.get("amount", 0)) / 1e6
                            owner_addr = c_param.get("owner_address", address)
                            to_addr = c_param.get("to_address", "contract")

                        transactions.append(Transaction(
                            tx_hash=tx_id,
                            chain=Chain.TRON_TRC20,
                            from_address=owner_addr,
                            to_address=to_addr,
                            amount=round(val_trx, 4),
                            token_symbol="TRX",
                            timestamp=ts,
                            fee=1.0
                        ))

            transactions.sort(key=lambda t: t.timestamp, reverse=True)
            return transactions[:limit]

        except ProviderError:
            raise
        except Exception as e:
            logger.warning("Failed to fetch Tron transactions for %s: %s", address, e)
            raise ProviderError("TronProvider", str(e))
