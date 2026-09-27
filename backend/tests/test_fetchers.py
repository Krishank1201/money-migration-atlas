import pytest
import time
from fastapi.testclient import TestClient

from app.core.schemas import Chain, DataSource, FetchResult, Transaction
from app.fetchers.bitcoin_provider import BitcoinProvider
from app.fetchers.ethereum_provider import EthereumProvider
from app.fetchers.tron_provider import TronProvider
from app.fetchers.cache import FileCache
from app.fetchers.orchestrator import FetchOrchestrator
from app.config import get_settings
from app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def temp_cache(tmp_path):
    """Provides an isolated cache directory for testing."""
    cache = FileCache(cache_dir=str(tmp_path / "cache"), ttl_hours=24)
    yield cache
    cache.clear()


@pytest.mark.anyio
async def test_btc_provider_parsing(httpx_mock):
    """Test: BTC provider parses a mock Blockchair response correctly."""
    address = "bc1qtestaddress1234567890abcdef"
    mock_blockchair = {
        "data": {
            address: {
                "address": {
                    "balance": 150000000,  # 1.5 BTC
                    "transaction_count": 2
                },
                "transactions": [
                    {
                        "hash": "btc_tx_hash_001",
                        "time": 1710000000,
                        "balance_change": 1.25,
                        "from_address": address,
                        "to_address": "bc1qcounterparty",
                        "fee": 0.0002
                    },
                    {
                        "hash": "btc_tx_hash_002",
                        "time": 1710001000,
                        "balance_change": 0.25,
                        "from_address": "bc1qcounterparty",
                        "to_address": address,
                        "fee": 0.0001
                    }
                ]
            }
        }
    }

    httpx_mock.add_response(
        url=f"https://api.blockchair.com/bitcoin/dashboards/address/{address}",
        json=mock_blockchair,
        is_reusable=True
    )

    provider = BitcoinProvider(timeout=5)
    balance = await provider.get_wallet_balance(address)
    txs = await provider.get_wallet_transactions(address)

    assert balance == 1.5
    assert len(txs) == 2
    assert txs[0].tx_hash == "btc_tx_hash_001"
    assert txs[0].amount == 1.25
    assert txs[0].chain == Chain.BTC


@pytest.mark.anyio
async def test_eth_provider_parsing(httpx_mock):
    """Test: ETH provider parses mock Etherscan response (ETH + ERC20) correctly."""
    address = "0x71c67d8e1c39e08736602382d625114498ca78a0"

    mock_txlist = {
        "status": "1",
        "message": "OK",
        "result": [
            {
                "hash": "0xeth_tx_hash_001",
                "from": address,
                "to": "0xcounterparty_eth",
                "value": "2500000000000000000",  # 2.5 ETH
                "gasUsed": "21000",
                "gasPrice": "20000000000",
                "timeStamp": "1710100000"
            }
        ]
    }

    mock_tokentx = {
        "status": "1",
        "message": "OK",
        "result": [
            {
                "hash": "0xerc20_tx_hash_002",
                "from": "0xcounterparty_eth",
                "to": address,
                "value": "10000000000",  # 10,000 USDT (6 decimals)
                "tokenSymbol": "USDT",
                "tokenDecimal": "6",
                "timeStamp": "1710100500"
            }
        ]
    }

    mock_balance = {
        "status": "1",
        "message": "OK",
        "result": "5000000000000000000"  # 5 ETH
    }

    httpx_mock.add_response(
        url=f"https://api.etherscan.io/v2/api?module=account&action=balance&address={address}&tag=latest&chainid=1",
        json=mock_balance,
        is_reusable=True
    )
    httpx_mock.add_response(
        url=f"https://api.etherscan.io/v2/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&page=1&offset=50&sort=desc&chainid=1",
        json=mock_txlist,
        is_reusable=True
    )
    httpx_mock.add_response(
        url=f"https://api.etherscan.io/v2/api?module=account&action=tokentx&address={address}&startblock=0&endblock=99999999&page=1&offset=50&sort=desc&chainid=1",
        json=mock_tokentx,
        is_reusable=True
    )

    provider = EthereumProvider(timeout=5)
    balance = await provider.get_wallet_balance(address)
    txs = await provider.get_wallet_transactions(address)

    assert balance == 5.0
    assert len(txs) == 2
    symbols = {t.token_symbol for t in txs}
    assert "ETH" in symbols
    assert "USDT" in symbols


@pytest.mark.anyio
async def test_tron_provider_parsing(httpx_mock):
    """Test: Tron provider parses mock TronGrid response correctly."""
    address = "TLyqz4YGLFiEd9MzgDpBt9pW"

    mock_account = {
        "data": [
            {"address": address, "balance": 45000000}  # 45 TRX
        ]
    }

    mock_trc20 = {
        "data": [
            {
                "transaction_id": "tron_trc20_hash_001",
                "token_info": {"symbol": "USDT", "decimals": 6},
                "from": address,
                "to": "TDestinationMuleAddress",
                "value": "25000000000",  # 25,000 USDT
                "block_timestamp": 1710200000000
            }
        ]
    }

    httpx_mock.add_response(
        url=f"https://api.trongrid.io/v1/accounts/{address}",
        json=mock_account,
        is_reusable=True
    )
    httpx_mock.add_response(
        url=f"https://api.trongrid.io/v1/accounts/{address}/transactions/trc20?limit=50",
        json=mock_trc20,
        is_reusable=True
    )
    httpx_mock.add_response(
        url=f"https://api.trongrid.io/v1/accounts/{address}/transactions?limit=50",
        json={"data": []},
        is_reusable=True
    )

    provider = TronProvider(timeout=5)
    balance = await provider.get_wallet_balance(address)
    txs = await provider.get_wallet_transactions(address)

    assert balance == 45.0
    assert len(txs) >= 1
    assert txs[0].amount == 25000.0
    assert txs[0].token_symbol == "USDT"
    assert txs[0].chain == Chain.TRON_TRC20


@pytest.mark.anyio
async def test_orchestrator_fallback_on_provider_error(temp_cache, monkeypatch):
    """Test: Orchestrator falls back to synthetic_fallback when provider raises without crashing."""
    settings = get_settings()
    monkeypatch.setattr(settings, "DEMO_MODE", False)

    orchestrator = FetchOrchestrator(cache=temp_cache)

    async def broken_fetch(addr, limit=100):
        raise ConnectionError("Simulated Network Failure")

    monkeypatch.setattr(orchestrator.providers[Chain.BTC], "get_wallet_transactions", broken_fetch)

    result = await orchestrator.fetch_wallet("bc1qfailedaddress", Chain.BTC)

    assert result is not None
    assert result.data_source == DataSource.SYNTHETIC_FALLBACK.value
    assert len(result.transactions) > 0
    assert "Live provider error" in (result.error_message or "")


@pytest.mark.anyio
async def test_eth_provider_raises_on_missing_api_key(httpx_mock):
    """Test: ETH provider raises ProviderError on Etherscan status == '0'."""
    from app.fetchers.base import ProviderError
    address = "0x71c67d8e1c39e08736602382d625114498ca78a0"

    mock_error_resp = {
        "status": "0",
        "message": "NOTOK",
        "result": "Missing/Invalid API Key"
    }

    httpx_mock.add_response(
        url=f"https://api.etherscan.io/v2/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&page=1&offset=50&sort=desc&chainid=1",
        json=mock_error_resp,
        is_reusable=True
    )

    provider = EthereumProvider(timeout=5)
    with pytest.raises(ProviderError) as exc_info:
        await provider.get_wallet_transactions(address)

    assert "Missing/Invalid API Key" in str(exc_info.value)
    assert exc_info.value.provider == "EthereumProvider"


@pytest.mark.anyio
async def test_btc_provider_raises_on_blockchair_error(httpx_mock):
    """Test: BTC provider raises ProviderError when Blockchair reports context error."""
    from app.fetchers.base import ProviderError
    address = "bc1qinvalidaddress999"

    mock_error_resp = {
        "data": {},
        "context": {
            "code": 404,
            "error": "Address not found or invalid"
        }
    }

    httpx_mock.add_response(
        url=f"https://api.blockchair.com/bitcoin/dashboards/address/{address}",
        json=mock_error_resp,
        is_reusable=True
    )

    provider = BitcoinProvider(timeout=5)
    with pytest.raises(ProviderError) as exc_info:
        await provider.get_wallet_transactions(address)

    assert "Address not found or invalid" in str(exc_info.value)
    assert exc_info.value.provider == "BitcoinProvider"


@pytest.mark.anyio
async def test_tron_provider_raises_on_failure(httpx_mock):
    """Test: Tron provider raises ProviderError when TronGrid reports success == False."""
    from app.fetchers.base import ProviderError
    address = "TInvalidTronAddress999"

    mock_failure_resp = {
        "success": False,
        "error": "class java.lang.NullPointerException"
    }

    httpx_mock.add_response(
        url=f"https://api.trongrid.io/v1/accounts/{address}/transactions/trc20?limit=50",
        json=mock_failure_resp,
        is_reusable=True
    )

    provider = TronProvider(timeout=5)
    with pytest.raises(ProviderError) as exc_info:
        await provider.get_wallet_transactions(address)

    assert exc_info.value.provider == "TronProvider"


@pytest.mark.anyio
async def test_orchestrator_falls_back_on_empty_live_result(temp_cache, monkeypatch):
    """Test: When live provider returns 0 tx and 0 balance, orchestrator triggers synthetic_fallback."""
    settings = get_settings()
    monkeypatch.setattr(settings, "DEMO_MODE", False)

    orchestrator = FetchOrchestrator(cache=temp_cache)

    # Mock provider returning 0 transactions and 0 balance
    async def empty_txs(addr, limit=100):
        return []

    async def zero_balance(addr):
        return 0.0

    monkeypatch.setattr(orchestrator.providers[Chain.ETH], "get_wallet_transactions", empty_txs)
    monkeypatch.setattr(orchestrator.providers[Chain.ETH], "get_wallet_balance", zero_balance)

    result = await orchestrator.fetch_wallet("0xEmptyWalletAddress", Chain.ETH)

    assert result.data_source == DataSource.SYNTHETIC_FALLBACK.value
    assert len(result.transactions) > 0
    assert "Live provider error" in (result.error_message or "")


@pytest.mark.anyio
async def test_cache_hit_returns_without_refetch(temp_cache, monkeypatch):
    """Test: Cache hit returns same data without re-fetching."""
    settings = get_settings()
    monkeypatch.setattr(settings, "DEMO_MODE", False)

    orchestrator = FetchOrchestrator(cache=temp_cache)

    mock_result = FetchResult(
        address="0xCachedAddress123",
        chain=Chain.ETH,
        data_source=DataSource.LIVE.value,
        transactions=[
            Transaction(
                tx_hash="0xcached_tx_001",
                chain=Chain.ETH,
                from_address="0xCachedAddress123",
                to_address="0xRecipient",
                amount=10.0,
                token_symbol="ETH",
                timestamp=int(time.time()),
                fee=0.001
            )
        ],
        balance=10.0,
        cached_at=int(time.time())
    )

    temp_cache.set("0xCachedAddress123", Chain.ETH, mock_result)

    fetched = await orchestrator.fetch_wallet("0xCachedAddress123", Chain.ETH)
    assert fetched.data_source == DataSource.CACHE.value
    assert fetched.balance == 10.0
    assert len(fetched.transactions) == 1
    assert fetched.transactions[0].tx_hash == "0xcached_tx_001"


@pytest.mark.anyio
async def test_demo_mode_bypasses_live_providers(temp_cache, monkeypatch):
    """Test: DEMO_MODE=true bypasses all live network providers."""
    settings = get_settings()
    monkeypatch.setattr(settings, "DEMO_MODE", True)

    orchestrator = FetchOrchestrator(cache=temp_cache)

    result = await orchestrator.fetch_wallet("0xSomeArbitraryWallet", Chain.ETH)
    assert result.data_source == DataSource.SYNTHETIC.value
    assert len(result.transactions) > 0


def test_api_fetch_endpoints():
    """Test: API endpoints /api/v1/fetch/{chain}/{address} and /balance respond cleanly."""
    with TestClient(app) as client:
        # 1. Fetch transactions
        resp = client.get("/api/v1/fetch/ETH/0x71c67d8e1c39e08736602382d625114498ca78a0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["address"] == "0x71c67d8e1c39e08736602382d625114498ca78a0"
        assert data["chain"] == "ETH"
        assert data["data_source"] in ("synthetic", "live", "cache", "synthetic_fallback")
        assert len(data["transactions"]) > 0

        # 2. Fetch balance
        bal_resp = client.get("/api/v1/fetch/ETH/0x71c67d8e1c39e08736602382d625114498ca78a0/balance")
        assert bal_resp.status_code == 200
        bal_data = bal_resp.json()
        assert "balance" in bal_data
        assert bal_data["data_source"] in ("synthetic", "live", "cache", "synthetic_fallback")

        # 3. Invalid chain returns 400
        bad_resp = client.get("/api/v1/fetch/DODGECOIN/someaddress")
        assert bad_resp.status_code == 400
