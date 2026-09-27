from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Query, Depends

from app.core.schemas import Chain, FetchResult
from app.fetchers.orchestrator import FetchOrchestrator

router = APIRouter(prefix="/fetch", tags=["Multi-Chain Blockchain Ingestion"])

# Global orchestrator instance injected at startup
_orchestrator: FetchOrchestrator = None


def set_fetch_orchestrator(orchestrator: FetchOrchestrator):
    global _orchestrator
    _orchestrator = orchestrator


def get_orchestrator() -> FetchOrchestrator:
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Fetch orchestrator not initialized")
    return _orchestrator


def parse_chain(chain_str: str) -> Chain:
    normalized = chain_str.upper().replace("-", "_")
    try:
        return Chain(normalized)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported chain '{chain_str}'. Supported chains: {[c.value for c in Chain]}"
        )


@router.get("/{chain}/{address}", response_model=FetchResult)
async def fetch_wallet_transactions(
    chain: str,
    address: str,
    limit: int = Query(default=50, ge=1, le=100),
    bypass_cache: bool = Query(default=False),
    orchestrator: FetchOrchestrator = Depends(get_orchestrator)
):
    """
    Fetch wallet transactions across BTC, ETH, or TRON_TRC20.
    Returns live on-chain data, cached result, or synthetic fallback with transparent data_source attribution.
    """
    chain_enum = parse_chain(chain)
    return await orchestrator.fetch_wallet(address, chain_enum, limit=limit, bypass_cache=bypass_cache)


@router.get("/{chain}/{address}/balance")
async def fetch_wallet_balance(
    chain: str,
    address: str,
    orchestrator: FetchOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """Fetch wallet native asset balance with data source provenance."""
    chain_enum = parse_chain(chain)
    res = await orchestrator.fetch_wallet(address, chain_enum, limit=1)
    return {
        "address": address,
        "chain": chain_enum.value,
        "balance": res.balance,
        "data_source": res.data_source,
        "cached_at": res.cached_at
    }
