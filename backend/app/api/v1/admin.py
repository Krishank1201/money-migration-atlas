import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query

from app.graph.store import GraphStore
from app.graph.networkx_store import NetworkXStore
from app.graph.neo4j_store import Neo4jStore
from app.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Administration & Switcher"])

# Store references
_networkx_store: Optional[NetworkXStore] = None
_neo4j_store: Optional[Neo4jStore] = None
_active_store: Optional[GraphStore] = None
_switch_listener: Optional[Any] = None


def set_admin_context(
    nx_store: NetworkXStore,
    neo_store: Neo4jStore,
    active_store: GraphStore,
    on_switch_callback: Optional[Any] = None
):
    global _networkx_store, _neo4j_store, _active_store, _switch_listener
    _networkx_store = nx_store
    _neo4j_store = neo_store
    _active_store = active_store
    _switch_listener = on_switch_callback


@router.get("/active-store")
async def get_active_store() -> Dict[str, Any]:
    """Returns the currently active graph store backend and connectivity status."""
    settings = get_settings()
    backend_name = "networkx" if isinstance(_active_store, NetworkXStore) else "neo4j"
    
    neo4j_online = False
    if _neo4j_store is not None:
        neo4j_online = _neo4j_store.is_available()

    reason = (
        "Operating on in-memory NetworkX store for zero-dependency local execution."
        if backend_name == "networkx"
        else "Operating on Neo4j enterprise graph database."
    )

    return {
        "active_backend": backend_name,
        "neo4j_available": neo4j_online,
        "demo_mode": settings.DEMO_MODE,
        "reason": reason,
        "active_stats": _active_store.stats() if _active_store else {}
    }


@router.post("/switch-store")
async def switch_graph_store(
    backend: str = Query(..., pattern="^(networkx|neo4j)$", description="Target backend: 'networkx' or 'neo4j'")
) -> Dict[str, Any]:
    """
    Dynamically switch active graph engine between NetworkX and Neo4j without restarting the service.
    """
    global _active_store
    target = backend.lower()

    if target == "neo4j":
        if _neo4j_store is None:
            raise HTTPException(status_code=500, detail="Neo4j store instance uninitialized")
        if not _neo4j_store.is_available():
            raise HTTPException(
                status_code=400,
                detail=f"Cannot switch to Neo4j: database unreachable at {_neo4j_store.uri}. Ensure container is running."
            )
        _active_store = _neo4j_store
        logger.info("Switched active graph store to Neo4j")
    else:
        if _networkx_store is None:
            raise HTTPException(status_code=500, detail="NetworkX store instance uninitialized")
        _active_store = _networkx_store
        logger.info("Switched active graph store to NetworkX")

    # Notify listeners/main app of switch
    if _switch_listener:
        _switch_listener(_active_store)

    return {
        "status": "success",
        "switched_to": target,
        "message": f"Active graph store successfully switched to {target}."
    }
