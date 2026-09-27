import pytest
from fastapi.testclient import TestClient

from app.main import app, nx_store
from app.config import get_settings
from app.core.schemas import Chain, Wallet, Transaction
from app.graph.networkx_store import NetworkXStore
from app.graph.neo4j_store import Neo4jStore
from app.graph.ingestion import GraphIngestionPipeline
from app.graph.analytics import (
    degree_distribution,
    betweenness_centrality,
    detect_mixer_candidates,
    vasp_reachability_map
)
from app.data.synthetic_generator import SyntheticGenerator

settings = get_settings()


def check_neo4j_online() -> bool:
    try:
        neo = Neo4jStore(
            uri=settings.NEO4J_URI,
            user=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD,
            database=settings.NEO4J_DATABASE,
            timeout=1.0
        )
        return neo.is_available()
    except Exception:
        return False


neo4j_available = check_neo4j_online()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.skipif(not neo4j_available, reason="Neo4j not reachable (optional Docker backend)")
def test_neo4j_index_creation():
    store = Neo4jStore(
        uri=settings.NEO4J_URI,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DATABASE,
        timeout=3.0
    )
    assert store.is_available()
    store._init_schema()
    store.close()


@pytest.mark.skipif(not neo4j_available, reason="Neo4j not reachable (optional Docker backend)")
def test_neo4j_bulk_ingest_1000_wallets():
    store = Neo4jStore(
        uri=settings.NEO4J_URI,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DATABASE,
        timeout=3.0
    )
    assert store.is_available()
    store.clear()

    wallets = [
        Wallet(
            address=f"0xbulktest{i:04d}",
            chain=Chain.ETH,
            risk_score=0.1
        )
        for i in range(1000)
    ]
    txs = [
        Transaction(
            tx_hash=f"0xtxbulk{i:04d}",
            chain=Chain.ETH,
            from_address=f"0xbulktest{i:04d}",
            to_address=f"0xbulktest{(i+1)%1000:04d}",
            amount=1.5,
            token_symbol="ETH",
            timestamp=1700000000 + i
        )
        for i in range(999)
    ]

    store.bulk_ingest(wallets, txs)
    stats = store.stats()
    assert stats["node_count"] == 1000
    assert stats["edge_count"] == 999

    store.clear()
    store.close()


@pytest.mark.skipif(not neo4j_available, reason="Neo4j not reachable (optional Docker backend)")
def test_networkx_neo4j_parity_12_cases():
    nx = NetworkXStore()
    neo = Neo4jStore(
        uri=settings.NEO4J_URI,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DATABASE,
        timeout=3.0
    )
    assert neo.is_available()
    neo.clear()

    gen = SyntheticGenerator(seed=settings.RANDOM_SEED)
    data = gen.generate()

    for v in data.vasps:
        nx.add_vasp(v)
        neo.add_vasp(v)

    for w in data.wallets:
        nx.add_wallet(w)
    for tx in data.transactions:
        nx.add_transaction(tx)

    neo.bulk_ingest(data.wallets, data.transactions)

    for case in data.benchmark_cases:
        nx_c = nx.find_nearest_vasp(case.suspect_wallet, max_hops=6)
        neo_c = neo.find_nearest_vasp(case.suspect_wallet, max_hops=6)

        if nx_c:
            assert bool(neo_c), f"Neo4j found no candidates for case {case.case_id}"
            assert nx_c[0]["vasp_id"] == neo_c[0]["vasp_id"], (
                f"Parity mismatch in {case.case_id}: NX={nx_c[0]['vasp_id']} vs Neo4j={neo_c[0]['vasp_id']}"
            )
            assert nx_c[0]["proximity_rank"] == neo_c[0]["proximity_rank"], (
                f"Proximity rank mismatch in {case.case_id}"
            )

    neo.clear()
    neo.close()


def test_subgraph_endpoint_returns_valid_json(client):
    # Fetch cases to get an address in the graph
    resp_cases = client.get("/api/v1/demo/test-cases")
    assert resp_cases.status_code == 200
    cases = resp_cases.json()
    suspect_addr = cases[0]["suspect_wallet"]
    chain = cases[0]["chain"]

    resp = client.get(f"/api/v1/graph/subgraph/{chain}/{suspect_addr}?hops=2")
    assert resp.status_code == 200
    data = resp.json()

    assert data["center_address"] == suspect_addr
    assert data["hops"] == 2
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) >= 1


def test_ingestion_pipeline_deduplicates():
    store = NetworkXStore()
    pipeline = GraphIngestionPipeline()

    gen = SyntheticGenerator(seed=settings.RANDOM_SEED)
    data = gen.generate()

    # Initial ingestion
    report1 = pipeline.ingest_from_synthetic(store, data)
    assert report1.wallets_added == len(data.wallets)
    assert report1.txs_added == len(data.transactions)

    # Second ingestion of identical dataset
    report2 = pipeline.ingest_from_synthetic(store, data)
    assert report2.wallets_added == 0
    assert report2.txs_added == 0


def test_mixer_detection_finds_known_mixers_from_synthetic():
    store = NetworkXStore()
    gen = SyntheticGenerator(seed=settings.RANDOM_SEED)
    data = gen.generate()

    for v in data.vasps:
        store.add_vasp(v)
    for w in data.wallets:
        store.add_wallet(w)
    for tx in data.transactions:
        store.add_transaction(tx)

    mixers = detect_mixer_candidates(store)
    assert len(mixers) > 0
    # Confirm known mixer labels or flags detected
    has_known = any(m["is_known_mixer"] for m in mixers)
    assert has_known, "Mixer candidate detection should identify at least one synthetic mixer pool"


def test_reachability_map_covers_all_vasps(client):
    resp_cases = client.get("/api/v1/demo/test-cases")
    cases = resp_cases.json()
    suspect_addr = cases[0]["suspect_wallet"]
    chain = cases[0]["chain"]

    resp = client.get(f"/api/v1/graph/analytics/reachability/{chain}/{suspect_addr}")
    assert resp.status_code == 200
    reachability = resp.json()

    assert isinstance(reachability, dict)
    assert len(reachability) > 0
    # The expected VASP should be in reachability
    expected_vasp = cases[0]["expected_vasp"]
    assert expected_vasp in reachability
    assert reachability[expected_vasp] >= 1


def test_admin_switch_store_endpoint(client):
    # Check active store
    resp = client.get("/api/v1/admin/active-store")
    assert resp.status_code == 200
    info = resp.json()
    assert "active_backend" in info
    assert "neo4j_available" in info

    # Switch to networkx (always succeeds)
    resp_switch = client.post("/api/v1/admin/switch-store?backend=networkx")
    assert resp_switch.status_code == 200
    assert resp_switch.json()["switched_to"] == "networkx"

    # Switch to invalid backend (fails validation)
    resp_bad = client.post("/api/v1/admin/switch-store?backend=invalid")
    assert resp_bad.status_code in (400, 422)

    # Switch to neo4j (if offline, returns graceful 400 error)
    resp_neo = client.post("/api/v1/admin/switch-store?backend=neo4j")
    if not info["neo4j_available"]:
        assert resp_neo.status_code == 400
        assert "unreachable" in resp_neo.json()["detail"].lower()
    else:
        assert resp_neo.status_code == 200
