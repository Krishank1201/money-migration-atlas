import pytest
from app.config import get_settings
from app.graph.networkx_store import NetworkXStore
from app.graph.neo4j_store import Neo4jStore
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
def populated_stores():
    """Populates both NetworkX and Neo4j stores with identical synthetic data."""
    if not neo4j_available:
        pytest.skip("Neo4j database not reachable; skipping parity test fixture.")

    nx_store = NetworkXStore()
    neo_store = Neo4jStore(
        uri=settings.NEO4J_URI,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DATABASE,
        timeout=3.0
    )
    assert neo_store.is_available()
    neo_store.clear()

    gen = SyntheticGenerator(seed=settings.RANDOM_SEED)
    data = gen.generate()

    # Populate NetworkX
    for v in data.vasps:
        nx_store.add_vasp(v)
    for w in data.wallets:
        nx_store.add_wallet(w)
    for tx in data.transactions:
        nx_store.add_transaction(tx)

    # Populate Neo4j
    for v in data.vasps:
        neo_store.add_vasp(v)
    neo_store.bulk_ingest(data.wallets, data.transactions)

    yield nx_store, neo_store, data.benchmark_cases

    neo_store.clear()
    neo_store.close()


@pytest.mark.skipif(not neo4j_available, reason="Neo4j not reachable (optional Docker backend)")
def test_networkx_neo4j_stats_parity(populated_stores):
    nx_store, neo_store, _ = populated_stores
    nx_stats = nx_store.stats()
    neo_stats = neo_store.stats()

    assert nx_stats["node_count"] == neo_stats["node_count"], (
        f"Node count mismatch: NX={nx_stats['node_count']} vs Neo4j={neo_stats['node_count']}"
    )
    assert nx_stats["edge_count"] == neo_stats["edge_count"], (
        f"Edge count mismatch: NX={nx_stats['edge_count']} vs Neo4j={neo_stats['edge_count']}"
    )


@pytest.mark.skipif(not neo4j_available, reason="Neo4j not reachable (optional Docker backend)")
def test_networkx_neo4j_benchmark_nearest_vasp_parity(populated_stores):
    nx_store, neo_store, benchmark_cases = populated_stores

    for case in benchmark_cases:
        nx_candidates = nx_store.find_nearest_vasp(case.suspect_wallet, max_hops=6)
        neo_candidates = neo_store.find_nearest_vasp(case.suspect_wallet, max_hops=6)

        assert bool(nx_candidates) == bool(neo_candidates), (
            f"Case {case.case_id}: Candidate presence mismatch (NX has {len(nx_candidates)}, Neo4j has {len(neo_candidates)})"
        )
        if nx_candidates and neo_candidates:
            assert nx_candidates[0]["vasp_id"] == neo_candidates[0]["vasp_id"], (
                f"Case {case.case_id}: Top-1 VASP mismatch! NX={nx_candidates[0]['vasp_id']} vs Neo4j={neo_candidates[0]['vasp_id']}"
            )
            assert nx_candidates[0]["proximity_rank"] == neo_candidates[0]["proximity_rank"], (
                f"Case {case.case_id}: Proximity rank mismatch! NX={nx_candidates[0]['proximity_rank']} vs Neo4j={neo_candidates[0]['proximity_rank']}"
            )


@pytest.mark.skipif(not neo4j_available, reason="Neo4j not reachable (optional Docker backend)")
def test_networkx_neo4j_shortest_path_parity(populated_stores):
    nx_store, neo_store, benchmark_cases = populated_stores

    for case in benchmark_cases:
        nx_candidates = nx_store.find_nearest_vasp(case.suspect_wallet, max_hops=6)
        if nx_candidates:
            target_wallet = nx_candidates[0]["target_wallet"]
            nx_path = nx_store.shortest_path(case.suspect_wallet, target_wallet)
            neo_path = neo_store.shortest_path(case.suspect_wallet, target_wallet)

            assert nx_path is not None
            assert neo_path is not None
            assert len(nx_path) == len(neo_path), (
                f"Case {case.case_id}: Path length mismatch! NX={len(nx_path)} vs Neo4j={len(neo_path)}"
            )
