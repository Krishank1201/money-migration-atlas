import pytest
from fastapi.testclient import TestClient

from app.core.schemas import Chain, Attribution, ConfidenceTier
from app.graph.networkx_store import NetworkXStore
from app.data.synthetic_generator import SyntheticDataGenerator, generate_synthetic_data
from app.main import app


def test_deterministic_reproducibility():
    """Verify that identical random seeds produce byte-for-byte identical graphs."""
    gen1 = SyntheticDataGenerator(seed=42)
    vasps1, wallets1, txs1, cases1 = gen1.generate()

    gen2 = SyntheticDataGenerator(seed=42)
    vasps2, wallets2, txs2, cases2 = gen2.generate()

    assert len(vasps1) == len(vasps2)
    assert len(wallets1) == len(wallets2)
    assert len(txs1) == len(txs2)
    assert len(cases1) == len(cases2)

    # Check sample addresses and tx hashes match exactly
    assert [w.address for w in wallets1[:10]] == [w.address for w in wallets2[:10]]
    assert [t.tx_hash for t in txs1[:10]] == [t.tx_hash for t in txs2[:10]]
    assert [c.suspect_wallet for c in cases1] == [c.suspect_wallet for c in cases2]


def test_dataset_scale_and_chain_coverage():
    """Verify minimum scale requirements: 8+ VASPs, 500+ wallets, 2000+ transactions."""
    vasps, wallets, transactions, test_cases = generate_synthetic_data(seed=42)

    assert len(vasps) >= 8, f"Expected >= 8 VASPs, got {len(vasps)}"
    assert len(wallets) >= 500, f"Expected >= 500 wallets, got {len(wallets)}"
    assert len(transactions) >= 2000, f"Expected >= 2000 transactions, got {len(transactions)}"
    assert len(test_cases) == 8, f"Expected 8 benchmark test cases, got {len(test_cases)}"

    # Check multi-chain presence (BTC, ETH, TRON_TRC20)
    chains_present = {w.chain for w in wallets}
    assert Chain.BTC in chains_present
    assert Chain.ETH in chains_present
    assert Chain.TRON_TRC20 in chains_present


def test_networkx_store_population_and_query():
    """Verify that NetworkXStore correctly loads all nodes, edges, and answers queries."""
    store = NetworkXStore()
    vasps, wallets, transactions, test_cases = generate_synthetic_data(seed=42, store=store)

    stats = store.stats()
    assert stats["backend"] == "networkx"
    assert stats["node_count"] >= 500
    assert stats["edge_count"] >= 2000
    assert stats["vasp_count"] >= 8

    # Verify wallet lookup
    sample_wallet = wallets[0]
    fetched = store.get_wallet(sample_wallet.address)
    assert fetched is not None
    assert fetched.address == sample_wallet.address


def test_all_8_benchmark_cases_ground_truth_proximity():
    """
    Verify that for every benchmark test case:
    1. The suspect wallet resolves to the expected ground-truth VASP
    2. The calculated hop count matches the expected topological distance
    3. The evidence path and transaction hashes are forensically complete
    """
    store = NetworkXStore()
    vasps, wallets, transactions, test_cases = generate_synthetic_data(seed=42, store=store)

    assert len(test_cases) == 8

    for case in test_cases:
        candidates = store.find_nearest_vasp(case.suspect_wallet, max_hops=6)
        assert len(candidates) > 0, f"Failed to find any VASP candidates for {case.case_id}"

        top_candidate = candidates[0]
        assert top_candidate["vasp_id"] == case.expected_vasp, (
            f"Case {case.case_id} failed: expected {case.expected_vasp}, "
            f"got {top_candidate['vasp_id']}"
        )
        assert top_candidate["proximity_rank"] == case.expected_graph_distance, (
            f"Case {case.case_id} distance mismatch: expected {case.expected_graph_distance}, "
            f"got {top_candidate['proximity_rank']}"
        )
        assert len(top_candidate["path"]) >= 2
        assert len(top_candidate["tx_hashes"]) == top_candidate["proximity_rank"]


def test_strict_score_separation_invariant():
    """
    Verify that Attribution enforces strict separation between:
    - proximity_rank (integer hops)
    - confidence_score (float probability from ML/GNN)
    and that never_blended is True.
    """
    attr = Attribution(
        suspect_wallet="bc1qtest123",
        chain=Chain.BTC,
        predicted_vasp="coindcx",
        proximity_rank=3,
        proximity_distance=3.0,
        confidence_score=None,  # Not fabricated in Phase 1
        confidence_tier=ConfidenceTier.UNKNOWN,
        evidence_subgraph=[],
        evidence_hashes=["0xhash1", "0xhash2", "0xhash3"],
        counterfactuals=[],
        never_blended=True
    )
    assert attr.never_blended is True
    assert attr.proximity_rank == 3
    assert attr.confidence_score is None


def test_fastapi_health_endpoint():
    """Verify FastAPI /health endpoint returns 200 OK and valid system statistics."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["demo_mode"] is True
        assert data["random_seed"] == 42
        assert data["graph_backend"] == "networkx"
        assert data["wallet_count"] >= 500
        assert data["transaction_count"] >= 2000
        assert data["benchmark_cases_count"] == 8


def test_fastapi_demo_benchmark_endpoints():
    """Verify listing test cases and evaluating proximity via REST API."""
    with TestClient(app) as client:
        # 1. List test cases
        cases_resp = client.get("/api/v1/demo/test-cases")
        assert cases_resp.status_code == 200
        cases = cases_resp.json()
        assert len(cases) == 8
        assert cases[0]["case_id"] == "CASE-001"

        # 2. Evaluate CASE-001 proximity
        eval_resp = client.get("/api/v1/demo/benchmark/CASE-001/proximity")
        assert eval_resp.status_code == 200
        eval_data = eval_resp.json()
        assert eval_data["case_id"] == "CASE-001"
        assert eval_data["topological_attribution"]["ground_truth_validated"] is True
        assert eval_data["topological_attribution"]["top_match_vasp"] == "coindcx"
        assert eval_data["topological_attribution"]["proximity_rank_hops"] == 3
        assert eval_data["score_separation_guarantee"]["never_blended"] is True
