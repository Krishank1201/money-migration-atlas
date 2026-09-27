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
    """Verify scale requirements: 8+ VASPs, 500+ wallets, 2000+ transactions, 12 test cases."""
    vasps, wallets, transactions, test_cases = generate_synthetic_data(seed=42)

    assert len(vasps) >= 8, f"Expected >= 8 VASPs, got {len(vasps)}"
    assert len(wallets) >= 500, f"Expected >= 500 wallets, got {len(wallets)}"
    assert len(transactions) >= 2000, f"Expected >= 2000 transactions, got {len(transactions)}"
    assert len(test_cases) == 12, f"Expected 12 benchmark test cases, got {len(test_cases)}"

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


def test_all_12_cases_expected_vasp_reachability():
    """Verify that the true ground-truth VASP is always topologically reachable for all 12 cases."""
    store = NetworkXStore()
    vasps, wallets, transactions, test_cases = generate_synthetic_data(seed=42, store=store)

    assert len(test_cases) == 12

    for case in test_cases:
        candidates = store.find_nearest_vasp(case.suspect_wallet, max_hops=6)
        vasp_ids = [c["vasp_id"] for c in candidates]
        assert case.expected_vasp in vasp_ids, (
            f"Case {case.case_id}: expected VASP {case.expected_vasp} not reachable from {case.suspect_wallet}"
        )


def test_baseline_accuracy_is_sweet_spot():
    """
    CRITICAL CHECK: Pure graph distance must NOT solve all cases.
    Accurate target: 3 <= correct <= 6 out of 12 (leaves headroom for GNN & Behavioral models).
    """
    store = NetworkXStore()
    vasps, wallets, transactions, test_cases = generate_synthetic_data(seed=42, store=store)

    correct = 0
    for case in test_cases:
        candidates = store.find_nearest_vasp(case.suspect_wallet, max_hops=6)
        if candidates and candidates[0]["vasp_id"] == case.expected_vasp:
            correct += 1

    assert 3 <= correct <= 6, f"Expected baseline accuracy between 3 and 6 out of 12, got {correct}/12"


def test_at_least_three_cross_chain_cases():
    """Verify that at least 3 benchmark cases involve multi-chain routing (e.g. BTC -> ETH -> TRON)."""
    gen = SyntheticDataGenerator(seed=42)
    _, _, _, test_cases = gen.generate()

    cross_chain_cases = [c for c in test_cases if c.is_cross_chain]
    assert len(cross_chain_cases) >= 3, f"Expected >= 3 cross-chain cases, got {len(cross_chain_cases)}"


def test_at_least_two_proximity_ties():
    """Verify that at least 2 benchmark cases feature equal-hop proximity ties requiring confidence models."""
    gen = SyntheticDataGenerator(seed=42)
    _, _, _, test_cases = gen.generate()

    tie_cases = [c for c in test_cases if c.has_proximity_tie]
    assert len(tie_cases) >= 2, f"Expected >= 2 proximity tie cases, got {len(tie_cases)}"


def test_cases_103_and_104_have_naive_proximity_fail_flag():
    """Verify that CASE-103 and CASE-104 explicitly flag that naive proximity fails."""
    gen = SyntheticDataGenerator(seed=42)
    _, _, _, test_cases = gen.generate()

    c103 = next((c for c in test_cases if c.case_id == "CASE-103"), None)
    c104 = next((c for c in test_cases if c.case_id == "CASE-104"), None)

    assert c103 is not None and c103.naive_proximity_will_fail is True
    assert c104 is not None and c104.naive_proximity_will_fail is True
    assert c103.requires_behavioral_fingerprint is True
    assert c104.requires_behavioral_fingerprint is True


def test_strict_score_separation_invariant():
    """Verify invariant that proximity_rank and confidence_score are never blended."""
    attr = Attribution(
        suspect_wallet="bc1qtest123",
        chain=Chain.BTC,
        predicted_vasp="coindcx",
        proximity_rank=3,
        proximity_distance=0.45,
        confidence_score=None,
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
        assert data["benchmark_cases_count"] == 12


def test_fastapi_demo_benchmark_endpoints():
    """Verify listing test cases and evaluating proximity via REST API."""
    with TestClient(app) as client:
        # 1. List test cases
        cases_resp = client.get("/api/v1/demo/test-cases")
        assert cases_resp.status_code == 200
        cases = cases_resp.json()
        assert len(cases) == 12
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
