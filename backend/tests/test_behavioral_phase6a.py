"""
Unit and integration tests for Phase 6a: Behavioral Fingerprinting.
Validates 64-dim vector extraction, similarity metrics, independent behavioral attribution,
and 3-way multi-model consensus.
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient

from app.core.schemas import (
    Chain,
    BehavioralFingerprint,
    ConsensusTier,
    NearestVASPCandidate,
    ConfidenceTier
)
from app.data.synthetic_generator import generate_synthetic_data
from app.graph.networkx_store import NetworkXStore
from app.behavioral.fingerprint import (
    extract_fingerprint,
    extract_fingerprint_safe
)
from app.behavioral.similarity import (
    cosine_similarity,
    euclidean_distance,
    manhattan_distance,
    find_similar_wallets
)
from app.behavioral.attributor import BehavioralAttributor
from app.ml.consensus import ConsensusScorer
from app.main import app


@pytest.fixture(scope="module")
def populated_store():
    vasps, wallets, txs, cases = generate_synthetic_data(seed=42)
    store = NetworkXStore()
    for v in vasps:
        store.add_vasp(v)
    for w in wallets:
        store.add_wallet(w)
    for t in txs:
        store.add_transaction(t)
    return store, cases


def test_fingerprint_extraction_returns_valid_64_dim_vector(populated_store):
    store, _ = populated_store
    # Find a wallet with >= 3 outgoing transactions
    target_wallet = None
    for addr in store.wallets.keys():
        out_txs = list(store.graph.out_edges(addr))
        if len(out_txs) >= 3:
            target_wallet = addr
            break

    assert target_wallet is not None
    fp = extract_fingerprint(store, target_wallet, min_txs=3)
    assert isinstance(fp, BehavioralFingerprint)
    vec = fp.to_vector()
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (64,)
    assert not np.isnan(vec).any()
    assert len(fp.timing_vector) == 16
    assert len(fp.gas_vector) == 16
    assert len(fp.amount_vector) == 16
    assert len(fp.chain_vector) == 16


def test_fingerprint_maturity_check(populated_store):
    store, cases = populated_store
    # Benchmark suspect wallets have only 1 out-tx directly from anchor
    suspect = cases[0].suspect_wallet
    fp = extract_fingerprint(store, suspect, min_txs=3, direction="out")
    assert fp.summary.get("confidence") == "INSUFFICIENT_DATA"
    assert fp.tx_count < 3
    # With strict=True, it raises ValueError
    with pytest.raises(ValueError, match="minimum 3 required"):
        extract_fingerprint(store, suspect, min_txs=3, strict=True, direction="out")


def test_cosine_similarity_is_symmetric():
    np.random.seed(42)
    v1 = np.random.uniform(0.0, 1.0, 64).astype(np.float32)
    v2 = np.random.uniform(0.0, 1.0, 64).astype(np.float32)

    sim12 = cosine_similarity(v1, v2)
    sim21 = cosine_similarity(v2, v1)
    assert abs(sim12 - sim21) < 1e-6


def test_cosine_similarity_range_0_to_1():
    v1 = np.ones(64, dtype=np.float32)
    v2 = np.ones(64, dtype=np.float32)
    assert abs(cosine_similarity(v1, v2) - 1.0) < 1e-5

    v_random = np.random.uniform(-1.0, 1.0, 64).astype(np.float32)
    sim = cosine_similarity(v1, v_random)
    assert 0.0 <= sim <= 1.0


def test_find_similar_wallets_returns_top_k(populated_store):
    store, _ = populated_store
    wallets_with_tx = [w for w in store.wallets.keys() if len(list(store.graph.out_edges(w))) >= 3]
    assert len(wallets_with_tx) >= 10
    target = wallets_with_tx[0]

    top_5 = find_similar_wallets(target, store, top_k=5)
    assert len(top_5) == 5
    for addr, score in top_5:
        assert addr != target
        assert 0.0 <= score <= 1.0
    # Assert descending order
    scores = [s for _, s in top_5]
    assert scores == sorted(scores, reverse=True)


def test_behavioral_confidence_insufficient_for_benchmark_suspects(populated_store):
    store, cases = populated_store
    att = BehavioralAttributor(store)
    for c in cases:
        score, wallets, sim_scores = att.compute_behavioral_score(
            suspect_wallet=c.suspect_wallet,
            candidate_vasp_id=c.expected_vasp,
            path=[c.suspect_wallet]
        )
        # All 12 benchmark suspect wallets have <3 outgoing txs -> returns None
        assert score is None
        assert wallets == []
        assert sim_scores == []


def test_behavioral_confidence_on_mature_wallets(populated_store):
    store, _ = populated_store
    att = BehavioralAttributor(store)
    mature_wallets = [w for w in store.wallets.keys() if len(list(store.graph.out_edges(w))) >= 3]
    assert len(mature_wallets) > 0
    score, wallets, sim_scores = att.compute_behavioral_score(
        suspect_wallet=mature_wallets[0],
        candidate_vasp_id="wazirx"
    )
    if score is not None:
        assert 0.0 <= score <= 1.0


def test_behavioral_api_endpoints_return_valid_json():
    with TestClient(app) as client:
        # 1. Health check
        h_resp = client.get("/health")
        assert h_resp.status_code == 200

        # 2. Get demo benchmark cases
        b_resp = client.get("/api/v1/demo/test-cases")
        assert b_resp.status_code == 200
        cases = b_resp.json()
        assert len(cases) >= 12
        case_002 = next(c for c in cases if c["case_id"] == "CASE-002")
        suspect = case_002["suspect_wallet"]
        chain = case_002["chain"]

        # 3. GET /fingerprint
        fp_resp = client.get(f"/api/v1/behavioral/fingerprint/{chain}/{suspect}")
        assert fp_resp.status_code == 200
        fp_data = fp_resp.json()
        assert "timing_vector" in fp_data
        assert "gas_vector" in fp_data
        assert "amount_vector" in fp_data
        assert "chain_vector" in fp_data
        assert len(fp_data["timing_vector"]) == 16

        # 4. GET /similar
        sim_resp = client.get(f"/api/v1/behavioral/similar/{chain}/{suspect}?top_k=5")
        assert sim_resp.status_code == 200
        sim_data = sim_resp.json()
        assert "similar_wallets" in sim_data
        assert len(sim_data["similar_wallets"]) <= 5

        # 5. POST /predict
        pred_resp = client.post(f"/api/v1/behavioral/predict/{chain}/{suspect}")
        assert pred_resp.status_code == 200
        pred_data = pred_resp.json()
        assert "candidates" in pred_data
        assert len(pred_data["candidates"]) > 0
        top_cand = pred_data["candidates"][0]
        assert "behavioral_confidence_score" in top_cand
        assert top_cand["never_blended"] is True


def test_consensus_now_includes_behavioral(populated_store):
    store, cases = populated_store
    scorer = ConsensusScorer()
    # Benchmark suspect wallet has < 3 txs -> behavioral is None (Fix 2)
    res = scorer.predict(cases[0].suspect_wallet, store, include_behavioral=True)
    assert len(res) > 0
    top = res[0]
    assert top.behavioral_confidence_score is None
    assert top.confidence_score is not None
    assert top.gnn_confidence_score is not None
    assert top.consensus_score is not None
    assert top.consensus_score == max(top.confidence_score or 0.0, top.gnn_confidence_score or 0.0)


def test_consensus_score_is_max_of_three_signals(populated_store):
    store, cases = populated_store
    scorer = ConsensusScorer()
    for c in cases:
        candidates = scorer.predict(c.suspect_wallet, store, include_behavioral=True)
        for cand in candidates:
            expected_max = round(max(
                cand.confidence_score or 0.0,
                cand.gnn_confidence_score or 0.0,
                cand.behavioral_confidence_score or 0.0
            ), 4)
            assert abs(cand.consensus_score - expected_max) < 1e-4


def test_never_blended_invariant_still_holds(populated_store):
    store, cases = populated_store
    scorer = ConsensusScorer()
    for c in cases:
        candidates = scorer.predict(c.suspect_wallet, store, include_behavioral=True)
        for cand in candidates:
            assert cand.never_blended is True
            # Invariant check: proximity_rank remains int hops, confidence scores remain unblended floats
            assert isinstance(cand.proximity_rank, int)
            assert cand.proximity_rank >= 1


def test_behavioral_only_tier_assigned_correctly():
    scorer = ConsensusScorer()
    # When behavioral is high (>= 0.60) and both XGB and GNN are low (< 0.60)
    tier = scorer.determine_consensus_tier(xgb_conf=0.15, gnn_conf=0.35, behavioral_conf=0.85)
    assert tier == ConsensusTier.BEHAVIORAL_ONLY


def test_case_104_must_not_be_confirmed(populated_store):
    store, cases = populated_store
    scorer = ConsensusScorer()
    c104 = [c for c in cases if c.case_id == "CASE-104"][0]
    cands = scorer.predict(c104.suspect_wallet, store, include_behavioral=True)
    top_cand = cands[0]
    # Under Fix 1 and Fix 2: CASE-104 must NOT be CONFIRMED
    assert top_cand.consensus_tier != ConsensusTier.CONFIRMED


def test_tier_logic_requires_no_contradiction_for_confirmed():
    scorer = ConsensusScorer()
    # 1. Contradiction from another candidate scoring >= 0.60 -> AMBIGUOUS
    tier = scorer.determine_consensus_tier(
        xgb_conf=0.70,
        gnn_conf=0.72,
        behavioral_conf=None,
        other_candidates_max_conf=0.75
    )
    assert tier == ConsensusTier.AMBIGUOUS

    # 2. No contradiction -> CONFIRMED
    tier_no_contra = scorer.determine_consensus_tier(
        xgb_conf=0.70,
        gnn_conf=0.72,
        behavioral_conf=None,
        other_candidates_max_conf=0.30
    )
    assert tier_no_contra == ConsensusTier.CONFIRMED

    # 3. Severe internal disagreement (diff >= 0.35) -> AMBIGUOUS
    tier_internal = scorer.determine_consensus_tier(
        xgb_conf=0.85,
        gnn_conf=0.40,
        behavioral_conf=None,
        other_candidates_max_conf=0.20
    )
    assert tier_internal == ConsensusTier.AMBIGUOUS


def test_behavioral_none_excluded_from_consensus_and_tier(populated_store):
    store, cases = populated_store
    scorer = ConsensusScorer()
    # For any benchmark suspect wallet with < 3 outgoing txs
    suspect = cases[0].suspect_wallet
    cands = scorer.predict(suspect, store, include_behavioral=True)
    for c in cands:
        assert c.behavioral_confidence_score is None
        # consensus_score must be max(xgb, gnn) when behavioral is None
        active = [s for s in [c.confidence_score, c.gnn_confidence_score] if s is not None]
        expected_score = round(max(active), 4) if active else 0.0
        assert c.consensus_score == expected_score
        assert c.consensus_tier != ConsensusTier.BEHAVIORAL_ONLY
