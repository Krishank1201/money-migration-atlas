"""
Tests for Phase 7 Agentic AI & Plain-Language Explanation Layer.
Money Migration Atlas (SIH26182).
"""

import pytest
from fastapi.testclient import TestClient

from app.core.schemas import (
    Chain,
    NearestVASPCandidate,
    ConsensusTier,
    RecommendedAction,
    InvestigationReport,
    ConfidenceTier
)
from app.data.synthetic_generator import generate_synthetic_data
from app.graph.networkx_store import NetworkXStore
from app.agentic.llm_client import LLMClient
from app.agentic.counterfactual import generate_counterfactuals
from app.agentic.agent import ForensicAgent
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


def test_llm_client_falls_back_without_api_key():
    client = LLMClient(api_key=None)
    data = {
        "vasp_name": "CoinDCX",
        "proximity_rank": 3,
        "confidence_score": 0.47,
        "gnn_confidence_score": 0.46,
        "behavioral_confidence_score": None,
        "consensus_score": 0.47,
        "consensus_tier": "UNCERTAIN"
    }
    exp, source = client.generate_explanation(data)
    assert source == "template_fallback"
    assert "CoinDCX" in exp
    assert "3 hops" in exp


def test_llm_client_returns_data_source_flag():
    client = LLMClient(api_key=None)
    data = {"vasp_name": "WazirX", "proximity_rank": 2, "consensus_score": 0.85, "consensus_tier": "CONFIRMED"}
    _, source = client.generate_explanation(data)
    assert source in ("llm", "template_fallback")


def test_template_fallback_produces_valid_explanation():
    client = LLMClient(api_key=None)
    data = {
        "vasp_name": "WazirX",
        "proximity_rank": 2,
        "confidence_score": 0.85,
        "gnn_confidence_score": 0.80,
        "behavioral_confidence_score": None,
        "consensus_score": 0.85,
        "consensus_tier": "CONFIRMED"
    }
    context = {
        "counterfactuals": ["If the path had passed through a privacy mixer, topological penalty would have suppressed confidence."]
    }
    exp, source = client.generate_explanation(data, context=context)
    assert source == "template_fallback"
    # Verify >= 3 sentences (at least 3 period terminators)
    sentences = [s.strip() for s in exp.split(".") if len(s.strip()) > 5]
    assert len(sentences) >= 3
    # Check that required scores & entities are mentioned
    assert "WazirX" in exp
    assert "2 hops" in exp
    assert "0.85" in exp
    assert "CONFIRMED" in exp
    assert "INSUFFICIENT_DATA" in exp
    assert "unblended" in exp.lower() or "never blended" in exp.lower()


def test_counterfactual_generator_returns_max_3(populated_store):
    store, cases = populated_store
    c = cases[0]
    # Build candidate
    cand = NearestVASPCandidate(
        vasp_id="coindcx",
        vasp_name="CoinDCX",
        proximity_rank=3,
        path=[c.suspect_wallet, "0xnode1", "0xnode2"],
        confidence_score=0.47,
        gnn_confidence_score=0.46,
        consensus_score=0.47,
        consensus_tier=ConsensusTier.UNCERTAIN,
        never_blended=True
    )
    cfs = generate_counterfactuals(cand, [cand], store=store)
    assert 1 <= len(cfs) <= 3


def test_counterfactuals_are_deterministic(populated_store):
    store, cases = populated_store
    c = cases[1]
    cand = NearestVASPCandidate(
        vasp_id="wazirx",
        vasp_name="WazirX",
        proximity_rank=3,
        path=[c.suspect_wallet, "0xmid1", "0xmid2"],
        confidence_score=0.44,
        gnn_confidence_score=0.86,
        consensus_score=0.86,
        consensus_tier=ConsensusTier.AMBIGUOUS,
        never_blended=True
    )
    run1 = generate_counterfactuals(cand, [cand], store=store)
    run2 = generate_counterfactuals(cand, [cand], store=store)
    assert run1 == run2


def test_recommended_action_logic():
    agent = ForensicAgent()

    # Branch 1: CONFIRMED + FIU-registered -> SEND_DISCLOSURE_REQUEST
    cand_fiu_confirmed = NearestVASPCandidate(
        vasp_id="wazirx",
        vasp_name="WazirX",
        proximity_rank=2,
        confidence_score=0.85,
        gnn_confidence_score=0.80,
        consensus_score=0.85,
        consensus_tier=ConsensusTier.CONFIRMED,
        fiu_ind_registered=True,
        never_blended=True
    )
    act1 = agent.determine_recommended_action([cand_fiu_confirmed])
    assert act1 == RecommendedAction.SEND_DISCLOSURE_REQUEST

    # Branch 2: AMBIGUOUS -> HUMAN_REVIEW_REQUIRED
    cand_ambiguous = NearestVASPCandidate(
        vasp_id="wazirx",
        vasp_name="WazirX",
        proximity_rank=2,
        confidence_score=0.85,
        gnn_confidence_score=0.40,
        consensus_score=0.85,
        consensus_tier=ConsensusTier.AMBIGUOUS,
        fiu_ind_registered=True,
        never_blended=True
    )
    act2 = agent.determine_recommended_action([cand_ambiguous])
    assert act2 == RecommendedAction.HUMAN_REVIEW_REQUIRED

    # Branch 2b: SINGLE_MODEL -> HUMAN_REVIEW_REQUIRED
    cand_single = NearestVASPCandidate(
        vasp_id="zebpay",
        vasp_name="ZebPay",
        proximity_rank=3,
        confidence_score=0.35,
        gnn_confidence_score=0.65,
        consensus_score=0.65,
        consensus_tier=ConsensusTier.SINGLE_MODEL,
        fiu_ind_registered=True,
        never_blended=True
    )
    act2b = agent.determine_recommended_action([cand_single])
    assert act2b == RecommendedAction.HUMAN_REVIEW_REQUIRED

    # Branch 3: All UNCERTAIN -> INSUFFICIENT_SIGNAL
    cand_uncertain = NearestVASPCandidate(
        vasp_id="mudrex",
        vasp_name="Mudrex",
        proximity_rank=4,
        confidence_score=0.20,
        gnn_confidence_score=0.22,
        consensus_score=0.22,
        consensus_tier=ConsensusTier.UNCERTAIN,
        never_blended=True
    )
    act3 = agent.determine_recommended_action([cand_uncertain])
    assert act3 == RecommendedAction.INSUFFICIENT_SIGNAL

    # Branch 3b: Top score < 0.30 -> INSUFFICIENT_SIGNAL
    cand_low = NearestVASPCandidate(
        vasp_id="mudrex",
        vasp_name="Mudrex",
        proximity_rank=3,
        confidence_score=0.15,
        gnn_confidence_score=0.10,
        consensus_score=0.15,
        consensus_tier=ConsensusTier.SINGLE_MODEL,
        never_blended=True
    )
    act3b = agent.determine_recommended_action([cand_low])
    assert act3b == RecommendedAction.INSUFFICIENT_SIGNAL


def test_investigation_report_has_all_fields(populated_store):
    store, cases = populated_store
    agent = ForensicAgent()
    c2 = [c for c in cases if c.case_id == "CASE-002"][0]

    report = agent.investigate(c2.suspect_wallet, store, chain=Chain.ETH)
    assert isinstance(report, InvestigationReport)
    assert report.suspect_wallet == c2.suspect_wallet
    assert report.chain == Chain.ETH
    assert len(report.plain_language_summary) > 50
    assert report.data_source in ("llm", "template_fallback")
    assert len(report.top_3_candidates) > 0
    assert len(report.counterfactuals) > 0
    assert report.evidence_package is not None
    assert report.recommended_action == RecommendedAction.HUMAN_REVIEW_REQUIRED
    assert report.never_blended is True


def test_agentic_investigate_api_endpoint(populated_store):
    _, cases = populated_store
    c2 = [c for c in cases if c.case_id == "CASE-002"][0]

    with TestClient(app) as client:
        resp = client.post(f"/api/v1/agentic/investigate/ETH/{c2.suspect_wallet}?max_hops=6")
        assert resp.status_code == 200
        data = resp.json()
        assert data["suspect_wallet"] == c2.suspect_wallet
        assert "plain_language_summary" in data
        assert "data_source" in data
        assert "top_3_candidates" in data
        assert "evidence_package" in data
        assert data["recommended_action"] == "HUMAN_REVIEW_REQUIRED"
        assert data["never_blended"] is True


def test_counterfactual_never_names_suspect_as_mixer(populated_store):
    store, cases = populated_store
    c104 = [c for c in cases if c.case_id == "CASE-104"][0]
    scorer = ConsensusScorer()
    cands = scorer.predict(c104.suspect_wallet, store, include_behavioral=True)
    cfs = generate_counterfactuals(cands[0], cands, store=store)
    for cf in cfs:
        text = cf.text if hasattr(cf, "text") else cf["text"]
        assert c104.suspect_wallet not in text
        assert c104.suspect_wallet[:10] not in text

    # Adversarial test: suspect wallet marked is_mixer=True
    adversarial_cand = NearestVASPCandidate(
        vasp_id="binance_offshore",
        vasp_name="Binance (Offshore)",
        proximity_rank=1,
        path=[c104.suspect_wallet, "0x7aaf4f02cd791871d819a7c354152aebca4fc844"],
        target_wallet="0x7aaf4f02cd791871d819a7c354152aebca4fc844",
        confidence_score=0.45,
        gnn_confidence_score=0.55,
        consensus_score=0.55,
        consensus_tier=ConsensusTier.UNCERTAIN,
        never_blended=True
    )
    suspect_w = store.get_wallet(c104.suspect_wallet)
    orig_mixer = suspect_w.is_mixer
    try:
        suspect_w.is_mixer = True
        adv_cfs = generate_counterfactuals(adversarial_cand, [adversarial_cand], store=store)
        for cf in adv_cfs:
            text = cf.text if hasattr(cf, "text") else cf["text"]
            assert c104.suspect_wallet not in text
            assert c104.suspect_wallet[:10] not in text
            assert "touched mixer" not in text
    finally:
        suspect_w.is_mixer = orig_mixer


def test_counterfactual_never_names_target_vasp_as_mixer(populated_store):
    store, cases = populated_store
    c2 = [c for c in cases if c.case_id == "CASE-002"][0]
    scorer = ConsensusScorer()
    cands = scorer.predict(c2.suspect_wallet, store, include_behavioral=True)
    top = cands[0]
    target_addr = top.target_wallet or top.path[-1]

    target_w = store.get_wallet(target_addr)
    orig_mixer = target_w.is_mixer
    try:
        target_w.is_mixer = True
        cfs = generate_counterfactuals(top, cands, store=store)
        for cf in cfs:
            text = cf.text if hasattr(cf, "text") else cf["text"]
            assert target_addr not in text
            assert target_addr[:10] not in text
    finally:
        target_w.is_mixer = orig_mixer


def test_counterfactuals_have_provenance_field(populated_store):
    store, cases = populated_store
    c2 = [c for c in cases if c.case_id == "CASE-002"][0]
    scorer = ConsensusScorer()
    cands = scorer.predict(c2.suspect_wallet, store, include_behavioral=True)
    cfs = generate_counterfactuals(cands[0], cands, store=store)
    assert len(cfs) > 0
    for cf in cfs:
        assert hasattr(cf, "provenance") or "provenance" in cf
        prov = cf.provenance if hasattr(cf, "provenance") else cf["provenance"]
        assert prov in ("recomputed", "estimated_shap", "qualitative")
        assert hasattr(cf, "method") or "method" in cf
        method = cf.method if hasattr(cf, "method") else cf["method"]
        assert len(method) > 0

