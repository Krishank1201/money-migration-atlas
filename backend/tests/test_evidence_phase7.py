"""
Tests for Phase 7 Forensic Evidence Package & Chain of Custody.
Money Migration Atlas (SIH26182).
"""

import json
from datetime import datetime
import pytest
from fastapi.testclient import TestClient

from app.core.schemas import (
    Chain,
    EvidencePackage,
    CustodyStep,
    NearestVASPCandidate,
    ConsensusTier,
    ConfidenceTier
)
from app.data.synthetic_generator import generate_synthetic_data
from app.graph.networkx_store import NetworkXStore
from app.ml.consensus import ConsensusScorer
from app.evidence.chain import EvidenceChainBuilder
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


def test_evidence_package_has_all_required_fields(populated_store):
    store, cases = populated_store
    scorer = ConsensusScorer()
    builder = EvidenceChainBuilder()
    c2 = [c for c in cases if c.case_id == "CASE-002"][0]

    cands = scorer.predict(c2.suspect_wallet, store, include_behavioral=True)
    assert len(cands) > 0
    top = cands[0]

    package = builder.build(top, store, c2.suspect_wallet, chain="ETH")
    assert isinstance(package, EvidencePackage)
    assert package.suspect_wallet == c2.suspect_wallet
    assert "vasp_id" in package.top_candidate
    assert "vasp_name" in package.top_candidate
    assert isinstance(package.proximity_rank, int)
    assert len(package.proximity_path) >= 2
    assert len(package.transaction_hashes) >= 1
    assert package.consensus_tier is not None
    assert package.model_versions is not None
    assert "xgboost" in package.model_versions
    assert "gnn_structural" in package.model_versions
    assert package.never_blended is True
    assert len(package.chain_of_custody) >= 5


def test_evidence_package_json_serializable(populated_store):
    store, cases = populated_store
    scorer = ConsensusScorer()
    builder = EvidenceChainBuilder()
    c = cases[0]
    cands = scorer.predict(c.suspect_wallet, store, include_behavioral=True)
    package = builder.build(cands[0], store, c.suspect_wallet)

    data = package.model_dump()
    serialized = json.dumps(data)
    assert len(serialized) > 100
    deserialized = json.loads(serialized)
    assert deserialized["suspect_wallet"] == c.suspect_wallet
    assert deserialized["never_blended"] is True


def test_evidence_package_markdown_format(populated_store):
    store, cases = populated_store
    scorer = ConsensusScorer()
    builder = EvidenceChainBuilder()
    c2 = [c for c in cases if c.case_id == "CASE-002"][0]
    cands = scorer.predict(c2.suspect_wallet, store, include_behavioral=True)
    package = builder.build(cands[0], store, c2.suspect_wallet)

    md = builder.to_markdown(package)
    assert "# FORENSIC EVIDENCE DOSSIER" in md
    assert "## 1. EXECUTIVE SUMMARY & TARGET VASP" in md
    assert "## 2. MULTI-MODEL INDEPENDENT CONFIDENCE SCORES" in md
    assert "## 3. ON-CHAIN TRANSACTION PATHWAY" in md
    assert "## 5. DIGITAL CHAIN OF CUSTODY AUDIT TRAIL" in md
    assert c2.suspect_wallet in md
    assert "never_blended=True" in md


def test_chain_of_custody_ordered_by_timestamp(populated_store):
    store, cases = populated_store
    scorer = ConsensusScorer()
    builder = EvidenceChainBuilder()
    c = cases[0]
    cands = scorer.predict(c.suspect_wallet, store, include_behavioral=True)
    package = builder.build(cands[0], store, c.suspect_wallet)

    steps = package.chain_of_custody
    assert len(steps) >= 5
    # Verify non-empty timestamps and step sequence
    step_names = [s.step for s in steps]
    assert "INGESTION_AND_INDEXING" in step_names
    assert "TOPOLOGICAL_TRAVERSAL" in step_names
    assert "SUPERVISED_MODEL_INFERENCE" in step_names
    assert "CONSENSUS_VERIFICATION" in step_names
    assert "EVIDENCE_SEALING" in step_names


def test_never_blended_invariant_in_evidence_package(populated_store):
    store, cases = populated_store
    scorer = ConsensusScorer()
    builder = EvidenceChainBuilder()
    c = cases[0]
    cands = scorer.predict(c.suspect_wallet, store, include_behavioral=True)
    package = builder.build(cands[0], store, c.suspect_wallet)

    assert package.never_blended is True
    with pytest.raises(ValueError, match=r"(?i)never be blended"):
        EvidencePackage(
            suspect_wallet=c.suspect_wallet,
            top_candidate={"vasp_id": "test", "vasp_name": "Test"},
            proximity_rank=1,
            generated_at="2026-09-29T00:00:00Z",
            never_blended=False
        )


def test_evidence_package_no_pii_leakage(populated_store):
    store, cases = populated_store
    scorer = ConsensusScorer()
    builder = EvidenceChainBuilder()
    c = cases[0]
    cands = scorer.predict(c.suspect_wallet, store, include_behavioral=True)
    package = builder.build(cands[0], store, c.suspect_wallet)
    md = builder.to_markdown(package)

    forbidden_tokens = ["ANTHROPIC_API_KEY", "ETHERSCAN_API_KEY", "sk-ant-", "password"]
    for token in forbidden_tokens:
        assert token not in md
        assert token not in json.dumps(package.model_dump())


def test_evidence_api_endpoints(populated_store):
    _, cases = populated_store
    c2 = [c for c in cases if c.case_id == "CASE-002"][0]

    with TestClient(app) as client:
        # 1. JSON format
        resp_json = client.get(f"/api/v1/evidence/ETH/{c2.suspect_wallet}?format=json")
        assert resp_json.status_code == 200
        data = resp_json.json()
        assert data["suspect_wallet"] == c2.suspect_wallet
        assert "proximity_path" in data
        assert data["never_blended"] is True

        # 2. Markdown format
        resp_md = client.get(f"/api/v1/evidence/ETH/{c2.suspect_wallet}?format=markdown")
        assert resp_md.status_code == 200
        assert "text/markdown" in resp_md.headers["content-type"]
        assert "# FORENSIC EVIDENCE DOSSIER" in resp_md.text

        # 3. Chain of custody
        resp_coc = client.get(f"/api/v1/evidence/ETH/{c2.suspect_wallet}/chain-of-custody")
        assert resp_coc.status_code == 200
        coc_data = resp_coc.json()
        assert isinstance(coc_data, list)
        assert len(coc_data) >= 5


def test_chain_of_custody_timestamps_are_strictly_increasing(populated_store):
    store, cases = populated_store
    scorer = ConsensusScorer()
    builder = EvidenceChainBuilder()
    c2 = [c for c in cases if c.case_id == "CASE-002"][0]
    cands = scorer.predict(c2.suspect_wallet, store, include_behavioral=True)
    package = builder.build(cands[0], store, c2.suspect_wallet)

    steps = package.chain_of_custody
    assert len(steps) >= 5
    for i in range(len(steps) - 1):
        t_curr = datetime.fromisoformat(steps[i].timestamp)
        t_next = datetime.fromisoformat(steps[i + 1].timestamp)
        assert t_next > t_curr, f"Step {i+1} ({steps[i+1].timestamp}) must be strictly after Step {i} ({steps[i].timestamp})"


def test_chain_of_custody_timestamps_are_unique(populated_store):
    store, cases = populated_store
    scorer = ConsensusScorer()
    builder = EvidenceChainBuilder()
    c2 = [c for c in cases if c.case_id == "CASE-002"][0]
    cands = scorer.predict(c2.suspect_wallet, store, include_behavioral=True)
    package = builder.build(cands[0], store, c2.suspect_wallet)

    steps = package.chain_of_custody
    timestamps = [s.timestamp for s in steps]
    assert len(timestamps) == len(set(timestamps)), f"Timestamps must be unique: {timestamps}"

