import time
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app
from app.core.schemas import NearestVASPCandidate, EvidencePackage, SahyogAction
from app.integrations.sahyog_mock import get_sahyog_router, SahyogMockRouter

client = TestClient(app)


from app.core.schemas import NearestVASPCandidate, EvidencePackage, SahyogAction, ConsensusTier, ConfidenceTier


def create_mock_candidate(vasp_id="coindcx", vasp_name="CoinDCX", score=0.88, tier="CONFIRMED"):
    c_tier = ConsensusTier.CONFIRMED if tier == "CONFIRMED" else ConsensusTier.AMBIGUOUS
    return NearestVASPCandidate(
        vasp_id=vasp_id,
        vasp_name=vasp_name,
        proximity_rank=1,
        distance=1.0,
        tx_hashes=["0xtx_a", "0xtx_b", "0xtx_c"],
        fiu_ind_registered=True,
        confidence_score=0.82,
        gnn_confidence_score=0.88,
        consensus_score=score,
        consensus_tier=c_tier,
    )


def create_mock_evidence(suspect_wallet="0x1111111111111111111111111111111111111111", target_vasp="coindcx"):
    return EvidencePackage(
        suspect_wallet=suspect_wallet,
        top_candidate={"vasp_id": target_vasp, "vasp_name": target_vasp.capitalize(), "fiu_ind_registered": True},
        proximity_rank=2,
        proximity_path=[suspect_wallet, "0xmid", "0xdest"],
        transaction_hashes=["0xtx1", "0xtx2"],
        chain_path=["ETH", "ETH"],
        consensus_score=0.88,
        generated_at="2026-09-29T12:00:00Z",
        never_blended=True,
        chain_of_custody=[]
    )


def test_vasp_registry_loads_9_vasps():
    router = SahyogMockRouter()
    registry = router.list_vasps()
    assert len(registry) == 9
    fiu_approved = [v for v in registry if v.get("fiu_registered")]
    assert len(fiu_approved) == 7  # 7 Indian registered VASPs
    offshore = [v for v in registry if not v.get("fiu_registered")]
    assert len(offshore) == 2  # Binance offshore & Bybit


def test_disclosure_request_created_for_fiu_vasp():
    router = SahyogMockRouter()
    candidate = create_mock_candidate(vasp_id="coindcx", vasp_name="CoinDCX")
    evidence = create_mock_evidence(target_vasp="coindcx")
    
    req = router.submit_disclosure_request(
        candidate=candidate,
        evidence_package=evidence,
        requested_action=SahyogAction.DISCLOSURE,
        timestamp_fixed="2026-09-29T12:00:00Z"
    )
    assert req.target_vasp.name == "CoinDCX"
    assert req.target_vasp.fiu_registered is True
    assert "FIU" in req.target_vasp.fiu_registration_number
    assert req.requested_action == SahyogAction.DISCLOSURE
    assert req.is_mock is True
    assert req.status in ["QUEUED", "ROUTED", "ACKNOWLEDGED"]


def test_disclosure_request_rejected_for_non_fiu_vasp():
    router = SahyogMockRouter()
    candidate = create_mock_candidate(vasp_id="bybit", vasp_name="Bybit")
    evidence = create_mock_evidence(target_vasp="bybit")
    
    with pytest.raises(ValueError) as excinfo:
        router.submit_disclosure_request(
            candidate=candidate,
            evidence_package=evidence,
            requested_action=SahyogAction.DISCLOSURE,
        )
    assert "not an FIU-IND registered VASP" in str(excinfo.value)


def test_request_id_is_deterministic_for_same_input():
    router = SahyogMockRouter()
    candidate = create_mock_candidate(vasp_id="wazirx", vasp_name="WazirX")
    evidence = create_mock_evidence(suspect_wallet="0xaaaa", target_vasp="wazirx")
    fixed_ts = "2026-09-29T10:00:00Z"

    req1 = router.submit_disclosure_request(
        candidate=candidate,
        evidence_package=evidence,
        requested_action=SahyogAction.FREEZE_AND_DISCLOSURE,
        timestamp_fixed=fixed_ts
    )
    req2 = router.submit_disclosure_request(
        candidate=candidate,
        evidence_package=evidence,
        requested_action=SahyogAction.FREEZE_AND_DISCLOSURE,
        timestamp_fixed=fixed_ts
    )
    assert req1.request_id == req2.request_id
    assert req1.request_id.startswith("sahyog-")


def test_status_progression_queued_to_acknowledged():
    router = SahyogMockRouter()
    candidate = create_mock_candidate(vasp_id="coindcx")
    evidence = create_mock_evidence(suspect_wallet="0xbbbb", target_vasp="coindcx")
    
    # Force created_at to 10 seconds ago
    old_ts = "2020-01-01T00:00:00Z"
    req = router.submit_disclosure_request(
        candidate=candidate,
        evidence_package=evidence,
        requested_action=SahyogAction.DISCLOSURE,
        timestamp_fixed=old_ts
    )
    # When polled, time elapsed > 3.0s, so status transitions to ACKNOWLEDGED
    polled = router.get_request_status(req.request_id)
    assert polled is not None
    assert polled.status == "ACKNOWLEDGED"


def test_sahyog_response_includes_mock_disclaimer():
    router = SahyogMockRouter()
    candidate = create_mock_candidate(vasp_id="mudrex")
    evidence = create_mock_evidence(target_vasp="mudrex")
    
    req = router.submit_disclosure_request(
        candidate=candidate,
        evidence_package=evidence,
        requested_action=SahyogAction.DISCLOSURE
    )
    assert req.is_mock is True
    assert "MOCK INTEGRATION" in req.disclaimer
    assert "sahyog.gov.in" in req.production_endpoint


def test_no_network_calls_made():
    """Ensure no socket or HTTP network calls are made during SAHYOG mock submission."""
    with patch("socket.socket.connect") as mock_connect, \
         patch("urllib.request.urlopen") as mock_urlopen:
        router = SahyogMockRouter()
        candidate = create_mock_candidate(vasp_id="zebpay")
        evidence = create_mock_evidence(target_vasp="zebpay")
        req = router.submit_disclosure_request(
            candidate=candidate,
            evidence_package=evidence,
            requested_action=SahyogAction.DISCLOSURE
        )
        assert req is not None
        mock_connect.assert_not_called()
        mock_urlopen.assert_not_called()
