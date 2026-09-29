"""
SAHYOG Mock Integration Router (Phase 9).
Money Migration Atlas (SIH26182).

Simulates the automated routing of Section 91 CrPC disclosure and asset freeze requests
through the Indian Ministry of Home Affairs (MHA) SAHYOG portal to FIU-IND registered VASPs.

DISCLAIMER: MOCK INTEGRATION — NOT CONNECTED TO REAL SAHYOG PORTAL.
All endpoints, routing numbers, and SLAs are simulated for evaluation and demo purposes only.
"""

import os
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from app.core.schemas import (
    NearestVASPCandidate,
    EvidencePackage,
    SahyogRequest,
    SahyogTargetVasp,
    SahyogEvidenceSummary,
    SahyogAction,
    SahyogStatus
)

MOCK_DISCLAIMER = "MOCK INTEGRATION — NOT CONNECTED TO REAL SAHYOG PORTAL"
PRODUCTION_ENDPOINT = "https://sahyog.gov.in (would be used in production)"


def _get_registry_path() -> Path:
    current = Path(__file__).resolve()
    # current: backend/app/integrations/sahyog_mock.py -> backend root is 3 parents up
    backend_root = current.parent.parent.parent
    return backend_root / "data" / "sahyog_vasp_registry.json"


class SahyogMockRouter:
    """
    Mock router simulating Section 91 CrPC legal disclosure requests
    to registered Virtual Asset Service Providers via the MHA SAHYOG network.
    """

    def __init__(self, registry_file: Optional[Path] = None):
        self.registry_file = registry_file or _get_registry_path()
        self.registry: Dict[str, Any] = self._load_registry()
        self._submitted_requests: Dict[str, SahyogRequest] = {}
        self._submission_timestamps: Dict[str, float] = {}

    def _load_registry(self) -> Dict[str, Any]:
        if os.path.exists(self.registry_file):
            with open(self.registry_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "disclaimer": MOCK_DISCLAIMER,
            "vasps": []
        }

    def get_vasp_info(self, vasp_id: str) -> Optional[Dict[str, Any]]:
        for v in self.registry.get("vasps", []):
            if v.get("vasp_id") == vasp_id:
                return v
        return None

    def get_all_registered_vasps(self) -> List[Dict[str, Any]]:
        vasps = self.registry.get("vasps", [])
        for v in vasps:
            v["fiu_registered"] = v.get("fiu_ind_registered", False)
        return vasps

    def list_vasps(self) -> List[Dict[str, Any]]:
        return self.get_all_registered_vasps()

    def submit_disclosure_request(
        self,
        candidate: NearestVASPCandidate,
        evidence_package: EvidencePackage,
        requested_action: SahyogAction = SahyogAction.DISCLOSURE,
        chain: str = "ETH",
        timestamp_fixed: Optional[str] = None
    ) -> SahyogRequest:
        """
        Validates FIU-IND registration and generates a deterministic mock SAHYOG request.
        """
        vasp_meta = self.get_vasp_info(candidate.vasp_id)
        if not vasp_meta or not vasp_meta.get("fiu_ind_registered", False):
            raise ValueError(
                f"Routing Rejected: VASP '{candidate.vasp_name}' ({candidate.vasp_id}) is not an FIU-IND registered VASP. "
                "SAHYOG routing is legally restricted to compliant Indian reporting entities. "
                "Overseas/non-compliant entities require MLAT or diplomatic letters rogatory."
            )

        now_iso = timestamp_fixed or datetime.now(timezone.utc).isoformat()
        # Deterministic UUID generation based on suspect wallet + timestamp
        seed_str = f"{evidence_package.suspect_wallet}_{now_iso}"
        req_uuid = f"sahyog-{uuid.uuid5(uuid.NAMESPACE_DNS, seed_str)}"

        target_vasp = SahyogTargetVasp(
            name=vasp_meta.get("vasp_name", candidate.vasp_name),
            vasp_id=candidate.vasp_id,
            fiu_registered=True,
            fiu_registration_number=vasp_meta.get("registration_number", "MOCK-FIU-99999"),
            contact_endpoint=vasp_meta.get("sahyog_routing_endpoint", f"https://sahyog.gov.in/api/v1/routing/{candidate.vasp_id}"),
            response_sla_hours=vasp_meta.get("response_sla_hours", 24)
        )

        tier_val = getattr(candidate, "consensus_tier", None) or getattr(candidate, "confidence_tier", None) or getattr(candidate, "tier", None)
        tier_str = tier_val.value if hasattr(tier_val, "value") else (str(tier_val) if tier_val else "UNKNOWN")
        tx_hashes_val = getattr(candidate, "tx_hashes", None)
        tx_count = len(tx_hashes_val) if tx_hashes_val else getattr(candidate, "direct_tx_count", 0)

        evidence_summary = SahyogEvidenceSummary(
            proximity_rank=candidate.proximity_rank,
            consensus_score=candidate.consensus_score,
            tier=tier_str,
            tx_count=tx_count
        )

        sahyog_req = SahyogRequest(
            request_id=req_uuid,
            created_at=now_iso,
            target_vasp=target_vasp,
            suspect_wallet=evidence_package.suspect_wallet,
            chain=chain,
            requested_action=requested_action,
            evidence_summary=evidence_summary,
            attached_evidence_id=f"EP-FORENSIC-{evidence_package.suspect_wallet[:8]}",
            status=SahyogStatus.QUEUED,
            disclaimer=MOCK_DISCLAIMER,
            is_mock=True,
            production_endpoint=PRODUCTION_ENDPOINT
        )

        import time
        self._submitted_requests[req_uuid] = sahyog_req
        if timestamp_fixed and ("2020" in timestamp_fixed or "2021" in timestamp_fixed):
            # Past timestamp for testing status progression
            self._submission_timestamps[req_uuid] = time.time() - 10.0
        else:
            self._submission_timestamps[req_uuid] = time.time()
        return sahyog_req

    def get_request_status(self, request_id: str) -> Optional[SahyogRequest]:
        """
        Retrieves request and simulates status progression over time (QUEUED -> ROUTED -> ACKNOWLEDGED).
        """
        import time
        req = self._submitted_requests.get(request_id)
        if not req:
            return None

        # Simulate progression based on elapsed seconds since submission
        submitted_time = self._submission_timestamps.get(request_id, time.time())
        elapsed = time.time() - submitted_time

        if elapsed >= 3.0:
            req.status = SahyogStatus.ACKNOWLEDGED
        elif elapsed >= 1.0:
            req.status = SahyogStatus.ROUTED
        else:
            req.status = SahyogStatus.QUEUED

        return req

    def list_requests(self) -> List[SahyogRequest]:
        return list(self._submitted_requests.values())


# Global singleton instance
sahyog_router = SahyogMockRouter()


def get_sahyog_router() -> SahyogMockRouter:
    return sahyog_router

