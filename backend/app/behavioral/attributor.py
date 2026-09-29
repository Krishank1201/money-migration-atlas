"""
Behavioral Attributor (Phase 6a).

Attribution engine leveraging behavioral habit fingerprints to link
suspect laundering paths to candidate Virtual Asset Service Providers (VASPs).

Operates as a strictly independent third signal alongside topological proximity
and supervised ML models.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np

from app.core.schemas import NearestVASPCandidate, BehavioralFingerprint
from app.graph.networkx_store import NetworkXStore
from app.behavioral.fingerprint import extract_fingerprint, extract_fingerprint_safe
from app.behavioral.similarity import cosine_similarity

logger = logging.getLogger("mma.behavioral.attributor")


class BehavioralAttributor:
    """
    Computes behavioral confidence scores for candidate VASPs by comparing
    syndicate transaction signatures against VASP deposit sweeper & hot wallet behaviors.
    """

    def __init__(self, store: NetworkXStore):
        self.store = store
        self._fp_cache: Dict[str, Optional[BehavioralFingerprint]] = {}

    def _get_fp(
        self,
        address: str,
        path_context: Optional[List[str]] = None,
        direction: str = "out"
    ) -> Optional[BehavioralFingerprint]:
        cache_key = f"{address}_{direction}"
        if cache_key in self._fp_cache:
            return self._fp_cache[cache_key]
        fp = extract_fingerprint_safe(self.store, address, fallback_path=path_context, direction=direction)
        self._fp_cache[cache_key] = fp
        return fp

    def get_vasp_reference_wallets(self, vasp_id: str) -> List[str]:
        """Collects all known hot wallets and deposit sweepers for candidate VASP."""
        wallets: List[str] = []
        vasp = self.store.get_vasp(vasp_id)
        if vasp:
            wallets.extend(vasp.deposit_sweepers)
            wallets.extend(vasp.hot_wallets)

        # Also search labeled wallet registry
        for addr, w in self.store.wallets.items():
            if w.vasp_id == vasp_id and (w.is_vasp_deposit_sweeper or w.is_vasp_hot_wallet):
                if addr not in wallets:
                    wallets.append(addr)
        return wallets

    def compute_behavioral_score(
        self,
        suspect_wallet: str,
        candidate_vasp_id: str,
        path: Optional[List[str]] = None
    ) -> Tuple[Optional[float], List[str], List[float]]:
        """
        Computes behavioral attribution confidence score (0.0 to 1.0) or None.
        If suspect's fingerprint is INSUFFICIENT_DATA (< 3 outgoing transactions),
        returns (None, [], []).
        """
        path = path or [suspect_wallet]

        # a) Extract suspect fingerprint (strictly outgoing transactions)
        suspect_fp = self._get_fp(suspect_wallet, path_context=path, direction="out")

        # FIX 2: Require minimum wallet maturity. If suspect has < 3 outgoing txs, return None.
        if (
            suspect_fp is None
            or suspect_fp.summary is None
            or suspect_fp.summary.get("confidence") == "INSUFFICIENT_DATA"
            or suspect_fp.tx_count < 3
        ):
            return None, [], []

        # b) Extract fingerprints of last 3 hops on path before final node
        path_fps: List[BehavioralFingerprint] = []
        if len(path) > 2:
            hops_to_check = path[1:-1][-3:]
            for h in hops_to_check:
                h_fp = self._get_fp(h, path_context=path)
                if h_fp and h_fp.summary and h_fp.summary.get("confidence") != "INSUFFICIENT_DATA":
                    path_fps.append(h_fp)

        query_fps = [suspect_fp]
        query_fps.extend([p for p in path_fps if p not in query_fps])

        if not query_fps:
            return None, [], []

        # c) Candidate VASP reference wallets
        vasp_wallets = self.get_vasp_reference_wallets(candidate_vasp_id)
        if not vasp_wallets:
            return 0.10, [], []

        best_sim = 0.0
        similar_items: List[Tuple[str, float]] = []

        for vw in vasp_wallets:
            v_fp = self._get_fp(vw, direction="both")
            if not v_fp:
                continue

            for q_fp in query_fps:
                sim = cosine_similarity(q_fp, v_fp)
                similar_items.append((vw, sim))
                if sim > best_sim:
                    best_sim = sim

        # Sort similar wallets descending
        similar_items.sort(key=lambda x: x[1], reverse=True)
        top_similar = similar_items[:3]
        top_addrs = [item[0] for item in top_similar]
        top_scores = [round(item[1], 4) for item in top_similar]

        # Calibrate raw similarity score:
        # Cosine similarity on non-negative feature spaces tends to baseline around 0.5-0.7.
        # Shift and scale so dissimilar wallets fall to 0.1-0.3 and high matches reach 0.7-0.95.
        if best_sim > 0.0:
            calibrated = (best_sim - 0.50) / 0.50  # maps [0.5, 1.0] -> [0.0, 1.0]
            confidence_score = float(np.clip(calibrated, 0.05, 0.98))
        else:
            confidence_score = 0.05

        return round(confidence_score, 4), top_addrs, top_scores

    def attribute_candidates(
        self,
        suspect_wallet: str,
        candidates: List[NearestVASPCandidate]
    ) -> List[NearestVASPCandidate]:
        """Enriches all NearestVASPCandidate items with behavioral attribution metrics."""
        for c in candidates:
            score, sim_wallets, sim_scores = self.compute_behavioral_score(
                suspect_wallet=suspect_wallet,
                candidate_vasp_id=c.vasp_id,
                path=c.path
            )
            c.behavioral_confidence_score = score
            c.behavioral_similar_wallets = sim_wallets
            c.behavioral_similarity_scores = sim_scores
        return candidates
