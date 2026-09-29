"""
Forensic Agent & Autonomous Co-Investigator (Phase 7).
Money Migration Atlas (SIH26182).

Synthesizes graph traversal, XGBoost tabular features, GNN structural embeddings,
and behavioral habit metrics into actionable investigator briefings.
Determines court-ready triage actions and assembles evidence packages.
"""

import logging
from typing import List, Optional

from app.core.schemas import (
    Chain,
    NearestVASPCandidate,
    ConsensusTier,
    RecommendedAction,
    CandidateExplanation,
    InvestigationReport
)
from app.graph.store import GraphStore
from app.ml.consensus import ConsensusScorer
from app.agentic.counterfactual import generate_counterfactuals
from app.agentic.llm_client import LLMClient
from app.evidence.chain import EvidenceChainBuilder

logger = logging.getLogger("mma.agentic.agent")


class ForensicAgent:
    """
    Autonomous investigator providing plain-language explanations, counterfactual analysis,
    and action recommendations for cryptocurrency laundering investigations.
    """

    def __init__(
        self,
        consensus_scorer: Optional[ConsensusScorer] = None,
        llm_client: Optional[LLMClient] = None,
        evidence_builder: Optional[EvidenceChainBuilder] = None
    ):
        self.consensus_scorer = consensus_scorer or ConsensusScorer()
        self.llm_client = llm_client or LLMClient()
        self.evidence_builder = evidence_builder or EvidenceChainBuilder()

    @staticmethod
    def determine_recommended_action(
        candidates: List[NearestVASPCandidate]
    ) -> RecommendedAction:
        """
        Determines investigative next step according to regulatory triage rules:
        - CONFIRMED tier + FIU-registered VASP -> SEND_DISCLOSURE_REQUEST
        - AMBIGUOUS or SINGLE_MODEL -> HUMAN_REVIEW_REQUIRED
        - All candidates UNCERTAIN or top score < 0.30 -> INSUFFICIENT_SIGNAL
        """
        if not candidates:
            return RecommendedAction.INSUFFICIENT_SIGNAL

        top = candidates[0]
        top_score = top.consensus_score if top.consensus_score is not None else 0.0

        # Branch 3: All candidates UNCERTAIN or top score < 0.30
        if top_score < 0.30 or all(c.consensus_tier == ConsensusTier.UNCERTAIN for c in candidates):
            return RecommendedAction.INSUFFICIENT_SIGNAL

        # Branch 1: CONFIRMED tier + FIU-registered VASP
        if top.consensus_tier == ConsensusTier.CONFIRMED and top.fiu_ind_registered:
            return RecommendedAction.SEND_DISCLOSURE_REQUEST

        # Branch 2: AMBIGUOUS, SINGLE_MODEL, or non-FIU CONFIRMED -> HUMAN_REVIEW_REQUIRED
        return RecommendedAction.HUMAN_REVIEW_REQUIRED

    def investigate(
        self,
        suspect_wallet: str,
        store: GraphStore,
        chain: Chain = Chain.ETH,
        max_hops: int = 6
    ) -> InvestigationReport:
        """
        Executes end-to-end attribution analysis, plain-language generation,
        counterfactual reasoning, and court-ready evidence package construction.
        """
        # 1. Run multi-model consensus prediction
        candidates = self.consensus_scorer.predict(
            suspect_wallet=suspect_wallet,
            store=store,
            max_hops=max_hops,
            include_behavioral=True
        )

        if not candidates:
            # Fallback for unlinked wallets
            empty_cand = NearestVASPCandidate(
                vasp_id="unknown",
                vasp_name="Unknown VASP",
                proximity_rank=999,
                path=[suspect_wallet],
                never_blended=True
            )
            evidence = self.evidence_builder.build(empty_cand, store, suspect_wallet, chain.value)
            return InvestigationReport(
                suspect_wallet=suspect_wallet,
                chain=chain,
                plain_language_summary=f"No candidate Virtual Asset Service Providers were reached within {max_hops} hops from {suspect_wallet}.",
                data_source="template_fallback",
                top_3_candidates=[],
                counterfactuals=["If the graph depth were expanded to 12 hops, deeper laundering clusters might be discovered."],
                evidence_package=evidence,
                recommended_action=RecommendedAction.INSUFFICIENT_SIGNAL,
                never_blended=True
            )

        top_cand = candidates[0]

        # 2. Process top 3 candidates with counterfactuals and plain-language explanations
        candidate_explanations: List[CandidateExplanation] = []
        for c in candidates[:3]:
            c_cfs = generate_counterfactuals(candidate=c, all_candidates=candidates, store=store)
            c_dict = c.model_dump()
            exp_text, src = self.llm_client.generate_explanation(
                candidate_data=c_dict,
                context={
                    "counterfactuals": c_cfs,
                    "suspect_wallet": suspect_wallet,
                    "all_candidates": [cand.vasp_id for cand in candidates[:3]]
                }
            )
            candidate_explanations.append(
                CandidateExplanation(
                    candidate=c,
                    explanation=exp_text,
                    counterfactuals=c_cfs,
                    data_source=src
                )
            )

        # 3. Build court-admissible evidence package for top candidate
        evidence_package = self.evidence_builder.build(
            candidate=top_cand,
            store=store,
            suspect_wallet=suspect_wallet,
            chain=chain.value
        )

        # 4. Triage action recommendation
        rec_action = self.determine_recommended_action(candidates)

        top_exp = candidate_explanations[0]

        report = InvestigationReport(
            suspect_wallet=suspect_wallet,
            chain=chain,
            plain_language_summary=top_exp.explanation,
            data_source=top_exp.data_source,
            top_3_candidates=candidate_explanations,
            counterfactuals=top_exp.counterfactuals,
            evidence_package=evidence_package,
            recommended_action=rec_action,
            never_blended=True
        )

        assert report.never_blended is True, "Invariant violated: never_blended must be True"
        return report
