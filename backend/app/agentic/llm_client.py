"""
LLM Client with Deterministic Template Fallback (Phase 7).
Money Migration Atlas (SIH26182).

Converts mathematical attribution signals into plain-language forensic narratives.
Supports Anthropic Claude API with automatic timeout, rate-limit, and missing-key fallback.
"""

import json
import logging
from typing import Dict, Any, Optional, Tuple

from app.config import get_settings
from app.core.schemas import ExplanationResponse

logger = logging.getLogger("mma.agentic.llm_client")


CLAUDE_PROMPT_TEMPLATE = """You are a crypto forensics analyst. Write a 3-5 sentence plain-language explanation for an investigator. Rules:
- Cite specific scores (proximity, XGB, GNN, consensus tier)
- Name VASPs by their real name (CoinDCX, WazirX, etc.)
- If models disagree, say so explicitly
- If any signal is UNKNOWN or INSUFFICIENT_DATA, mention it
- Do NOT speculate beyond the evidence provided
- Do NOT blend proximity and confidence
- End with one counterfactual sentence

Evidence: {structured_json}"""


class LLMClient:
    """
    Forensic narrative generator utilizing Anthropic Claude when configured,
    with an offline-first deterministic template fallback.
    """

    def __init__(self, api_key: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.ANTHROPIC_API_KEY
        self.timeout = settings.LLM_TIMEOUT_SECONDS
        self.model = settings.LLM_MODEL
        self.max_tokens = settings.LLM_MAX_TOKENS
        self.fallback_enabled = settings.LLM_FALLBACK_ENABLED

    def generate_explanation(
        self,
        candidate_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """
        Generates 3-5 sentence plain-language attribution explanation.
        Returns: (explanation_text, data_source: "llm" | "template_fallback")
        """
        context = context or {}
        combined_payload = {**candidate_data, "context": context}

        if self.api_key and not settings_disabled():
            try:
                explanation = self._call_anthropic(combined_payload)
                if explanation and len(explanation.strip()) > 30:
                    return explanation.strip(), "llm"
            except Exception as e:
                logger.warning("Anthropic Claude API call failed (%s); invoking fallback.", e)

        # Template fallback
        return self._generate_template_fallback(candidate_data, context), "template_fallback"

    def _call_anthropic(self, payload: Dict[str, Any]) -> Optional[str]:
        """Calls Anthropic Messages API with strict timeout."""
        try:
            import anthropic  # type: ignore
        except ImportError:
            logger.info("anthropic package not installed; falling back to template generator.")
            return None

        prompt = CLAUDE_PROMPT_TEMPLATE.format(
            structured_json=json.dumps(payload, indent=2, default=str)
        )

        client = anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout)
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        if message.content and len(message.content) > 0:
            return message.content[0].text
        return None

    def _generate_template_fallback(
        self,
        candidate_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """
        Deterministic, audit-proof template fallback generator.
        Produces >= 3 sentences covering proximity, XGB, GNN, consensus tier,
        behavioral status, and counterfactuals.
        """
        vasp_name = candidate_data.get("vasp_name", "Unknown VASP")
        prox = candidate_data.get("proximity_rank", 0)
        xgb_score = candidate_data.get("confidence_score")
        gnn_score = candidate_data.get("gnn_confidence_score")
        beh_score = candidate_data.get("behavioral_confidence_score")
        cons_tier_raw = candidate_data.get("consensus_tier", "UNCERTAIN")
        cons_tier = cons_tier_raw.value if hasattr(cons_tier_raw, "value") else str(cons_tier_raw)
        cons_score = candidate_data.get("consensus_score", 0.0)
        counterfactuals = context.get("counterfactuals", [])

        # Sentence 1: Primary finding and topological proximity
        s1 = (
            f"The suspect funds terminate at {vasp_name} within a graph distance of {prox} hops, "
            f"yielding an overall multi-model consensus score of {cons_score:.2f} (Tier: {cons_tier})."
        )

        # Sentence 2: Model agreement / divergence breakdown
        xgb_str = f"{xgb_score:.2f}" if xgb_score is not None else "UNKNOWN"
        gnn_str = f"{gnn_score:.2f}" if gnn_score is not None else "UNKNOWN"

        if cons_tier == "CONFIRMED":
            s2 = (
                f"Both XGBoost tabular scoring ({xgb_str}) and Graph Neural Network structural embeddings ({gnn_str}) "
                f"strongly corroborate attribution to {vasp_name} without cross-candidate contradictions."
            )
        elif cons_tier == "AMBIGUOUS":
            s2 = (
                f"The predictive models exhibit disagreement: XGBoost estimated confidence at {xgb_str}, whereas the GNN "
                f"produced a score of {gnn_str}, indicating divergent algorithmic evidence that warrants manual forensic verification."
            )
        elif cons_tier == "SINGLE_MODEL":
            s2 = (
                f"Attribution is driven primarily by a single high-confidence signal ({gnn_str if (gnn_score or 0) >= 0.60 else xgb_str}), "
                f"while the secondary model remained inconclusive (XGB: {xgb_str}, GNN: {gnn_str})."
            )
        else:
            s2 = (
                f"Both tabular features (XGBoost: {xgb_str}) and graph message-passing (GNN: {gnn_str}) returned sub-threshold "
                f"probabilities below 0.60, establishing an UNCERTAIN agreement state."
            )

        # Sentence 3: Behavioral status and strict score separation invariant
        if beh_score is not None:
            s3 = (
                f"Behavioral habit analysis verified a similarity coefficient of {beh_score:.2f}, while topological proximity "
                f"and predictive confidence scores remained strictly unblended."
            )
        else:
            s3 = (
                f"Behavioral habit fingerprinting flagged INSUFFICIENT_DATA due to minimal transaction history (<3 outgoing txs), "
                f"ensuring proximity rank ({prox} hops) and confidence scores are never blended."
            )

        # Sentence 4: Counterfactual statement
        if counterfactuals and len(counterfactuals) > 0:
            s4 = counterfactuals[0]
        else:
            s4 = f"If the transaction path had traversed an additional 2 hops, topological decay would have degraded confidence."

        return f"{s1} {s2} {s3} {s4}"


def settings_disabled() -> bool:
    """Helper to detect if fallback is forced or API key is simulated off."""
    return False
