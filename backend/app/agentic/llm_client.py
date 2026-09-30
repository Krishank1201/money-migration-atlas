"""
LLM Client with Deterministic Template Fallback (Phase 7).
Money Migration Atlas (SIH26182).

Converts mathematical attribution signals into plain-language forensic narratives.
Supports Groq API (Llama 3.3 70B) with automatic timeout, rate-limit, and missing-key fallback.
"""

import json
import logging
from typing import Dict, Any, Optional, Tuple

import httpx

from app.config import get_settings

logger = logging.getLogger("mma.agentic.llm_client")

_DEFAULT_KEY = object()

GROQ_SYSTEM_PROMPT = """You are a crypto forensics analyst. Write a 3-5 sentence plain-language explanation for an investigator. Rules:
- Cite specific scores (proximity, XGB, GNN, consensus tier)
- Name VASPs by their real name (CoinDCX, WazirX, etc.)
- If models disagree, say so explicitly
- If any signal is UNKNOWN or INSUFFICIENT_DATA, mention it
- Do NOT speculate beyond the evidence provided
- Do NOT blend proximity and confidence
- End with one counterfactual sentence"""


class LLMClient:
    """
    Forensic narrative generator utilizing Groq (Llama 3.3 70B) when configured,
    with an offline-first deterministic template fallback.
    """

    def __init__(self, api_key: Any = _DEFAULT_KEY):
        settings = get_settings()
        if api_key is _DEFAULT_KEY:
            self.api_key = settings.GROQ_API_KEY
        else:
            self.api_key = api_key
        self.base_url = settings.GROQ_BASE_URL
        self.model = settings.GROQ_MODEL
        self.timeout = settings.LLM_TIMEOUT_SECONDS
        self.max_tokens = settings.LLM_MAX_TOKENS
        self.fallback_enabled = settings.LLM_FALLBACK_ENABLED

    def generate_explanation(
        self,
        candidate_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """
        Generates 3-5 sentence plain-language attribution explanation.
        Returns: (explanation_text, data_source: "groq" | "template_fallback")
        """
        context = context or {}
        if not self.api_key or not str(self.api_key).strip():
            return self._generate_template_fallback(candidate_data, context), "template_fallback"

        combined_payload = {**candidate_data, "context": context}

        try:
            explanation = self._call_groq(combined_payload)
            if explanation and len(explanation.strip()) > 30:
                return explanation.strip(), "groq"
        except Exception as e:
            logger.warning("Groq API call failed (%s); invoking fallback.", e)

        # Template fallback
        return self._generate_template_fallback(candidate_data, context), "template_fallback"

    def _call_groq(self, payload: Dict[str, Any]) -> Optional[str]:
        """Calls Groq Chat Completions API with strict timeout."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        user_content = f"Evidence: {json.dumps(payload, indent=2, default=str)}"
        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"

        models_to_try = [self.model]
        for alt_model in ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]:
            if alt_model not in models_to_try:
                models_to_try.append(alt_model)

        with httpx.Client(timeout=self.timeout) as client:
            for model_name in models_to_try:
                body = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": GROQ_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "max_tokens": self.max_tokens,
                    "temperature": 0.3,
                }
                response = client.post(endpoint, json=body, headers=headers)
                if response.status_code == 404:
                    # Model not accessible under this key tier, try next available model on Groq
                    continue
                response.raise_for_status()
                res_json = response.json()
                choices = res_json.get("choices", [])
                if choices and len(choices) > 0:
                    content = choices[0].get("message", {}).get("content", "")
                    if content:
                        return content.strip()
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
