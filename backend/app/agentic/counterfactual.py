"""
Counterfactual Generator (Phase 7).
Money Migration Atlas (SIH26182).

Produces deterministic, logic-driven counterfactual reasoning ("what-if" statements)
for cryptocurrency attribution cases without relying on LLMs.
"""

from typing import List, Optional
from app.core.schemas import NearestVASPCandidate, ConsensusTier
from app.graph.store import GraphStore


def generate_counterfactuals(
    candidate: NearestVASPCandidate,
    all_candidates: List[NearestVASPCandidate],
    store: Optional[GraphStore] = None
) -> List[str]:
    """
    Generates up to 3 deterministic counterfactual statements explaining what would
    change the attribution decision or confidence level.

    Guarantees:
    - Exactly 1 to 3 statements.
    - Pure deterministic logic (no LLM, no randomness).
    - Cites specific scores, hops, mixers, and alternate candidates.
    """
    counterfactuals: List[str] = []

    # 1. Mixer Presence / Absence Counterfactual
    has_mixer = False
    mixer_addr = None
    if store and hasattr(store, "wallets"):
        for addr in candidate.path:
            w = store.get_wallet(addr)
            if w and (getattr(w, "is_mixer", False) or "mixer" in getattr(w, "label", "").lower()):
                has_mixer = True
                mixer_addr = addr
                break

    c_score = candidate.consensus_score or candidate.confidence_score or 0.50

    if has_mixer and mixer_addr:
        higher_score = min(0.95, round(c_score + 0.25, 2))
        counterfactuals.append(
            f"If the path had not touched mixer {mixer_addr[:10]}..., "
            f"confidence would rise from {c_score:.2f} to {higher_score:.2f}."
        )
    else:
        lower_score = max(0.05, round(c_score - 0.25, 2))
        counterfactuals.append(
            f"If the path had passed through a privacy mixer, "
            f"topological penalty would have suppressed confidence from {c_score:.2f} to {lower_score:.2f}."
        )

    # 2. Model Divergence / Algorithm Contrast Counterfactual
    if all_candidates:
        xgb_best = max(all_candidates, key=lambda x: x.confidence_score if x.confidence_score is not None else -1.0)
        gnn_best = max(all_candidates, key=lambda x: x.gnn_confidence_score if x.gnn_confidence_score is not None else -1.0)

        if xgb_best.vasp_id != candidate.vasp_id and (xgb_best.confidence_score or 0) > 0.30:
            counterfactuals.append(
                f"If XGBoost alone made this decision, it would attribute to {xgb_best.vasp_name} "
                f"({xgb_best.confidence_score:.2f}), whereas consensus selected {candidate.vasp_name}."
            )
        elif gnn_best.vasp_id != candidate.vasp_id and (gnn_best.gnn_confidence_score or 0) > 0.30:
            counterfactuals.append(
                f"If GNN structural embeddings alone decided, attribution would favor {gnn_best.vasp_name} "
                f"({gnn_best.gnn_confidence_score:.2f}) over {candidate.vasp_name}."
            )
        else:
            # Both models agree or single candidate
            counterfactuals.append(
                f"If both XGBoost ({candidate.confidence_score or 0.0:.2f}) and GNN ({candidate.gnn_confidence_score or 0.0:.2f}) "
                f"scored below 0.60, consensus tier would downgrade from {candidate.consensus_tier.value if candidate.consensus_tier else 'CURRENT'} to UNCERTAIN."
            )

    # 3. Proximity / Alternate Candidate Sensitivity Counterfactual
    competitors = [c for c in all_candidates if c.vasp_id != candidate.vasp_id]
    if competitors:
        # Find closest or highest runner up
        runner_up = max(
            competitors,
            key=lambda x: (
                x.consensus_score if x.consensus_score is not None else 0.0,
                -x.proximity_rank
            )
        )
        if runner_up.proximity_rank > candidate.proximity_rank:
            counterfactuals.append(
                f"If the suspect wallet had reached {runner_up.vasp_name} at {candidate.proximity_rank} hops "
                f"instead of {runner_up.proximity_rank}, attribution would shift toward {runner_up.vasp_name}."
            )
        elif runner_up.proximity_rank < candidate.proximity_rank:
            counterfactuals.append(
                f"If {candidate.vasp_name} were reachable in {runner_up.proximity_rank} hops rather than "
                f"{candidate.proximity_rank}, proximity ranking would match {runner_up.vasp_name}."
            )
        else:
            counterfactuals.append(
                f"If the laundering path to {runner_up.vasp_name} were 1 hop shorter, "
                f"topological proximity would tilt attribution toward {runner_up.vasp_name}."
            )
    else:
        counterfactuals.append(
            f"If the laundering path had extended beyond {candidate.proximity_rank + 2} hops, "
            f"multi-hop decay would have substantially suppressed confidence."
        )

    # Strictly cap at 3 counterfactuals
    return counterfactuals[:3]
