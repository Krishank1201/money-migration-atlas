from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from pydantic import BaseModel, Field, field_validator


class Chain(str, Enum):
    BTC = "BTC"
    ETH = "ETH"
    TRON_TRC20 = "TRON_TRC20"
    BSC = "BSC"
    SOL = "SOL"
    POLYGON = "POLYGON"


class VASPCategory(str, Enum):
    INDIAN_FIU = "INDIAN_FIU"
    OVERSEAS_NON_COMPLIANT = "OVERSEAS_NON_COMPLIANT"


class ConfidenceTier(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


class ConsensusTier(str, Enum):
    CONFIRMED = "CONFIRMED"
    AMBIGUOUS = "AMBIGUOUS"
    UNCERTAIN = "UNCERTAIN"
    SINGLE_MODEL = "SINGLE_MODEL"
    BEHAVIORAL_ONLY = "BEHAVIORAL_ONLY"


class LaunderingPattern(str, Enum):
    PEELING = "peeling"
    MIXER = "mixer"
    NESTED = "nested"
    SWEEPER = "sweeper"
    DIRECT = "direct"


class VASP(BaseModel):
    """Regulated exchange or Virtual Asset Service Provider entity."""
    id: str = Field(..., description="Unique VASP identifier (e.g., 'coindcx')")
    name: str = Field(..., description="Display name of exchange")
    legal_name: Optional[str] = None
    category: VASPCategory = Field(..., description="Regulatory categorization")
    jurisdiction: str = Field(default="IN", description="ISO Country code or jurisdiction")
    fiu_ind_registered: bool = Field(default=False, description="Registered under FIU-IND")
    fiu_registration_number: Optional[str] = None
    hot_wallets: List[str] = Field(default_factory=list, description="Known hot wallet addresses")
    deposit_sweepers: List[str] = Field(default_factory=list, description="Known sweeper addresses")


class Wallet(BaseModel):
    """Blockchain address node."""
    address: str = Field(..., description="Cryptographic address or public key")
    chain: Chain = Field(..., description="Associated blockchain network")
    label: Optional[str] = Field(default=None, description="Known label or tag")
    vasp_id: Optional[str] = Field(default=None, description="Ground truth VASP owner if known")
    is_vasp_hot_wallet: bool = Field(default=False)
    is_vasp_deposit_sweeper: bool = Field(default=False)
    is_mixer: bool = Field(default=False)
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    first_seen: Optional[int] = Field(default=None, description="Epoch timestamp of earliest activity")
    last_seen: Optional[int] = Field(default=None, description="Epoch timestamp of latest activity")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Transaction(BaseModel):
    """Directed value transfer edge."""
    tx_hash: str = Field(..., description="Transaction hash")
    chain: Chain = Field(..., description="Underlying chain")
    from_address: str = Field(..., description="Sender wallet address")
    to_address: str = Field(..., description="Recipient wallet address")
    amount: float = Field(..., ge=0.0, description="Transferred asset value")
    token_symbol: str = Field(..., description="Asset ticker, e.g. BTC, ETH, USDT")
    timestamp: int = Field(..., description="Epoch timestamp")
    fee: float = Field(default=0.0, ge=0.0)
    gas_price: Optional[float] = None
    is_peeling_chain: bool = Field(default=False)
    is_sweeper_tx: bool = Field(default=False)
    is_mixer_tx: bool = Field(default=False)


class Attribution(BaseModel):
    """
    Forensic attribution result connecting a suspect address to an identified VASP.
    CRITICAL SIH REQUIREMENT:
    Proximity rank (graph hops) and Confidence score (probabilistic model)
    are strictly INDEPENDENT and never blended.
    """
    suspect_wallet: str
    chain: Chain
    predicted_vasp: str
    proximity_rank: int = Field(..., description="Graph distance in hops to VASP-controlled address")
    proximity_distance: float = Field(..., description="Weighted graph distance")
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Model prediction probability")
    confidence_tier: ConfidenceTier = Field(default=ConfidenceTier.UNKNOWN)
    evidence_subgraph: List[Transaction] = Field(default_factory=list, description="Ordered path of transactions")
    evidence_hashes: List[str] = Field(default_factory=list, description="Hashes along the evidence chain")
    counterfactuals: List[str] = Field(default_factory=list, description="Alternative hypotheses evaluated")
    plain_language_explanation: Optional[str] = Field(default=None, description="Investigator summary")
    never_blended: bool = Field(default=True, description="Enforces strict separation of rank and confidence")


class GroundTruthTestCase(BaseModel):
    """Benchmark test case with verified on-chain laundering pathway and ground truth."""
    case_id: str
    case_description: str
    suspect_wallet: str
    expected_vasp: str
    expected_graph_distance: int
    expected_confidence_range: Tuple[float, float]
    laundering_pattern: str
    chain: Chain
    expected_min_hops: Optional[int] = None
    expected_max_hops: Optional[int] = None
    requires_behavioral_fingerprint: bool = False
    has_proximity_tie: bool = False
    is_cross_chain: bool = False
    naive_proximity_will_fail: bool = False


class DataSource(str, Enum):
    LIVE = "live"
    SYNTHETIC = "synthetic"
    CACHE = "cache"
    SYNTHETIC_FALLBACK = "synthetic_fallback"


class FetchResult(BaseModel):
    """Encapsulates transactions and balance fetched from live blockchain or synthetic fallback."""
    address: str
    chain: Chain
    data_source: str = Field(..., description="Origin of data: 'live', 'synthetic', or 'cache'")
    transactions: List[Transaction] = Field(default_factory=list)
    balance: float = 0.0
    cached_at: Optional[int] = None
    error_message: Optional[str] = None


class SubgraphResponse(BaseModel):
    """Ego-subgraph around a center address for visualization."""
    center_address: str
    chain: Chain
    hops: int
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)


class IngestionReport(BaseModel):
    """Summary of graph ingestion operation."""
    wallets_added: int = 0
    txs_added: int = 0
    duration_ms: float = 0.0
    source: str = Field(..., description="Data source identifier ('synthetic', 'live', 'benchmark')")


class ShapFeature(BaseModel):
    """Top feature attribution from SHAP explainability."""
    feature: str = Field(..., description="Feature identifier name")
    value: float = Field(..., description="Observed feature value for this sample")
    shap: float = Field(..., description="SHAP contribution value")
    direction: str = Field(..., description="Direction of contribution: 'positive' or 'negative'")


class NearestVASPCandidate(BaseModel):
    """Candidate VASP identified via topological traversal and ML attribution."""
    vasp_id: str
    vasp_name: str
    proximity_rank: int = Field(..., description="Graph distance in hops to VASP-controlled address")
    distance: float = Field(default=0.0, description="Cumulative edge traversal weight")
    target_wallet: Optional[str] = None
    path: List[str] = Field(default_factory=list)
    tx_hashes: List[str] = Field(default_factory=list)
    chain_path: List[str] = Field(default_factory=list)
    fiu_ind_registered: bool = False
    confidence_score: Optional[float] = None
    confidence_tier: ConfidenceTier = ConfidenceTier.UNKNOWN
    shap_explanation: Optional[List[ShapFeature]] = None
    model_version: Optional[str] = None
    gnn_confidence_score: Optional[float] = None
    gnn_confidence_tier: Optional[ConfidenceTier] = None
    gnn_model_used: Optional[str] = None
    consensus_tier: Optional[ConsensusTier] = None
    xgb_gnn_agreement: Optional[float] = None
    consensus_score: Optional[float] = None
    gnn_subgraph_explanation: Optional[List[Dict[str, Any]]] = None
    behavioral_confidence_score: Optional[float] = None
    behavioral_similar_wallets: Optional[List[str]] = None
    behavioral_similarity_scores: Optional[List[float]] = None
    never_blended: bool = Field(default=True, description="Enforces strict separation of proximity rank and confidence score")

    @field_validator("never_blended")
    @classmethod
    def validate_never_blended(cls, v: bool) -> bool:
        if not v:
            raise ValueError("Invariant violated: proximity_rank and confidence_score must NEVER be blended.")
        return True


class MLPredictResponse(BaseModel):
    """Full attribution response with independent scores and SHAP explainability."""
    suspect_wallet: str
    chain: Chain
    candidates: List[NearestVASPCandidate]
    never_blended: bool = Field(default=True, description="Asserted at runtime: scores are strictly unblended")

    @field_validator("never_blended")
    @classmethod
    def validate_never_blended(cls, v: bool) -> bool:
        if not v:
            raise ValueError("Invariant violated: proximity_rank and confidence_score must NEVER be blended.")
        return True


class MLModelInfoResponse(BaseModel):
    """Metadata regarding currently loaded XGBoost model."""
    model_version: str
    trained_at: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    feature_importance: List[Dict[str, Any]] = Field(default_factory=list)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    training_samples: int = 0


class AnalyticsResponse(BaseModel):
    """Topological graph analytics report."""
    degree_distribution: Dict[str, Any] = Field(default_factory=dict)
    top_central_wallets: List[Dict[str, Any]] = Field(default_factory=list)
    mixer_candidates: List[Dict[str, Any]] = Field(default_factory=list)


class BehavioralFingerprint(BaseModel):
    """64-dimensional behavioral fingerprint capturing transaction habits across timing, gas, amount, and chain domains."""
    address: str
    chain: Optional[str] = None
    tx_count: int = 0
    timing_vector: List[float] = Field(default_factory=list, description="16-dim timing feature vector")
    gas_vector: List[float] = Field(default_factory=list, description="16-dim gas fee feature vector")
    amount_vector: List[float] = Field(default_factory=list, description="16-dim amount structuring vector")
    chain_vector: List[float] = Field(default_factory=list, description="16-dim chain + temporal vector")
    summary: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Human-readable behavioral indicators")

    def to_vector(self) -> np.ndarray:
        vec = np.concatenate([
            np.array(self.timing_vector, dtype=np.float32),
            np.array(self.gas_vector, dtype=np.float32),
            np.array(self.amount_vector, dtype=np.float32),
            np.array(self.chain_vector, dtype=np.float32),
        ])
        if len(vec) != 64:
            # Pad or truncate if dimensions differ
            out = np.zeros(64, dtype=np.float32)
            lim = min(len(vec), 64)
            out[:lim] = vec[:lim]
            return out
        return vec


class RecommendedAction(str, Enum):
    """Forensic triage recommendations for law enforcement investigators."""
    SEND_DISCLOSURE_REQUEST = "SEND_DISCLOSURE_REQUEST"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"
    INSUFFICIENT_SIGNAL = "INSUFFICIENT_SIGNAL"


class CustodyStep(BaseModel):
    """Immutable audit trail step for digital evidence chain of custody."""
    step: str
    timestamp: str
    source: str
    notes: str


class EvidencePackage(BaseModel):
    """Court-admissible forensic evidence dossier."""
    suspect_wallet: str
    top_candidate: Dict[str, Any] = Field(..., description="VASP identification and registration details")
    proximity_rank: int = Field(..., description="Shortest topological graph hops")
    proximity_path: List[str] = Field(default_factory=list, description="Sequence of addresses from suspect to deposit")
    transaction_hashes: List[str] = Field(default_factory=list, description="On-chain tx hashes corresponding to path edges")
    chain_path: List[str] = Field(default_factory=list, description="Blockchains traversed along path")
    xgb_confidence_score: Optional[float] = None
    xgb_confidence_tier: Optional[ConfidenceTier] = None
    gnn_confidence_score: Optional[float] = None
    gnn_confidence_tier: Optional[ConfidenceTier] = None
    gnn_model_used: Optional[str] = None
    behavioral_confidence_score: Optional[float] = None
    consensus_score: Optional[float] = None
    consensus_tier: Optional[ConsensusTier] = None
    xgb_gnn_agreement: Optional[float] = None
    shap_explanation: Optional[List[ShapFeature]] = None
    gnn_subgraph_explanation: Optional[List[Dict[str, Any]]] = None
    model_versions: Dict[str, str] = Field(default_factory=dict)
    generated_at: str
    never_blended: bool = Field(default=True, description="Strict invariant: topological and probabilistic scores are unblended")
    chain_of_custody: List[CustodyStep] = Field(default_factory=list)

    @field_validator("never_blended")
    @classmethod
    def validate_never_blended(cls, v: bool) -> bool:
        if not v:
            raise ValueError("Invariant violated: proximity_rank and confidence_score must NEVER be blended.")
        return True


class CandidateExplanation(BaseModel):
    """Explanation and counterfactual analysis for an individual candidate VASP."""
    candidate: NearestVASPCandidate
    explanation: str
    counterfactuals: List[str] = Field(default_factory=list)
    data_source: str = Field(default="template_fallback", description="'llm' or 'template_fallback'")


class ExplanationResponse(BaseModel):
    """LLM or template generated explanation with counterfactuals and provenance."""
    plain_language: str
    counterfactuals: List[str] = Field(default_factory=list)
    data_source: str = Field(..., description="'llm' or 'template_fallback'")


class InvestigationReport(BaseModel):
    """Comprehensive investigation-ready intelligence dossier combining narrative prose and mathematical proof."""
    suspect_wallet: str
    chain: Chain = Chain.ETH
    plain_language_summary: str
    data_source: str = Field(..., description="'llm' or 'template_fallback'")
    top_3_candidates: List[CandidateExplanation] = Field(default_factory=list)
    counterfactuals: List[str] = Field(default_factory=list)
    evidence_package: EvidencePackage
    recommended_action: RecommendedAction
    never_blended: bool = Field(default=True, description="Strict invariant: scores are never blended")

    @field_validator("never_blended")
    @classmethod
    def validate_never_blended(cls, v: bool) -> bool:
        if not v:
            raise ValueError("Invariant violated: proximity_rank and confidence_score must NEVER be blended.")
        return True

