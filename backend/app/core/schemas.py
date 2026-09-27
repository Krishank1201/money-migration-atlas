from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field


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
