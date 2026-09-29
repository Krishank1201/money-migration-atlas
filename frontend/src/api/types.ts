export type Chain = 'ETH' | 'BTC' | 'TRC-20';

export type ConfidenceTier = 'HIGH' | 'MEDIUM' | 'LOW' | 'UNKNOWN';

export type ConsensusTier = 'CONFIRMED' | 'AMBIGUOUS' | 'UNCERTAIN' | 'SINGLE_MODEL' | 'BEHAVIORAL_ONLY';

export type RecommendedAction = 'SEND_DISCLOSURE_REQUEST' | 'HUMAN_REVIEW_REQUIRED' | 'INSUFFICIENT_SIGNAL';

export interface ShapFeature {
  feature: string;
  value: number;
  shap: number;
  direction: 'positive' | 'negative';
}

export interface NearestVASPCandidate {
  vasp_id: string;
  vasp_name: string;
  proximity_rank: number;
  distance: number;
  target_wallet?: string;
  path: string[];
  tx_hashes: string[];
  chain_path: string[];
  fiu_ind_registered: boolean;
  confidence_score?: number | null;
  confidence_tier?: ConfidenceTier;
  shap_explanation?: ShapFeature[];
  model_version?: string;
  gnn_confidence_score?: number | null;
  gnn_confidence_tier?: ConfidenceTier;
  gnn_model_used?: string;
  consensus_tier?: ConsensusTier;
  xgb_gnn_agreement?: number;
  consensus_score?: number | null;
  gnn_subgraph_explanation?: Array<{
    node_address: string;
    importance_score: number;
    role: string;
  }>;
  behavioral_confidence_score?: number | null;
  behavioral_similar_wallets?: string[];
  behavioral_similarity_scores?: number[];
  never_blended: boolean;
}

export interface CustodyStep {
  step: string;
  timestamp: string;
  source: string;
  notes: string;
}

export interface EvidencePackage {
  suspect_wallet: string;
  top_candidate: {
    vasp_id: string;
    vasp_name: string;
    fiu_ind_registered?: boolean;
    target_wallet?: string;
  };
  proximity_rank: number;
  proximity_path: string[];
  transaction_hashes: string[];
  chain_path: string[];
  xgb_confidence_score?: number | null;
  xgb_confidence_tier?: ConfidenceTier;
  gnn_confidence_score?: number | null;
  gnn_confidence_tier?: ConfidenceTier;
  gnn_model_used?: string;
  behavioral_confidence_score?: number | null;
  consensus_score?: number | null;
  consensus_tier?: ConsensusTier;
  xgb_gnn_agreement?: number;
  shap_explanation?: ShapFeature[];
  gnn_subgraph_explanation?: any[];
  model_versions?: Record<string, string>;
  generated_at: string;
  never_blended: boolean;
  chain_of_custody: CustodyStep[];
}

export interface CounterfactualItem {
  text: string;
  provenance: 'recomputed' | 'estimated_shap' | 'qualitative';
  method: string;
}

export interface CandidateExplanation {
  candidate: NearestVASPCandidate;
  explanation: string;
  counterfactuals: CounterfactualItem[];
  data_source: string;
}

export interface InvestigationReport {
  suspect_wallet: string;
  chain: Chain;
  plain_language_summary: string;
  data_source: 'llm' | 'template_fallback';
  top_3_candidates: CandidateExplanation[];
  counterfactuals: CounterfactualItem[];
  evidence_package: EvidencePackage;
  recommended_action: RecommendedAction;
  never_blended: boolean;
}

export interface TestCase {
  case_id: string;
  name: string;
  suspect_wallet: string;
  chain: string;
  expected_vasp_id: string;
  expected_vasp_name: string;
  expected_tier: string;
  notes?: string;
  expected_hops?: number;
}

export interface ModelInfoResponse {
  model_version: string;
  trained_at?: string;
  metrics: Record<string, any>;
  feature_importance: Array<{ feature: string; importance: number }>;
  hyperparameters: Record<string, any>;
  training_samples: number;
}

export interface GNNModelInfoResponse {
  model_family: string;
  architectures: string[];
  feature_dimensions: Record<string, number>;
  metrics: Record<string, any>;
}
