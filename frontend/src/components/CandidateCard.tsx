import React, { useState } from 'react';
import { NearestVASPCandidate } from '../api/types';
import { ScoreBar } from './ScoreBar';
import { ConsensusBadge } from './ConsensusBadge';
import { ShieldCheck, ShieldAlert, CheckCircle, ChevronDown, ChevronUp, Layers } from 'lucide-react';

interface CandidateCardProps {
  candidate: NearestVASPCandidate;
  rankIndex: number;
  isSelected?: boolean;
  onSelect?: () => void;
}

export const CandidateCard: React.FC<CandidateCardProps> = ({
  candidate,
  rankIndex,
  isSelected = false,
  onSelect,
}) => {
  const [showShap, setShowShap] = useState(false);

  return (
    <div
      onClick={onSelect}
      className={`rounded-xl border transition-all cursor-pointer p-4 mb-4 ${
        isSelected
          ? 'bg-slate-800/90 border-cyan-500 shadow-[0_0_20px_rgba(6,182,212,0.15)] ring-1 ring-cyan-500/50'
          : 'bg-slate-850/80 hover:bg-slate-800/60 border-slate-700/80 hover:border-slate-600'
      }`}
    >
      {/* Header: Rank + VASP Name + FIU Status */}
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex items-center gap-2.5">
          <span className="flex items-center justify-center w-6 h-6 rounded-md bg-slate-900 border border-slate-700 text-xs font-mono font-bold text-cyan-400">
            #{rankIndex + 1}
          </span>
          <div>
            <h3 className="font-semibold text-slate-100 text-base flex items-center gap-2">
              {candidate.vasp_name}
              {candidate.fiu_ind_registered ? (
                <span className="inline-flex items-center gap-1 text-[11px] font-mono font-medium text-emerald-400 bg-emerald-950/70 border border-emerald-600/40 px-2 py-0.5 rounded-full">
                  <ShieldCheck className="w-3 h-3" /> FIU-IND Reg.
                </span>
              ) : (
                <span className="inline-flex items-center gap-1 text-[11px] font-mono font-medium text-amber-400 bg-amber-950/70 border border-amber-600/40 px-2 py-0.5 rounded-full">
                  <ShieldAlert className="w-3 h-3" /> Offshore / Non-FIU
                </span>
              )}
            </h3>
            <p className="text-xs text-slate-400 font-mono mt-0.5">
              Target Deposit: {candidate.target_wallet ? `${candidate.target_wallet.slice(0, 10)}...${candidate.target_wallet.slice(-6)}` : 'Identified Deposit Pool'}
            </p>
          </div>
        </div>

        {/* Invariant Assertion Badge */}
        <div className="text-right">
          <span
            className="inline-flex items-center gap-1 text-[11px] font-mono text-emerald-400 bg-emerald-950/50 border border-emerald-700/50 px-2 py-0.5 rounded"
            title="Forensic Invariant: Proximity rank and model confidence scores are mathematically separated and never blended."
          >
            <CheckCircle className="w-3 h-3" />
            <span>never_blended: ✓</span>
          </span>
        </div>
      </div>

      {/* Consensus Tier Summary Strip */}
      <div className="flex items-center justify-between p-2.5 mb-3 rounded-lg bg-slate-900/80 border border-slate-800">
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400 font-medium">Agreement Tier:</span>
          <ConsensusBadge tier={candidate.consensus_tier} size="sm" />
        </div>
        <div className="text-right font-mono text-xs">
          <span className="text-slate-400">Consensus (Max): </span>
          <span className="font-bold text-cyan-400 text-sm">
            {candidate.consensus_score !== null && candidate.consensus_score !== undefined
              ? candidate.consensus_score.toFixed(2)
              : 'N/A'}
          </span>
        </div>
      </div>

      {/* Independent Score Bars Section (Strictly Separated) */}
      <div className="space-y-2">
        {/* 1. Proximity Rank (Discrete Hops) */}
        <ScoreBar
          type="proximity"
          modelName="Dijkstra Shortest Path Traversal"
          value={candidate.proximity_rank}
          subtitle="Topological network graph distance"
        />

        {/* 2. XGBoost Tabular Confidence */}
        <ScoreBar
          type="xgboost"
          modelName="XGBoost Classifier v1.0"
          value={candidate.confidence_score}
          tierLabel={candidate.confidence_tier}
          subtitle="20 tabular path quality & volume features"
        />

        {/* 3. GNN Structural Confidence */}
        <ScoreBar
          type="gnn"
          modelName={candidate.gnn_model_used || "GraphSAGE / GATv2 Ensemble"}
          value={candidate.gnn_confidence_score}
          tierLabel={candidate.gnn_confidence_tier}
          subtitle="Direct relational graph embeddings & message passing"
        />

        {/* 4. Behavioral Habit Confidence */}
        <ScoreBar
          type="behavioral"
          modelName="Behavioral Fingerprint Attributor"
          value={candidate.behavioral_confidence_score ?? 'INSUFFICIENT_DATA'}
          subtitle="64-dim timing, gas & structuring habit vector"
        />
      </div>

      {/* Local Explainability (SHAP Top Features) Accordion */}
      {candidate.shap_explanation && candidate.shap_explanation.length > 0 && (
        <div className="mt-3 pt-2 border-t border-slate-800">
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              setShowShap(!showShap);
            }}
            className="flex items-center justify-between w-full text-xs text-slate-400 hover:text-slate-200 transition-colors py-1"
          >
            <span className="flex items-center gap-1.5 font-mono">
              <Layers className="w-3.5 h-3.5 text-cyan-400" />
              SHAP Local Feature Attributions (Top {candidate.shap_explanation.length})
            </span>
            {showShap ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>

          {showShap && (
            <div className="mt-2 space-y-1.5 bg-slate-900/90 p-2.5 rounded border border-slate-800 text-xs">
              {candidate.shap_explanation.map((sf, idx) => (
                <div key={idx} className="flex items-center justify-between font-mono text-[11px]">
                  <span className="text-slate-300 truncate max-w-[200px]" title={sf.feature}>
                    {sf.feature}
                  </span>
                  <div className="flex items-center gap-2">
                    <span className="text-slate-500">val: {Number(sf.value).toFixed(2)}</span>
                    <span
                      className={`font-semibold ${
                        sf.direction === 'positive' ? 'text-emerald-400' : 'text-rose-400'
                      }`}
                    >
                      {sf.direction === 'positive' ? '+' : ''}{Number(sf.shap).toFixed(4)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
