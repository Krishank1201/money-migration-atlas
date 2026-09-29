import React from 'react';
import { CounterfactualItem } from '../api/types';
import { GitFork, CheckCircle, Sliders, AlertCircle } from 'lucide-react';

interface CounterfactualPanelProps {
  counterfactuals: CounterfactualItem[];
}

export const CounterfactualPanel: React.FC<CounterfactualPanelProps> = ({ counterfactuals }) => {
  if (!counterfactuals || counterfactuals.length === 0) {
    return null;
  }

  const getProvenanceBadge = (prov: string) => {
    switch (prov) {
      case 'recomputed':
        return (
          <span className="inline-flex items-center gap-1 text-[11px] font-mono px-2 py-0.5 rounded-full bg-emerald-950/70 border border-emerald-600/50 text-emerald-400">
            <CheckCircle className="w-3 h-3" /> recomputed
          </span>
        );
      case 'estimated_shap':
        return (
          <span className="inline-flex items-center gap-1 text-[11px] font-mono px-2 py-0.5 rounded-full bg-amber-950/70 border border-amber-600/50 text-amber-400">
            <Sliders className="w-3 h-3" /> estimated_shap
          </span>
        );
      case 'qualitative':
      default:
        return (
          <span className="inline-flex items-center gap-1 text-[11px] font-mono px-2 py-0.5 rounded-full bg-cyan-950/70 border border-cyan-700/50 text-cyan-400">
            <AlertCircle className="w-3 h-3" /> qualitative
          </span>
        );
    }
  };

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/90 p-5 shadow-lg">
      <div className="flex items-center gap-2 pb-3 mb-4 border-b border-slate-800">
        <GitFork className="w-4 h-4 text-cyan-400" />
        <h2 className="text-sm font-semibold tracking-wide text-slate-200 uppercase font-mono">
          Counterfactual "What-If" Forensic Scenarios
        </h2>
      </div>

      <div className="space-y-3">
        {counterfactuals.map((cf, idx) => {
          const text = typeof cf === 'string' ? cf : cf.text;
          const prov = typeof cf === 'object' && cf.provenance ? cf.provenance : 'qualitative';
          const method = typeof cf === 'object' && cf.method ? cf.method : 'Forensic projection';

          return (
            <div
              key={idx}
              className="p-3.5 rounded-lg bg-slate-950/70 border border-slate-800/80 hover:border-slate-700 transition-colors"
            >
              <div className="flex items-start justify-between gap-3 mb-1.5 flex-wrap">
                <span className="text-xs font-mono font-bold text-slate-400">
                  Scenario #{idx + 1}
                </span>
                <div className="flex items-center gap-2">
                  {getProvenanceBadge(prov)}
                </div>
              </div>

              <p className="text-sm text-slate-200 leading-relaxed font-sans mb-2">
                "{text}"
              </p>

              <div className="text-[11px] font-mono text-slate-500 flex items-center gap-1">
                <span>Provenance Method:</span>
                <span className="text-slate-400">{method}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
