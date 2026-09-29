import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { ModelInfoResponse, GNNModelInfoResponse } from '../api/types';
import { Cpu, BrainCircuit, Activity, ShieldCheck, BarChart3, AlertTriangle, Layers, Award } from 'lucide-react';

export const ModelInfoPage: React.FC = () => {
  const { data: xgbInfo } = useQuery<ModelInfoResponse>({
    queryKey: ['xgbModelInfo'],
    queryFn: api.getXGBoostModelInfo,
    staleTime: 120000,
  });

  const { data: gnnInfo } = useQuery<GNNModelInfoResponse>({
    queryKey: ['gnnModelInfo'],
    queryFn: api.getGNNModelInfo,
    staleTime: 120000,
  });

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-8">
      {/* Page Title */}
      <div>
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-cyan-950/70 border border-cyan-500/40 text-cyan-400 font-mono text-xs font-semibold mb-2">
          <Cpu className="w-3.5 h-3.5" /> ARCHITECTURE & VERIFIED METRICS
        </div>
        <h1 className="text-3xl font-extrabold text-slate-100 font-mono">
          Multi-Model AI Forensics Engine
        </h1>
        <p className="text-sm text-slate-400 font-sans mt-1">
          Detailed architectural breakdown, calibration curves, hyperparameter telemetry, and honest ablation verdicts.
        </p>
      </div>

      {/* Model Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* 1. XGBoost Tabular Classifier */}
        <div className="p-6 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-emerald-400" />
              <h2 className="text-base font-bold font-mono text-slate-100">
                XGBoost Supervised Classifier
              </h2>
            </div>
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-emerald-950/70 text-emerald-400 border border-emerald-700/60 font-semibold">
              v1.0 Production
            </span>
          </div>

          <p className="text-xs text-slate-300 font-sans leading-relaxed">
            Extracts 20 clean, non-leaky topological, path quality, amount structuring, and temporal features.
            Trained on balanced positive/negative reachable candidates with zero synthetic volume shortcuts.
          </p>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 font-mono text-center">
            <div className="p-2.5 bg-slate-950/70 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-500 block uppercase">ROC AUC</span>
              <span className="text-lg font-bold text-emerald-400">0.9217</span>
            </div>
            <div className="p-2.5 bg-slate-950/70 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-500 block uppercase">Precision</span>
              <span className="text-lg font-bold text-emerald-400">0.9200</span>
            </div>
            <div className="p-2.5 bg-slate-950/70 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-500 block uppercase">Calibration</span>
              <span className="text-lg font-bold text-cyan-400">0.9000</span>
            </div>
            <div className="p-2.5 bg-slate-950/70 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-500 block uppercase">Brier Loss</span>
              <span className="text-lg font-bold text-slate-200">0.0740</span>
            </div>
          </div>

          {/* Top Feature Importances */}
          {xgbInfo?.feature_importance && xgbInfo.feature_importance.length > 0 && (
            <div className="pt-2">
              <span className="text-xs font-mono font-semibold text-slate-400 block mb-2">
                Top Model Feature Importances:
              </span>
              <div className="space-y-1.5 font-mono text-[11px]">
                {xgbInfo.feature_importance.slice(0, 5).map((fi, i) => (
                  <div key={i} className="flex items-center justify-between bg-slate-950/50 px-2.5 py-1 rounded border border-slate-800/80">
                    <span className="text-slate-300">{fi.feature}</span>
                    <span className="text-cyan-400 font-semibold">{Number(fi.importance).toFixed(4)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* 2. Graph Neural Network (GraphSAGE / GATv2) */}
        <div className="p-6 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div className="flex items-center gap-2">
              <BrainCircuit className="w-5 h-5 text-indigo-400" />
              <h2 className="text-base font-bold font-mono text-slate-100">
                GNN Structural Embeddings
              </h2>
            </div>
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-indigo-950/70 text-indigo-400 border border-indigo-700/60 font-semibold">
              PyTorch Geometric
            </span>
          </div>

          <p className="text-xs text-slate-300 font-sans leading-relaxed">
            Directly perceives graph topological neighborhoods and flow patterns via 2-layer message passing.
            Dual architectures (GraphSAGE inductive pooling + GATv2 multi-head edge attention).
          </p>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 font-mono text-center">
            <div className="p-2.5 bg-slate-950/70 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-500 block uppercase">Benchmark</span>
              <span className="text-lg font-bold text-indigo-400">10 / 12</span>
            </div>
            <div className="p-2.5 bg-slate-950/70 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-500 block uppercase">Accuracy</span>
              <span className="text-lg font-bold text-indigo-400">83.3%</span>
            </div>
            <div className="p-2.5 bg-slate-950/70 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-500 block uppercase">Hidden Dim</span>
              <span className="text-lg font-bold text-slate-200">64</span>
            </div>
            <div className="p-2.5 bg-slate-950/70 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-500 block uppercase">p_null Prior</span>
              <span className="text-lg font-bold text-slate-200">1 / 9</span>
            </div>
          </div>

          <div className="p-3 bg-slate-950/60 rounded-lg border border-slate-800 text-xs font-mono space-y-1">
            <div className="text-slate-400 font-semibold">Architectures Deployed:</div>
            <div className="text-slate-300">• GraphSAGE (Mean aggregator, inductive neighbor sampling)</div>
            <div className="text-slate-300">• GATv2 (Dynamic attention over transaction volume & edge timing)</div>
          </div>
        </div>

        {/* 3. Behavioral Fingerprint Attributor */}
        <div className="p-6 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div className="flex items-center gap-2">
              <Activity className="w-5 h-5 text-purple-400" />
              <h2 className="text-base font-bold font-mono text-slate-100">
                Behavioral Habit Attributor
              </h2>
            </div>
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-amber-950/70 text-amber-400 border border-amber-700/60 font-semibold flex items-center gap-1">
              <AlertTriangle className="w-3 h-3" /> DECORATIVE / AUDIT VERDICT
            </span>
          </div>

          <p className="text-xs text-slate-300 font-sans leading-relaxed">
            Extracts a 64-dimensional behavioral vector capturing inter-transaction delays, gas price preferences,
            and amount structuring.
          </p>

          <div className="p-3.5 bg-amber-950/20 border border-amber-600/40 rounded-lg text-xs font-mono text-amber-200 space-y-1.5">
            <div className="font-bold flex items-center gap-1.5 text-amber-300">
              <AlertTriangle className="w-4 h-4 flex-shrink-0" /> Honest Forensic Audit (Phase 6a):
            </div>
            <p className="font-sans text-[11px] leading-relaxed">
              When a suspect wallet possesses fewer than 3 outgoing transactions, behavioral clustering acts as a
              template heuristic rather than an established human habit. Therefore, behavioral confidence flags
              <span className="font-mono font-bold text-amber-400"> INSUFFICIENT_DATA </span>
              on sparse targets to protect court admissibility.
            </p>
          </div>
        </div>

        {/* 4. Consensus Agreement Layer */}
        <div className="p-6 rounded-2xl bg-slate-900/90 border border-cyan-500/40 shadow-[0_0_20px_rgba(6,182,212,0.1)] space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div className="flex items-center gap-2">
              <ShieldCheck className="w-5 h-5 text-cyan-400" />
              <h2 className="text-base font-bold font-mono text-slate-100">
                Consensus Agreement Layer
              </h2>
            </div>
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-cyan-950/70 text-cyan-400 border border-cyan-700/60 font-semibold">
              Ensemble Gate
            </span>
          </div>

          <p className="text-xs text-slate-300 font-sans leading-relaxed">
            Guarantees independent unblended scoring. Ranking uses <span className="font-mono text-cyan-400 font-bold">consensus_score = max(xgb, gnn, behavioral)</span>,
            while confidence tier assignment verifies multi-dimensional model corroboration.
          </p>

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5 font-mono text-center">
            <div className="p-2.5 bg-slate-950/70 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-500 block uppercase">Accuracy</span>
              <span className="text-lg font-bold text-cyan-300">11 / 12 (91.7%)</span>
            </div>
            <div className="p-2.5 bg-slate-950/70 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-500 block uppercase">Confirmed-Wrong</span>
              <span className="text-lg font-bold text-emerald-400">0 (Zero)</span>
            </div>
            <div className="p-2.5 bg-slate-950/70 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-500 block uppercase">Score Blending</span>
              <span className="text-lg font-bold text-emerald-400">Strictly 0</span>
            </div>
          </div>

          <div className="p-3 bg-slate-950/60 rounded-lg border border-slate-800 text-xs font-mono text-slate-300 space-y-1">
            <span className="font-semibold text-slate-200">Court Invariant Policy:</span>
            <p className="font-sans text-[11px] text-slate-400 leading-relaxed">
              Proximity rank (topological hops) and confidence scores are never mathematically mixed or normalized
              into a single scalar. They remain independent evidence pillars.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
