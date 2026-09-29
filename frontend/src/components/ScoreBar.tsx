import React from 'react';
import { Network, BarChart3, BrainCircuit, Activity } from 'lucide-react';

export type ScoreType = 'proximity' | 'xgboost' | 'gnn' | 'behavioral' | 'consensus';

interface ScoreBarProps {
  type: ScoreType;
  modelName: string;
  value?: number | null | string;
  tierLabel?: string;
  subtitle?: string;
}

export const ScoreBar: React.FC<ScoreBarProps> = ({
  type,
  modelName,
  value,
  tierLabel,
  subtitle,
}) => {
  // Proximity Rank is discrete graph topological distance (hops) - rendered distinctly from probability bars!
  if (type === 'proximity') {
    const hops = typeof value === 'number' ? value : parseInt(String(value || '0'), 10);
    const maxHops = 6;
    return (
      <div className="p-3 bg-slate-900/60 rounded-lg border border-slate-800">
        <div className="flex items-center justify-between text-xs mb-1.5">
          <div className="flex items-center gap-1.5 text-slate-300 font-medium">
            <Network className="w-3.5 h-3.5 text-cyan-400" />
            <span>Topological Proximity</span>
          </div>
          <span className="text-[11px] font-mono text-cyan-400 font-semibold bg-cyan-950/70 border border-cyan-800/60 px-2 py-0.5 rounded">
            {hops} {hops === 1 ? 'hop' : 'hops'}
          </span>
        </div>
        <div className="flex items-center justify-between text-[11px] text-slate-500 mb-2">
          <span>Model: <span className="text-slate-400 font-mono">{modelName}</span></span>
          <span>{subtitle || 'Shortest path traversal'}</span>
        </div>

        {/* Discrete Hop Indicator Segments (Never a continuous probability bar) */}
        <div className="grid grid-cols-6 gap-1.5 h-2">
          {Array.from({ length: maxHops }).map((_, i) => {
            const step = i + 1;
            const isFilled = step <= hops;
            const isTarget = step === hops;
            return (
              <div
                key={i}
                className={`rounded-sm transition-all ${
                  isTarget
                    ? 'bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.6)]'
                    : isFilled
                    ? 'bg-cyan-900 border border-cyan-700/50'
                    : 'bg-slate-800/50'
                }`}
                title={`Hop ${step}`}
              />
            );
          })}
        </div>
        <div className="flex justify-between text-[10px] text-slate-500 font-mono mt-1">
          <span>1 hop (direct)</span>
          <span>6 hops (limit)</span>
        </div>
      </div>
    );
  }

  // Model Probabilistic Confidence Scores (0.0 to 1.0)
  const isInsufficient = value === null || value === undefined || value === 'INSUFFICIENT_DATA';
  const numericVal = typeof value === 'number' ? value : 0;
  const pct = Math.max(0, Math.min(100, Math.round(numericVal * 100)));

  let barColor = 'bg-cyan-500';
  let textColor = 'text-cyan-400';
  let Icon = BarChart3;

  if (type === 'xgboost') {
    Icon = BarChart3;
    barColor = numericVal >= 0.60 ? 'bg-emerald-500' : numericVal >= 0.35 ? 'bg-amber-500' : 'bg-slate-500';
    textColor = numericVal >= 0.60 ? 'text-emerald-400' : numericVal >= 0.35 ? 'text-amber-400' : 'text-slate-400';
  } else if (type === 'gnn') {
    Icon = BrainCircuit;
    barColor = numericVal >= 0.60 ? 'bg-emerald-500' : numericVal >= 0.35 ? 'bg-amber-500' : 'bg-slate-500';
    textColor = numericVal >= 0.60 ? 'text-emerald-400' : numericVal >= 0.35 ? 'text-amber-400' : 'text-slate-400';
  } else if (type === 'behavioral') {
    Icon = Activity;
    barColor = numericVal >= 0.60 ? 'bg-purple-500' : 'bg-purple-800';
    textColor = 'text-purple-400';
  } else if (type === 'consensus') {
    barColor = numericVal >= 0.60 ? 'bg-emerald-400' : 'bg-amber-400';
    textColor = numericVal >= 0.60 ? 'text-emerald-400' : 'text-amber-400';
  }

  return (
    <div className="p-3 bg-slate-900/60 rounded-lg border border-slate-800">
      <div className="flex items-center justify-between text-xs mb-1">
        <div className="flex items-center gap-1.5 text-slate-300 font-medium">
          <Icon className={`w-3.5 h-3.5 ${textColor}`} />
          <span>{type.toUpperCase()} Confidence</span>
        </div>
        <div className="flex items-center gap-2">
          {tierLabel && (
            <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
              {tierLabel}
            </span>
          )}
          <span className={`font-mono text-sm font-bold ${textColor}`}>
            {isInsufficient ? (
              <span className="text-xs text-slate-500 font-normal">INSUFFICIENT_DATA</span>
            ) : (
              `${numericVal.toFixed(2)} (${pct}%)`
            )}
          </span>
        </div>
      </div>

      <div className="flex items-center justify-between text-[11px] text-slate-500 mb-2">
        <span>Model: <span className="text-slate-400 font-mono">{modelName}</span></span>
        <span>{subtitle || 'Probability [0.0 - 1.0]'}</span>
      </div>

      {/* Continuous Probability Bar */}
      <div className="h-2 w-full bg-slate-800/80 rounded-full overflow-hidden p-0.5 border border-slate-700/50">
        {!isInsufficient && (
          <div
            className={`h-full rounded-full transition-all duration-300 ${barColor}`}
            style={{ width: `${pct}%` }}
          />
        )}
      </div>
      <div className="flex justify-between text-[10px] text-slate-500 font-mono mt-1">
        <span>0.00 (Unlikely)</span>
        <span>0.60 (Threshold)</span>
        <span>1.00 (Certain)</span>
      </div>
    </div>
  );
};
