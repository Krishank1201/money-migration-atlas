import React from 'react';
import { CustodyStep } from '../api/types';
import { Shield, Clock, CheckCircle2 } from 'lucide-react';
import { formatTimestamp } from '../utils/format';

interface ChainOfCustodyProps {
  steps: CustodyStep[];
}

export const ChainOfCustody: React.FC<ChainOfCustodyProps> = ({ steps }) => {
  if (!steps || steps.length === 0) return null;

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/90 p-5 shadow-lg">
      <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <Shield className="w-4 h-4 text-emerald-400" />
          <h2 className="text-sm font-semibold tracking-wide text-slate-200 uppercase font-mono">
            Digital Chain of Custody Audit Trail
          </h2>
        </div>
        <span className="text-xs font-mono text-emerald-400 bg-emerald-950/60 border border-emerald-700/50 px-2 py-0.5 rounded">
          Strictly Monotonic ({steps.length} Steps)
        </span>
      </div>

      <div className="relative pl-6 space-y-4 before:content-[''] before:absolute before:left-2 before:top-2 before:bottom-2 before:w-0.5 before:bg-slate-700">
        {steps.map((step, idx) => (
          <div key={idx} className="relative group">
            {/* Step Marker Node */}
            <div className="absolute -left-[23px] top-1.5 w-3.5 h-3.5 rounded-full bg-slate-900 border-2 border-emerald-400 shadow-[0_0_8px_rgba(16,185,129,0.5)] flex items-center justify-center">
              <div className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
            </div>

            <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800 text-xs">
              <div className="flex items-center justify-between gap-2 flex-wrap mb-1">
                <span className="font-mono font-bold text-slate-200 flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                  {step.step}
                </span>
                <span className="font-mono text-slate-400 flex items-center gap-1 text-[11px]">
                  <Clock className="w-3 h-3 text-cyan-400" />
                  {formatTimestamp(step.timestamp)}
                </span>
              </div>

              <div className="text-[11px] text-slate-400 font-mono mb-1">
                Source: <span className="text-cyan-400">{step.source}</span>
              </div>

              <p className="text-slate-300 font-sans text-xs mt-1">
                {step.notes}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
