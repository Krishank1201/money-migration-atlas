import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { SearchBar } from '../components/SearchBar';
import { Chain, TestCase } from '../api/types';
import { Shield, Sparkles, Database, ArrowRight, Activity, Terminal } from 'lucide-react';
import { ConsensusBadge } from '../components/ConsensusBadge';

interface HomePageProps {
  onInvestigate: (chain: Chain | string, address: string) => void;
}

export const HomePage: React.FC<HomePageProps> = ({ onInvestigate }) => {
  const { data: testCases, isLoading: casesLoading } = useQuery<TestCase[]>({
    queryKey: ['testCases'],
    queryFn: api.getTestCases,
    staleTime: 60000,
  });

  return (
    <div className="max-w-6xl mx-auto px-4 py-12">
      {/* Hero Section */}
      <div className="text-center max-w-3xl mx-auto mb-12">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-950/70 border border-cyan-500/40 text-cyan-400 font-mono text-xs font-semibold mb-6 shadow-[0_0_15px_rgba(6,182,212,0.2)]">
          <Terminal className="w-3.5 h-3.5" />
          <span>MINISTRY OF HOME AFFAIRS (MHA) / FIU-IND CYBER FORENSICS</span>
        </div>

        <h1 className="text-4xl sm:text-5xl font-extrabold text-slate-100 tracking-tight mb-4">
          Money Migration Atlas
        </h1>

        <p className="text-base sm:text-lg text-slate-400 leading-relaxed font-sans max-w-2xl mx-auto">
          AI-powered crypto forensics and multi-model attribution engine. Unblended graph traversal,
          supervised machine learning, and structural GNN message-passing for court-admissible evidence.
        </p>
      </div>

      {/* Main Search Bar */}
      <div className="max-w-3xl mx-auto mb-14">
        <SearchBar
          onSearch={(chain, address) => onInvestigate(chain, address)}
        />
        <div className="flex items-center justify-between text-xs text-slate-500 font-mono mt-2.5 px-2">
          <span>Supported: ETH (Mainnet), BTC, TRC-20</span>
          <span>Dual independent scoring: Proximity + Confidence</span>
        </div>
      </div>

      {/* Quick-Launch 12 Benchmark Cases */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 sm:p-8 backdrop-blur shadow-2xl">
        <div className="flex items-center justify-between pb-4 mb-6 border-b border-slate-800 flex-wrap gap-2">
          <div>
            <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2 font-mono">
              <Database className="w-5 h-5 text-cyan-400" />
              BENCHMARK VALIDATION CASES (12 Scenarios)
            </h2>
            <p className="text-xs text-slate-400 font-sans mt-0.5">
              Click any verified benchmark case to launch immediate forensic multi-model attribution.
            </p>
          </div>

          <span className="text-xs font-mono px-3 py-1 rounded bg-slate-800 border border-slate-700 text-cyan-400">
            100% Synthetic Ground Truth
          </span>
        </div>

        {casesLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="h-28 rounded-xl bg-slate-800/40 animate-pulse border border-slate-800" />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5">
            {testCases?.map((c) => (
              <div
                key={c.case_id}
                onClick={() => onInvestigate(c.chain || 'ETH', c.suspect_wallet)}
                className="group p-4 rounded-xl bg-slate-950/70 border border-slate-800/90 hover:border-cyan-500/80 hover:bg-slate-800/50 transition-all cursor-pointer shadow-md hover:shadow-[0_0_20px_rgba(6,182,212,0.15)] flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-center justify-between gap-2 mb-2">
                    <span className="font-mono font-bold text-xs text-cyan-400 bg-cyan-950/80 border border-cyan-800/50 px-2 py-0.5 rounded">
                      {c.case_id}
                    </span>
                    <span className="text-[11px] font-mono font-medium text-slate-400">
                      {c.chain || 'ETH'} • {c.expected_hops || 3} hops
                    </span>
                  </div>

                  <h3 className="font-medium text-slate-200 text-sm mb-1 group-hover:text-cyan-300 transition-colors">
                    {c.name}
                  </h3>

                  <div className="text-xs text-slate-400 font-mono mb-2">
                    Expected: <span className="text-emerald-400 font-semibold">{c.expected_vasp_name}</span>
                  </div>
                </div>

                <div className="flex items-center justify-between pt-2 border-t border-slate-900 mt-2">
                  <ConsensusBadge tier={c.expected_tier} size="sm" />
                  <span className="text-xs text-cyan-400 flex items-center gap-1 font-mono font-medium opacity-0 group-hover:opacity-100 transition-opacity">
                    Inspect <ArrowRight className="w-3.5 h-3.5" />
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
