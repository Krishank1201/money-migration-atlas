import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { Chain, InvestigationReport } from '../api/types';
import { GraphView } from '../components/GraphView';
import { CandidateCard } from '../components/CandidateCard';
import { ExplanationPanel } from '../components/ExplanationPanel';
import { CounterfactualPanel } from '../components/CounterfactualPanel';
import { ChainOfCustody } from '../components/ChainOfCustody';
import { EvidencePanel } from '../components/EvidencePanel';
import { DataSourceBadge } from '../components/DataSourceBadge';
import { copyToClipboard } from '../utils/format';
import { Copy, Check, ArrowLeft, Loader2, Cpu, Network, ShieldCheck, AlertCircle } from 'lucide-react';

interface InvestigationPageProps {
  chain: Chain | string;
  address: string;
  onBack: () => void;
}

export const InvestigationPage: React.FC<InvestigationPageProps> = ({
  chain,
  address,
  onBack,
}) => {
  const [selectedCandidateIdx, setSelectedCandidateIdx] = useState<number>(0);
  const [copied, setCopied] = useState<boolean>(false);

  const {
    data: report,
    isLoading,
    isError,
    error,
  } = useQuery<InvestigationReport>({
    queryKey: ['investigation', chain, address],
    queryFn: () => api.investigate(chain, address, 6),
    staleTime: 120000,
  });

  const handleCopy = async () => {
    await copyToClipboard(address);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Loading View with Multi-Model Step Indicator
  if (isLoading) {
    return (
      <div className="max-w-6xl mx-auto px-4 py-20 text-center">
        <div className="max-w-md mx-auto p-8 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-2xl backdrop-blur">
          <div className="w-14 h-14 rounded-2xl bg-cyan-950/80 border border-cyan-500/50 flex items-center justify-center mx-auto mb-6 text-cyan-400 animate-pulse">
            <Loader2 className="w-7 h-7 animate-spin" />
          </div>

          <h2 className="text-xl font-bold font-mono text-slate-100 mb-2">
            Forensic Investigation in Progress
          </h2>
          <p className="text-xs text-slate-400 font-mono mb-6">
            Analyzing target: {address.slice(0, 10)}...{address.slice(-6)}
          </p>

          <div className="space-y-3 text-left font-mono text-xs">
            <div className="flex items-center gap-2.5 text-slate-300">
              <Network className="w-4 h-4 text-cyan-400 animate-spin" />
              <span>1. Dijkstra Topological Traversal</span>
            </div>
            <div className="flex items-center gap-2.5 text-slate-300">
              <Cpu className="w-4 h-4 text-emerald-400 animate-pulse" />
              <span>2. XGBoost 20-Feature Inference</span>
            </div>
            <div className="flex items-center gap-2.5 text-slate-300">
              <Cpu className="w-4 h-4 text-indigo-400 animate-pulse" />
              <span>3. GraphSAGE Structural Message Passing</span>
            </div>
            <div className="flex items-center gap-2.5 text-slate-300">
              <ShieldCheck className="w-4 h-4 text-amber-400 animate-pulse" />
              <span>4. Multi-Model Consensus Agreement</span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Error State View
  if (isError || !report) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-20 text-center">
        <div className="p-8 rounded-2xl bg-red-950/30 border border-red-800 text-slate-200">
          <AlertCircle className="w-10 h-10 text-red-400 mx-auto mb-3" />
          <h2 className="text-xl font-bold font-mono text-red-400 mb-2">Attribution Pipeline Failed</h2>
          <p className="text-xs font-mono text-slate-400 mb-6">
            {(error as any)?.response?.data?.detail || 'Unable to connect to backend forensic engine at http://localhost:8000.'}
          </p>
          <button
            onClick={onBack}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-mono font-semibold"
          >
            <ArrowLeft className="w-4 h-4" /> Return to Search
          </button>
        </div>
      </div>
    );
  }

  const candidateExplanations = report.top_3_candidates || [];
  const currentCandidate = candidateExplanations[selectedCandidateIdx]?.candidate || report.evidence_package.top_candidate;
  const currentPath = candidateExplanations[selectedCandidateIdx]?.candidate?.path || report.evidence_package.proximity_path || [];
  const currentTxHashes = candidateExplanations[selectedCandidateIdx]?.candidate?.tx_hashes || report.evidence_package.transaction_hashes || [];

  return (
    <div className="max-w-7xl mx-auto px-4 py-6 space-y-6">
      {/* Back button and navigation */}
      <div>
        <button
          onClick={onBack}
          className="inline-flex items-center gap-1.5 text-xs font-mono text-slate-400 hover:text-cyan-400 transition-colors py-1"
        >
          <ArrowLeft className="w-4 h-4" /> Back to Dashboard
        </button>
      </div>

      {/* TOP ROW: Header Bar */}
      <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/90 shadow-xl backdrop-blur flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1.5 flex-wrap">
            <span className="text-xs font-mono text-slate-500 uppercase tracking-wider font-semibold">
              Suspect Target Wallet
            </span>
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-cyan-400 font-semibold">
              {report.chain}
            </span>
            <DataSourceBadge source={report.data_source || 'synthetic'} />
          </div>

          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-base sm:text-lg font-mono font-bold text-slate-100 tracking-tight">
              {report.suspect_wallet}
            </span>
            <button
              onClick={handleCopy}
              className="p-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-400 hover:text-cyan-400 transition-colors"
              title="Copy Suspect Address"
            >
              {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {/* Recommended Action Badge */}
        <div className="flex-shrink-0">
          <div className="text-right">
            <span className="text-[11px] font-mono text-slate-500 uppercase block mb-1">
              Forensic Triage Verdict
            </span>
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-bold bg-cyan-950/80 border border-cyan-500/60 text-cyan-300">
              {report.recommended_action}
            </span>
          </div>
        </div>
      </div>

      {/* CENTER GRID: 60% Left Column / 40% Right Column */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* LEFT COLUMN (60% -> 7 Cols) */}
        <div className="lg:col-span-7 space-y-6">
          {/* Cytoscape Graph Visualization */}
          <div>
            <div className="flex items-center justify-between mb-2 px-1">
              <span className="text-xs font-mono font-bold uppercase text-slate-400">
                Topological Traversal Pathway ({currentCandidate?.vasp_name || 'VASP'})
              </span>
              <span className="text-xs font-mono text-slate-500">
                {currentPath.length - 1} Hops • {currentPath.length} Nodes
              </span>
            </div>
            <GraphView
              path={currentPath}
              txHashes={currentTxHashes}
              suspectWallet={report.suspect_wallet}
              targetWallet={currentCandidate?.target_wallet}
              vaspName={currentCandidate?.vasp_name}
            />
          </div>

          {/* Plain-Language Forensic Explanation Panel */}
          <ExplanationPanel
            summary={report.plain_language_summary}
            dataSource={report.data_source}
            recommendedAction={report.recommended_action}
          />
        </div>

        {/* RIGHT COLUMN (40% -> 5 Cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="flex items-center justify-between px-1 mb-1">
            <h2 className="text-xs font-mono font-bold uppercase text-slate-400 tracking-wider">
              Candidate VASPs (Top {candidateExplanations.length})
            </h2>
            <span className="text-xs font-mono text-cyan-400">
              Ranked by Consensus
            </span>
          </div>

          {/* Candidate Cards List */}
          {candidateExplanations.map((item, idx) => (
            <CandidateCard
              key={item.candidate.vasp_id}
              candidate={item.candidate}
              rankIndex={idx}
              isSelected={selectedCandidateIdx === idx}
              onSelect={() => setSelectedCandidateIdx(idx)}
            />
          ))}
        </div>
      </div>

      {/* BOTTOM FORENSIC DOSSIER SECTIONS */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 pt-2">
        {/* Counterfactual Panel */}
        <CounterfactualPanel counterfactuals={report.counterfactuals} />

        {/* Chain of Custody Timeline */}
        <ChainOfCustody steps={report.evidence_package.chain_of_custody} />
      </div>

      {/* Court Brief Download Bar */}
      <EvidencePanel
        evidencePackage={report.evidence_package}
        chain={report.chain}
        suspectWallet={report.suspect_wallet}
      />
    </div>
  );
};
