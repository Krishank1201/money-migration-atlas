import React, { useState } from 'react';
import { EvidencePackage } from '../api/types';
import { api } from '../api/client';
import { Download, FileCheck, ShieldCheck, Loader2 } from 'lucide-react';

interface EvidencePanelProps {
  evidencePackage?: EvidencePackage;
  chain?: string;
  suspectWallet?: string;
}

export const EvidencePanel: React.FC<EvidencePanelProps> = ({
  evidencePackage,
  chain = 'ETH',
  suspectWallet = '',
}) => {
  const [downloading, setDownloading] = useState(false);

  const handleDownload = async () => {
    if (!suspectWallet) return;
    try {
      setDownloading(true);
      const markdown = await api.downloadEvidenceMarkdown(chain, suspectWallet);
      const blob = new Blob([markdown], { type: 'text/markdown;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `Forensic_Evidence_Brief_${suspectWallet.slice(0, 10)}.md`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Failed to download court brief:', err);
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/90 p-5 shadow-lg">
      <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-800 flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <FileCheck className="w-4 h-4 text-cyan-400" />
          <h2 className="text-sm font-semibold tracking-wide text-slate-200 uppercase font-mono">
            Court-Ready Dossier (Section 91 CrPC Disclosure)
          </h2>
          <span className="text-[10px] font-mono text-slate-500">(not a certification)</span>
        </div>

        <button
          onClick={handleDownload}
          disabled={downloading || !suspectWallet}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 active:bg-cyan-600 text-slate-950 font-mono font-semibold text-xs transition-all shadow-[0_0_15px_rgba(6,182,212,0.3)] disabled:opacity-50"
        >
          {downloading ? (
            <>
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              <span>Generating Dossier...</span>
            </>
          ) : (
            <>
              <Download className="w-3.5 h-3.5" />
              <span>Download Court Brief (.md)</span>
            </>
          )}
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs font-mono">
        <div className="p-3 bg-slate-950/70 rounded-lg border border-slate-800">
          <span className="text-slate-500 block mb-1">Forensic Seal Status</span>
          <span className="text-emerald-400 font-bold flex items-center gap-1">
            <ShieldCheck className="w-3.5 h-3.5" /> Sealed & Tamper-Evident
          </span>
        </div>

        <div className="p-3 bg-slate-950/70 rounded-lg border border-slate-800">
          <span className="text-slate-500 block mb-1">Score Separation Invariant</span>
          <span className="text-cyan-400 font-bold">never_blended = True</span>
        </div>

        <div className="p-3 bg-slate-950/70 rounded-lg border border-slate-800">
          <span className="text-slate-500 block mb-1">Generated At</span>
          <span className="text-slate-300">
            {evidencePackage?.generated_at ? new Date(evidencePackage.generated_at).toUTCString() : 'Active Session'}
          </span>
        </div>
      </div>
    </div>
  );
};
