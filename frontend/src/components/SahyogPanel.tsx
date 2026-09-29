import React, { useState, useEffect } from 'react';
import {
  Send,
  ShieldAlert,
  CheckCircle2,
  Clock,
  ExternalLink,
  Copy,
  Check,
  AlertTriangle,
  Loader2,
  Building2
} from 'lucide-react';
import { api } from '../api/client';
import { NearestVASPCandidate, SahyogRequest, SahyogAction, SahyogStatus } from '../api/types';

interface SahyogPanelProps {
  candidate: NearestVASPCandidate | null;
  suspectWallet: string;
  chain: string;
}

export const SahyogPanel: React.FC<SahyogPanelProps> = ({
  candidate,
  suspectWallet,
  chain,
}) => {
  const [action, setAction] = useState<SahyogAction>('DISCLOSURE');
  const [request, setRequest] = useState<SahyogRequest | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const isFiuRegistered = candidate?.fiu_ind_registered ?? false;

  // Poll status progression if active
  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | null = null;
    if (request && request.status !== 'ACKNOWLEDGED') {
      interval = setInterval(async () => {
        try {
          const updated = await api.getSahyogStatus(request.request_id);
          setRequest(updated);
          if (updated.status === 'ACKNOWLEDGED' && interval) {
            clearInterval(interval);
          }
        } catch (e) {
          console.error('Error polling SAHYOG status:', e);
        }
      }, 1000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [request]);

  const handleSubmit = async () => {
    if (!candidate || !isFiuRegistered) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.submitSahyogRequest(chain, suspectWallet, action);
      setRequest(res);
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Failed to submit SAHYOG request');
    } finally {
      setLoading(false);
    }
  };

  const copyRequestId = () => {
    if (request?.request_id) {
      navigator.clipboard.writeText(request.request_id);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  if (!candidate) {
    return null;
  }

  return (
    <div className="rounded-xl border border-indigo-900/50 bg-slate-900/90 p-5 shadow-xl mt-6">
      {/* Header with MOCK badge */}
      <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-800 flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <Building2 className="w-5 h-5 text-indigo-400" />
          <h3 className="font-semibold text-slate-100 text-base">
            Route to SAHYOG (MHA / I4C Portal)
          </h3>
          <span className="text-xs px-2 py-0.5 rounded-full font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40 tracking-wider">
            MOCK GATEWAY
          </span>
        </div>
        <span className="text-xs font-mono text-slate-400">
          Target VASP: <strong className="text-slate-200">{candidate.vasp_name}</strong>
        </span>
      </div>

      {/* VASP Regulatory Status */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <div className="rounded-lg bg-slate-950/60 border border-slate-800 p-3.5 flex items-center justify-between">
          <div>
            <div className="text-xs text-slate-400">FIU-IND Regulatory Status</div>
            <div className="text-sm font-semibold text-slate-200 mt-0.5">
              {candidate.vasp_name} ({candidate.vasp_id})
            </div>
          </div>
          {isFiuRegistered ? (
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold bg-emerald-950/80 text-emerald-300 border border-emerald-500/40">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              FIU-IND Registered
            </span>
          ) : (
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold bg-rose-950/80 text-rose-300 border border-rose-500/40">
              <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
              Offshore / Non-Reporting
            </span>
          )}
        </div>

        {/* Action Selection */}
        <div className="rounded-lg bg-slate-950/60 border border-slate-800 p-3.5 flex items-center gap-3">
          <div className="text-xs text-slate-400 shrink-0">Lawful Directive:</div>
          <div className="flex gap-2 w-full">
            <button
              type="button"
              onClick={() => setAction('DISCLOSURE')}
              className={`flex-1 py-1.5 px-2 text-xs rounded font-medium transition-colors border ${
                action === 'DISCLOSURE'
                  ? 'bg-indigo-600/30 border-indigo-500 text-indigo-200'
                  : 'bg-slate-800 border-slate-700 text-slate-400 hover:text-slate-200'
              }`}
            >
              Disclosure Only
            </button>
            <button
              type="button"
              onClick={() => setAction('FREEZE_AND_DISCLOSURE')}
              className={`flex-1 py-1.5 px-2 text-xs rounded font-medium transition-colors border ${
                action === 'FREEZE_AND_DISCLOSURE'
                  ? 'bg-red-600/30 border-red-500 text-red-200'
                  : 'bg-slate-800 border-slate-700 text-slate-400 hover:text-slate-200'
              }`}
            >
              Freeze & Disclosure
            </button>
          </div>
        </div>
      </div>

      {/* Action Button & Warning if offshore */}
      {!isFiuRegistered && (
        <div className="mb-4 p-3 rounded-lg bg-rose-950/30 border border-rose-900/50 flex items-start gap-2.5 text-xs text-rose-300">
          <ShieldAlert className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
          <div>
            <strong>Automated Routing Prohibited:</strong> {candidate.vasp_name} is not registered with FIU-IND. 
            MHA SAHYOG routing is legally reserved for compliant Indian reporting entities. 
            For overseas exchanges, please initiate an MLAT (Mutual Legal Assistance Treaty) or diplomatic letter rogatory.
          </div>
        </div>
      )}

      {error && (
        <div className="mb-4 p-3 rounded-lg bg-rose-950/40 border border-rose-800 text-xs text-rose-300">
          {error}
        </div>
      )}

      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={handleSubmit}
          disabled={!isFiuRegistered || loading}
          className={`flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-xs font-semibold uppercase tracking-wider transition-all shadow-md ${
            isFiuRegistered && !loading
              ? 'bg-indigo-600 hover:bg-indigo-500 text-white cursor-pointer active:scale-95'
              : 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
          }`}
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Routing Request...
            </>
          ) : (
            <>
              <Send className="w-4 h-4" />
              Submit Disclosure Request
            </>
          )}
        </button>

        <span className="text-xs text-slate-400">
          Simulates automated handoff to {candidate.vasp_name} Law Enforcement Desk.
        </span>
      </div>

      {/* Active Request Details & Timeline */}
      {request && (
        <div className="mt-5 p-4 rounded-lg bg-slate-950/80 border border-indigo-500/30">
          <div className="flex items-center justify-between pb-2 mb-3 border-b border-slate-800 flex-wrap gap-2">
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-400">Request ID:</span>
              <code className="text-xs font-mono text-indigo-300 font-bold bg-indigo-950/50 px-2 py-0.5 rounded border border-indigo-800">
                {request.request_id}
              </code>
              <button
                type="button"
                onClick={copyRequestId}
                className="text-slate-400 hover:text-slate-200 transition-colors"
                title="Copy Request ID"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              </button>
            </div>
            <div className="text-xs font-mono text-slate-400">
              Submitted: {new Date(request.created_at).toLocaleTimeString()}
            </div>
          </div>

          {/* Status Timeline */}
          <div className="my-4">
            <div className="text-xs font-medium text-slate-300 mb-2">Routing Lifecycle:</div>
            <div className="grid grid-cols-3 gap-2">
              {/* Step 1: QUEUED */}
              <div
                className={`p-2.5 rounded-lg border text-center transition-all ${
                  request.status === 'QUEUED' || request.status === 'ROUTED' || request.status === 'ACKNOWLEDGED'
                    ? 'bg-indigo-950/40 border-indigo-500 text-indigo-200'
                    : 'bg-slate-900 border-slate-800 text-slate-500'
                }`}
              >
                <div className="text-xs font-bold flex items-center justify-center gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5 text-indigo-400" />
                  1. QUEUED
                </div>
                <div className="text-[10px] text-slate-400 mt-1">Gateway Ingestion</div>
              </div>

              {/* Step 2: ROUTED */}
              <div
                className={`p-2.5 rounded-lg border text-center transition-all ${
                  request.status === 'ROUTED' || request.status === 'ACKNOWLEDGED'
                    ? 'bg-blue-950/40 border-blue-500 text-blue-200'
                    : 'bg-slate-900 border-slate-800 text-slate-500'
                }`}
              >
                <div className="text-xs font-bold flex items-center justify-center gap-1">
                  {request.status === 'ROUTED' || request.status === 'ACKNOWLEDGED' ? (
                    <CheckCircle2 className="w-3.5 h-3.5 text-blue-400" />
                  ) : (
                    <Clock className="w-3.5 h-3.5 text-slate-500" />
                  )}
                  2. ROUTED
                </div>
                <div className="text-[10px] text-slate-400 mt-1">Sent to VASP Nodal</div>
              </div>

              {/* Step 3: ACKNOWLEDGED */}
              <div
                className={`p-2.5 rounded-lg border text-center transition-all ${
                  request.status === 'ACKNOWLEDGED'
                    ? 'bg-emerald-950/40 border-emerald-500 text-emerald-200 shadow-md shadow-emerald-950/50'
                    : 'bg-slate-900 border-slate-800 text-slate-500'
                }`}
              >
                <div className="text-xs font-bold flex items-center justify-center gap-1">
                  {request.status === 'ACKNOWLEDGED' ? (
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                  ) : (
                    <Clock className="w-3.5 h-3.5 text-slate-500" />
                  )}
                  3. ACKNOWLEDGED
                </div>
                <div className="text-[10px] text-slate-400 mt-1">24h SLA Active</div>
              </div>
            </div>
          </div>

          {/* Target VASP Metadata Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs pt-2 border-t border-slate-850">
            <div>
              <span className="text-slate-500">Legal Target:</span>
              <div className="font-medium text-slate-200 truncate">{request.target_vasp.name}</div>
            </div>
            <div>
              <span className="text-slate-500">FIU Reg Number:</span>
              <div className="font-mono text-slate-200 text-[11px] truncate">
                {request.target_vasp.fiu_registration_number}
              </div>
            </div>
            <div>
              <span className="text-slate-500">Evidence Link:</span>
              <div className="font-mono text-indigo-300 text-[11px] truncate">
                {request.attached_evidence_id}
              </div>
            </div>
            <div>
              <span className="text-slate-500">VASP SLA:</span>
              <div className="font-medium text-emerald-300">
                {request.target_vasp.response_sla_hours} Hours
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Mandatory Mock Integration Disclaimer */}
      <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between text-[11px] text-amber-400/90 flex-wrap gap-2">
        <div className="flex items-center gap-1.5">
          <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
          <span>
            <strong>DISCLAIMER:</strong> MOCK INTEGRATION — NOT CONNECTED TO REAL SAHYOG PORTAL.
          </span>
        </div>
        <span className="text-slate-500 text-[10px] font-mono">
          Production endpoint: https://sahyog.gov.in
        </span>
      </div>
    </div>
  );
};
