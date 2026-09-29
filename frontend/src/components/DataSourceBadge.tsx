import React from 'react';
import { Database, ShieldAlert, Cpu } from 'lucide-react';

interface DataSourceBadgeProps {
  source?: string;
  className?: string;
}

export const DataSourceBadge: React.FC<DataSourceBadgeProps> = ({ source = 'synthetic', className = '' }) => {
  const s = source.toLowerCase();

  let config = {
    bg: 'bg-indigo-950/50',
    border: 'border-indigo-600/40',
    text: 'text-indigo-400',
    label: 'SYNTHETIC GRAPH (DEMO)',
    icon: Database,
  };

  if (s.includes('live') || s === 'live_rpc') {
    config = {
      bg: 'bg-emerald-950/50',
      border: 'border-emerald-600/50 shadow-[0_0_8px_rgba(16,185,129,0.2)]',
      text: 'text-emerald-400',
      label: 'LIVE ON-CHAIN RPC',
      icon: Cpu,
    };
  } else if (s.includes('fallback') || s === 'synthetic_fallback') {
    config = {
      bg: 'bg-amber-950/50',
      border: 'border-amber-600/40',
      text: 'text-amber-400',
      label: 'SYNTHETIC FALLBACK',
      icon: ShieldAlert,
    };
  }

  const Icon = config.icon;

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full border text-xs font-mono font-medium tracking-wide ${config.bg} ${config.border} ${config.text} ${className}`}
      title={`Data Ingestion Source: ${source}`}
    >
      <Icon className="w-3 h-3" />
      <span>{config.label}</span>
    </span>
  );
};
