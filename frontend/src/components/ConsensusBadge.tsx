import React from 'react';
import { ConsensusTier } from '../api/types';
import { CheckCircle2, AlertTriangle, HelpCircle, Zap, Activity } from 'lucide-react';

interface ConsensusBadgeProps {
  tier?: ConsensusTier | string;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export const ConsensusBadge: React.FC<ConsensusBadgeProps> = ({ tier, size = 'md', className = '' }) => {
  const t = tier ? tier.toUpperCase() : 'UNKNOWN';

  let config = {
    bg: 'bg-slate-800',
    border: 'border-slate-700',
    text: 'text-slate-400',
    icon: HelpCircle,
    label: t,
  };

  switch (t) {
    case 'CONFIRMED':
      config = {
        bg: 'bg-emerald-950/60',
        border: 'border-emerald-500/60 shadow-[0_0_12px_rgba(16,185,129,0.25)]',
        text: 'text-emerald-400',
        icon: CheckCircle2,
        label: 'CONFIRMED (Multi-Model Agreement)',
      };
      break;
    case 'AMBIGUOUS':
      config = {
        bg: 'bg-amber-950/60',
        border: 'border-amber-500/60 shadow-[0_0_12px_rgba(245,158,11,0.25)]',
        text: 'text-amber-400',
        icon: AlertTriangle,
        label: 'AMBIGUOUS (Model Divergence)',
      };
      break;
    case 'SINGLE_MODEL':
      config = {
        bg: 'bg-cyan-950/60',
        border: 'border-cyan-500/60 shadow-[0_0_12px_rgba(6,182,212,0.25)]',
        text: 'text-cyan-400',
        icon: Zap,
        label: 'SINGLE_MODEL (High Disagreement)',
      };
      break;
    case 'BEHAVIORAL_ONLY':
      config = {
        bg: 'bg-purple-950/60',
        border: 'border-purple-500/60',
        text: 'text-purple-400',
        icon: Activity,
        label: 'BEHAVIORAL_ONLY (Habitual Match)',
      };
      break;
    case 'UNCERTAIN':
    default:
      config = {
        bg: 'bg-slate-800/80',
        border: 'border-slate-700',
        text: 'text-slate-400',
        icon: HelpCircle,
        label: 'UNCERTAIN (Sub-Threshold)',
      };
      break;
  }

  const Icon = config.icon;
  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5 gap-1',
    md: 'text-sm px-3 py-1 gap-1.5',
    lg: 'text-base px-4 py-1.5 gap-2 font-semibold',
  }[size];

  return (
    <span
      className={`inline-flex items-center rounded-full border font-mono tracking-wide ${config.bg} ${config.border} ${config.text} ${sizeClasses} ${className}`}
      title={`Consensus Agreement Tier: ${t}`}
    >
      <Icon className={size === 'sm' ? 'w-3 h-3' : size === 'lg' ? 'w-5 h-5' : 'w-4 h-4'} />
      <span>{config.label}</span>
    </span>
  );
};
