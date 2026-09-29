import React, { useState } from 'react';
import { Search, Loader2 } from 'lucide-react';
import { Chain } from '../api/types';

interface SearchBarProps {
  initialChain?: Chain | string;
  initialAddress?: string;
  onSearch: (chain: Chain, address: string) => void;
  isLoading?: boolean;
}

export const SearchBar: React.FC<SearchBarProps> = ({
  initialChain = 'ETH',
  initialAddress = '',
  onSearch,
  isLoading = false,
}) => {
  const [chain, setChain] = useState<Chain>(initialChain as Chain);
  const [address, setAddress] = useState<string>(initialAddress);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!address.trim()) return;
    onSearch(chain, address.trim());
  };

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div className="flex flex-col sm:flex-row items-center gap-2.5 p-2 bg-slate-900/90 rounded-2xl border border-slate-700/80 shadow-2xl backdrop-blur-lg">
        {/* Chain Dropdown */}
        <div className="w-full sm:w-auto">
          <select
            value={chain}
            onChange={(e) => setChain(e.target.value as Chain)}
            disabled={isLoading}
            className="w-full sm:w-36 bg-slate-800 text-slate-200 font-mono text-xs px-3.5 py-3 rounded-xl border border-slate-700 focus:outline-none focus:border-cyan-400 font-semibold cursor-pointer"
          >
            <option value="ETH">ETH (Ethereum)</option>
            <option value="BTC">BTC (Bitcoin)</option>
            <option value="TRC-20">TRC-20 (Tron)</option>
          </select>
        </div>

        {/* Address Input */}
        <div className="relative flex-1 w-full">
          <input
            type="text"
            placeholder="Enter suspect wallet address (e.g. 0x1072... or 0x3abb...)"
            value={address}
            onChange={(e) => setAddress(e.target.value)}
            disabled={isLoading}
            className="w-full bg-slate-950/80 text-slate-100 placeholder-slate-500 font-mono text-xs sm:text-sm px-4 py-3 rounded-xl border border-slate-700/80 focus:outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 transition-all"
          />
        </div>

        {/* Action Button */}
        <button
          type="submit"
          disabled={isLoading || !address.trim()}
          className="w-full sm:w-auto flex items-center justify-center gap-2 px-6 py-3 rounded-xl bg-cyan-500 hover:bg-cyan-400 active:bg-cyan-600 disabled:opacity-50 text-slate-950 font-mono font-bold text-xs uppercase tracking-wider transition-all shadow-[0_0_20px_rgba(6,182,212,0.35)] flex-shrink-0"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Attributing...</span>
            </>
          ) : (
            <>
              <Search className="w-4 h-4" />
              <span>Investigate</span>
            </>
          )}
        </button>
      </div>
    </form>
  );
};
