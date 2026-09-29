import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from './api/client';
import { HomePage } from './pages/HomePage';
import { InvestigationPage } from './pages/InvestigationPage';
import { BenchmarkPage } from './pages/BenchmarkPage';
import { ModelInfoPage } from './pages/ModelInfoPage';
import { Chain } from './api/types';
import {
  Compass,
  Search,
  BarChart2,
  Cpu,
  Shield,
  Activity,
  CheckCircle,
  AlertCircle
} from 'lucide-react';

type Page = 'home' | 'investigation' | 'benchmark' | 'model-info';

export const App: React.FC = () => {
  const [activePage, setActivePage] = useState<Page>('home');
  const [currentChain, setCurrentChain] = useState<Chain | string>('ETH');
  const [currentAddress, setCurrentAddress] = useState<string>('0x107266dbbe3f4b03adb553953e065039657e11da');

  // Backend Health Ping
  const { data: health, isError: healthError } = useQuery({
    queryKey: ['backendHealth'],
    queryFn: api.checkHealth,
    refetchInterval: 30000,
    retry: 1,
  });

  const isOnline = !healthError && health?.status === 'ok';

  const navigateToInvestigation = (chain: Chain | string, address: string) => {
    setCurrentChain(chain);
    setCurrentAddress(address);
    setActivePage('investigation');
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 selection:bg-cyan-500 selection:text-slate-950 font-sans">
      {/* Top Forensic Navigation Header */}
      <header className="sticky top-0 z-50 border-b border-slate-800 bg-slate-900/90 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between gap-4">
          {/* Logo & Platform Name */}
          <div
            onClick={() => setActivePage('home')}
            className="flex items-center gap-3 cursor-pointer group"
          >
            <div className="w-9 h-9 rounded-xl bg-cyan-950/80 border border-cyan-500/50 flex items-center justify-center text-cyan-400 group-hover:scale-105 transition-all shadow-[0_0_12px_rgba(6,182,212,0.3)]">
              <Compass className="w-5 h-5" />
            </div>
            <div>
              <span className="font-mono font-bold text-sm text-slate-100 flex items-center gap-1.5">
                MONEY MIGRATION ATLAS
                <span className="text-[10px] font-normal px-1.5 py-0.2 rounded bg-cyan-950 border border-cyan-800 text-cyan-400">
                  SIH26182
                </span>
              </span>
              <span className="text-[10px] font-mono text-slate-400 block -mt-0.5">
                AI-Powered Crypto Forensics Engine
              </span>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="flex items-center gap-1 sm:gap-2 text-xs font-mono">
            <button
              onClick={() => setActivePage('home')}
              className={`flex items-center gap-1.5 px-3 py-2 rounded-lg transition-colors ${
                activePage === 'home'
                  ? 'bg-slate-800 text-cyan-400 font-bold border border-slate-700'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-850'
              }`}
            >
              <Search className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Search & Cases</span>
            </button>

            {currentAddress && (
              <button
                onClick={() => setActivePage('investigation')}
                className={`flex items-center gap-1.5 px-3 py-2 rounded-lg transition-colors ${
                  activePage === 'investigation'
                    ? 'bg-slate-800 text-cyan-400 font-bold border border-slate-700'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-850'
                }`}
              >
                <Activity className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Active Case</span>
              </button>
            )}

            <button
              onClick={() => setActivePage('benchmark')}
              className={`flex items-center gap-1.5 px-3 py-2 rounded-lg transition-colors ${
                activePage === 'benchmark'
                  ? 'bg-slate-800 text-cyan-400 font-bold border border-slate-700'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-850'
              }`}
            >
              <BarChart2 className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Benchmark Matrix</span>
            </button>

            <button
              onClick={() => setActivePage('model-info')}
              className={`flex items-center gap-1.5 px-3 py-2 rounded-lg transition-colors ${
                activePage === 'model-info'
                  ? 'bg-slate-800 text-cyan-400 font-bold border border-slate-700'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-850'
              }`}
            >
              <Cpu className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Models & Metrics</span>
            </button>
          </nav>

          {/* Backend Status Indicator */}
          <div className="flex items-center gap-2 text-xs font-mono">
            {isOnline ? (
              <span className="hidden md:inline-flex items-center gap-1.5 text-emerald-400 bg-emerald-950/60 border border-emerald-800/60 px-2.5 py-1 rounded-full text-[11px]">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
                API Online (8000)
              </span>
            ) : (
              <span className="hidden md:inline-flex items-center gap-1.5 text-amber-400 bg-amber-950/60 border border-amber-800/60 px-2.5 py-1 rounded-full text-[11px]">
                <AlertCircle className="w-3.5 h-3.5 text-amber-400" />
                API Offline / Check Port 8000
              </span>
            )}
          </div>
        </div>
      </header>

      {/* Main View Area */}
      <main className="flex-1">
        {activePage === 'home' && (
          <HomePage onInvestigate={navigateToInvestigation} />
        )}

        {activePage === 'investigation' && (
          <InvestigationPage
            chain={currentChain}
            address={currentAddress}
            onBack={() => setActivePage('home')}
          />
        )}

        {activePage === 'benchmark' && (
          <BenchmarkPage
            onSelectCase={(chain, addr) => navigateToInvestigation(chain, addr)}
          />
        )}

        {activePage === 'model-info' && <ModelInfoPage />}
      </main>

      {/* Forensic Footer */}
      <footer className="border-t border-slate-800 bg-slate-950 py-6 text-xs font-mono text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Shield className="w-4 h-4 text-cyan-400" />
            <span>Smart India Hackathon (SIH26182) • Forensic Multi-Model Architecture</span>
          </div>
          <div>
            <span>Strict Invariant: never_blended = True</span>
          </div>
        </div>
      </footer>
    </div>
  );
};
