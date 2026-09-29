import React, { useEffect, useRef, useState } from 'react';
import cytoscape, { Core } from 'cytoscape';
import { ZoomIn, ZoomOut, Maximize2, RotateCcw, Info, Copy, Check } from 'lucide-react';
import { copyToClipboard } from '../utils/format';

interface GraphNodeData {
  id: string;
  label: string;
  role: 'suspect' | 'intermediate' | 'mixer' | 'vasp';
  address: string;
  isMixer?: boolean;
  vaspName?: string;
}

interface GraphEdgeData {
  id: string;
  source: string;
  target: string;
  amount?: string;
  txHash?: string;
}

interface GraphViewProps {
  path: string[];
  txHashes?: string[];
  suspectWallet: string;
  targetWallet?: string;
  vaspName?: string;
  mixerAddresses?: string[];
}

export const GraphView: React.FC<GraphViewProps> = ({
  path,
  txHashes = [],
  suspectWallet,
  targetWallet,
  vaspName = 'VASP',
  mixerAddresses = [],
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNodeData | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!containerRef.current || !path || path.length === 0) return;

    // Convert path into Cytoscape elements
    const elements: cytoscape.ElementDefinition[] = [];

    path.forEach((addr, idx) => {
      const isSuspect = addr.toLowerCase() === suspectWallet.toLowerCase() || idx === 0;
      const isVasp = (targetWallet && addr.toLowerCase() === targetWallet.toLowerCase()) || idx === path.length - 1;
      const isMixer = mixerAddresses.some((m) => m.toLowerCase() === addr.toLowerCase());

      let role: GraphNodeData['role'] = 'intermediate';
      let label = `Hop ${idx}`;

      if (isSuspect) {
        role = 'suspect';
        label = `SUSPECT (${addr.slice(0, 6)}...)`;
      } else if (isVasp) {
        role = 'vasp';
        label = `${vaspName.toUpperCase()} DEPOSIT`;
      } else if (isMixer) {
        role = 'mixer';
        label = `MIXER (${addr.slice(0, 6)}...)`;
      } else {
        label = `Intermed. (${addr.slice(0, 6)}...)`;
      }

      elements.push({
        data: {
          id: addr,
          label,
          role,
          address: addr,
          isMixer,
          vaspName: isVasp ? vaspName : undefined,
        },
      });

      // Directed edge to next node
      if (idx < path.length - 1) {
        const nextAddr = path[idx + 1];
        const txHash = txHashes[idx] || `tx_${idx}`;
        elements.push({
          data: {
            id: `edge_${addr}_${nextAddr}`,
            source: addr,
            target: nextAddr,
            txHash,
            amount: `Tx #${idx + 1}`,
          },
        });
      }
    });

    // Initialize Cytoscape
    const cy = cytoscape({
      container: containerRef.current,
      elements,
      layout: {
        name: 'grid', // initial fallback
      },
      style: [
        {
          selector: 'node',
          style: {
            'content': 'data(label)',
            'text-valign': 'bottom',
            'text-margin-y': 8,
            'color': '#cbd5e1',
            'font-family': 'JetBrains Mono, monospace',
            'font-size': '11px',
            'font-weight': 'bold',
            'width': 44,
            'height': 44,
            'border-width': 2,
            'border-color': '#475569',
            'transition-property': 'background-color, border-color, border-width',
            'transition-duration': 0.2,
          },
        },
        {
          selector: 'node[role = "suspect"]',
          style: {
            'background-color': '#ef4444',
            'border-color': '#f87171',
            'border-width': 3,
            'shadow-blur': 15,
            'shadow-color': '#ef4444',
            'shadow-opacity': 0.6,
          },
        },
        {
          selector: 'node[role = "intermediate"]',
          style: {
            'background-color': '#475569',
            'border-color': '#64748b',
          },
        },
        {
          selector: 'node[role = "mixer"]',
          style: {
            'background-color': '#f97316',
            'border-color': '#fb923c',
            'border-width': 3,
            'shadow-blur': 15,
            'shadow-color': '#f97316',
            'shadow-opacity': 0.5,
          },
        },
        {
          selector: 'node[role = "vasp"]',
          style: {
            'background-color': '#10b981',
            'border-color': '#34d399',
            'border-width': 3,
            'shadow-blur': 20,
            'shadow-color': '#10b981',
            'shadow-opacity': 0.7,
          },
        },
        {
          selector: 'node:selected',
          style: {
            'border-width': 4,
            'border-color': '#22d3ee',
            'shadow-blur': 25,
            'shadow-color': '#22d3ee',
            'shadow-opacity': 0.9,
          },
        },
        {
          selector: 'edge',
          style: {
            'width': 2.5,
            'line-color': '#0ea5e9',
            'target-arrow-color': '#0ea5e9',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'arrow-scale': 1.3,
            'label': 'data(amount)',
            'font-family': 'JetBrains Mono, monospace',
            'font-size': '10px',
            'color': '#94a3b8',
            'text-background-color': '#0f172a',
            'text-background-opacity': 0.85,
            'text-background-padding': '2px',
            'text-rotation': 'autorotate',
          },
        },
      ] as any,
      userZoomingEnabled: true,
      userPanningEnabled: true,
      boxSelectionEnabled: false,
    });

    // Run breadthfirst layout horizontally (Left to Right)
    const layout = cy.layout({
      name: 'breadthfirst',
      directed: true,
      roots: [suspectWallet],
      spacingFactor: 1.4,
      padding: 40,
    } as any);

    layout.run();
    cy.fit();

    // Event listeners
    cy.on('tap', 'node', (evt) => {
      const node = evt.target;
      setSelectedNode(node.data());
    });

    cy.on('tap', (evt) => {
      if (evt.target === cy) {
        setSelectedNode(null);
      }
    });

    cyRef.current = cy;

    return () => {
      cy.destroy();
    };
  }, [path, txHashes, suspectWallet, targetWallet, vaspName, mixerAddresses]);

  const handleZoomIn = () => cyRef.current?.zoom(cyRef.current.zoom() * 1.25);
  const handleZoomOut = () => cyRef.current?.zoom(cyRef.current.zoom() * 0.8);
  const handleFit = () => cyRef.current?.fit();
  const handleReset = () => {
    cyRef.current?.reset();
    cyRef.current?.fit();
  };

  const handleCopy = async (text: string) => {
    await copyToClipboard(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative rounded-xl border border-slate-800 bg-slate-950/80 overflow-hidden shadow-xl">
      {/* Top Legend Bar */}
      <div className="flex items-center justify-between px-4 py-2.5 bg-slate-900/90 border-b border-slate-800 text-xs font-mono">
        <div className="flex items-center gap-4 flex-wrap">
          <span className="text-slate-400 font-sans font-medium">Network Legend:</span>
          <span className="flex items-center gap-1.5 text-slate-300">
            <span className="w-2.5 h-2.5 rounded-full bg-red-500 shadow-[0_0_6px_#ef4444]" /> Suspect
          </span>
          <span className="flex items-center gap-1.5 text-slate-300">
            <span className="w-2.5 h-2.5 rounded-full bg-slate-500" /> Intermediate Hop
          </span>
          <span className="flex items-center gap-1.5 text-slate-300">
            <span className="w-2.5 h-2.5 rounded-full bg-orange-500 shadow-[0_0_6px_#f97316]" /> Mixer / Pool
          </span>
          <span className="flex items-center gap-1.5 text-slate-300">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-[0_0_6px_#10b981]" /> Target VASP
          </span>
        </div>

        {/* Viewport Control Buttons */}
        <div className="flex items-center gap-1 bg-slate-800/80 p-1 rounded-lg border border-slate-700/60">
          <button
            onClick={handleZoomIn}
            className="p-1 text-slate-400 hover:text-cyan-400 transition-colors"
            title="Zoom In"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button
            onClick={handleZoomOut}
            className="p-1 text-slate-400 hover:text-cyan-400 transition-colors"
            title="Zoom Out"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <button
            onClick={handleFit}
            className="p-1 text-slate-400 hover:text-cyan-400 transition-colors"
            title="Fit to Canvas"
          >
            <Maximize2 className="w-4 h-4" />
          </button>
          <button
            onClick={handleReset}
            className="p-1 text-slate-400 hover:text-cyan-400 transition-colors"
            title="Reset View"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Cytoscape Canvas Container */}
      <div ref={containerRef} className="w-full h-[420px] bg-slate-950" />

      {/* Node Click Inspector Drawer / Overlay */}
      {selectedNode && (
        <div className="absolute bottom-4 left-4 right-4 md:right-auto md:w-96 bg-slate-900/95 backdrop-blur border border-slate-700 p-3.5 rounded-xl shadow-2xl z-20 text-xs font-mono">
          <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-800">
            <span className="font-bold text-cyan-400 flex items-center gap-1.5">
              <Info className="w-4 h-4" /> NODE INSPECTOR
            </span>
            <span className="text-[10px] uppercase px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
              Role: {selectedNode.role}
            </span>
          </div>

          <div className="space-y-1.5 text-slate-300">
            <div className="flex items-center justify-between">
              <span className="text-slate-500">Address:</span>
              <button
                onClick={() => handleCopy(selectedNode.address)}
                className="flex items-center gap-1 text-slate-200 hover:text-cyan-400"
                title="Copy Address"
              >
                <span>{selectedNode.address.slice(0, 14)}...{selectedNode.address.slice(-8)}</span>
                {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
              </button>
            </div>
            {selectedNode.vaspName && (
              <div className="flex items-center justify-between">
                <span className="text-slate-500">VASP Entity:</span>
                <span className="text-emerald-400 font-semibold">{selectedNode.vaspName}</span>
              </div>
            )}
            <div className="flex items-center justify-between">
              <span className="text-slate-500">Node Status:</span>
              <span>{selectedNode.isMixer ? 'Identified Privacy Mixer' : 'Standard Address'}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
