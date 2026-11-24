import React, { useState, useMemo, useCallback } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  MarkerType,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { CheckCircle2, RefreshCw, AlertCircle, XCircle } from 'lucide-react';

// ---------------------------------------------------------------------------
// PXP tag colour definitions
// ---------------------------------------------------------------------------
const TAG_META = {
  RATIFY: { border: '#10b981', bg: '#064e3b', text: '#6ee7b7', icon: CheckCircle2, minimap: '#10b981' },
  REVISE: { border: '#3b82f6', bg: '#1e3a5f', text: '#93c5fd', icon: RefreshCw,    minimap: '#3b82f6' },
  REFUTE: { border: '#f59e0b', bg: '#451a03', text: '#fcd34d', icon: AlertCircle,  minimap: '#f59e0b' },
  REJECT: { border: '#f43f5e', bg: '#4c0519', text: '#fca5a5', icon: XCircle,      minimap: '#f43f5e' },
};

const SANDBOX_BORDER  = '#a855f7';
const SANDBOX_BG      = '#2e1065';
const SANDBOX_TEXT    = '#d8b4fe';

// ---------------------------------------------------------------------------
// Custom React Flow node renderer
// ---------------------------------------------------------------------------
function TurnNode({ data }) {
  const meta = TAG_META[data.tag] || TAG_META.RATIFY;
  const Icon = meta.icon;
  const isCF = data.is_counterfactual;
  const border = isCF ? SANDBOX_BORDER : meta.border;
  const bg     = isCF ? SANDBOX_BG     : meta.bg;
  const text   = isCF ? SANDBOX_TEXT   : meta.text;

  return (
    <div
      style={{
        border: `1.5px solid ${border}`,
        backgroundColor: bg,
        borderRadius: 10,
        padding: '10px 12px',
        width: 230,
        boxShadow: data.isActive
          ? `0 0 12px 2px ${border}60`
          : '0 2px 8px rgba(0,0,0,0.5)',
        opacity: data.dimmed ? 0.3 : 1,
        transition: 'opacity 0.2s, box-shadow 0.2s',
        cursor: 'pointer',
        userSelect: 'none',
      }}
    >
      {/* Header row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
        <span style={{ fontSize: 10, fontFamily: 'monospace', color: '#94a3b8' }}>
          T{data.turn_index}
        </span>
        <span
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 3,
            fontSize: 10,
            fontWeight: 700,
            color: text,
            background: `${border}22`,
            border: `1px solid ${border}55`,
            borderRadius: 4,
            padding: '1px 5px',
          }}
        >
          <Icon size={10} />
          {data.tag}
        </span>
      </div>

      {/* Agent name */}
      <div style={{ fontSize: 11, fontWeight: 600, color: '#e2e8f0', marginBottom: 4, lineHeight: 1.3 }}>
        {data.agent_id}
      </div>

      {/* Claim text */}
      <div
        style={{
          fontSize: 10.5,
          color: '#cbd5e1',
          lineHeight: 1.4,
          display: '-webkit-box',
          WebkitLineClamp: 3,
          WebkitBoxOrient: 'vertical',
          overflow: 'hidden',
          marginBottom: 6,
        }}
      >
        {data.claim || 'No claim recorded.'}
      </div>

      {/* Confidence bar */}
      <div style={{ marginBottom: isCF ? 4 : 0 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: '#64748b', marginBottom: 2, fontFamily: 'monospace' }}>
          <span>Confidence</span>
          <span>{data.confidence}%</span>
        </div>
        <div style={{ background: '#1e293b', borderRadius: 9999, height: 4, overflow: 'hidden' }}>
          <div style={{ width: `${data.confidence}%`, height: '100%', background: border, borderRadius: 9999 }} />
        </div>
      </div>

      {isCF && (
        <div style={{ marginTop: 5, fontSize: 9, color: SANDBOX_TEXT, background: `${SANDBOX_BORDER}20`, border: `1px solid ${SANDBOX_BORDER}44`, borderRadius: 4, padding: '2px 6px', fontWeight: 600 }}>
          ⚡ Sandbox Counterfactual
        </div>
      )}
    </div>
  );
}

const NODE_TYPES = { turnNode: TurnNode };

// ---------------------------------------------------------------------------
// Build React Flow graph from contributions array
// ---------------------------------------------------------------------------
function buildGraph(contributions, activeIndex) {
  const nodes = [];
  const edges = [];

  contributions.forEach((c, idx) => {
    const tag     = c.tag || 'RATIFY';
    const claim   = c.payload?.prediction?.claim || c.prediction || '';
    const conf    = Math.round((c.payload?.prediction?.confidence ?? c.confidence ?? 0.8) * 100);
    const isActive = activeIndex === null || activeIndex >= idx;
    const isCF    = Boolean(c.is_counterfactual);

    // Layout: counterfactual nodes on second row (y = 220), normal on first (y = 0)
    nodes.push({
      id: String(c.contribution_id ?? idx),
      type: 'turnNode',
      position: { x: idx * 270, y: isCF ? 220 : 0 },
      data: {
        turn_index: c.turn_index ?? idx + 1,
        agent_id:   c.agent_id || 'Agent',
        tag,
        claim,
        confidence: conf,
        is_counterfactual: isCF,
        isActive,
        dimmed: !isActive,
      },
    });

    // Edge to previous non-CF node or explicit parent
    const parentId = c.target_contribution_id
      ? String(c.target_contribution_id)
      : idx > 0
      ? String(contributions[idx - 1].contribution_id ?? idx - 1)
      : null;

    if (parentId) {
      const meta = TAG_META[tag] || TAG_META.RATIFY;
      edges.push({
        id: `e_${parentId}_${c.contribution_id ?? idx}`,
        source: parentId,
        target: String(c.contribution_id ?? idx),
        label: tag,
        animated: isCF,
        style: { stroke: isCF ? SANDBOX_BORDER : meta.border, strokeWidth: 1.5 },
        labelStyle: { fill: isCF ? SANDBOX_TEXT : meta.text, fontSize: 9, fontWeight: 700 },
        labelBgStyle: { fill: '#0f172a' },
        markerEnd: { type: MarkerType.ArrowClosed, color: isCF ? SANDBOX_BORDER : meta.border },
      });
    }
  });

  return { nodes, edges };
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export function GraphViewer({ contributions = [], activeIndex = null, onSelectNode = () => {} }) {
  const [selectedId, setSelectedId] = useState(null);

  const { nodes: baseNodes, edges: baseEdges } = useMemo(
    () => buildGraph(contributions, activeIndex),
    [contributions, activeIndex]
  );

  const [nodes, , onNodesChange] = useNodesState(baseNodes);
  const [edges, , onEdgesChange] = useEdgesState(baseEdges);

  // Keep nodes/edges in sync when contributions or activeIndex change
  const syncedNodes = useMemo(() => buildGraph(contributions, activeIndex).nodes, [contributions, activeIndex]);
  const syncedEdges = useMemo(() => buildGraph(contributions, activeIndex).edges, [contributions, activeIndex]);

  const handleNodeClick = useCallback((_evt, node) => {
    setSelectedId(node.id);
    const contrib = contributions.find((c) => String(c.contribution_id) === node.id);
    if (contrib) onSelectNode(contrib);
  }, [contributions, onSelectNode]);

  if (!contributions || contributions.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center p-12 border border-dashed border-slate-800 rounded-xl bg-slate-950/40 text-slate-500 text-sm min-h-[380px]">
        <div className="p-3 bg-slate-900 rounded-full mb-3 text-slate-400">
          <RefreshCw className="w-6 h-6 animate-spin" />
        </div>
        <span>Awaiting agent submissions on the blackboard…</span>
      </div>
    );
  }

  const selectedContrib = selectedId
    ? contributions.find((c) => String(c.contribution_id) === selectedId)
    : null;

  return (
    <div className="flex flex-col h-full space-y-4">
      {/* Toolbar */}
      <div className="flex items-center justify-between pb-2 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-slate-200">Reasoning Trajectory Graph</h2>
          <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">
            {contributions.length} Turns
          </span>
        </div>
        <div className="flex items-center gap-3 text-xs text-slate-400">
          {Object.entries(TAG_META).map(([tag, m]) => (
            <span key={tag} className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: m.minimap }} />
              {tag}
            </span>
          ))}
        </div>
      </div>

      {/* React Flow canvas */}
      <div className="relative rounded-xl border border-slate-800 bg-slate-950/70 overflow-hidden" style={{ height: 420 }}>
        <ReactFlow
          nodes={syncedNodes}
          edges={syncedEdges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={handleNodeClick}
          nodeTypes={NODE_TYPES}
          fitView
          fitViewOptions={{ padding: 0.25 }}
          attributionPosition="bottom-right"
          style={{ background: 'transparent' }}
        >
          <Background color="#1e293b" gap={20} size={1} />
          <Controls
            style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
            showInteractive={false}
          />
          <MiniMap
            nodeColor={(n) => {
              const tag = n.data?.tag || 'RATIFY';
              return n.data?.is_counterfactual ? SANDBOX_BORDER : (TAG_META[tag]?.minimap || '#94a3b8');
            }}
            style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8 }}
            maskColor="rgba(0,0,0,0.6)"
          />
        </ReactFlow>
      </div>

      {/* Selected node detail drawer */}
      {selectedContrib && (
        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/90 text-xs space-y-2">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2">
            <span className="font-semibold text-slate-200">
              Contribution Details: Turn {selectedContrib.turn_index || 1} ({selectedContrib.agent_id})
            </span>
            <button onClick={() => setSelectedId(null)} className="text-slate-400 hover:text-white">
              Close
            </button>
          </div>
          <div>
            <span className="text-slate-400 font-mono">Claim: </span>
            <span className="text-slate-200">
              {selectedContrib.payload?.prediction?.claim || selectedContrib.prediction}
            </span>
          </div>
          <div>
            <span className="text-slate-400 font-mono">Explanation: </span>
            <span className="text-slate-300">
              {selectedContrib.payload?.explanation?.rationale || selectedContrib.explanation || 'No rationale logged.'}
            </span>
          </div>
          {selectedContrib.payload?.explanation?.evidence?.length > 0 && (
            <div>
              <span className="text-slate-400 font-mono">Evidence: </span>
              <ul className="list-disc list-inside text-slate-300 pl-2">
                {selectedContrib.payload.explanation.evidence.map((ev, i) => <li key={i}>{ev}</li>)}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
