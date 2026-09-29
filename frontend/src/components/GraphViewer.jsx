import React, { useState } from 'react';
import { CheckCircle2, RefreshCw, AlertCircle, XCircle, ArrowRight, ShieldCheck } from 'lucide-react';

const TAG_STYLES = {
  RATIFY: {
    bg: 'bg-emerald-950/70',
    border: 'border-emerald-500/60',
    badge: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40',
    icon: CheckCircle2,
    dot: 'bg-emerald-400',
  },
  REVISE: {
    bg: 'bg-blue-950/70',
    border: 'border-blue-500/60',
    badge: 'bg-blue-500/20 text-blue-400 border-blue-500/40',
    icon: RefreshCw,
    dot: 'bg-blue-400',
  },
  REFUTE: {
    bg: 'bg-amber-950/70',
    border: 'border-amber-500/60',
    badge: 'bg-amber-500/20 text-amber-400 border-amber-500/40',
    icon: AlertCircle,
    dot: 'bg-amber-400',
  },
  REJECT: {
    bg: 'bg-rose-950/70',
    border: 'border-rose-500/60',
    badge: 'bg-rose-500/20 text-rose-400 border-rose-500/40',
    icon: XCircle,
    dot: 'bg-rose-400',
  },
};

export function GraphViewer({ contributions = [], activeIndex = null, onSelectNode = () => {} }) {
  const [selectedContribution, setSelectedContribution] = useState(null);

  if (!contributions || contributions.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center p-12 border border-dashed border-slate-800 rounded-xl bg-slate-950/40 text-slate-500 text-sm">
        <div className="p-3 bg-slate-900 rounded-full mb-3 text-slate-400">
          <RefreshCw className="w-6 h-6 animate-spin" />
        </div>
        <span>Awaiting agent submissions on the blackboard...</span>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full space-y-4">
      <div className="flex items-center justify-between pb-2 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-slate-200">Reasoning Trajectory Graph</h2>
          <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">
            {contributions.length} Turns
          </span>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-emerald-400"></span> RATIFY</span>
          <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-blue-400"></span> REVISE</span>
          <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-amber-400"></span> REFUTE</span>
          <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-rose-400"></span> REJECT</span>
        </div>
      </div>

      {/* Trajectory Flow Container */}
      <div className="relative overflow-x-auto p-4 bg-slate-950/70 border border-slate-800 rounded-xl min-h-[380px] flex items-center gap-4">
        {contributions.map((turn, idx) => {
          const tag = turn.tag || 'RATIFY';
          const style = TAG_STYLES[tag] || TAG_STYLES.RATIFY;
          const TagIcon = style.icon;
          const isActive = activeIndex === null || activeIndex >= idx;
          const isSelected = selectedContribution?.contribution_id === turn.contribution_id;

          const claimText = turn.payload?.prediction?.claim || turn.prediction || 'Proposal asserting claim';
          const confidence = Math.round(((turn.payload?.prediction?.confidence ?? turn.confidence ?? 0.8) * 100));

          return (
            <React.Fragment key={turn.contribution_id || idx}>
              {/* Turn Node Card */}
              <div
                onClick={() => {
                  setSelectedContribution(turn);
                  onSelectNode(turn);
                }}
                className={`flex-shrink-0 w-72 p-4 rounded-xl border cursor-pointer transition-all duration-200 ${
                  style.bg
                } ${style.border} ${
                  isSelected ? 'ring-2 ring-blue-400 shadow-lg shadow-blue-500/20 scale-102' : 'hover:border-slate-500'
                } ${!isActive ? 'opacity-30' : 'opacity-100'}`}
              >
                {/* Node Header */}
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono font-bold text-slate-400">
                      T{turn.turn_index ?? idx + 1}
                    </span>
                    <span className="text-xs font-medium text-slate-300 truncate max-w-[100px]">
                      {turn.agent_id || 'Agent'}
                    </span>
                  </div>
                  <span className={`flex items-center gap-1 text-[11px] font-bold px-2 py-0.5 rounded border ${style.badge}`}>
                    <TagIcon className="w-3 h-3" />
                    {tag}
                  </span>
                </div>

                {/* Claim Body */}
                <p className="text-xs text-slate-200 line-clamp-3 mb-3 leading-relaxed font-sans">
                  {claimText}
                </p>

                {/* Confidence Bar */}
                <div className="space-y-1 pt-2 border-t border-slate-800/80">
                  <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                    <span>Confidence</span>
                    <span>{confidence}%</span>
                  </div>
                  <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${style.dot}`}
                      style={{ width: `${confidence}%` }}
                    />
                  </div>
                </div>

                {turn.is_counterfactual && (
                  <div className="mt-2 text-[10px] text-purple-300 bg-purple-950/60 border border-purple-800/40 rounded px-1.5 py-0.5 font-medium">
                    ⚡ Sandbox Counterfactual Injection
                  </div>
                )}
              </div>

              {/* Transition Edge */}
              {idx < contributions.length - 1 && (
                <div className="flex-shrink-0 flex items-center justify-center text-slate-600">
                  <ArrowRight className="w-5 h-5 animate-pulse" />
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>

      {/* Selected Node Details Drawer */}
      {selectedContribution && (
        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/90 text-xs space-y-2">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2">
            <span className="font-semibold text-slate-200">
              Contribution Details: Turn {selectedContribution.turn_index || 1} ({selectedContribution.agent_id})
            </span>
            <button
              onClick={() => setSelectedContribution(null)}
              className="text-slate-400 hover:text-white"
            >
              Close
            </button>
          </div>
          <div>
            <span className="text-slate-400 font-mono">Claim: </span>
            <span className="text-slate-200">
              {selectedContribution.payload?.prediction?.claim || selectedContribution.prediction}
            </span>
          </div>
          <div>
            <span className="text-slate-400 font-mono">Explanation Rationale: </span>
            <span className="text-slate-300">
              {selectedContribution.payload?.explanation?.rationale || selectedContribution.explanation || 'No rationale logged.'}
            </span>
          </div>
          {selectedContribution.payload?.explanation?.evidence?.length > 0 && (
            <div>
              <span className="text-slate-400 font-mono">Evidence Citations: </span>
              <ul className="list-disc list-inside text-slate-300 pl-2">
                {selectedContribution.payload.explanation.evidence.map((ev, i) => (
                  <li key={i}>{ev}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
