import React from 'react';
import { AlertTriangle, Clock, Layers, Users, ShieldAlert } from 'lucide-react';

export function BottleneckMonitor({
  deadlockState,
  contributions = [],
  activeAgents = [],
  lockStatus = 'IDLE',
}) {
  // Compute tag breakdown
  const counts = { RATIFY: 0, REVISE: 0, REFUTE: 0, REJECT: 0 };
  contributions.forEach((c) => {
    const t = c.tag;
    if (counts[t] !== undefined) counts[t] += 1;
  });

  const total = contributions.length || 1;
  const refuteRejectRatio = Math.round(((counts.REFUTE + counts.REJECT) / total) * 100);

  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
      {/* Bottleneck Alert */}
      <div className={`p-4 rounded-xl border flex flex-col justify-between ${
        deadlockState
          ? 'bg-rose-950/40 border-rose-500/60 text-rose-200'
          : 'bg-slate-950/70 border-slate-800 text-slate-300'
      }`}>
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Deadlock Detector
          </span>
          {deadlockState ? (
            <ShieldAlert className="w-5 h-5 text-rose-400 animate-bounce" />
          ) : (
            <Clock className="w-4 h-4 text-emerald-400" />
          )}
        </div>
        <div className="mt-2">
          <div className="text-base font-bold">
            {deadlockState ? 'Impasse Triggered' : 'Smooth Deliberation'}
          </div>
          <p className="text-[11px] text-slate-400 mt-1 line-clamp-2">
            {deadlockState?.reason || 'Protocol state machine is progressing without circular contradiction.'}
          </p>
        </div>
      </div>

      {/* Disagreement Ratio */}
      <div className="p-4 rounded-xl border border-slate-800 bg-slate-950/70 flex flex-col justify-between text-slate-300">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Contradiction Ratio
          </span>
          <span className="text-xs font-mono text-amber-400 font-bold">{refuteRejectRatio}%</span>
        </div>
        <div className="mt-2">
          <div className="text-lg font-bold text-slate-100">
            {counts.REFUTE + counts.REJECT} / {contributions.length}
          </div>
          <p className="text-[11px] text-slate-400 mt-1">
            REFUTE + REJECT volume vs overall turns.
          </p>
        </div>
      </div>

      {/* Registered Agents */}
      <div className="p-4 rounded-xl border border-slate-800 bg-slate-950/70 flex flex-col justify-between text-slate-300">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Active Agent Panel
          </span>
          <Users className="w-4 h-4 text-blue-400" />
        </div>
        <div className="mt-2">
          <div className="text-lg font-bold text-slate-100">
            {activeAgents.length > 0 ? activeAgents.length : 3} Agents
          </div>
          <p className="text-[11px] text-slate-400 mt-1">
            Ensemble: Generalist, Domain Critic, Counterfactual
          </p>
        </div>
      </div>

      {/* Lock State */}
      <div className="p-4 rounded-xl border border-slate-800 bg-slate-950/70 flex flex-col justify-between text-slate-300">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Concurrency Write Lock
          </span>
          <Layers className="w-4 h-4 text-purple-400" />
        </div>
        <div className="mt-2">
          <div className="text-base font-bold text-slate-100 uppercase font-mono">
            {lockStatus}
          </div>
          <p className="text-[11px] text-slate-400 mt-1">
            Thread-safe transactional lock arbiter.
          </p>
        </div>
      </div>
    </div>
  );
}
