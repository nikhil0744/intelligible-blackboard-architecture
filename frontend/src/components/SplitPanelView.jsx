import React from 'react';
import { GitBranch, Clock, AlertOctagon, CheckCircle, RefreshCw, Zap } from 'lucide-react';

export function SplitPanelView({
  liveSnapshot,
  counterfactualBranch,
  creditAssignment,
}) {
  const liveContributions = liveSnapshot?.contributions || [];
  const status = liveSnapshot?.status || 'ACTIVE';

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* LEFT PANEL: Live Blackboard Timeline */}
      <div className="flex flex-col rounded-xl border border-slate-800 bg-slate-950/70 p-4 space-y-3">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-blue-400" />
            <h3 className="text-sm font-bold text-slate-100">Live Blackboard Trajectory</h3>
          </div>
          <span className="text-xs px-2.5 py-0.5 rounded-full font-mono font-semibold bg-slate-800 text-slate-300">
            Status: {status}
          </span>
        </div>

        {/* Task Prompt Overview */}
        <div className="bg-slate-900/80 rounded-lg p-3 border border-slate-800 text-xs">
          <span className="text-slate-400 font-mono block mb-1">Problem Statement:</span>
          <p className="text-slate-200">
            {liveSnapshot?.task_description || 'Active multi-agent deliberation task.'}
          </p>
        </div>

        {/* Contributions List */}
        <div className="space-y-2 overflow-y-auto max-h-[420px] pr-1">
          {liveContributions.length === 0 ? (
            <p className="text-xs text-slate-500 italic text-center py-6">
              No live contributions recorded yet.
            </p>
          ) : (
            liveContributions.map((c, i) => (
              <div
                key={c.contribution_id || i}
                className="p-3 rounded-lg border border-slate-800/80 bg-slate-900/60 text-xs flex flex-col gap-1"
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-slate-400">
                    Turn {c.turn_index ?? i + 1} • <strong className="text-slate-300">{c.agent_id}</strong>
                  </span>
                  <span className="font-mono px-2 py-0.5 rounded text-[10px] bg-slate-800 text-slate-300">
                    {c.tag}
                  </span>
                </div>
                <p className="text-slate-200">
                  {c.payload?.prediction?.claim || c.prediction}
                </p>
              </div>
            ))
          )}
        </div>
      </div>

      {/* RIGHT PANEL: Retrospective Sandbox Simulation */}
      <div className="flex flex-col rounded-xl border border-purple-900/50 bg-slate-950/70 p-4 space-y-3">
        <div className="flex items-center justify-between border-b border-purple-900/40 pb-3">
          <div className="flex items-center gap-2">
            <GitBranch className="w-4 h-4 text-purple-400" />
            <h3 className="text-sm font-bold text-purple-200">
              Retrospective Counterfactual Sandbox
            </h3>
          </div>
          <span className="text-xs px-2.5 py-0.5 rounded-full font-mono font-semibold bg-purple-950/80 text-purple-300 border border-purple-800/50">
            Fork Branch: {counterfactualBranch?.branch_id ? counterfactualBranch.branch_id.slice(0, 8) : 'SANDBOX_IDLE'}
          </span>
        </div>

        {/* Blame Attribution & Divergence Point */}
        <div className="bg-purple-950/30 rounded-lg p-3 border border-purple-900/40 text-xs space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-purple-300 flex items-center gap-1.5">
              <Zap className="w-3.5 h-3.5 text-amber-400" />
              Deadlock Divergence Hook
            </span>
            <span className="font-mono text-[11px] text-purple-400">
              Forked at Turn: {counterfactualBranch?.forked_at_turn_index ?? 'N/A'}
            </span>
          </div>
          <p className="text-slate-300 text-[11px]">
            {creditAssignment?.rationale ||
              'When the board enters a circular REFUTE/REJECT loop, the sandbox clones history and calculates credit attribution to determine which prior assertion caused the impasse.'}
          </p>
          {creditAssignment && (
            <div className="mt-2 pt-2 border-t border-purple-900/40 flex items-center justify-between">
              <span className="text-slate-400 font-mono text-[10px]">Causality Blame Score:</span>
              <span className="font-mono text-amber-400 font-bold text-xs">
                {(creditAssignment.causality_score * 100).toFixed(1)}%
              </span>
            </div>
          )}
        </div>

        {/* Simulated Alternative Trajectory */}
        <div className="space-y-2 overflow-y-auto max-h-[420px] pr-1">
          {counterfactualBranch?.simulated_trajectory?.length > 0 ? (
            counterfactualBranch.simulated_trajectory.map((turn, idx) => (
              <div
                key={idx}
                className="p-3 rounded-lg border border-purple-800/40 bg-purple-950/20 text-xs space-y-1"
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-purple-400">
                    Sim Turn {turn.turn_index ?? idx + 1} • <strong className="text-slate-300">{turn.agent_id}</strong>
                  </span>
                  <span className="font-mono px-2 py-0.5 rounded text-[10px] bg-purple-900/60 text-purple-300">
                    {turn.tag}
                  </span>
                </div>
                <p className="text-slate-200">
                  {turn.payload?.prediction?.claim || turn.prediction}
                </p>
              </div>
            ))
          ) : (
            <div className="text-center py-10 space-y-2 text-slate-500 text-xs">
              <RefreshCw className="w-5 h-5 mx-auto text-purple-400/60 animate-spin" />
              <p>Sandbox standby. Simulation triggers automatically upon deadlock.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
