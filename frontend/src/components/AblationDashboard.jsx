import React, { useState } from 'react';
import { BarChart3, TrendingUp, DollarSign, Award, Play } from 'lucide-react';

const SAMPLE_ABLATION_DATA = [
  {
    density: '0% (Baseline)',
    densityVal: 0.0,
    consensusRate: 54.2,
    meanTurns: 7.8,
    meanTokens: 3850,
    deadlockRecoveryRate: 12.5,
    ultraStrongRate: 0.0,
    color: '#64748B',
  },
  {
    density: '33% (1 Agent)',
    densityVal: 0.33,
    consensusRate: 72.8,
    meanTurns: 6.4,
    meanTokens: 4920,
    deadlockRecoveryRate: 64.0,
    ultraStrongRate: 38.5,
    color: '#3B82F6',
  },
  {
    density: '66% (2 Agents)',
    densityVal: 0.66,
    consensusRate: 88.5,
    meanTurns: 5.9,
    meanTokens: 5640,
    deadlockRecoveryRate: 87.5,
    ultraStrongRate: 74.0,
    color: '#8B5CF6',
  },
  {
    density: '100% (Full)',
    densityVal: 1.0,
    consensusRate: 96.4,
    meanTurns: 5.2,
    meanTokens: 6310,
    deadlockRecoveryRate: 98.2,
    ultraStrongRate: 94.6,
    color: '#10B981',
  },
];

export function AblationDashboard({ onTriggerBatchRun = () => {} }) {
  const [isRunning, setIsRunning] = useState(false);
  const [ablationData, setAblationData] = useState(SAMPLE_ABLATION_DATA);

  const handleRun = async () => {
    setIsRunning(true);
    await onTriggerBatchRun();
    setTimeout(() => {
      setIsRunning(false);
    }, 2000);
  };

  return (
    <div className="space-y-6">
      {/* Header and Trigger Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-5 rounded-xl border border-slate-800 bg-slate-950/70">
        <div>
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-blue-400" />
            Counterfactual Density Ablation Matrix
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Evaluating multi-agent convergence across KramaBench, MSCoRe, and MedAgentBench.
          </p>
        </div>
        <button
          onClick={handleRun}
          disabled={isRunning}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold shadow disabled:opacity-50"
        >
          <Play className="w-3.5 h-3.5" />
          {isRunning ? 'Running Benchmark Batch...' : 'Run Ablation Batch'}
        </button>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 space-y-1">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Peak Convergence</span>
            <TrendingUp className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold text-white">96.4%</div>
          <span className="text-[11px] text-emerald-400">+42.2% over 0% baseline</span>
        </div>

        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 space-y-1">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Deadlock Recovery</span>
            <Award className="w-4 h-4 text-purple-400" />
          </div>
          <div className="text-2xl font-bold text-white">98.2%</div>
          <span className="text-[11px] text-purple-400">At 100% counterfactual density</span>
        </div>

        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 space-y-1">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Mean Token Overhead</span>
            <DollarSign className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-bold text-white">+63.8%</div>
          <span className="text-[11px] text-slate-400">Sandbox replay cost</span>
        </div>

        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 space-y-1">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Ultra-Strong Tier</span>
            <Award className="w-4 h-4 text-blue-400" />
          </div>
          <div className="text-2xl font-bold text-white">94.6%</div>
          <span className="text-[11px] text-blue-400">Michie-Baskar transfer achieved</span>
        </div>
      </div>

      {/* Comparative Grid Table */}
      <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/70">
        <table className="w-full text-left text-xs">
          <thead className="bg-slate-900 border-b border-slate-800 text-slate-400 font-mono">
            <tr>
              <th className="p-3">Counterfactual Density</th>
              <th className="p-3">Consensus Rate</th>
              <th className="p-3">Mean Turns</th>
              <th className="p-3">Token Cost</th>
              <th className="p-3">Deadlock Recovery</th>
              <th className="p-3">Ultra-Strong Intelligibility</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {ablationData.map((row) => (
              <tr key={row.density} className="hover:bg-slate-900/40">
                <td className="p-3 font-semibold text-slate-200 flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: row.color }} />
                  {row.density}
                </td>
                <td className="p-3 font-mono font-bold text-emerald-400">{row.consensusRate}%</td>
                <td className="p-3 font-mono text-slate-300">{row.meanTurns}</td>
                <td className="p-3 font-mono text-slate-300">{row.meanTokens.toLocaleString()}</td>
                <td className="p-3 font-mono text-purple-400">{row.deadlockRecoveryRate}%</td>
                <td className="p-3 font-mono text-blue-400">{row.ultraStrongRate}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
