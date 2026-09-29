import React from 'react';
import { Activity, Radio, GitBranch, Cpu, Database, AlertTriangle } from 'lucide-react';

export function Header({
  connectionStatus,
  sessionId,
  availableSessions = [],
  onSelectSession,
  activeTab,
  onSelectTab,
  deadlockState,
}) {
  const isConnected = connectionStatus === 'CONNECTED';

  return (
    <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur sticky top-0 z-50 px-6 py-3.5">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        {/* Title and Branding */}
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-blue-600/20 text-blue-400 border border-blue-500/30">
            <Radio className="w-5 h-5 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-bold text-white tracking-tight">
                Intelligible Blackboard
              </h1>
              <span className="text-xs px-2 py-0.5 rounded-full font-mono bg-blue-900/50 text-blue-300 border border-blue-700/50">
                PXP 2.0
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Student 4 Real-Time Reasoning Graph & Ablation Dashboard
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex items-center bg-slate-900 p-1 rounded-lg border border-slate-800">
          <button
            onClick={() => onSelectTab('reasoning')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
              activeTab === 'reasoning'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Live Reasoning & Sandbox
          </button>
          <button
            onClick={() => onSelectTab('ablation')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
              activeTab === 'ablation'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Ablation Benchmarks
          </button>
          <button
            onClick={() => onSelectTab('telemetry')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
              activeTab === 'telemetry'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Raw Telemetry Feed
          </button>
        </div>

        {/* Status Indicators & Session Selector */}
        <div className="flex items-center gap-3">
          {/* Deadlock Badge */}
          {deadlockState && (
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-amber-500/20 text-amber-300 border border-amber-500/40 text-xs font-semibold animate-pulse">
              <AlertTriangle className="w-3.5 h-3.5" />
              <span>DEADLOCK ACTIVE</span>
            </div>
          )}

          {/* Session Dropdown */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">Session:</span>
            <select
              value={sessionId || ''}
              onChange={(e) => onSelectSession(e.target.value)}
              className="bg-slate-900 border border-slate-700 text-xs text-slate-200 rounded-md px-2.5 py-1 focus:ring-1 focus:ring-blue-500 outline-none"
            >
              <option value="">Global Stream</option>
              {availableSessions.map((s) => (
                <option key={s.session_id} value={s.session_id}>
                  {s.session_id.slice(0, 18)}... ({s.event_count} evts)
                </option>
              ))}
            </select>
          </div>

          {/* Connection Status Pill */}
          <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-slate-900 border border-slate-800 text-xs">
            <span
              className={`w-2 h-2 rounded-full ${
                isConnected ? 'bg-emerald-500 animate-ping' : 'bg-rose-500'
              }`}
            />
            <span className={isConnected ? 'text-emerald-400 font-medium' : 'text-rose-400 font-medium'}>
              {isConnected ? 'LIVE WS' : 'DISCONNECTED'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
