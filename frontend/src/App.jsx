import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { GraphViewer } from './components/GraphViewer';
import { SplitPanelView } from './components/SplitPanelView';
import { BottleneckMonitor } from './components/BottleneckMonitor';
import { PlaybackControls } from './components/PlaybackControls';
import { AblationDashboard } from './components/AblationDashboard';
import { useWebSocket } from './hooks/useWebSocket';

// Pre-packaged demo session for instant inspection when backend is offline
const DEMO_SNAPSHOT = {
  session_id: 'demo_session_pe_diagnosis',
  task_id: 'mscore_sample_01',
  task_description:
    'A patient with acute pulmonary embolism is hemodynamically unstable (BP 80/50). What is the first-line intervention?',
  status: 'RESOLVED',
  active_agents: ['Dr. Cardiologist', 'Dr. Pulmonologist', 'Counterfactual Architect'],
  contributions: [
    {
      contribution_id: 'turn_1',
      turn_index: 1,
      agent_id: 'Dr. Pulmonologist',
      tag: 'REVISE',
      prediction: 'Initiate Subcutaneous Low-Molecular Weight Heparin (LMWH).',
      explanation: 'Anticoagulation halts thrombus propagation while minimizing acute bleed risk.',
      confidence: 0.85,
    },
    {
      contribution_id: 'turn_2',
      turn_index: 2,
      agent_id: 'Dr. Cardiologist',
      tag: 'REFUTE',
      prediction: 'Subcutaneous LMWH is inadequate for massive PE with shock.',
      explanation: 'Patient has hemodynamic collapse (BP 80/50, lactate 4.2). Systemic thrombolysis (Alteplase) is required.',
      confidence: 0.94,
    },
    {
      contribution_id: 'turn_3',
      turn_index: 3,
      agent_id: 'Dr. Pulmonologist',
      tag: 'REJECT',
      prediction: 'Reject Alteplase; bleeding risk outweighs immediate benefit without CT angiography confirmation.',
      explanation: 'Thrombolysis carries a 2-3% intracranial hemorrhage risk; heparin infusion is the safer immediate stabilizer.',
      confidence: 0.89,
    },
    {
      contribution_id: 'turn_4',
      turn_index: 4,
      agent_id: 'Dr. Cardiologist',
      tag: 'REFUTE',
      prediction: 'Patient cannot tolerate diagnostic delays; obstructive shock will cause arrest within minutes.',
      explanation: 'Guidelines mandate immediate thrombolytic reperfusion when systemic hypotension is documented.',
      confidence: 0.97,
    },
    {
      contribution_id: 'turn_5',
      turn_index: 5,
      agent_id: 'Counterfactual Architect',
      tag: 'REVISE',
      is_counterfactual: true,
      prediction: 'Adopt Systemic Thrombolysis (Alteplase) with bedside echocardiogram verification.',
      explanation: 'Counterfactual retrospective simulation demonstrates that withholding thrombolysis leads to terminal cardiac arrest with 88% probability.',
      confidence: 0.99,
    },
    {
      contribution_id: 'turn_6',
      turn_index: 6,
      agent_id: 'Dr. Pulmonologist',
      tag: 'RATIFY',
      prediction: 'Ratify Alteplase systemic thrombolysis.',
      explanation: 'Bedside echo confirms right ventricular strain; counterfactual projection proves survival benefit justifies risk.',
      confidence: 0.98,
    },
  ],
};

const DEMO_COUNTERFACTUAL_BRANCH = {
  branch_id: 'cf_branch_b49e21',
  session_id: 'demo_session_pe_diagnosis',
  forked_at_turn_index: 3,
  divergence_agent_id: 'Dr. Pulmonologist',
  simulated_trajectory: [
    {
      turn_index: 4,
      agent_id: 'Simulated Arbiter',
      tag: 'REVISE',
      prediction: 'Hypothetical: Pulmonologist evaluates echocardiographic RV overload.',
      explanation: 'Simulated trajectory demonstrates immediate hemodynamic stabilization.',
    },
    {
      turn_index: 5,
      agent_id: 'Dr. Cardiologist',
      tag: 'RATIFY',
      prediction: 'Ratify consensus protocol.',
      explanation: 'Resolution achieved; deadlock avoided.',
    },
  ],
};

const DEMO_CREDIT_ASSIGNMENT = {
  agent_id: 'Dr. Pulmonologist',
  turn_index: 3,
  contribution_id: 'turn_3',
  causality_score: 0.82,
  rationale: 'Turn 3 (REJECT) initiated the circular contradiction by anchoring on hemorrhagic risk without acknowledging obstructive shock severity.',
};

export default function App() {
  const [activeTab, setActiveTab] = useState('reasoning');
  const [selectedSessionId, setSelectedSessionId] = useState('');
  const [currentScrubTurn, setCurrentScrubTurn] = useState(6);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [useDemoFallback, setUseDemoFallback] = useState(false);

  const wsUrl = `ws://${window.location.hostname || 'localhost'}:8000/ws/telemetry`;
  const {
    status: wsStatus,
    events,
    latestSnapshot: liveSnapshot,
    activeSessions,
    deadlockState,
    counterfactualBranch: liveCounterfactualBranch,
    creditAssignment: liveCreditAssignment,
  } = useWebSocket(wsUrl, selectedSessionId || null);

  // Fallback to demo snapshot if no live websocket session is active
  const activeSnapshot = liveSnapshot || (useDemoFallback || events.length === 0 ? DEMO_SNAPSHOT : null);
  const activeCounterfactualBranch = liveCounterfactualBranch || (useDemoFallback || events.length === 0 ? DEMO_COUNTERFACTUAL_BRANCH : null);
  const activeCreditAssignment = liveCreditAssignment || (useDemoFallback || events.length === 0 ? DEMO_CREDIT_ASSIGNMENT : null);
  const contributions = activeSnapshot?.contributions || [];
  const totalTurns = contributions.length;

  useEffect(() => {
    setCurrentScrubTurn(totalTurns);
  }, [totalTurns]);

  // Handle Playback Loop
  useEffect(() => {
    let interval;
    if (isPlaying) {
      interval = setInterval(() => {
        setCurrentScrubTurn((prev) => {
          if (prev >= totalTurns) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 1000 / playbackSpeed);
    }
    return () => clearInterval(interval);
  }, [isPlaying, playbackSpeed, totalTurns]);

  const visibleContributions = contributions.slice(0, currentScrubTurn);

  return (
    <div className="min-h-screen bg-[#090d16] text-slate-100 flex flex-col font-sans">
      <Header
        connectionStatus={wsStatus}
        sessionId={selectedSessionId}
        availableSessions={activeSessions}
        onSelectSession={setSelectedSessionId}
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        deadlockState={deadlockState}
      />

      <main className="flex-1 p-6 max-w-7xl w-full mx-auto space-y-6">
        {/* Offline / Demo Notice Bar */}
        {wsStatus !== 'CONNECTED' && (
          <div className="flex items-center justify-between p-3.5 bg-blue-950/40 border border-blue-800/40 rounded-xl text-xs text-blue-300">
            <span>
              ℹ️ WebSocket streaming server is not detected on port 8000. Displaying pre-loaded demo session.
            </span>
            <button
              onClick={() => setUseDemoFallback((v) => !v)}
              className="px-3 py-1 bg-blue-800/60 hover:bg-blue-700 text-white rounded-md text-[11px] font-medium"
            >
              Toggle Demo Data
            </button>
          </div>
        )}

        {/* Tab 1: Live Reasoning & Sandbox View */}
        {activeTab === 'reasoning' && (
          <>
            <BottleneckMonitor
              deadlockState={deadlockState}
              contributions={visibleContributions}
              activeAgents={activeSnapshot?.active_agents || []}
            />

            <PlaybackControls
              currentTurn={currentScrubTurn}
              totalTurns={totalTurns}
              isPlaying={isPlaying}
              playbackSpeed={playbackSpeed}
              onPlayToggle={() => setIsPlaying((p) => !p)}
              onStepBack={() => setCurrentScrubTurn((t) => Math.max(0, t - 1))}
              onStepForward={() => setCurrentScrubTurn((t) => Math.min(totalTurns, t + 1))}
              onReset={() => setCurrentScrubTurn(0)}
              onSpeedChange={setPlaybackSpeed}
              onScrub={setCurrentScrubTurn}
            />

            <GraphViewer
              contributions={visibleContributions}
              activeIndex={currentScrubTurn - 1}
            />

            <SplitPanelView
              liveSnapshot={activeSnapshot}
              counterfactualBranch={activeCounterfactualBranch}
              creditAssignment={activeCreditAssignment}
            />
          </>
        )}

        {/* Tab 2: Ablation Benchmarks */}
        {activeTab === 'ablation' && (
          <AblationDashboard
            onTriggerBatchRun={async () => {
              console.log('Batch run triggered');
            }}
          />
        )}

        {/* Tab 3: Raw Telemetry Stream Feed */}
        {activeTab === 'telemetry' && (
          <div className="p-4 rounded-xl border border-slate-800 bg-slate-950/80 space-y-3 font-mono text-xs">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2">
              <span className="font-bold text-slate-300">Raw Telemetry Event Stream</span>
              <span className="text-slate-500">{events.length} frames received</span>
            </div>
            <div className="max-h-[500px] overflow-y-auto space-y-2">
              {events.length === 0 ? (
                <div className="text-slate-500 py-10 text-center">
                  No live frames captured yet. Events will appear here as the scheduler dispatches turns.
                </div>
              ) : (
                events.map((e, idx) => (
                  <div key={idx} className="p-2.5 rounded bg-slate-900 border border-slate-800 text-[11px]">
                    <span className="text-blue-400 font-bold">[{e.event_type}]</span>{' '}
                    <span className="text-slate-400">({e.timestamp})</span>:
                    <pre className="mt-1 text-slate-300 overflow-x-auto whitespace-pre-wrap">
                      {JSON.stringify(e.payload, null, 2)}
                    </pre>
                  </div>
                ))
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
