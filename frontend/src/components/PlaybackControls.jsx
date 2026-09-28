import React from 'react';
import { Play, Pause, SkipBack, SkipForward, RotateCcw } from 'lucide-react';

export function PlaybackControls({
  currentTurn,
  totalTurns,
  isPlaying,
  playbackSpeed,
  onPlayToggle,
  onStepBack,
  onStepForward,
  onReset,
  onSpeedChange,
  onScrub,
}) {
  return (
    <div className="flex flex-col sm:flex-row items-center justify-between gap-4 p-4 rounded-xl border border-slate-800 bg-slate-950/80">
      {/* Buttons */}
      <div className="flex items-center gap-2">
        <button
          onClick={onReset}
          title="Reset to Start"
          className="p-2 rounded-lg bg-slate-900 border border-slate-700 text-slate-300 hover:text-white hover:bg-slate-800"
        >
          <RotateCcw className="w-4 h-4" />
        </button>
        <button
          onClick={onStepBack}
          disabled={currentTurn <= 0}
          title="Step Backward"
          className="p-2 rounded-lg bg-slate-900 border border-slate-700 text-slate-300 hover:text-white disabled:opacity-40"
        >
          <SkipBack className="w-4 h-4" />
        </button>
        <button
          onClick={onPlayToggle}
          title={isPlaying ? 'Pause' : 'Play Live'}
          className="p-2 px-3 rounded-lg bg-blue-600 text-white hover:bg-blue-500 font-medium text-xs flex items-center gap-1.5 shadow"
        >
          {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          <span>{isPlaying ? 'Pause' : 'Play'}</span>
        </button>
        <button
          onClick={onStepForward}
          disabled={currentTurn >= totalTurns}
          title="Step Forward"
          className="p-2 rounded-lg bg-slate-900 border border-slate-700 text-slate-300 hover:text-white disabled:opacity-40"
        >
          <SkipForward className="w-4 h-4" />
        </button>
      </div>

      {/* Scrubber Slider */}
      <div className="flex-1 w-full flex items-center gap-3">
        <span className="text-xs font-mono text-slate-400 min-w-[50px]">
          Turn {currentTurn}/{totalTurns}
        </span>
        <input
          type="range"
          min="0"
          max={totalTurns}
          value={currentTurn}
          onChange={(e) => onScrub(Number(e.target.value))}
          className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
        />
      </div>

      {/* Speed Controls */}
      <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-lg border border-slate-800 text-xs">
        {[0.5, 1, 2, 5].map((speed) => (
          <button
            key={speed}
            onClick={() => onSpeedChange(speed)}
            className={`px-2 py-0.5 rounded font-mono ${
              playbackSpeed === speed
                ? 'bg-blue-600 text-white font-bold'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            {speed}x
          </button>
        ))}
      </div>
    </div>
  );
}
