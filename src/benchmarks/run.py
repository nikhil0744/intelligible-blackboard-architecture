"""API-free benchmark CLI. Historical mscore simulations are a separate entry point."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_broker.config import InferenceSettings, build_broker

from .batch import check_readiness, resume_batch, run_batch
from .persistence import RunStore
from .preparation import PreparedBenchmark, prepare, resources
from .runtime import Limits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--cpu-smoke", action="store_true")
    prep = commands.add_parser("prepare")
    prep.add_argument("--benchmark", choices=["kramabench", "medagentbench"], default="kramabench")
    prep.add_argument("--cache-dir", type=Path, default=Path("outputs/benchmark-cache"))
    run = commands.add_parser("run")
    run.add_argument("--benchmark", choices=["kramabench", "medagentbench"], default="kramabench")
    run.add_argument("--cache-dir", type=Path, default=Path("outputs/benchmark-cache"))
    run.add_argument("--results-dir", type=Path, required=True)
    run.add_argument("--limit", type=int, default=3)
    run.add_argument("--task-ids", nargs="+")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--max-turns", type=int, default=15)
    run.add_argument("--task-timeout-s", type=float, default=1200)
    run.add_argument("--run-timeout-s", type=float, default=3600)
    run.add_argument("--no-recovery", action="store_true", help="Disable recovery for a baseline comparison")
    run.add_argument("--cpu-smoke", action="store_true")
    run.add_argument("--temporary", action="store_true", help="Explicit ephemeral smoke mode; download results before disconnecting")
    resume = commands.add_parser("resume")
    resume.add_argument("--run-dir", type=Path, required=True)
    resume.add_argument("--cache-dir", type=Path, default=Path("outputs/benchmark-cache"))
    resume.add_argument("--retry-failures", action="store_true")
    resume.add_argument("--cpu-smoke", action="store_true")
    summary = commands.add_parser("summarize")
    summary.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "summarize":
            with RunStore(args.run_dir).writer() as store:
                report = store.summarize()
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "prepare":
            root = prepare(args.cache_dir, args.benchmark)
            print(json.dumps({"cache": str(root), "coverage": PreparedBenchmark.open(args.cache_dir, args.benchmark).coverage()}, indent=2))
            return 0
        if args.command == "run" and args.benchmark == "medagentbench":
            raise ValueError(resources()["medagentbench"]["reason"])
        if args.command == "resume":
            saved = RunStore(args.run_dir).manifest["configuration"]
            settings = InferenceSettings(**saved["inference"])
        else:
            settings = InferenceSettings.from_env()
        with build_broker(settings) as broker:
            if args.command == "doctor":
                report = check_readiness(broker, settings, args.cpu_smoke)
                print(json.dumps({"inference": report, "benchmarks": resources()}, indent=2))
            else:
                prepared = PreparedBenchmark.open(args.cache_dir)
                if args.command == "resume":
                    store, report = resume_batch(args.run_dir, prepared, broker, settings,
                                                  cpu_smoke=args.cpu_smoke, retry_failures=args.retry_failures)
                else:
                    store, report = run_batch(prepared, broker, settings, args.results_dir,
                        limit=args.limit, task_ids=args.task_ids, seed=args.seed,
                        limits=Limits(recovery_enabled=not args.no_recovery, max_turns=args.max_turns, task_timeout_s=args.task_timeout_s,
                                      run_timeout_s=args.run_timeout_s),
                        cpu_smoke=args.cpu_smoke, temporary=args.temporary)
                print(f"Saved run: {store.root}")
                print(json.dumps(report, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Benchmark stopped: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
