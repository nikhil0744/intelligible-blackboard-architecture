"""Inference preflight: `python -m scripts.s2_preflight [--backend ollama] [--model ...] [--json PATH]`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from llm_broker import InferenceSettings, build_broker
from llm_broker.preflight import format_report, run_preflight


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default=None, help="ollama | litellm | mock (overrides .env)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-probe", action="store_true", help="skip the real model call")
    ap.add_argument("--json", default=None, help="also write the report to this path")
    args = ap.parse_args(argv)

    settings = InferenceSettings.from_env(backend=args.backend, model=args.model)
    broker = build_broker(settings)
    report = run_preflight(broker, settings, probe=not args.no_probe)
    print(format_report(report))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
