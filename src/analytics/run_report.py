"""Generate a LaTeX table from a saved benchmark run directory or its ZIP."""
import argparse
from collections import Counter
import json
from pathlib import Path
from zipfile import ZipFile, Path as ZipPath


def render(root):
    summary = json.loads((root / "summary.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    tasks = [r["task_id"] for r in summary["outcomes"]]
    if tasks != [r["task_id"] for r in selection] or len(tasks) != summary["selected"]:
        raise ValueError("Summary outcomes disagree with the saved task selection.")

    outcomes = Counter()
    for record in summary["outcomes"]:
        grade = record.get("grade", {})
        if grade.get("status") == "scored":
            label = ("Full credit" if grade["correct"] else
                     "Partial credit" if grade["score"] > 0 else "Zero credit")
        elif grade.get("status") == "no_final_submission":
            label = "No submission"
        else:
            label = "Pending or ungraded"
        outcomes[label, bool(record.get("consensus"))] += 1

    phases = {}
    for attempt in (root / "attempts").iterdir():
        ledger = attempt / "calls.jsonl"
        if not ledger.is_file():
            continue
        phase_names = {"deliberation": "Main deliberation", "candidate_replay": "Recovery"}
        with ledger.open("r") as stream:
            for line in stream:
                call = json.loads(line)
                counts = phases.setdefault(phase_names.get(call.get("phase"), "Other"), Counter())
                counts["calls"] += 1
                for key in ("prompt_tokens", "completion_tokens"):
                    if call[key] is None:
                        counts["unknown"] += 1
                    else:
                        counts["tokens"] += call[key]

    usage = summary["usage_all_attempts"]
    calls = sum(c["calls"] for c in phases.values())
    tokens = sum(c["tokens"] for c in phases.values())
    if (calls != usage["calls"] or
            tokens != usage["known_prompt_tokens"] + usage["known_completion_tokens"]):
        raise ValueError("Call ledgers disagree with the saved usage summary.")
    consensus = sum(count for (_, voted), count in outcomes.items() if voted)
    if consensus != summary["consensus"]:
        raise ValueError("Consensus totals disagree with the summary.")

    rows = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule",
            r"Task outcome & Consensus & No consensus \\", r"\midrule"]
    def row(label, a, b):
        rows.append(f"{label} & {a} & {b}" + r" \\")
    for label in ("Full credit", "Partial credit", "Zero credit", "No submission", "Pending or ungraded"):
        if outcomes[label, True] or outcomes[label, False]:
            row(label, outcomes[label, True], outcomes[label, False])
    row("All selected", consensus, len(tasks) - consensus)
    rows.extend([r"\midrule", r"Phase & Calls & Tokens \\", r"\midrule"])
    for phase in ("Main deliberation", "Recovery", "Other"):
        if phase in phases:
            c = phases[phase]
            row(phase, c["calls"], "Unknown" if c["unknown"] else f'{c["tokens"]:,}')
    row("All attempts", calls, "Unknown" if usage["total_tokens"] is None else f"{tokens:,}")
    return "\n".join([*rows, r"\bottomrule", r"\end{tabular}", ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.run.is_dir():
        table = render(args.run)
    else:
        with ZipFile(args.run) as archive:
            roots = [name[:-len("summary.json")] for name in archive.namelist()
                     if name.split("/")[-1] == "summary.json"]
            if len(roots) != 1:
                parser.error("Expected exactly one saved run summary in the ZIP.")
            table = render(ZipPath(archive, roots[0]))
    if args.output:
        args.output.write_text(table, encoding="utf-8")
    else:
        print(table, end="")


if __name__ == "__main__":
    main()
