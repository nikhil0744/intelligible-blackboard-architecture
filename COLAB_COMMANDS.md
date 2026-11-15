# Colab benchmark commands

Start with [notebooks/colab_benchmarks.ipynb](notebooks/colab_benchmarks.ipynb). It handles the pinned checkout, Drive mount, local Ollama/model setup, data preparation, execution and recorded review. Fresh-Colab/GPU acceptance is pending.

After notebook setup, the equivalent commands are:

```bash
python -m benchmarks.run doctor
python -m benchmarks.run prepare --cache-dir /content/drive/MyDrive/blackboard_benchmarks/cache
python -m benchmarks.run run --limit 1 --cache-dir /content/drive/MyDrive/blackboard_benchmarks/cache --results-dir /content/drive/MyDrive/blackboard_benchmarks/runs
```

Use `--limit 3` for the whole supported profile. Stop after an `attempt_saved` message, reconnect, rerun setup at the same revision, then resume:

```bash
python -m benchmarks.run resume --run-dir /content/drive/MyDrive/blackboard_benchmarks/runs/RUN_ID --cache-dir /content/drive/MyDrive/blackboard_benchmarks/cache
python -m benchmarks.run summarize --run-dir /content/drive/MyDrive/blackboard_benchmarks/runs/RUN_ID
```

Resume skips committed attempts, including incorrect answers. Partial tasks restart with new IDs. Keep the original code, packages and model digest; use one writer per run at a time.

The pilot covers three actual KramaBench legal CSV tasks with local numeric grading. Recovery is disabled. MedAgentBench stays unavailable until its official environment/evaluator work in the same runtime. Historical `benchmarks.mscore` and `scripts.s2_smoke` commands are demonstrations, not official execution of either benchmark.

See the [usage guide](docs/guides/colab-benchmarks.md) for scope, limits, saved files, retries and the acceptance checklist.
