# Saved benchmark examples

These CSV/JSON results and the images in `figures/` preserve the repository's existing simulation outputs unchanged. They demonstrate result formats and publication plotting; they are not evidence of live LLM or end-to-end system performance. Their original exact run configuration was not recorded here.

From the repository root, after installing the project, generate a new small offline example with:

```bash
python -m benchmarks.mscore --mock --trials 8
```

New results go to the ignored `outputs/` directory. To make plots from the preserved JSON:

```python
import json
from pathlib import Path
from contracts.schemas import TrialResult
from analytics.plotting import generate_all_ablation_plots

raw = json.loads(Path("examples/benchmark-results/results.json").read_text())
results = [TrialResult.model_validate(item) for item in raw]
generate_all_ablation_plots(results)  # outputs/figures/
```

Regeneration can differ from these historical outputs; the files here are retained as reference artifacts.
