# OpenGLT Model Selection

## Quick Start

From the repository root:

```bash
cd selection
python run_selection.py
```

Results are written to `results/`:

- `main_results.csv`: main LOGO and LODO results
- `fold_results.csv`: fold-level results
- `domain_results.csv`: domain-level results

The `target` column distinguishes the MedQ and P95 experiments.

## Project Structure

```text
.
├── data/
│   └── openglt_selection.npz
├── results/               # Generated result CSV files
├── selection_bench/
│   ├── config.py          # Datasets, methods, and metrics
│   ├── data.py            # Data loading
│   ├── metrics.py         # Evaluation metrics
│   ├── protocol.py        # Evaluation workflow
│   └── selectors.py       # Selector implementations
└── run_selection.py
```
