# Best-configuration manifests

The nine YAML files are grouped by dataset family. They use JSON-compatible
YAML so the runners can read them with Python's standard library. Each entry
contains either:

- `params`: the Optuna parameter dictionary replayed by the model tuner; or
- `config`: a complete nested configuration override.

An entry may also contain `train`, which records runtime-sensitive training
settings such as batch size, maximum epochs, patience, or scheduler period.
The fixed runner applies these values before command-line overrides, so an
explicit CLI option always takes precedence.

The public fixed runner resolves the appropriate family automatically:

```bash
python example/run.py --dataset NCI1 --model GINE --use_best_hyper_params
```

Configurations are intentionally absent for combinations that exhausted the
24 GB memory limit or exceeded the runtime limit. Requesting one through
`--use_best_hyper_params` exits with an explicit OOM/TLE message. CardBench
transfer experiments tune on the source-validation task and therefore do not
use this manifest interface.
