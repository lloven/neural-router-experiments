# Neural Router Experiment Protocol

This document describes the protocol for reproducing the Neural Router
experiments, both locally and on a remote GPU host. It covers setup,
execution, monitoring, result collection, and troubleshooting.

## Overview

The experiment pipeline is manifest-driven: a JSON manifest tracks all
runs with atomic persistence. Runs are dispatched as subprocesses by the
orchestrator (`scripts/run_all.py`), each executing `scripts/run_one.py`
for a single `(stage, dataset, config, seed, model)` combination. Results
are written incrementally to CSV files; the system supports crash
recovery via manifest resume.

### Experiment types

| Stage       | Script                          | Purpose                                                              | Where to run                   |
|-------------|---------------------------------|----------------------------------------------------------------------|--------------------------------|
| ablation    | `run_all.py`                    | 7 configs × 3 datasets × 3 backends × 5 seeds                        | Local (API) + Remote (Ollama)  |
| sensitivity | `run_all.py`                    | 4 parameter sweeps × 3 datasets × 3 backends                         | Local (API) + Remote (Ollama)  |
| scaling     | `run_all.py`                    | Event-count scaling up to 5,000                                      | Remote (Ollama)                |
| crossover   | `run_all.py` / `run_crossover.py`| A0 vs A4 under a constrained context window                          | Remote (Ollama, requires GPU)  |
| qoe         | `run_all.py` / `run_qoe.py`     | Heterogeneous backend assignment                                     | Remote (requires all backends) |
| baselines   | `run_all.py`                    | BM25, SBERT, cross-encoder, TF-IDF, GloVe, Word2Vec, BART-MNLI       | Local (CPU/GPU, no API)        |

### Test levels

The pipeline supports three levels that share the same code paths and
differ only in scope. Run the smaller levels first when iterating; each
catches the failure modes of the next.

| Level             | `mode`              | Scope                                       | Duration   | When to use                          |
|-------------------|---------------------|---------------------------------------------|------------|--------------------------------------|
| Unit smoke        | `unit_smoke`        | 20 events, 2 configs, 1 dataset, 1 seed     | Seconds    | After any code change                |
| Integration smoke | `integration_smoke` | 100 events, 3 configs, 3 datasets, 1 seed   | Minutes    | Before deploying to a remote host    |
| Full              | `full`              | All events, 7 configs, 3 datasets, 5 seeds  | Hours-days | Final experimental run               |

## Configuration

### `params.yaml` (root source of truth)

Switch test level by changing `mode`:

```yaml
mode: full   # unit_smoke | integration_smoke | full
```

Per-model cost control:

```yaml
llm_models:
  sonnet:
    seed_override: [42]           # single seed (cost)
    max_events_override:
      D2: 5000                    # stratified subsample
```

Experiment sections:

```yaml
crossover:
  sub_volumes: [50, 100, 200, 500, 1000, 2000]
  configs: [A0, A4]
  max_context_tokens: 4096

qoe:
  strategies: [homogeneous, round_robin, qoe_optimised]
  weight_presets: [accuracy_first, balanced, cost_first]
  calibration_fraction: 0.10

scaling_subs:
  sub_counts: [50, 100, 200, 500, 1000, 2000, 5000]
  configs: [A0, A4]
  fixed_events: 500
```

### `remote.yaml` (remote-host connection)

Copy from `remote.yaml.example` and fill in for your environment:

```yaml
ssh_host: <SSH alias for your remote host>
remote_dir: <path to checkout on the remote host>
venv_path: <path to virtualenv on the remote host>
ollama_model: qwen2.5:7b
```

Environment variables (`NROUTER_SSH_HOST`, `NROUTER_REMOTE_DIR`, etc.)
override the file values.

## Local protocol

### Prerequisites

- Python 3.10+ with a virtualenv at `.venv/`.
- `ANTHROPIC_API_KEY` (and/or `OPENAI_API_KEY`) in `.env` or the environment
  if running the API stages.
- `transformers`, `sentence-transformers` installed for the baselines.

### Running locally

```bash
# 1. Smoke test (always run first after code changes)
.venv/bin/python scripts/run_all.py --mode unit_smoke --dry-run
.venv/bin/python scripts/run_all.py --mode unit_smoke

# 2. Integration smoke (before deploying to a remote host)
.venv/bin/python scripts/run_all.py --mode integration_smoke

# 3. Full run, API-only stages (ablation with Haiku / Sonnet)
.venv/bin/python scripts/run_all.py --mode full --ollama-slots 0 --api-slots 2

# 4. Resume after interruption
.venv/bin/python scripts/run_all.py --mode full --resume
```

### Monitoring locally

```bash
# Live dashboard
.venv/bin/python scripts/monitor.py

# Inspect the manifest directly
python -c "
import json
m = json.load(open('results/full/manifest.json'))
runs = m['runs']
done = sum(1 for v in runs.values() if v['status'] == 'done')
failed = sum(1 for v in runs.values() if v['status'] == 'failed')
pending = sum(1 for v in runs.values() if v['status'] == 'pending')
print(f'Done: {done}, Failed: {failed}, Pending: {pending}, Total: {len(runs)}')
"
```

### Slot management

- `--ollama-slots N`: maximum concurrent local Ollama runs (default 1).
- `--api-slots N`: maximum concurrent API calls (default 2).
- Set `--ollama-slots 0` when running locally without a GPU.

## Remote protocol

### First-time host setup

```bash
# 1. Add your remote host to ~/.ssh/config
# 2. Install Python, Ollama, the model, and create the venv:
./scripts/remote/setup-vm.sh
# 3. Deploy code, environment, and data:
./scripts/remote/deploy.sh --setup --env --data
```

### Deployment (before every experiment run)

```bash
# 1. Commit local changes
git add -A && git commit -m "experiment update"

# 2. Deploy to the remote host
./scripts/remote/deploy.sh

# Optional flags
./scripts/remote/deploy.sh --env    # also transfer .env
./scripts/remote/deploy.sh --data   # also rsync data/
```

`deploy.sh` tests the SSH connection, pushes code, pulls on the remote
host, reinstalls dependencies if `requirements.txt` changed, and
verifies that Ollama is running with the expected model.

### Running experiments remotely

```bash
./scripts/remote/run-experiments.sh smoke       # smoke test
./scripts/remote/run-experiments.sh full        # full sweep
./scripts/remote/run-experiments.sh full --resume
./scripts/remote/run-experiments.sh crossover   # crossover only
./scripts/remote/run-experiments.sh full --dry-run
./scripts/remote/run-experiments.sh full --no-sync --resume
```

Experiments run inside named `tmux` sessions on the remote host
(`nrouter-smoke`, `nrouter-full`, `nrouter-crossover`); the script
refuses to launch if a session already exists, preventing duplicate
runs.

### Monitoring remote experiments

```bash
./scripts/remote/run-experiments.sh status        # quick status check
ssh <host> -t 'tmux attach -t nrouter-full'       # attach to session
./scripts/remote/check-vm.sh                      # health check
```

### Stopping remote experiments

```bash
./scripts/remote/run-experiments.sh stop          # graceful kill
```

### Collecting results

```bash
./scripts/remote/collect-results.sh                       # everything
./scripts/remote/collect-results.sh --mode full           # one mode only
./scripts/remote/collect-results.sh --mode full --logs    # include logs
```

Results are merged into the local `results/` directory via `rsync`;
conflicts are backed up with a `.vm-backup` suffix.

## Manifest system

### Run ID format

```
{stage}__{dataset}__{config}__{seed}__{model}
```

Examples:

```
ablation__D1__A3__seed42__qwen7b
sensitivity__D2__sweep_k__seed0__haiku
crossover__D1__A0__sub200__seed42__qwen7b
qoe__D1__qoe_balanced__seed42__mixed
```

### Run lifecycle

```
pending → running → done
                  → failed (with error message)
```

On crash recovery (`--resume`), stale `running` entries are reset to
`pending`.

### Result file paths

```
results/{mode}/ablation/{dataset}_ablation_{model}_results.csv
results/{mode}/sensitivity/{model}/sensitivity_{param}_{dataset}.csv
results/{mode}/scaling/{model}/scaling_events_{dataset}.csv
results/{mode}/crossover/crossover_{dataset}.csv
results/{mode}/qoe/qoe_{dataset}.csv
```

## Troubleshooting

### API rate limit hit

The orchestrator detects API rate-limit errors and writes a note to
`OPERATIONS_LOG.md`. To recover, wait for the limit to reset (or add
credits) and resume: `python scripts/run_all.py --mode full --resume`.
Failed runs are retried on resume.

### Stale `running` entries

If the orchestrator crashes, runs may remain in the `running` state. On
next launch with `--resume`, these are automatically reset to `pending`.

### Ollama out of memory

If a local model runs out of GPU memory: inspect with `nvidia-smi`, kill
stale processes with `pkill ollama`, restart with `ollama serve &`, and
resume the experiment.

### API model snapshot drift

Record the API model snapshot identifiers in results. Cloud models may
be updated by providers between runs; the unit-smoke test can detect
unexpected behaviour changes.

### Manifest corruption

The manifest uses atomic write (temp file + rename). If a corrupted
manifest does appear, delete it and let the orchestrator regenerate the
state from existing CSV files via `migrate_existing_results()`.

## Reproducibility checklist

Before publishing or sharing results:

- [ ] All non-skipped runs in the manifest are `done` (no `pending`,
      no `failed`; `skipped` is acceptable when justified).
- [ ] Smoke tests pass both locally and on the remote host.
- [ ] Results collected from the remote host:
      `./scripts/remote/collect-results.sh --mode full`.
- [ ] Per-class F1 breakdown generated for the relevant datasets.
- [ ] Wilcoxon signed-rank p-values computed where reported.
- [ ] Crossover figure generated (A0 vs A4 at increasing `|S|`).
- [ ] QoE results table populated.
- [ ] Machine specs confirmed (GPU model, VRAM, API model snapshot dates).
- [ ] Single-seed caveats noted in any table that reports a single-seed
      cell.
- [ ] Any deliberate scope reductions (e.g. configurations skipped on a
      particular dataset to control cost) noted in the experiment section
      of the paper.
