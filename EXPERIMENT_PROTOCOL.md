# Neural Router Experiment Protocol

This document describes the complete protocol for running Neural Router experiments, both locally and on a remote GPU VM. It covers setup, execution, monitoring, result collection, and troubleshooting.

## Overview

The experiment pipeline is manifest-driven: a JSON manifest tracks all runs with atomic persistence. Runs are dispatched as subprocesses by the orchestrator (`scripts/run_all.py`), each executing `scripts/run_one.py` for a single (stage, dataset, config, seed, model) combination. Results are written incrementally to CSV files. The system supports crash recovery via manifest resume.

### Experiment types

| Stage | Script | Purpose | Where to run |
|---|---|---|---|
| ablation | run_all.py | 7 configs x 3 datasets x 3 backends x 5 seeds | Local (API) + Remote (Ollama) |
| sensitivity | run_all.py | 4 parameter sweeps x 3 datasets x 3 backends | Local (API) + Remote (Ollama) |
| scaling | run_all.py | Event-count scaling up to 5000 | Remote (Ollama) |
| crossover | run_all.py / run_crossover.py | A0 vs A4 under constrained context window | Remote (Ollama, requires GPU) |
| qoe | run_all.py / run_qoe.py | Heterogeneous backend assignment | Remote (needs all 3 backends) |
| baselines | run_all.py | BM25, SBERT, cross-encoder, TF-IDF, GloVe, Word2Vec, BART-MNLI | Local (CPU/GPU, no API) |

### Test levels (per L24)

| Level | Mode | Scope | Duration | When to use |
|---|---|---|---|---|
| Unit smoke | `unit_smoke` | 20 events, 2 configs, 1 dataset, 1 seed | Seconds | After any code change |
| Integration smoke | `integration_smoke` | 100 events, 3 configs, 3 datasets, 1 seed | Minutes | Before deploying to VM |
| Full | `full` | All events, 7 configs, 3 datasets, 5 seeds | Hours-days | Final experiment run |

## Configuration

### params.yaml (root source of truth)

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
      D2: 5000                    # 7.7% stratified subsample
```

New experiment sections:
```yaml
crossover:
  sub_volumes: [50, 100, 200, 500, 1000, 2000]
  configs: [A0, A4]
  max_context_tokens: 4096

qoe:
  strategies: [homogeneous_qwen, homogeneous_haiku, homogeneous_sonnet,
               round_robin, qoe_accuracy_first, qoe_balanced, qoe_cost_first]
  calibration_fraction: 0.1

scaling_subs:
  sub_counts: [50, 100, 200, 500, 1000, 2000, 5000]
  configs: [A0, A4]
  fixed_events: 500
```

### remote.yaml (VM connection)

Copy from `remote.yaml.example` and fill in:
```yaml
ssh_host: nrouter-vm           # SSH config alias
remote_dir: ~/neural-router
venv_path: ~/neural-router/.venv
ollama_model: qwen2.5:7b
```

Or use environment variables: `NROUTER_SSH_HOST`, `NROUTER_REMOTE_DIR`, etc.

## Local Experiment Protocol

### Prerequisites

- Python 3.10+ with venv at `.venv/`
- Anthropic API key in `.env` or environment
- For baselines: `transformers`, `sentence-transformers` installed

### Running locally

```bash
# 1. Smoke test (always run first after code changes)
.venv/bin/python scripts/run_all.py --mode unit_smoke --dry-run
.venv/bin/python scripts/run_all.py --mode unit_smoke

# 2. Integration smoke (before deploying)
.venv/bin/python scripts/run_all.py --mode integration_smoke

# 3. Full run (API-only stages: ablation with Haiku/Sonnet)
.venv/bin/python scripts/run_all.py --mode full --ollama-slots 0 --api-slots 2

# 4. Resume after interruption
.venv/bin/python scripts/run_all.py --mode full --resume
```

### Monitoring locally

```bash
# Watch the manifest
.venv/bin/python scripts/monitor.py

# Check manifest directly
python3 -c "
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

- `--ollama-slots N`: Max concurrent local Ollama runs (default 1, limited by GPU)
- `--api-slots N`: Max concurrent API calls (default 2, limited by rate limits)
- Set `--ollama-slots 0` when running locally without GPU (API-only mode)

## Remote Experiment Protocol

### First-time VM setup

```bash
# 1. Configure SSH access (add to ~/.ssh/config)
Host nrouter-vm
    HostName <VM_IP>
    User <username>
    IdentityFile ~/.ssh/<key>

# 2. Run automated setup (installs Python, Ollama, model, venv)
./scripts/remote/setup-vm.sh

# 3. Deploy code + data + env
./scripts/remote/deploy.sh --setup --env --data
```

### Deployment protocol (before every experiment run)

```bash
# 1. Commit local changes
git add -A && git commit -m "experiment update"

# 2. Deploy to VM (git push + pull + dependency check + Ollama health)
./scripts/remote/deploy.sh

# Or with optional flags:
./scripts/remote/deploy.sh --env    # also transfer .env
./scripts/remote/deploy.sh --data   # also rsync data/
```

The deploy script:
1. Tests SSH connection
2. Pushes code via git remote `vm`
3. Pulls on VM (`git pull --ff-only`)
4. Checks if `requirements.txt` changed, reinstalls if needed
5. Verifies Ollama is running and model available

### Running experiments on VM

```bash
# Smoke test (always first!)
./scripts/remote/run-experiments.sh smoke

# Full experiment sweep
./scripts/remote/run-experiments.sh full

# Resume interrupted run
./scripts/remote/run-experiments.sh full --resume

# Crossover experiments only
./scripts/remote/run-experiments.sh crossover

# Dry run (show plan, don't execute)
./scripts/remote/run-experiments.sh full --dry-run

# Skip code sync (if already deployed)
./scripts/remote/run-experiments.sh full --no-sync --resume
```

All experiments run inside a named tmux session on the VM (`nrouter-smoke`, `nrouter-full`, `nrouter-crossover`). The script refuses to launch if a session already exists (preventing duplicate runs).

### Monitoring remote experiments

```bash
# Quick status check
./scripts/remote/run-experiments.sh status

# Attach to tmux session
ssh nrouter-vm -t 'tmux attach -t nrouter-full'

# VM health check (SSH, GPU, Ollama, disk, manifest)
./scripts/remote/check-vm.sh
```

### Stopping remote experiments

```bash
# Graceful stop (kills orchestrator + run_one processes + tmux sessions)
./scripts/remote/run-experiments.sh stop
```

### Collecting results

```bash
# Pull all results
./scripts/remote/collect-results.sh

# Pull specific mode only
./scripts/remote/collect-results.sh --mode full

# Also pull logs
./scripts/remote/collect-results.sh --mode full --logs
```

Results are merged into the local `results/` directory via rsync. Conflicts are backed up with `.vm-backup` suffix.

## Manifest System

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

On crash recovery (`--resume`), stale `running` entries are reset to `pending`.

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

The orchestrator detects "usage limit" errors and logs them to `OPERATIONS_LOG.md`. To recover:
1. Wait for rate limit reset (or add credits)
2. Resume: `python scripts/run_all.py --mode full --resume`
3. Failed runs are automatically retried on resume

### Stale "running" entries

If the orchestrator crashes, some runs may be stuck as "running". On next launch with `--resume`, these are automatically reset to "pending".

### Ollama out of memory

If Qwen-7B runs OOM on the VM:
1. Check GPU memory: `nvidia-smi`
2. Kill stale Ollama processes: `pkill ollama`
3. Restart: `ollama serve &`
4. Resume experiments

### Wrong model version

Record API model snapshot dates in results. Cloud models may change between runs. The `--mode unit_smoke` test can detect unexpected behavior changes.

### Manifest corruption

The manifest uses atomic write (temp file + rename). If corruption occurs:
1. Check `results/{mode}/manifest.json`
2. Delete the manifest and regenerate: the orchestrator will create a new one and mark existing CSV files as `done` via `migrate_existing_results()`

## Cost Control: Option B (2026-04-06)

Configs A5 (event clustering) and A6 (no cosine filter) are **skipped on D2** to control API cost. Rationale: the paper's revised theory (discrimination capacity, Section 3.6) establishes that D2 (|S|=201) exceeds the LLM's discrimination capacity. All configs struggle equally on D2; A5/A6 add marginal insight at disproportionate cost (~$260 saved). The 22 skipped runs have status `skipped` in the manifest and are excluded from the submission checklist.

Remaining D2 ablation coverage: A0-A4 (Haiku 5 seeds, Sonnet 1 seed) + all 7 baselines. Sufficient for the cross-dataset comparison table and discrimination capacity analysis.

## Checklist: Before Submitting Results

- [ ] All non-skipped runs in manifest are `done` (no `pending` or `failed`; `skipped` is OK)
- [ ] Smoke tests pass on both local and remote
- [ ] Results collected from VM: `./scripts/remote/collect-results.sh --mode full`
- [ ] Per-class F1 breakdown generated for D1 and D3
- [ ] Wilcoxon signed-rank p-values computed
- [ ] Crossover figure generated (A0 vs A4 at increasing |S|)
- [ ] QoE results table populated
- [ ] Machine specs confirmed (GPU model, VRAM, API model snapshot dates)
- [ ] Sonnet single-seed caveat noted in all relevant tables
- [ ] D2 A5/A6 skip noted in experiment section (Option B cost control)
