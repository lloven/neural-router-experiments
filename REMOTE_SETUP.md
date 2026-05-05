# Neural Router — Remote GPU Setup

## Status (2026-04-29)

The Qwen track runs on **CSC** (project `project_2018951`). The earlier FCG
vGPU desktop track is **dropped** — those VMs (`vgpu-host/9/10`) crashed
with container disk-full issues in early April 2026 (local-infra-admin diagnosis
2026-04-09) and are not maintained for this campaign.

Anthropic API runs (Haiku, Sonnet) continue to run from the laptop or the
CSC login nodes; they are independent of the GPU host.

## Active infrastructure: CSC Puhti + Mahti

| Cluster | SSH alias | GPU | Walltime cap | Partition (single GPU) | Use case |
|---|---|---|---|---|---|
| **Puhti** | `puhti` | V100 16 GB | 3 days | `gpu` | Primary Qwen-2.5-7B campaign (`scripts/slurm/puhti_qwen7b_ablation.sh`) |
| **Mahti** | `mahti` | A100 40 GB | 1.5 days | `gpusmall` (1 GPU) / `gpumedium` (≥4) | Tier 1c — Qwen-3-8B / Qwen-3-14B / Qwen-2.5-32B (`scripts/slurm/mahti_*.sh`) |

* User: `lovenlau`. Identity: `~/.ssh/csc`.
* Project scratch: `/scratch/project_2018951/neural-router/`
  * `code/` — repo checkout
  * `venv/py312-neural-router/` — Python venv
  * `weights/ollama/`, `weights/hf/` — model caches
  * `bin/ollama-install/` — local Ollama build (no admin rights needed)
  * `logs/` — per-task SLURM stdout/stderr + ollama logs
  * `code/results/full/ablation/by_task/<TAG>/` — per-task SLURM result dirs
* The SLURM `scripts/slurm/*.sh` are the canonical entry points; do **not**
  use `scripts/remote/run-experiments.sh` (it targets the deprecated FCG
  vGPU desktops).

## Submitting work

```bash
ssh puhti
cd /scratch/project_2018951/neural-router/code

# Qwen-2.5-7B full ablation (105-task array; 3-h walltime each).
sbatch scripts/slurm/puhti_qwen7b_ablation.sh
```

```bash
ssh mahti
cd /scratch/project_2018951/neural-router/code

# Tier 1c — newer-model discrimination-capacity test on D2.
# REQUIRED: smoke first (15 min, gputest partition).
sbatch scripts/slurm/mahti_qwen3_8b_smoke.sh
# After smoke succeeds (check the smoke CSV), submit the array:
sbatch scripts/slurm/mahti_qwen3_8b_ablation.sh   # 10 tasks
```

## Reconciling SLURM results back to the local manifest

The local `results/full/manifest.json` does NOT see Puhti/Mahti completions
automatically. Run the reconciliation tool from the laptop:

```bash
# Pull SLURM CSVs locally (~MB).
python scripts/reconcile_puhti.py --pull
# Equivalent for Mahti:
python scripts/reconcile_puhti.py --ssh-host mahti --pull \
    --local-mirror results/mahti_mirror/by_task

# Dry-run first.
python scripts/reconcile_puhti.py --dry-run

# Apply.
python scripts/reconcile_puhti.py --apply
```

The tool validates each per-task CSV per L30 (data row present, F1 non-null,
config/dataset/seed match) before marking the manifest entry done.

## Important non-equivalence with the manuscript matrix

The CSC SLURM scripts cap events to fit a **self-imposed** 3-h walltime
budget — `MAX_EVENTS=1000` for D1/D3 and `MAX_EVENTS=300` for D2
(puhti_qwen7b_ablation.sh, `#SBATCH --time=03:00:00`). This is **not** a
CSC ceiling: Puhti `gpu` partition allows up to 3 days, Mahti `gpusmall`
up to 1.5 days, MaxArraySize is 1001, the `normal` QoS has no per-job
cap. The 3-h budget was chosen for BU conservation and faster fair-share
scheduling, then events were capped to fit.

The manuscript currently says "all events". Before final submission,
either (a) accept the cap and revise the experiment-section text, or (b)
re-run with longer `--time=` and full event sets — Mahti's 1.5-day
walltime is more than enough for D2 at full corpus on A100s.

## Deprecated: FCG vGPU desktops (do not use)

| Host | SSH alias | GPU | Status |
|---|---|---|---|
| vgpu-host | `nrouter-vm` | RTX 2080 Ti | Disk-full / crashing |
| vgpu-host | `nrouter-vm` | RTX 2080 Ti | BLOCKED on Python |
| vgpu-host | `nrouter-vm` | RTX 2080 Ti | UNREACHABLE |

Symptoms (2026-04-07 to 2026-04-09): Python imports hung on `.pyc` writes
because the container disk was full; NFS retries piled up, load average
spiked into the thousands, processes wedged. Diagnosis from local infra admin
2026-04-09 — disk-full container, not vGPU/NFS as initially suspected.

The `scripts/remote/{check-vm,deploy,run-experiments,setup-vm}.sh` family
targets these dead VMs. Kept in-tree for reference only. **Do not run.**
