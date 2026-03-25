---
status: active
tags: [infrastructure, diagnostics]
---

# GPU VM Torch Import Cascade — Diagnostics Report

## Environment
- **Hosts:** vgpudesktop9 (nrouter-vm1), vgpudesktop10 (nrouter-vm2)
- **Type:** LXD containers on shared physical host
- **GPU:** RTX 2080 Ti (11GB), NVIDIA driver 570.211, CUDA 12.8
- **NFS:** v3 hard mount at /mnt/nfs/home (sync mode)
- **NVMe:** btrfs at /var/nvme-cache (371GB on vm2, 239GB on vm1)
- **PyTorch:** 2.5.1+cu121 installed on NVMe venv

## Root Cause (confirmed via strace)

`import torch` triggers CUDA initialization which spawns ~30 threads.
Each thread resolves HOME directory for config lookups. HOME points to
NFS (`/mnt/nfs/home/lloven`). Each `openat("/mnt/nfs/home/lloven")`
blocks on NFS v3 hard mount. Threads pile up waiting for NFS responses.

**Evidence (strace on VM2, 2026-03-25):**
- 84,951 total openat calls in 15 seconds
- 290 NFS accesses to `/mnt/nfs/home/lloven`
- PIDs 3396-3422 (30 threads) all blocking on same NFS directory open
- Most calls show `<unfinished ...>` (blocked in NFS)

**Setting HOME=/var/nvme-cache/lloven reduced but did not eliminate the cascade:**
- Load still reached 82 (watchdog killed at threshold 30)
- Remaining NFS accesses likely from: /etc/passwd HOME resolution, 
  venv activate script tilde expansion, Python site.py

## Hypotheses tested

| Hypothesis | Result |
|---|---|
| H1: CUDA version mismatch (cu121 + 12.8 driver) | Not primary cause (NFS is) |
| H2: Persistence mode disabled | Untested (needs sudo) |
| H3: vGPU driver issue | Not primary (NFS is the bottleneck) |
| H4: Process spawning cascade | CONFIRMED: 30 threads blocking on NFS |
| H5: Hidden NFS access | CONFIRMED: 290 NFS calls even with venv on NVMe |
| H6: Triton JIT | Partially mitigated by TRITON_CACHE_DIR on NVMe |
| H7: mmap through LXD overlay | Not tested (NFS was confirmed primary) |

## Watchdog effectiveness

Load-based watchdog (kill at load > 30) successfully killed the cascade
before it reached 10,000+. However, load still reaches 80+ before kill,
and strace adds enough overhead to cascade to 3000+.

## Resolution: Ollama-only architecture

Since `import torch` is fundamentally incompatible with NFS-home LXD containers:
- Use Ollama HTTP API for LLM inference (ollama serve works fine)
- Use Ollama embeddings API for sentence embeddings (replaces sentence-transformers)
- No direct PyTorch import needed in our experiment code
- Refactor: src/embeddings.py to use ollama instead of sentence-transformers

## Alternative fixes (require Jani)
1. `sudo nvidia-smi -pm 1` (persistence mode — may reduce thread spawn)
2. Mount /home as NFS with `soft` instead of `hard` (NFS calls would fail instead of block)
3. Bind-mount HOME to local disk in the LXD container config
4. Create a local user with HOME on /var/nvme-cache (cleanest but most effort)
