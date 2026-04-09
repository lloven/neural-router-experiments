# Neural Router — Remote VM Setup

## Infrastructure

| Host | SSH alias | GPU | NVMe | Status |
|---|---|---|---|---|
| vgpu-host | nrouter-vm | RTX 2080 Ti (11GB) | /var/nvme-cache (239GB) | BLOCKED — crashes on ANY Python execution |
| vgpu-host | nrouter-vm | RTX 2080 Ti (11GB) | /var/nvme-cache (371GB) | UNREACHABLE (2026-04-07) |

## Known issue (updated 2026-04-09)

**Likely root cause: container disk full** (local-infra-admin, 2026-04-09): "On GPU-Desktop6, your container had run out of disk space and may have been the cause of the strange hangs."

A full container disk explains the symptoms:
- Python imports hang on `.pyc` writes
- NFS retries pile up on blocked writes
- Apparent "fork bomb" is actually processes blocked on I/O, accumulating in load average

Confirmed 2026-04-07: `python -c 'print("hello")'` from NVMe venv triggered load 20,373 within 1 second on vm1.

**Before next remote attempt:**
1. Ask local-infra-admin which VMs are now clean (disk-wise)
2. SSH in and check `df -h` on all mounts before any Python execution
3. Clean up old experiment artifacts, logs, and HF cache if disk usage > 80%
4. Verify load is stable for 5+ minutes before launching anything

The previous diagnosis (NFS / vGPU / kernel) was probably wrong — disk-full is a much simpler explanation.

## When fixed — setup protocol (L44)

All operations on NVMe (`/var/nvme-cache/lloven/`), never NFS home.

1. Symlink `~/.cache -> /var/nvme-cache/lloven/.cache`
2. Set in venv activate: `export HF_HOME=/var/nvme-cache/lloven/.cache/huggingface`
3. ONE operation at a time in tmux
4. Check `uptime` between each step (load must be < 5)
5. Step sequence:
   a. `python -c 'print("hello")'` (verify Python works)
   b. `python -c 'import torch; print(torch.cuda.is_available())'` (CUDA)
   c. `python -c 'from sentence_transformers import SentenceTransformer; print("OK")'` (embeddings)
   d. `ollama run qwen2.5:7b "say hello"` (LLM inference)
   e. Full smoke test

## Alternative: Ollama-only architecture

If PyTorch remains unusable, the Neural Router can run without torch:
- LLM inference via Ollama HTTP API (litellm → ollama)
- Embeddings via Ollama embeddings API (ollama pull nomic-embed-text)
- No direct PyTorch import needed
- Requires refactoring src/embeddings.py to use Ollama instead of sentence-transformers
