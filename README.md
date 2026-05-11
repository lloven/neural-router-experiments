# Neural Router Experiments

Reference implementation and experiment harness for the Neural Router paper:
an LLM-driven semantic-matching engine for content-based publish/subscribe.

See the accompanying paper for the experimental design, cost model, and
discussion. This repository provides the source code, configuration, and
SLURM scripts needed to reproduce the empirical results.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure backends and API keys
cp .env.example .env
# Edit .env to set OPENAI_API_KEY / ANTHROPIC_API_KEY as needed

# Download the public datasets
python scripts/fetch_datasets.py
```

## Running experiments

### DVC pipeline (recommended)

The experiment pipeline is managed by [DVC](https://dvc.org/) with three
test levels that share the same code paths and differ only in scope:

| Mode                | Scope                                                | Approximate time |
|---------------------|------------------------------------------------------|------------------|
| `unit_smoke`        | D1 only, 2 configs, 20 events                        | ~1 minute        |
| `integration_smoke` | D1-D3, 3 configs, 100 events                         | ~30 minutes      |
| `full`              | D1-D3, 7 configs, 5 seeds, all baselines             | hours            |

Each level catches the failure modes of the next (imports / API keys /
basic logic at unit; all stage types and config combinations at
integration; full statistical design at `full`). Run the smaller levels
first when iterating.

```bash
# 1. Choose mode (rewrites params.yaml and dvc.yaml)
python scripts/generate_pipeline.py --mode unit_smoke

# 2. Run the pipeline
dvc repro                    # sequential
dvc repro -j 3               # up to 3 stages in parallel

# 3. Monitor progress
python scripts/monitor.py
```

Switching modes preserves prior results (each mode writes to
`results/{mode}/`). DVC caches completed stages.

### LLM backends

| Model                | Type            | Notes                                       |
|----------------------|-----------------|---------------------------------------------|
| `qwen2.5:7b`         | Local (Ollama)  | Open-weight; default for local development. |
| `qwen2.5:32b`        | Local (Ollama)  | Open-weight larger tier (~25 GB Q4).        |
| `qwen2.5:72b`        | Local (Ollama)  | Open-weight; requires two A100 40 GB GPUs.  |
| `claude-3-haiku`     | Anthropic API   | Inexpensive cloud baseline.                 |
| `claude-sonnet-4`    | Anthropic API   | Single-seed by default (cost control).      |

LLM calls flow through [LiteLLM](https://github.com/BerriAI/litellm); see
`src/llm.py`.

### Manual runs

The scripts under `scripts/` are runnable standalone:

```bash
# Single-config ablation
python scripts/run_experiment.py --dataset D1 --configs A3 --max-events 100 \
    --seeds 42 --llm-model ollama/qwen2.5:7b --mode integration_smoke --resume

# Parameter sensitivity sweeps
python scripts/run_sensitivity.py --dataset D1 --sweep all \
    --llm-model ollama/qwen2.5:7b

# Event-count scaling
python scripts/run_scaling.py --dataset D1 --dimension events \
    --llm-model ollama/qwen2.5:7b

# QoE heterogeneous backend assignment
python scripts/run_qoe.py --dataset D1 --strategies homogeneous,round_robin,qoe_optimised \
    --weight-presets balanced --seeds 42 --calibration-fraction 0.10 \
    --backends "tier_mid:ollama/qwen2.5:7b,tier_large:ollama/qwen2.5:32b" \
    --output-dir results/qoe_demo
```

### Monitoring

`python scripts/monitor.py` reads `params.yaml` and the per-stage
`.progress.json` files written by the runners; it reports per-stage
progress, within-stage event-level progress, and ETA estimates.

### Checkpointing and resume

All runners write intermediate results so an interrupted run can resume:

- **Ablation:** pickle checkpoints every N events (5 for Qwen, 25 for
  Haiku, 50 for Sonnet); `--resume` skips completed `(config, seed)` pairs.
- **Sensitivity / scaling / QoE:** rows appended to CSV immediately;
  resume skips completed cells.

## Datasets

| ID | Dataset                    | Events | Subscriptions | Domain                            |
|----|----------------------------|-------:|--------------:|-----------------------------------|
| D1 | CardiffNLP Tweet Topic     | ~6,000 |            19 | Social media (short text)         |
| D2 | EUR-Lex (MultiEURLEX)      | ~65,000|           201 | EU legislation (long documents)   |
| D3 | CASAS smart-home (hh113)   | ~11,000|            19 | IoT sensor logs (sensor → activity) |

All three datasets are publicly available; see `scripts/fetch_datasets.py`
for the download URLs.

## Ablation configurations

| Config              | Cluster | Cover / Merge | Reunite | Event clust. | Cosine filter |
|---------------------|:-------:|:-------------:|:-------:|:------------:|:-------------:|
| A0: Raw LLM         |    --   |       --      |    --   |      --      |       --      |
| A1: Cluster only    |    ✓    |       --      |    --   |      --      |       ✓       |
| A2: C&M only        |    --   |       ✓       |    --   |      --      |       --      |
| A3: Clust. + C&M    |    ✓    |       ✓       |    --   |      --      |       ✓       |
| A4: + Reunite       |    ✓    |       ✓       |    ✓    |      --      |       --      |
| A5: + Event clust.  |    ✓    |       ✓       |    --   |      ✓       |       ✓       |
| A6: No cosine filter|    ✓    |       ✓       |    --   |      --      |       --      |

Default parameters: `k=19`, `τ=0.3`, `κ=3`, embedding `all-MiniLM-L6-v2`.

## Baselines

BM25, Sentence-BERT cosine, Cross-encoder reranker, DistilBART-MNLI
zero-shot, GloVe, TF-IDF, Word2Vec cosine. See `src/baselines.py`.

## Parameter sensitivity sweeps

- `k`     ∈ {1, 5, 10, 15, 19, 25, 30}
- `τ`     ∈ {0.0, 0.1, ..., 0.9}
- `κ`     ∈ {1, 2, 3, 5, 7, 10}
- embedding ∈ {all-MiniLM-L6-v2, all-mpnet-base-v2, e5-large-v2, bge-base-en-v1.5}

## Metrics

- Matching accuracy: precision, recall, F1 (macro), FPR.
- System: latency, throughput, invocation count.
- Cost: compression ratio, tokens consumed, equivalent $/1k events.

All headline numbers are reported as mean ± 95% CI over 5 seeds (15 seeds
where indicated). Pairwise significance uses the Wilcoxon signed-rank
test at α=0.05.

## Directory layout

```
params.yaml          # DVC-tracked parameters (mode, profiles, LLM models)
dvc.yaml             # Pipeline definition (auto-generated)
configs/             # YAML experiment configs (reference)
src/                 # Core implementation
  config.py          #   Profile/mode loader
  progress.py        #   Per-stage progress tracking
  data.py            #   Dataset loading
  router.py          #   Neural Router engine (Algorithms 1-3)
  llm.py             #   LLM client (LiteLLM)
  llm_async.py       #   Async LLM client
  embeddings.py      #   Embedding model with disk cache
  baselines.py       #   Baseline implementations
  evaluation.py      #   Metrics and statistical tests
  qoe.py             #   QoE-based heterogeneous backend assignment
  manifest.py        #   Run manifest (atomic resume)
scripts/             # Experiment runners and helpers
  run_experiment.py  #   Ablation
  run_sensitivity.py #   Parameter sensitivity
  run_scaling.py     #   Event-count scaling
  run_qoe.py         #   QoE backend assignment
  run_crossover.py   #   Cost-model crossover validation
  monitor.py         #   Live dashboard
  fetch_datasets.py  #   Dataset download / verification
  slurm/             #   SLURM templates for HPC runs
analysis/            # Post-processing, statistics, figure generation
tests/               # pytest suite
results/             # Output files (gitignored; organised by mode)
figs/                # Generated PDFs (gitignored)
```

## Environment

- Python 3.10 or newer.
- DVC for pipeline orchestration.
- LLM serving: [Ollama](https://ollama.com) for local open-weight backends
  (Qwen 2.5 family), Anthropic API (Haiku, Sonnet) via LiteLLM.
- Embeddings: [sentence-transformers](https://www.sbert.net) (MiniLM-L6
  default).
- Clustering: scikit-learn `KMeans`.

## License

See `LICENSE`.
