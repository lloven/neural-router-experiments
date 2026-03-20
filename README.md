# Neural Router Experiments

Experiment code for the Neural Router paper (Elsevier DCN, under revision).

**Manuscript:** `Manuscripts/Neural Router (Elsevier DCN)/` (Overleaf-synced via Dropbox)
**Experiment design:** See `Manuscripts/Neural Router (Elsevier DCN)/txt/Experiment.tex`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy your OpenAI API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY

# Download datasets
python scripts/fetch_datasets.py
```

## Running experiments

### DVC pipeline (recommended)

The experiment pipeline is managed by [DVC](https://dvc.org/) with three test
levels (per L24 in lessons.md):

| Mode | Purpose | Scope | Time |
|---|---|---|---|
| `unit_smoke` | Catch imports, API keys, basic logic | D1 only, 2 configs, 20 events | ~1 min |
| `integration_smoke` | All stage types, representative subset | D1-D3, 3 configs, 100 events | ~30 min |
| `full` | Complete experimental design | D1-D3, 7 configs, 5 seeds, all baselines | hours |

```bash
# 1. Choose mode (edits params.yaml and regenerates dvc.yaml)
python scripts/generate_pipeline.py --mode unit_smoke

# 2. Run the pipeline
dvc repro                    # sequential
dvc repro -j 3               # parallel (3 concurrent stages)

# 3. Monitor progress (live dashboard)
python scripts/monitor.py
```

Switching modes preserves previous results (each mode writes to
`results/{mode}/`). DVC caches completed stages and skips them on rerun.

### LLM models

Three LLM backends run in parallel tracks:

| Model | Type | Stages | Notes |
|---|---|---|---|
| `qwen2.5:7b` | Local (Ollama) | all | Slow but free; checkpoint every 5 events |
| `claude-3-haiku` | API | all | Cheap; checkpoint every 25 events |
| `claude-sonnet-4` | API | all (single seed) | Expensive; seed override for cost control |

### Manual runs (scripts still work standalone)

```bash
# Ablation
python scripts/run_experiment.py --dataset D1 --configs A3 --max-events 100 --seeds 42 \
    --llm-model ollama/qwen2.5:7b --mode integration_smoke --resume

# Sensitivity
python scripts/run_sensitivity.py --dataset D1 --sweep all --llm-model ollama/qwen2.5:7b

# Scaling
python scripts/run_scaling.py --dataset D1 --dimension events --llm-model ollama/qwen2.5:7b
```

### Monitoring

```bash
python scripts/monitor.py                    # live dashboard (reads params.yaml for mode)
python scripts/monitor.py --log my.log       # custom log path
```

The monitor shows per-stage progress, within-stage event-level progress
(via `.progress/` JSON files), and ETA estimates. Pending stages are collapsed
into a single summary line.

### Checkpointing and resume

All scripts preserve intermediate results to avoid losing compute:

- **Ablation:** Pickle checkpoints every N events (5 for Qwen, 25 for Haiku,
  50 for Sonnet). `--resume` skips completed config x seed pairs.
- **Sensitivity:** Each parameter value appended to CSV immediately. Resume
  is automatic (skips completed values on restart).
- **Scaling:** Each scale point appended to CSV immediately. Same resume logic.

## Experiment plan

### Datasets (§4.1)

| ID | Dataset | Events | Subscriptions | Domain |
|---|---|---|---|---|
| D1 | CardiffNLP Tweet Topic | ~6,000 | 19 | Social media (short) |
| D2 | EUR-Lex (MultiEURLEX) | ~65,000 | 127 | EU legislation (long) |
| D3 | MN-DS (Multilabeled News) | ~10,917 | 109 | News articles (medium) |

### Ablation configurations (§4.2)

| Config | Cluster | C&M | Reunite | Evt. clust. | Cos. filter |
|---|---|---|---|---|---|
| A0: Raw LLM | -- | -- | -- | -- | -- |
| A1: Cluster only | ✓ | -- | -- | -- | ✓ |
| A2: C&M only | -- | ✓ | -- | -- | -- |
| A3: Clust. + C&M | ✓ | ✓ | -- | -- | ✓ |
| A4: + Reunite | ✓ | ✓ | ✓ | -- | -- |
| A5: + Evt. clust. | ✓ | ✓ | -- | ✓ | ✓ |
| A6: No cos. filter | ✓ | ✓ | -- | -- | -- |

Default parameters: k=19, τ=0.3, κ=3, LLM=GPT-4o-mini, embedding=all-MiniLM-L6-v2.

### Baselines (§4.3)

BM25, Sentence-BERT cosine, Cross-encoder reranker, GloVe, TF-IDF, Word2Vec.

### Parameter sensitivity (§4.4)

- k: {1, 5, 10, 15, 19, 25, 30}
- τ: {0.0, 0.1, 0.2, ..., 0.9}
- κ: {1, 2, 3, 5, 7, 10}
- Embedding model: {all-MiniLM-L6-v2, all-mpnet-base-v2, e5-large-v2, bge-base-en-v1.5}

### Metrics (§4.7)

- Matching accuracy: precision, recall, F1, FPR (per-event, macro-averaged)
- System: latency L, per-event latency L/m, throughput (events/s), invocation count I
- Cost: compression ratio ρ, tokens consumed, $/1k events

### Statistical reporting

All metrics: mean ± 95% CI over 5 seeds. Pairwise significance: Wilcoxon signed-rank (α=0.05).

## Directory structure

```
params.yaml        # DVC-tracked parameters (mode, profiles, LLM models)
dvc.yaml           # Auto-generated pipeline (via generate_pipeline.py)
configs/           # YAML experiment configs (reference)
src/               # Core implementation
  config.py        #   Profile/mode loader (reads params.yaml)
  progress.py      #   Fine-grained progress tracking for monitor
  data.py          #   Dataset loading (D1, D2, D3)
  router.py        #   Neural Router engine (Algorithms 1-3)
  llm.py           #   LLM client (LiteLLM)
  llm_async.py     #   Async LLM client
  embeddings.py    #   Embedding model with disk cache
  baselines.py     #   Six baseline methods
  evaluation.py    #   Metrics and statistical tests
scripts/           # Experiment runners
  generate_pipeline.py # Generates dvc.yaml from params.yaml
  run_experiment.py    # Ablation runner (A0-A6 + baselines)
  run_parallel.py      # Parallel runner (multiprocessing)
  run_sensitivity.py   # Parameter sensitivity sweeps
  run_scaling.py       # Scaling analysis
  monitor.py           # Live progress dashboard
  fetch_datasets.py    # Dataset download/verification
  fetch_eurovoc.py     # EUROVOC label descriptions for D2
results/           # Output files, organized by mode
  unit_smoke/      #   Quick validation results
  integration_smoke/ # Representative subset results
  full/            #   Complete experiment results
analysis/          # Jupyter notebooks for figures and tables
figs/              # Generated PDFs for paper
```

## Performance notes

- **Async LLM calls:** All cluster batches fire concurrently (up to `max_parallel=10`), typically 5-10x faster than sequential.
- **Embedding cache:** Disk-cached in `data/embedding_cache/`, avoiding recomputation across seeds.
- **Parallel runner:** Multiple (config, dataset, seed) jobs run in separate processes.

## Environment

- Python 3.10+
- DVC for pipeline management
- LLM: Ollama (qwen2.5:7b local), Anthropic API (Haiku, Sonnet) via LiteLLM
- Embeddings: sentence-transformers (all-MiniLM-L6-v2 default)
- Clustering: scikit-learn KMeans
