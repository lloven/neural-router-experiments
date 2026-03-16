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

### Quick start

```bash
# Smoke test (dry run, no API calls)
python scripts/run_experiment.py --dataset D1 --configs A3 --dry-run --max-events 50

# Real test with 100 events
python scripts/run_experiment.py --dataset D1 --configs A3 --max-events 100 --seeds 42

# Full ablation on D1 with all seeds
python scripts/run_experiment.py --dataset D1 --configs all --baselines all

# Full run on all datasets (parallel)
python scripts/run_parallel.py --dataset all --configs all --baselines all --workers 4
```

### Parameter sensitivity

```bash
python scripts/run_sensitivity.py --dataset D1 --sweep k         # cluster count
python scripts/run_sensitivity.py --dataset D1 --sweep tau       # cosine threshold
python scripts/run_sensitivity.py --dataset D1 --sweep kappa     # top-K matches
python scripts/run_sensitivity.py --dataset D1 --sweep embedding # embedding models
python scripts/run_sensitivity.py --dataset D1 --sweep all       # all sweeps
```

### Scaling analysis

```bash
python scripts/run_scaling.py --dataset D1 --dimension events
python scripts/run_scaling.py --dataset D2 --dimension subscriptions
```

### Monitoring

```bash
# One-shot progress check
../shared/monitor.sh results/

# Auto-refresh every 5 seconds
../shared/monitor.sh results/ --watch
```

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
configs/           # YAML experiment configs
src/               # Core implementation
  data.py          #   Dataset loading (D1, D2, D3)
  router.py        #   Neural Router engine (Algorithms 1-3)
  llm.py           #   LLM client (LiteLLM)
  llm_async.py     #   Async LLM client
  embeddings.py    #   Embedding model with disk cache
  baselines.py     #   Six baseline methods
  evaluation.py    #   Metrics and statistical tests
scripts/           # Experiment runners
  run_experiment.py    # Sequential runner
  run_parallel.py      # Parallel runner (multiprocessing)
  run_sensitivity.py   # Parameter sensitivity sweeps
  run_scaling.py       # Scaling analysis
  fetch_datasets.py    # Dataset download/verification
  fetch_eurovoc.py     # EUROVOC label descriptions for D2
results/           # Raw output files (timestamped CSVs)
analysis/          # Jupyter notebooks for figures and tables
figs/              # Generated PDFs for paper
```

## Performance notes

- **Async LLM calls:** All cluster batches fire concurrently (up to `max_parallel=10`), typically 5-10x faster than sequential.
- **Embedding cache:** Disk-cached in `data/embedding_cache/`, avoiding recomputation across seeds.
- **Parallel runner:** Multiple (config, dataset, seed) jobs run in separate processes.

## Environment

- Python 3.10+
- LLM: OpenAI GPT-4o-mini via LiteLLM
- Embeddings: sentence-transformers (all-MiniLM-L6-v2 default)
- Clustering: scikit-learn KMeans
- Record API model snapshot date for reproducibility
