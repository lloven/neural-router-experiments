# Neural Router Experiments

Experiment code for the Neural Router paper (Elsevier DCN, under revision).

**Manuscript:** `Manuscripts/Neural Router (Elsevier DCN)/` (Overleaf-synced via Dropbox)
**Experiment design:** See `Manuscripts/Neural Router (Elsevier DCN)/txt/Experiment.tex`

## Experiment plan

### Datasets (§4.1)

| ID | Dataset | Events | Subscriptions | Domain |
|---|---|---|---|---|
| D1 | CardiffNLP Tweet Topic | ~6,000 | 19 | Social media (short) |
| D2 | EUR-Lex (MultiEURLEX) | ~65,000 | 201 | EU legislation (long) |
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

- k: {1, 5, 10, 15, 19, 30, 50}
- τ: {0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9}
- κ: {1, 2, 3, 5, 10, adaptive}
- Embedding model: {all-MiniLM-L6-v2, all-mpnet-base-v2, e5-large-v2, bge-base-en-v1.5}

### Metrics (§4.7)

- Matching accuracy: precision, recall, F1, FPR (per-event, macro-averaged)
- System: latency L, per-event latency L/m, throughput (events/s), invocation count I
- Cost: compression ratio ρ, tokens consumed, $/1k events

### Statistical reporting

All metrics: mean ± 95% CI over 5 seeds. Pairwise significance: Wilcoxon signed-rank (α=0.05).

## Directory structure

```
configs/       # YAML configs for each experiment run
src/           # Core implementation
scripts/       # Experiment runners (run_ablation.py, run_baselines.py, etc.)
results/       # Raw output files (timestamped)
analysis/      # Jupyter notebooks / scripts for figures and tables
figs/          # Generated PDFs → copy to Manuscripts/.../figs/
```

## Setup

```bash
# TODO: Add setup instructions after implementation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running experiments

```bash
# TODO: Add run instructions after implementation
```

## Environment

- Python 3.10+
- LLM: OpenAI GPT-4o-mini via LiteLLM
- Embeddings: sentence-transformers
- Clustering: scikit-learn
- Record API model snapshot date for reproducibility
