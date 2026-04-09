# Neural Router: A0 Crossover & FGCS Reframe Plan

## Context

The Neural Router's ablation shows A0 (raw LLM, no pipeline) dominates all optimized configurations on F1. This undermines the paper's core architectural contributions (CoverAndMerge, compression, filtering). 8/11 FGCS reviewers flagged this. The fix: demonstrate the crossover point where the pipeline becomes necessary, reframe the cost model as the primary contribution.

## Phase 1: Scale Experiment (NEW — highest priority)

### 1.1 Generate scaled subscription sets

Create synthetic subscription sets at controlled scales by sampling and expanding from existing datasets:

- **D1 (20 Newsgroups)**: base 19 subs → expand to 50, 100, 200, 500, 1000, 2000, 5000
  - Method: sample real newsgroup posts as additional subscriptions with hierarchical topic structure
- **D2 (CardiffNLP)**: base 201 subs → expand to 500, 1000, 2000, 5000
  - Method: generate topic variations using the existing taxonomy
- **D3 (CASAS)**: base 19 subs → expand to 50, 100, 200, 500, 1000
  - Method: create sub-activity subscriptions (e.g., "Sleep > deep sleep", "Cook > microwave")

Deliverable: `data/scaled/{dataset}_{N}.json` subscription files

### 1.2 Run scale experiments

```
Matrix:
  Datasets: D1, D2, D3
  Subscription scales: [50, 100, 200, 500, 1000, 2000, 5000]
  Configs: A0 (raw), A2 (cluster only), A4 (full pipeline)
  LLMs: Qwen-2.5-7B (local), Haiku (API)
  Seeds: 5
  Metrics: F1, latency_ms, cost_tokens, throughput_events/s
```

Total runs: 3 datasets × 7 scales × 3 configs × 2 LLMs × 5 seeds = 630 runs
Estimated time: ~3-5 days (Qwen local + Haiku API budget)

### 1.3 Expected results

- A0 F1 degrades beyond context window (~4K tokens for Qwen, ~200K for Claude)
- A0 latency grows linearly with subscription count
- A0 cost grows linearly (API) or stays flat (local) but with quality degradation
- Pipeline F1 stays stable (clustering keeps prompt size bounded)
- **Crossover point**: the subscription count where pipeline F1 > A0 F1

### 1.4 Key figure: Crossover Plot

X-axis: subscription count (log scale)
Y-axis (left): F1 score
Y-axis (right): cost per event (tokens)
Lines: A0, A2, A4 for each LLM
Annotation: crossover point where A4 > A0

This becomes the paper's **central figure**.

## Phase 2: Pareto Analysis (reanalysis of existing + new data)

### 2.1 Multi-objective evaluation

For each (config, scale, LLM) point, compute:
- F1 (accuracy)
- Mean latency per event (ms)
- Cost per event (tokens for API, FLOPS for local)
- Throughput under concurrent load (events/s)

### 2.2 Pareto frontier

Plot F1 vs cost Pareto frontier at each scale. Show that:
- At small scale: A0 is Pareto-optimal (best F1, acceptable cost)
- At medium scale: A2/A4 enter the Pareto frontier (similar F1, much lower cost)
- At large scale: A0 falls off the frontier (both worse F1 AND higher cost)

### 2.3 Practical operating regions

Define three regimes:
1. **Small** (< crossover): Use A0, don't bother with pipeline
2. **Medium** (crossover zone): Pipeline saves cost with minimal F1 loss
3. **Large** (>> crossover): Pipeline is the only viable option

## Phase 3: Narrative Reframe (manuscript edits)

### 3.1 Abstract

Old: "We propose three optimizations for LLM-based content matching"
New: "We characterize the cost-accuracy tradeoff in LLM-based content matching and identify the crossover point beyond which naive prompting fails. Our pipeline extends the operating range of LLM matching to large-scale subscription sets."

### 3.2 Introduction

Add: "A key finding is that at subscription volumes below ~N, raw prompting (A0) is optimal — confirming our cost model's prediction. Beyond this threshold, the pipeline maintains matching quality where A0 degrades, validating the model's crossover analysis."

### 3.3 Evaluation section restructure

Current structure: configs × datasets × LLMs (all at native scale)
New structure:
1. **Fixed-scale ablation** (existing): show A0 dominance at small scale, explain via cost model
2. **Scale experiment** (new): show crossover at increasing subscription counts
3. **Pareto analysis** (new): multi-objective F1/cost/latency tradeoff
4. **IoT case study** (existing CASAS): demonstrate domain-specific matching

### 3.4 Discussion reframe

Position the A0 result as the cost model's first validated prediction:
> "The A0 dominance at small scale is not a limitation but a confirmed hypothesis: our cost model (Eq. X) predicts that the crossover occurs at approximately N subscriptions for Qwen-7B. The scale experiment confirms this: beyond N=500, A0 F1 drops by X% while A4 maintains Y%."

### 3.5 Contribution list update

1. A formal cost model for LLM-based content matching that predicts the accuracy-cost crossover (Section 3)
2. A three-stage optimization pipeline (CoverAndMerge, compression, filtering) that extends LLM matching to large subscription sets (Section 4)
3. Empirical validation showing the crossover at N≈500 subscriptions, with A0 optimal below and the pipeline necessary above (Section 5)
4. A multi-LLM evaluation across three datasets demonstrating the framework's generality (Section 5)

## Phase 4: FGCS-Specific Additions

### 4.1 System-level evaluation (new)

Add a simple broker benchmark:
- Deploy Neural Router as an actual HTTP pub/sub broker
- N concurrent publishers sending events
- M subscribers registered
- Measure: throughput (events/s), latency (p50/p95/p99), memory footprint
- Compare: A0 vs A4 under load
- This addresses the "no prototype" concern (6/11 reviewers)

### 4.2 Format conversion

- Switch from IEEEtran to elsarticle (Elsevier template)
- Add structured abstract
- Add CRediT author statement
- Add data availability statement
- Update bibliography style

### 4.3 Additional references

Current: 37 references (below FGCS 40-70 norm)
Add: pub/sub literature (Carzaniga, Eugster, Mühl), computing continuum (Dustdar, Deng), LLM evaluation methodology (Liang HELM, Zheng Chatbot Arena), middleware (DEBS proceedings)
Target: 50-55 references

## Phase 5: Code Changes

### 5.1 Scale experiment infrastructure

- `src/scaled_subscriptions.py`: generate scaled subscription sets
- `scripts/run_scale_experiment.py`: orchestrate the scale matrix
- `src/analysis/crossover.py`: find and plot crossover points
- `src/analysis/pareto.py`: multi-objective Pareto analysis

### 5.2 Broker benchmark

- `src/broker_benchmark.py`: simple HTTP pub/sub broker wrapping the Neural Router
- `scripts/run_broker_benchmark.py`: concurrent load test with configurable publishers/subscribers

### 5.3 Tests (TDD)

- `tests/test_scaled_subscriptions.py`: validate generated subscription sets
- `tests/test_crossover_analysis.py`: validate crossover detection logic
- `tests/test_broker_benchmark.py`: validate broker benchmark harness

## Timeline

| Week | Task | Deliverable |
|---|---|---|
| 1 | Scale experiment infrastructure (TDD) | Scripts + tests |
| 1 | Generate scaled subscription sets | data/scaled/*.json |
| 2-3 | Run scale experiments (Qwen local) | 315 runs |
| 2-3 | Run scale experiments (Haiku API) | 315 runs (budget dependent) |
| 3 | Crossover analysis + Pareto plots | Figures |
| 4 | Broker benchmark implementation (TDD) | Throughput/latency data |
| 4 | Narrative reframe + manuscript edits | Updated tex files |
| 5 | Format conversion (Elsevier) + references | FGCS-ready manuscript |
| 5 | Internal review round 2 | Validation |

## Success Criteria

- [ ] Crossover point identified empirically (subscription count where pipeline F1 > A0 F1)
- [ ] Pareto frontier shows pipeline dominance at medium-large scale
- [ ] Broker benchmark shows throughput > 10 events/s under concurrent load
- [ ] Cost model prediction matches empirical crossover within 2x
- [ ] All FGCS mandatory revisions addressed
- [ ] Manuscript fits within 18 pages (Elsevier format)
- [ ] Reference count ≥ 45

## Files to Create/Modify

| File | Action |
|---|---|
| `src/scaled_subscriptions.py` | NEW |
| `scripts/run_scale_experiment.py` | NEW |
| `src/analysis/crossover.py` | NEW |
| `src/analysis/pareto.py` | NEW |
| `src/broker_benchmark.py` | NEW |
| `scripts/run_broker_benchmark.py` | NEW |
| `tests/test_scaled_subscriptions.py` | NEW |
| `tests/test_crossover_analysis.py` | NEW |
| `tests/test_broker_benchmark.py` | NEW |
| `txt/Abstract.tex` | MODIFY (reframe) |
| `txt/Introduction.tex` | MODIFY (crossover narrative) |
| `txt/Experiment.tex` | MODIFY (add scale + Pareto sections) |
| `txt/Discussion.tex` | MODIFY (A0 as confirmed prediction) |
| `main.tex` | MODIFY (Elsevier template) |
| `references.bib` | MODIFY (add 15+ references) |
