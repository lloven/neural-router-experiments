---
title: "Theoretical Analysis: LLM Underperformance on D2 (EUR-Lex) and the Missing Accuracy Model"
date: 2026-04-06
source: experimental results vs theory predictions
manuscript: Manuscripts/Neural Router (Elsevier FGCS)/
status: draft
tags: [theoretical-analysis, accuracy-model, subscription-scaling]
---

# Phase 1: Deep Understanding

## 1.1 Gap Classification

**Gap type: Framework gap.**

This is not a proof error (the cost model's mathematics are correct) and not a related-work awareness issue. The gap is structural: the paper's formal apparatus models only *cost* (invocations, tokens, latency) but implicitly assumes that the LLM's *accuracy* is at least as good as baselines whenever all information fits in context. The D2 results falsify this assumption. The cost model correctly predicts that A0 is cost-optimal when W > W_cross, but says nothing about whether A0 (or any LLM configuration) produces good matches under those conditions.

The manuscript's Discussion (line 11) currently claims "Sonnet outperforms SBERT by a factor of 2.3x" on D2, which appears to be based on a Sonnet single-seed run on a 5,000-event subsample (7.7% of the corpus). The full Haiku ablation over 5 seeds tells a different story: the best LLM configuration (A1 Haiku) achieves F1 = 0.089, while TF-IDF achieves 0.162 and SBERT achieves 0.154. Even granting that Sonnet may perform better, the Haiku results (the only statistically robust D2 results with 5 seeds over the full corpus) show the LLM losing to simple baselines by nearly 2x.

The core question: **Why does the LLM, which dominates on D1 and D3, fail on D2?**

## 1.2 FUSILLI

Not applicable (no prior work directory).

## 1.3 Prior Work

No prior theoretical analysis exists for this specific issue.

---

# Phase 2: Expand Before Contracting

## 2.1 Four Expansion Questions

### Question 1: Can the fix be more general?

Yes. The D2 failure is not an isolated data point but reveals a **missing component in the paper's theoretical framework**: an accuracy model that complements the cost model. The cost model tells you *how many* LLM calls you need and *how much* they cost. It says nothing about *how well* the LLM performs those calls. The implicit assumption is monotonic: more context = better accuracy. D2 falsifies this.

A general fix would introduce an **accuracy scaling law** alongside the cost model, capturing how LLM matching quality degrades as a function of:

- |S| (subscription cardinality)
- n_s (mean labels per event, i.e., multi-label density)
- Document length and complexity
- Semantic discriminability between subscriptions

This would transform the paper's contribution from "here is a cost model that predicts when to compress" to "here is a joint cost-accuracy model that predicts when LLM matching is viable at all."

### Question 2: Does this connect to other issues?

**Yes, directly to the tau plateau.**

The parallel analysis of the cosine threshold sensitivity shows that F1 peaks at tau = 0.0-0.1 and is flat for tau >= 0.4 (on D2 Sonnet, F1 = 0.103 at tau=0.0 and 0.054 for tau >= 0.4). This plateau means the cosine filter operates in a binary regime: either all events pass (tau = 0) or a fixed subset passes (tau >= 0.4), with no intermediate discrimination.

**The shared root cause is the high-dimensional embedding geometry of a large subscription space.**

With |S| = 201 subscriptions embedded in the same vector space, the cluster centroids become diffuse. When subscriptions are numerous and semantically overlapping (as EUROVOC descriptors are), the centroid of a k-means cluster represents a vague semantic region. The cosine similarity between an event and any cluster centroid will be low and similar across clusters, producing the observed binary tau behaviour. Simultaneously, the LLM receiving 201 subscriptions in a single prompt faces a combinatorial discrimination problem that exceeds its effective working capacity, even though the subscriptions fit within the context window.

The connection can be stated precisely:

> **Both failures stem from the same mechanism: subscription-space density exceeding the system's discrimination capacity.** For the cosine pre-filter, discrimination capacity is limited by embedding geometry (centroids converge in high-|S| spaces). For the LLM, discrimination capacity is limited by attention distribution over a large candidate set.

### Question 3: Is there a deeper pattern?

Yes. The deeper pattern is a **discrimination complexity** scaling law that governs LLM-as-classifier performance. Consider the distinction:

| Aspect | D1 (success) | D2 (failure) | D3 (success) |
|--------|-------------|-------------|-------------|
| |S| | 19 | 201 | 19 |
| Mean labels/event | 1.6 | 2.2 | 1.0 |
| Label taxonomy | Coarse, disjoint topics | Hierarchical, overlapping EUROVOC | Coarse, disjoint activities |
| Document length | ~30 words | ~700+ words | ~45 words |
| Semantic gap | Low | Medium-high | Very high |
| LLM vs SBERT ratio | 1.55x (Haiku) | 0.53x (Haiku) | 1.76x (Haiku A0) |
| Decision complexity | 19 binary choices | 201 binary choices | 19 binary choices |

The LLM wins when |S| is small (19) regardless of whether the semantic gap is low (D1) or extreme (D3). The LLM loses when |S| is large (201) even though the semantic gap is moderate. This implicates **|S|, not semantic difficulty, as the binding constraint.**

Why? The matching prompt asks the LLM to make |S| binary relevance decisions simultaneously. For |S| = 19, this is a manageable multi-label classification with O(19) decision boundaries. For |S| = 201, this is O(201) simultaneous decisions, many involving fine-grained distinctions between similar EUROVOC descriptors (e.g., "monetary economics" vs. "economic policy" vs. "national accounts" vs. "prices"). The LLM's attention mechanism must distribute over all 201 subscription descriptions while processing a 700-word legal document; the effective per-subscription attention budget is 1/201th of what it was for D1.

This is analogous to the **set-size effect** in cognitive psychology and the **lost in the middle** phenomenon in LLM research (Liu et al., 2024). Large language models systematically underperform when required to attend to many items simultaneously, even when the items fit within the context window. The context window is a necessary but not sufficient condition for accurate processing.

**Formalisation attempt.** Let A(|S|) denote the LLM's matching accuracy as a function of subscription count. The evidence suggests:

```
A(|S|) ~ A_0 / (1 + |S| / S_half)
```

where A_0 is the accuracy on a single-subscription task and S_half is the subscription count at which accuracy halves. From the data:

- D1: A(19) ~ 0.656 (Haiku A0 F1)
- D2: A(201) ~ 0.082 (Haiku A0 F1)

If we assume A_0 ~ 0.72 (extrapolating from D1 single-seed best), then S_half ~ 19 * (0.72/0.082 - 1) / (201/19 - 1) ~ 15. This implies the LLM's effective discrimination capacity is approximately 15-20 subscriptions, and accuracy degrades rapidly beyond that.

This is consistent with D1 and D3 (both |S| = 19, near S_half) showing good performance, while D2 (|S| = 201, far beyond S_half) collapses.

### Question 4: Does it suggest a cross-field connection?

**Yes, to three bodies of work:**

1. **Extreme multi-label classification (XML).** The ML literature on XML (Bhatia et al., 2015; Liu et al., 2017; You et al., 2019) has long recognised that performance degrades with label cardinality, even for purpose-built classifiers. Standard approaches in XML use label trees, label embeddings, and attention-based label partitioning to manage large label spaces. The EUR-Lex dataset is, in fact, a standard XML benchmark, and state-of-the-art XML methods achieve F1 ~ 0.30-0.50 on it (with supervised training). The LLM achieving F1 ~ 0.08 in a zero-shot setting is not surprising from the XML perspective; what would be surprising is if it worked well.

2. **Attention saturation and "lost in the middle."** Liu et al. (2024) showed that LLMs struggle to retrieve information from the middle of long contexts, even when the context fits within the window. This is directly relevant: with 201 subscriptions listed in the prompt, subscriptions in the middle of the list receive less effective attention, leading to systematic misses.

3. **Information-theoretic bounds on zero-shot classification.** With 201 binary decisions per event, the LLM must produce approximately 201 bits of classification information per event. For a 700-word document, this requires extracting roughly 1 bit of relevance information per 3.5 words per subscription. The channel capacity of zero-shot LLM classification (without task-specific training) may simply be insufficient for this information rate.

## 2.2 Research Note: Proposed Resolution

### The Discrimination Complexity Hypothesis

**Thesis:** The cost model's implicit accuracy assumption fails because it conflates *context capacity* (can the prompt fit?) with *discrimination capacity* (can the LLM make accurate decisions from the prompt?). These are different resources with different scaling properties.

**Context capacity** scales linearly with the context window W and is well-modelled by the existing cost model (Eq. 1-7 in the paper). When W is large enough, all subscriptions fit, and the cost model correctly predicts A0 dominance.

**Discrimination capacity** scales inversely with |S| and is NOT modelled. The LLM's ability to make accurate binary relevance decisions degrades as the number of simultaneous decisions increases, even when all decisions fit in the context window. The degradation is more severe when:

- Subscriptions are semantically similar (fine-grained taxonomy, as in EUROVOC)
- Documents are long (more text to process per subscription)
- The task is multi-label (multiple correct answers per event)

**Formal characterisation.** Let us define the *effective discrimination load*:

```
D_eff = |S| * H(S) * L_doc
```

where |S| is the subscription count, H(S) is the semantic entropy of the subscription space (a measure of how similar subscriptions are to each other; high for EUROVOC, low for CardiffNLP), and L_doc is the normalised document length.

The LLM's accuracy degrades when D_eff exceeds a model-dependent threshold D_max:

```
F1(D_eff) ~ F1_base * exp(-D_eff / D_max)
```

For D1: D_eff ~ 19 * 0.3 * 1.0 = 5.7 (well below D_max, LLM dominates)
For D3: D_eff ~ 19 * 0.2 * 1.5 = 5.7 (well below D_max, LLM dominates)
For D2: D_eff ~ 201 * 0.8 * 23.3 = 3,750 (far above D_max, LLM fails)

The exact functional form requires more data points to calibrate, but the qualitative prediction is clear: **there exists a subscription-count regime where the LLM's discrimination capacity, not its context capacity, is the binding constraint.**

### Implications for the Paper

**What this means for the cost model:**

The cost model is correct as a cost model. It accurately predicts invocation counts and token costs. The gap is that the paper uses it to justify the A0 dominance finding without acknowledging that A0's cost optimality says nothing about A0's accuracy. The cost model should be explicitly scoped as a *cost* model, and the paper should acknowledge that an *accuracy* model is needed to determine when LLM matching is viable.

**What this means for the crossover analysis:**

The crossover analysis identifies W_cross as the context window below which compression helps. The D2 results suggest a second crossover point: |S|_cross, the subscription count above which LLM matching (regardless of configuration) loses to simpler baselines. For the current LLM backends, |S|_cross appears to be somewhere between 19 (D1/D3, LLM wins) and 201 (D2, LLM loses). The exact value depends on the LLM, the subscription taxonomy, and the document characteristics.

**What this means for the architecture:**

For large |S|, the clustering pipeline (A1-A6) should not merely compress subscriptions to save tokens; it should *partition* the decision space to keep per-prompt |S'| below |S|_cross. This reframes clustering from a cost-optimisation mechanism to a **discrimination-capacity management** mechanism. In this view, A1 (cluster only) should outperform A0 on D2, because it presents each LLM call with only |S|/k ~ 201/19 ~ 10 subscriptions. And indeed, the data partially supports this: A1 Haiku (F1 = 0.089) slightly outperforms A0 Haiku (F1 = 0.082).

However, A1's advantage is small because the cosine pre-filter (tau = 0.3) discards many relevant clusters. At tau = 0, A1 would send each event to all clusters, giving the LLM k separate decisions with |S|/k subscriptions each, but at the cost of k*m invocations. This is the trade-off the paper's cost model captures. What it does not capture is that this trade-off exists *for accuracy reasons*, not just cost reasons.

### Connection to the Tau Plateau

The tau plateau (F1 flat for tau >= 0.4) and the D2 underperformance share a root cause: **the embedding geometry fails to discriminate in high-|S| subscription spaces.**

With 201 subscriptions, the k-means cluster centroids are pulled toward the centre of the subscription embedding space. The cosine similarity between any event and any centroid is therefore low and similar across centroids. This means:

- tau = 0 sends events to all clusters (high recall, high invocations)
- tau > 0 sends events to few or no clusters (low recall, low invocations)
- There is no intermediate regime where tau selectively routes events to relevant clusters

This is exactly what the Sonnet D2 sensitivity data shows:
- tau = 0.0: F1 = 0.103, invocations = 118
- tau = 0.1: F1 = 0.080, invocations = 105
- tau = 0.3: F1 = 0.070, invocations = 92
- tau >= 0.4: F1 = 0.054, invocations = 88 (plateau)

The plateau occurs because all cosine scores fall below tau = 0.4 for most events, so increasing tau further has no effect (the same events are already excluded).

**Unified explanation:** In high-|S| subscription spaces, both the cosine pre-filter and the LLM operate in a degraded regime. The cosine pre-filter cannot discriminate between relevant and irrelevant clusters because centroid geometry is diffuse. The LLM cannot discriminate between relevant and irrelevant subscriptions because the attention budget is spread too thin. The root cause is the same: the system's discrimination resources (embedding geometry for the pre-filter; attention capacity for the LLM) are fixed, while the discrimination demand scales with |S|.

### Why TF-IDF and SBERT Win on D2

TF-IDF and SBERT make 201 *independent* pairwise comparisons, each requiring only a single similarity computation between the event and one subscription. The decision for subscription s_i does not compete for resources with the decision for s_j. This independence means their per-subscription accuracy does not degrade with |S|.

The LLM, by contrast, makes all 201 decisions in a single prompt. The decisions are entangled through the shared attention mechanism: attending to subscription s_i's text reduces the attention available for s_j's text. This entanglement causes accuracy degradation that accelerates with |S|.

For small |S| (D1, D3), the LLM's superior semantic understanding more than compensates for the attention dilution. For large |S| (D2), the attention dilution overwhelms the semantic advantage.

This framing yields a clean theoretical prediction:

> **There exists a crossover subscription count |S|_acc above which independent pairwise methods (TF-IDF, SBERT) outperform the LLM despite having weaker per-pair semantic understanding.** The crossover depends on the LLM's attention capacity, the subscription taxonomy's semantic density, and the document length.

### What the Data Does NOT Tell Us

Several confounds prevent a clean attribution of the D2 failure to |S| alone:

1. **Document length.** D2 documents are ~700 words vs. ~30 (D1) and ~45 (D3). Longer documents consume more of the LLM's context and attention budget, leaving less for subscription discrimination. To isolate |S| from document length, one would need D2 with truncated documents or D1 with artificially inflated |S|.

2. **Subscription taxonomy.** EUROVOC descriptors are hierarchical and overlapping, with many near-synonyms. CardiffNLP topics are coarse and disjoint. The fine-grained taxonomy may compound the |S| effect. However, the Qwen sensitivity_k_D2 data (F1 ~ 0 for all k values on D2) suggests the problem is more fundamental than taxonomy granularity.

3. **Multi-label density.** D2 has mean 2.2 labels per event vs. 1.6 (D1) and 1.0 (D3). Higher multi-label density means more true positives to find, potentially harder for the top-kappa mechanism. However, kappa = 3 should accommodate mean 2.2 labels adequately.

4. **Sonnet vs. Haiku.** The Discussion claims Sonnet achieves 2.3x over SBERT on D2, but this is based on a single seed with 5,000 events (not the full 65K). If confirmed, this would suggest that larger, more capable LLMs have higher |S|_cross thresholds, which is consistent with the discrimination capacity hypothesis (more capable attention mechanisms sustain accuracy at higher |S|).

5. **The |S| = 127 vs. 201 discrepancy.** The codebase loads EUROVOC level_2 labels, which yields 127 labels according to the CSV file. The paper states 201. If the actual subscription count is 127, this is still 6.7x larger than D1/D3 (19), and the qualitative analysis holds. The exact number affects the quantitative calibration of S_half but not the fundamental conclusion.

---

## 2.3 Five Evaluation Questions

### 1. Stronger or just longer?

**Stronger.** The analysis introduces a genuinely new theoretical concept (discrimination capacity as distinct from context capacity) that explains an otherwise puzzling empirical finding. It converts a potential weakness of the paper (LLM losing to baselines on D2) into a theoretical insight about the fundamental limits of LLM-as-classifier scaling. The proposed accuracy scaling law is falsifiable and connects to established work in extreme multi-label classification and attention saturation.

### 2. Changes contribution hierarchy?

**Yes, but constructively.** The paper's current contribution hierarchy is:
1. Architecture (Neural Router design)
2. Cost model (invocation/token scaling)
3. Empirical validation

Adding the discrimination complexity analysis would create:
1. Architecture
2. Joint cost-accuracy model (cost model + accuracy scaling law)
3. Empirical validation (now including D2 as a *predicted* boundary case, not an anomaly)

This strengthens the paper because it shows the authors understand the limits of their approach, which is exactly what reviewers value.

### 3. Creates new dependencies?

**Minimal.** The analysis relies on:
- The existing experimental data (already collected)
- The "lost in the middle" literature (Liu et al., 2024) -- well-known, no controversial dependency
- The extreme multi-label classification literature (standard ML) -- well-established
- No new experiments are strictly required, though a controlled |S|-sweep experiment would strengthen the argument substantially

### 4. Belongs in this paper?

**Yes.** The D2 results are already in the paper. Currently they are either misrepresented (the Discussion claims 2.3x advantage based on Sonnet single-seed subsample) or left unexplained. The discrimination complexity analysis turns an awkward result into a theoretical contribution. The alternative (dropping D2 or hiding the poor results) would be intellectually dishonest and would likely be caught by reviewers.

The analysis should appear in the Discussion section as a subsection on "Accuracy Scaling and Discrimination Complexity," approximately 1-1.5 pages. It does NOT require changes to the cost model derivation (which remains correct for cost). It requires:
- Honest reporting of D2 results in the Results section
- A new Discussion subsection explaining the discrimination capacity concept
- A note in the Limitations section acknowledging that the cost model does not capture accuracy scaling
- A Future Work item on controlled |S|-sweep experiments to calibrate the accuracy scaling law

### 5. Would a professor approve?

**Yes, with qualifications.** A professor would approve the core insight (discrimination capacity is distinct from context capacity) and the connection to established literature (XML, lost in the middle). They would want:
- Careful hedging (the proposed scaling law is a hypothesis, not a proven theorem)
- Clear separation between what the data shows and what is conjectured
- Acknowledgment that D1/D3 vs. D2 confounds |S| with document length and taxonomy
- A plan for controlled experiments in future work
- Correction of the Discussion's current 2.3x claim if the Haiku full-corpus results contradict it

A professor would NOT approve sweeping the D2 failure under the rug or claiming the cost model predicts accuracy when it does not.

---

## 2.4 Integration Plan

| Step | File | Section | Action | Net pages |
|------|------|---------|--------|-----------|
| 1 | Results.tex | Cross-Dataset Generalisability | Report D2 results honestly. Present the full Haiku results (best LLM config F1 = 0.089 vs. TF-IDF 0.162) alongside the Sonnet subsample results. Flag the discrepancy between full-corpus Haiku and subsampled Sonnet. | +0.25 |
| 2 | Discussion.tex | Analysis of Results (para 3) | Correct the "2.3x advantage" claim. Replace with honest characterisation: LLM advantage depends critically on |S|, with D2 representing a regime where baselines dominate. | +0.0 (net rewrite) |
| 3 | Discussion.tex | New subsection: "Discrimination Complexity" | Introduce the concept. Explain why context capacity (modelled) differs from discrimination capacity (not modelled). Present the evidence: |S| = 19 works, |S| = 201 fails. Connect to lost-in-the-middle and XML literature. Present the unified explanation with the tau plateau. Propose the accuracy scaling hypothesis with appropriate hedging. | +1.0 |
| 4 | Discussion.tex | Limitations | Add a paragraph acknowledging that the cost model does not capture accuracy scaling and that D2 results reveal this gap. Note the |S|/document-length/taxonomy confound. | +0.25 |
| 5 | Discussion.tex | Future Work | Add a bullet on controlled |S|-sweep experiments (varying |S| on D1 while holding other factors constant) to calibrate the discrimination capacity threshold. | +0.1 |
| 6 | Design.tex | Cost Model | Add a brief remark (1-2 sentences) after the crossover analysis noting that the cost model assumes the LLM's accuracy is independent of |S| given sufficient context window, and that this assumption is examined in the Discussion. | +0.1 |
| 7 | bibtex/ | References | Add Liu et al. 2024 (lost in the middle), Bhatia et al. 2015 or You et al. 2019 (XML). | +0.0 |
| | | | **Total** | **+1.7 pages** |

### Priority order

1. Steps 1-2 are **mandatory** (honest reporting).
2. Step 3 is **high value** (converts weakness to insight).
3. Steps 4-6 are **supporting** (scope the limitation, point to future work).
4. Step 7 is **mechanical** (add references).

### Risk assessment

- **Risk of over-claiming:** The scaling law is a hypothesis, not a theorem. Must be presented as such. The functional form (exponential decay, hyperbolic, etc.) cannot be calibrated from three data points (D1, D2, D3) with confounded variables.
- **Risk of undermining the paper:** If not framed carefully, the D2 analysis could read as "the system doesn't work for realistic subscription counts." The framing should be: "the system works well for the subscription counts typical of edge/IoT deployments (tens of subscriptions); for large taxonomy-based systems (hundreds of subscriptions), the cost model correctly identifies cost-optimal configurations, but an additional accuracy constraint emerges that motivates future work on hierarchical partitioning."
- **Risk of reviewer pushback on the 2.3x claim:** If the current Discussion says 2.3x but the full-corpus Haiku results say 0.53x (inverted), a reviewer who checks the numbers will see a discrepancy. Better to address it proactively than to be caught.

---

# Summary of Key Findings

1. **The cost model is correct but incomplete.** It models cost (invocations, tokens, latency) but not accuracy. The implicit assumption that LLM accuracy is invariant to |S| given sufficient W is falsified by D2.

2. **The D2 failure is primarily a function of |S|, not semantic difficulty.** D3 (extreme semantic gap, |S| = 19) works; D2 (moderate semantic gap, |S| = 201) fails. Subscription cardinality, not content difficulty, is the binding constraint.

3. **Discrimination capacity is distinct from context capacity.** The LLM can fit 201 subscriptions in its context window but cannot make 201 accurate binary decisions simultaneously. The attention mechanism's discrimination capacity degrades with |S|.

4. **The D2 failure and the tau plateau share a root cause:** subscription-space density exceeding the system's discrimination resources (attention for the LLM, embedding geometry for the cosine pre-filter).

5. **This reframes clustering from cost optimisation to accuracy management.** In high-|S| regimes, clustering should target per-cluster |S'| below the discrimination threshold, not just save tokens.

6. **Independent pairwise methods (TF-IDF, SBERT) do not suffer from |S|-scaling** because each (event, subscription) decision is independent. The LLM's decisions are entangled through the shared attention mechanism.

7. **The paper should report D2 results honestly** and present the discrimination complexity analysis as a theoretical contribution, not hide the failure.
