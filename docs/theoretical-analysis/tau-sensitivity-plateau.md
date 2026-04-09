---
title: "Theoretical Analysis: Cosine Threshold Plateau and the Geometry of Embedding-Based Pre-Filtering"
date: 2026-04-06
source: experimental tau sensitivity results vs. cost model predictions (Design.tex Section 3.5)
manuscript: Manuscripts/Neural Router (Elsevier FGCS)/
status: draft
tags: [theoretical-analysis, embedding-geometry, cosine-filter, tau-sensitivity]
---

# Cosine Threshold Plateau and the Geometry of Embedding-Based Pre-Filtering

## 1. The Anomaly

The Neural Router's cost model (Design.tex, Section 3.5) predicts that the cosine threshold tau controls a smooth precision-recall tradeoff in the event-to-cluster pre-filter: lower tau routes events to more clusters (higher recall, higher cost), while higher tau restricts routing (lower cost, potential recall loss). The paper's default is tau=0.3.

Experimental tau sensitivity sweeps on the A3 configuration (clustering + cover/merge, k=19, kappa=3) reveal a fundamentally different behavior. Three anomalies are observed consistently across all datasets and all LLM backends:

**Anomaly 1: Suboptimal default.** F1 peaks at tau=0.0-0.2, not at the paper's default tau=0.3, on every dataset and every backend tested.

**Anomaly 2: Binary plateau.** Above a dataset-specific critical threshold (approximately tau=0.3-0.4), F1, precision, recall, FPR, and the number of LLM invocations become completely identical for all higher tau values. The cosine filter becomes inert: raising tau from 0.4 to 0.9 changes nothing.

**Anomaly 3: Absence of smooth tradeoff.** Instead of the predicted continuous precision-recall curve, the data shows at most 3-4 distinct operating points before collapsing to a plateau. The filter has effectively binary behavior: events either pass (for their nearest cluster) or they don't, with no intermediate regime.

### 1.1 Raw Evidence

**D1 (CardiffNLP, |S|=19, Sonnet):**

| tau | F1    | Invocations | Recall | Precision |
|-----|-------|-------------|--------|-----------|
| 0.0 | 0.532 | 19          | 0.641  | 0.497     |
| 0.1 | 0.576 | 19          | 0.628  | 0.592     |
| 0.2 | 0.475 | 19          | 0.461  | 0.537     |
| 0.3 | 0.433 | 16          | 0.393  | 0.520     |
| 0.4 | 0.414 | 16          | 0.376  | 0.500     |
| 0.5-0.9 | 0.414 | 16     | 0.376  | 0.500     |

**D2 (EUR-Lex, |S|=201, Sonnet):**

| tau | F1    | Invocations |
|-----|-------|-------------|
| 0.0 | 0.103 | 118         |
| 0.1 | 0.080 | 105         |
| 0.2 | 0.109 | 101         |
| 0.3 | 0.070 | 92          |
| 0.4 | 0.054 | 88          |
| 0.5-0.9 | 0.054 | 88     |

**D3 (CASAS IoT, |S|=19, Sonnet):**

| tau | F1    | Invocations |
|-----|-------|-------------|
| 0.0 | 0.106 | 66          |
| 0.1 | 0.110 | 66          |
| 0.2 | 0.111 | 66          |
| 0.3 | 0.098 | 66          |
| 0.4 | 0.091 | 66          |
| 0.5-0.9 | 0.091 | 66     |

**D1, Qwen-7B (confirms cross-backend consistency):**

| tau | F1    | Invocations |
|-----|-------|-------------|
| 0.0 | 0.388 | 38          |
| 0.1 | 0.443 | 21          |
| 0.2 | 0.275 | 19          |
| 0.3 | 0.278 | 16          |
| 0.4-0.9 | 0.278 | 16     |

### 1.2 Gap Classification (Phase 1)

This is a **framework gap**, not a proof gap or a related-work awareness issue. The cost model's mathematical structure is internally consistent (the equations are correct given their assumptions), but the model assumes the cosine similarity between event embeddings and cluster centroids is distributed continuously and broadly enough that the threshold tau produces a meaningful gradation. The empirical data shows this assumption is violated: the similarity distribution is concentrated in a narrow band, making the threshold a near-binary switch rather than a continuous dial.

## 2. Deep Understanding: Why the Plateau Occurs

### 2.1 The Concentration of Measure in High-Dimensional Embedding Spaces

The root cause is a well-known phenomenon in high-dimensional geometry: **concentration of measure** (also called the "curse of dimensionality" for distance functions). In a d-dimensional unit sphere (which is the embedding space for L2-normalized sentence embeddings), the cosine similarity between a random unit vector and any fixed unit vector concentrates around zero as d grows, with variance proportional to 1/d.

For the all-MiniLM-L6-v2 model used in the experiments, d=384. The key theoretical result is:

**Theorem (Concentration of cosine similarity).** For two independent random unit vectors u, v drawn uniformly from S^{d-1}, the cosine similarity cos(u, v) = u . v satisfies:
- E[cos(u, v)] = 0
- Var[cos(u, v)] = 1/d
- P(|cos(u, v)| > epsilon) <= 2 exp(-d * epsilon^2 / 2) (sub-Gaussian tail)

For d=384, the standard deviation of random cosine similarity is 1/sqrt(384) = 0.051. This means that 95% of pairwise cosine similarities between random unit vectors fall within [-0.102, 0.102].

Of course, sentence embeddings are not random: semantically related texts cluster together, producing a non-uniform distribution. But the centroid of a k-means cluster is a mean of several subscription embeddings, and the event embedding must be compared to this centroid. The centroid averaging introduces an additional smoothing effect (the centroid is "closer to the origin" in terms of angular specificity), further compressing the similarity range.

### 2.2 The Centroid Dilution Effect

When k-means produces cluster centroids from semantically related subscriptions, the centroid vector is the mean of the member embeddings:

    centroid_c = (1/|c|) * sum_{s in c} theta(s)

Even within a semantically coherent cluster, individual subscription embeddings point in somewhat different directions. The mean operation dilutes the centroid: its norm (before re-normalization) is less than 1, and its direction represents a "compromise" that is not perfectly aligned with any member. After L2 normalization (which sklearn's cosine_similarity implicitly handles), the centroid points toward the geometric center of the cluster on the hypersphere.

The critical consequence: **the cosine similarity between an event embedding and a cluster centroid is systematically lower than the similarity between the event and any individual subscription in that cluster.** The centroid dilution compresses the range of meaningful similarities, pushing them toward lower values.

### 2.3 Quantitative Reconstruction of the Plateau

The data directly reveals the similarity distribution's structure:

**D1 analysis (k=19 clusters, Sonnet):**
- At tau=0.0: all 19 clusters receive events (invocations=19). Events go to every cluster.
- At tau=0.1: still 19 clusters, but invocations=19 (same). The filter is active but no events are excluded from any cluster they would reach. However, F1 improves (0.532 to 0.576), suggesting that some spurious multi-cluster assignments are removed.
- At tau=0.2: still 19 invocations but F1 drops to 0.475. Recall drops sharply (0.641 to 0.461), meaning events are now being excluded from clusters that contained their correct subscriptions.
- At tau=0.3: invocations drop to 16 (3 clusters become completely unreachable). F1 drops further.
- At tau=0.4-0.9: nothing changes. The 16 remaining clusters each have at least one event assigned via the fallback mechanism (nearest cluster), and no events have similarity >= 0.4 to any non-nearest centroid.

This tells us the similarity distribution has the following shape:
- The maximum event-to-centroid cosine similarity (for the nearest cluster) is typically in the range [0.2, 0.5].
- The second-highest similarity (the next-nearest cluster) is almost always below 0.3.
- No event has similarity >= 0.4 to any centroid other than its nearest one.

**D3 analysis (CASAS IoT, k=19):**
- Invocations=66 at ALL tau values (0.0 through 0.9).
- This means that even at tau=0.0, 66 cluster-event assignments are made (each event goes to all 19 clusters, but with batching, this produces 66 invocations).
- The invocation count never changes because the fallback mechanism assigns every event to at least its nearest cluster. With 19 clusters, the invocation count is determined by the batch size, not by tau.
- F1 still varies slightly (0.106 to 0.111 to 0.091), which must be due to the difference between "event goes to all clusters" (tau=0.0, all subscriptions evaluated) versus "event goes only to nearest cluster" (high tau, only that cluster's subscriptions evaluated).

Wait: re-reading the code more carefully, at tau=0.0 with `use_cosine_filter=True`, the condition `similarities[i, j] >= 0.0` is always true (cosine similarity of normalized vectors is in [-1, 1], and empirically positive for sentence embeddings). So tau=0.0 with the cosine filter enabled is equivalent to sending events to all clusters. But for D3, invocations=66 at all tau values, meaning all events always pass the filter for all clusters (the maximum similarity to any non-nearest centroid is still >= tau for all tested tau values), OR the fallback assigns them.

Actually, re-reading D3: the F1 and other metrics DO change between tau values (0.106 at tau=0.0, 0.111 at tau=0.2, 0.091 at tau=0.4), even though invocations stay at 66. This means the routing decisions change (events go to different sets of clusters), but the total invocation count stays constant because the batch structure absorbs the change. The plateau at tau=0.4-0.9 means that above 0.4, every event reaches only its nearest cluster and the fallback activates for all of them.

### 2.4 The Fallback Mechanism Masks the Plateau

A crucial implementation detail: when no cluster passes the cosine threshold, the router assigns the event to its nearest cluster (the fallback in router.py lines 466-469). This ensures every event is matched against at least one cluster. At high tau, the fallback fires for nearly every event, making tau irrelevant. The router degenerates to a nearest-centroid assignment, which is equivalent to hard Voronoi partitioning of the embedding space. The tau parameter becomes a no-op.

This means the practical operating regime of the cosine pre-filter is confined to a narrow band:
- **tau < 0.1**: Most events go to most/all clusters (expensive, high recall)
- **tau in [0.1, 0.3]**: The only range where tau actually modulates routing behavior
- **tau > 0.3**: Fallback dominates; equivalent to nearest-centroid assignment

The "smooth precision-recall tradeoff" predicted by the cost model exists only within the [0.1, 0.3] band, a range of width 0.2 that offers at most 2-3 meaningfully different operating points.

## 3. Expansion (Phase 2)

### 3.1 Can the Fix Be More General?

**Yes.** The plateau is not a bug in the Neural Router's implementation; it is a fundamental property of cosine similarity between embeddings and centroids in high-dimensional spaces. Any system that uses embedding cosine similarity as a routing pre-filter (with a fixed threshold) will encounter this behavior. This includes:

- **Retrieval-augmented generation (RAG)** systems that use cosine thresholds to filter candidate documents before LLM processing
- **Semantic routers** (e.g., Aurelio AI's Semantic Router) that classify intents using embedding distance thresholds
- **Vector database queries** with similarity thresholds (e.g., Pinecone, Weaviate score thresholds)
- **Content-based recommendation** systems using embedding similarity filters

The general principle is: **fixed cosine similarity thresholds are poor discriminators in high-dimensional embedding spaces because the similarity distribution concentrates in a narrow band.** The threshold either lets everything through or blocks everything, with a very narrow transition zone.

This is a known result in the information retrieval literature, where it manifests as the "hubness problem" (Radovanovic et al., 2010) and the concentration of distances in high-dimensional spaces (Beyer et al., 1999; Aggarwal et al., 2001). The phenomenon is sometimes called the "curse of dimensionality for nearest-neighbor search."

The general fix is to replace fixed absolute thresholds with **relative or rank-based filtering**:
- Top-k nearest centroids (already implicit in the fallback mechanism)
- Percentile-based thresholds (e.g., "route to all clusters within the top 10% of similarity scores for this event")
- Adaptive thresholds based on the event's similarity distribution (e.g., "route to all clusters where similarity > max_similarity - delta")

### 3.2 Does This Connect to Other Issues?

**Yes, directly.** A parallel analysis is examining D2 (EUR-Lex), where the LLM loses to TF-IDF and SBERT baselines. The connection:

With |S|=201 subscriptions clustered into k=19 groups, each cluster contains approximately 10-11 subscriptions. The cosine pre-filter's binary behavior means that in the A3 configuration, events effectively see only the subscriptions in their nearest cluster (approximately 10-11 out of 201). If the ground-truth matching subscriptions are spread across multiple clusters (which is likely for EUR-Lex's fine-grained, overlapping legal categories), the pre-filter eliminates them before the LLM ever sees them. The LLM cannot match what it never receives.

This explains why A0 (no clustering, no pre-filter) outperforms A3 on D2: A0 shows all 201 subscriptions to the LLM, while A3 shows only the 10-11 in the nearest cluster. The recall ceiling is structurally limited by the pre-filter, not by the LLM's reasoning ability.

The tau plateau exacerbates this: operators cannot tune tau to recover the lost recall because tau > 0.3 is already in the plateau zone. The only escape is tau < 0.2, which sends events to nearly all clusters (approaching A0's behavior but with the overhead of k separate LLM calls).

### 3.3 Is There a Deeper Pattern?

**Yes: the mismatch between continuous cost models and discrete routing decisions.** The cost model (Equations 2-7 in Design.tex) treats the number of active clusters per event as a continuous quantity modulated by tau. In reality, the cosine pre-filter makes a binary decision per cluster (pass/fail), and the concentration of similarities means these decisions flip together (as a group) rather than individually. The result is that the "number of clusters per event" is effectively quantized to a small number of values: {1, k}, with perhaps one or two intermediate points.

This is a specific instance of a broader pattern in computational systems: **optimization models that assume continuous control over discrete decisions produce misleading sensitivity predictions.** The cost model's tau derivative (dI/d_tau) is treated as smooth, but the actual function is piecewise constant with at most 3-4 steps.

The deeper pattern connects to phase-transition behavior in combinatorial optimization. The cosine filter acts like a percolation threshold on a bipartite graph (events x clusters): below a critical tau, almost all edges exist (events reach almost all clusters); above it, almost none do (events reach only their nearest cluster). The transition is sharp, not gradual, because the edge weights (cosine similarities) are concentrated.

### 3.4 Does It Suggest a Cross-Field Connection?

**Yes, to several fields:**

**1. High-dimensional geometry and the concentration of measure.** The mathematical foundation is the concentration of Lipschitz functions on the unit sphere (Levy's lemma). For any Lipschitz function f on S^{d-1} with Lipschitz constant L, and any epsilon > 0:

    P(|f(x) - E[f]| > epsilon) <= 2 exp(-c * d * epsilon^2 / L^2)

Cosine similarity is a Lipschitz function with L=1, giving exponential concentration around the mean. This is not a weakness of sentence-BERT specifically; it is a property of any L2-normalized embedding in R^d for large d. The phenomenon would be mitigated by lower-dimensional embeddings (d << 100) but exacerbated by higher-dimensional ones.

**2. Information retrieval and the hubness problem.** In high-dimensional spaces, some points ("hubs") appear as nearest neighbors of many other points, while most points appear as nearest neighbors of very few (Radovanovic et al., 2010; "Hubs in space: Popular nearest neighbors in high-dimensional data"). The dual phenomenon is that distances/similarities to non-nearest neighbors concentrate, making threshold-based discrimination ineffective. This is precisely what the tau plateau reveals.

**3. Clustering theory and the Johnson-Lindenstrauss lemma.** The JL lemma guarantees that random projections preserve pairwise distances up to (1 +/- epsilon), but the key insight is that in the projected (or original high-dimensional) space, the distance distribution concentrates. K-means centroids, being means of cluster members, are even more "central" and thus more similar to each other and to arbitrary points than individual data points are.

**4. Vector quantization and product quantization in approximate nearest-neighbor search.** The ANN search literature (Jegou et al., 2011; "Product quantization for nearest neighbor search") has long recognized that absolute distance thresholds are unreliable in high dimensions. State-of-the-art ANN systems (FAISS, ScaNN, HNSW) use rank-based retrieval (top-k) rather than threshold-based filtering for exactly this reason.

## 4. Proposed Resolution

### 4.1 Diagnosis

The cost model's tau treatment contains an implicit assumption that is violated in practice:

**Assumption (implicit in Design.tex Section 3.5):** The cosine similarities cos(theta(e), centroid_c) for an event e across clusters c in C are distributed broadly enough that varying tau from 0 to 1 produces a monotonic, approximately continuous variation in the number of active clusters per event.

**Reality:** Due to concentration of measure in d=384 dimensions, these similarities are concentrated in a narrow band (approximately [0.05, 0.35] for most event-centroid pairs). The centroid dilution effect further compresses this range. As a result, tau has at most 3-4 effective operating points before the fallback mechanism makes it inert.

### 4.2 Three-Level Resolution

The resolution has three levels, from simplest (acknowledge and scope) to most ambitious (redesign):

#### Level 1: Acknowledge the Effective Operating Range (Minimal Manuscript Change)

State explicitly in the Discussion that the cosine threshold's effective range is narrow due to concentration of measure, and that the practical recommendation is tau in [0.05, 0.2] rather than tau=0.3. This is honest reporting that strengthens the paper by showing the authors understand the limitation.

Specific language for the sensitivity analysis: "The cosine threshold tau has a narrow effective operating range (approximately 0.05-0.2) due to the concentration of cosine similarities in high-dimensional embedding spaces (d=384). Above approximately tau=0.3, the fallback mechanism assigns all events to their nearest centroid, making tau inert. This is a known consequence of the concentration of measure on the unit sphere (Levy's lemma) and applies broadly to any system using fixed cosine thresholds for pre-filtering in high-dimensional embedding spaces."

#### Level 2: Replace tau with Rank-Based Routing (Algorithmic Fix)

Replace the threshold-based cosine filter with a rank-based assignment:

    For each event e:
        similarities = [cos(theta(e), centroid_c) for c in C]
        route to top-r clusters by similarity

where r in {1, 2, ..., k} is the new hyperparameter. This provides exactly r routing targets per event, giving the operator a discrete but well-defined control over the recall-cost tradeoff. The value r=1 corresponds to nearest-centroid (the current high-tau behavior), and r=k corresponds to broadcast (the current tau=0 behavior).

Alternatively, a relative-threshold approach:

    route to all clusters c where cos(theta(e), centroid_c) >= max_c cos(theta(e), centroid_c) - delta

where delta controls how far from the best match an event is still routed. This adapts to each event's similarity profile and avoids the concentration problem.

#### Level 3: Characterize the Concentration and Integrate into the Cost Model (Theory Extension)

Derive the expected number of active clusters per event as a function of tau, d, and the intra-cluster variance, incorporating the concentration of measure. Specifically:

Let sigma_sim be the standard deviation of event-to-centroid cosine similarities for a typical event. Then the number of active clusters for threshold tau is approximately:

    n_active(tau) = k * Phi((mu_sim - tau) / sigma_sim)

where Phi is the standard normal CDF and mu_sim is the mean similarity. For concentrated distributions (sigma_sim << mu_sim), this function is a steep sigmoid rather than a gradual curve, explaining the near-binary behavior. The cost model can then incorporate this to give more accurate invocation predictions.

### 4.3 Recommended Resolution for This Paper

**Level 1 is sufficient and appropriate.** The paper's contribution is the Neural Router architecture and cost model, not the cosine pre-filter. The pre-filter is a design component, and its behavior at different tau values is an empirical finding that should be reported accurately.

Level 2 (rank-based routing) is a straightforward improvement that could be mentioned as future work. Level 3 (theory extension) would strengthen the cost model but may be more appropriate for a follow-up paper focused on embedding-based routing.

The key manuscript changes:
1. Acknowledge the narrow effective range in the sensitivity analysis discussion.
2. Update the default tau recommendation from 0.3 to a dataset-specific optimal (approximately 0.1).
3. Explain the mechanism (concentration of measure + centroid dilution + fallback).
4. Connect to the D2 underperformance: the pre-filter's binary behavior is one factor limiting A3 on large subscription spaces.
5. Mention rank-based routing as a natural improvement direction.

## 5. Evaluation (Phase 2.3)

### 5.1 Stronger or Just Longer?

**Stronger.** The analysis identifies a concrete mechanism (concentration of measure) that explains an otherwise puzzling empirical finding, connects it to established theory in high-dimensional geometry and information retrieval, and provides actionable recommendations (change the default tau, consider rank-based routing). This is not a padding exercise; it resolves a genuine gap between the cost model's predictions and the experimental data.

### 5.2 Changes Contribution Hierarchy?

**No.** The paper's main contributions are (1) the Neural Router architecture, (2) the cost model with crossover analysis, and (3) the empirical evaluation. This analysis refines contribution (2) by identifying a boundary condition of the cost model (the tau smoothness assumption fails due to concentration of measure) and refines contribution (3) by explaining an experimental anomaly. It does not introduce a new contribution; it strengthens the existing ones by showing the authors understand their system's limitations.

### 5.3 Creates New Dependencies?

**Minimal.** Level 1 requires citing the concentration of measure phenomenon (standard references: Ledoux 2001, "The Concentration of Measure Phenomenon"; Vershynin 2018, "High-Dimensional Probability"). Level 2 would require re-running the tau sensitivity with a rank-based parameter, which is feasible but not essential. Level 3 would require a new theoretical section, which is not recommended for this paper.

### 5.4 Belongs in This Paper?

**Yes, at Level 1.** The tau sensitivity analysis is already in the paper (Section 4.6, Figure 6b). The analysis explains the observed results rather than introducing new content. The explanation strengthens the Discussion section by connecting the empirical finding to established theory, which is exactly what reviewers expect.

### 5.5 Would a Professor Approve?

**Yes.** The analysis:
- Correctly identifies the root cause (concentration of measure in high-dimensional spaces)
- Provides quantitative reconstruction from the data
- Connects to established literature (Levy's lemma, hubness problem, ANN search)
- Gives actionable recommendations at appropriate scope
- Does not over-claim (acknowledges this is a known phenomenon, not a novel discovery)
- Recommends a proportionate response (Level 1 for this paper, Levels 2-3 for future work)

The only concern a professor might raise is whether the paper should have anticipated this issue during the design phase (since concentration of measure is well-known). The defense: the Neural Router is primarily an architectural contribution, and the cosine pre-filter is one component whose sensitivity characteristics are appropriately studied empirically. The finding that it has a narrow effective range is a useful empirical contribution in itself.

## 6. Integration Plan

| Step | File | Section | Action | Net pages |
|------|------|---------|--------|-----------|
| 1 | txt/Results.tex | Sec. 5.3 (Sensitivity) | Add 1 paragraph explaining the tau plateau with reference to concentration of measure. Note the effective operating range [0.05, 0.2]. Note that invocations freeze above tau~0.3. | +0.3 |
| 2 | txt/Discussion.tex | Sec. 6.1 (Analysis) | Add 1-2 paragraphs connecting tau plateau to (a) the cost model's implicit smoothness assumption, (b) the D2 pre-filter recall ceiling, (c) the general principle that fixed cosine thresholds are poor discriminators in high-d spaces. | +0.4 |
| 3 | txt/Discussion.tex | Sec. 6.4 (Limitations) | Add 1 sentence to an existing limitation bullet noting that the cosine pre-filter has a narrow effective range. | +0.05 |
| 4 | txt/Discussion.tex | Sec. 6.5 (Future Work) | Add "rank-based cluster routing" as a bullet in Future Work. | +0.1 |
| 5 | txt/Design.tex | Sec. 3.5 (Cost Model) | Add a remark noting that the smooth-tau assumption holds only when the similarity distribution is broad relative to the threshold range; in high-dimensional spaces, concentration of measure compresses this range. | +0.2 |
| 6 | main.tex | References | Add 2-3 references: concentration of measure (Vershynin 2018 or Ledoux 2001), hubness problem (Radovanovic et al. 2010), optional: Beyer et al. 1999. | +0.1 |
| **Total** | | | | **+1.1 pages** |

## 7. Summary of Key Findings

1. **The cosine threshold tau has a narrow effective operating range** (approximately [0.05, 0.2] for the default all-MiniLM-L6-v2 embeddings with d=384). Above tau~0.3, the fallback mechanism makes tau inert.

2. **The root cause is concentration of measure on the unit sphere.** In d=384 dimensions, cosine similarities between event embeddings and cluster centroids are concentrated in a narrow band. The centroid dilution effect (averaging member embeddings) further compresses the range.

3. **The cost model's implicit smoothness assumption is violated.** The model treats the number of active clusters per event as a smooth function of tau. In reality, it is a near-step function with at most 3-4 distinct levels.

4. **This is a general property of embedding-based pre-filtering**, not specific to the Neural Router or to sentence-BERT. Any system using fixed cosine thresholds in high-dimensional spaces will encounter this behavior.

5. **The fix is proportionate to the paper:** acknowledge the narrow effective range, update the default recommendation, explain the mechanism via concentration of measure, and mention rank-based routing as future work. This adds approximately 1 page and strengthens the Discussion without changing the contribution hierarchy.

6. **The D2 underperformance connection:** The pre-filter's binary behavior is one factor (among others) limiting A3 on EUR-Lex, because events see only their nearest cluster's subscriptions rather than the full 201-subscription space.
