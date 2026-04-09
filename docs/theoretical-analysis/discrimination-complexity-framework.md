---
title: "Theoretical Analysis: Discrimination Complexity Framework and Tractability via Graph Structure"
date: 2026-04-06
source: experimental results (D2 underperformance, tau plateau) + trilogy/TEAC tractability pattern
manuscript: Manuscripts/Neural Router (Elsevier FGCS)/
related:
  - docs/theoretical-analysis/d2-eurlex-underperformance.md
  - docs/theoretical-analysis/tau-sensitivity-plateau.md
  - Manuscripts/Trilogy Paper 1 (Proposition 1, 3)
  - Manuscripts/Trilogy Paper 2 (Theorem 1)
  - Manuscripts/Governance Duality (TEAC) (multi-level hierarchy results)
  - Manuscripts/Phase Transition (Econometrica) (phase transition theorem)
status: draft
tags: [theoretical-analysis, discrimination-complexity, tractability, polymatroid, graph-structure]
---

# Discrimination Complexity Framework and Tractability via Graph Structure

## 1. Summary of Prior Analysis

Two companion notes (d2-eurlex-underperformance.md, tau-sensitivity-plateau.md) established:

1. **The cost model is incomplete.** It models invocations and tokens (context capacity) but not LLM accuracy (discrimination capacity). These are distinct resources with different scaling.

2. **Discrimination capacity degrades with |S|.** The LLM dominates baselines at |S|=19 (D1, D3) but loses to TF-IDF/SBERT at |S|=201 (D2). The binding constraint is not the context window but the LLM's ability to make |S| simultaneous binary decisions.

3. **The cosine pre-filter operates in a binary regime.** Concentration of measure in d=384 embedding space means tau has a narrow effective range (~0.05-0.2). Above tau~0.3, the fallback fires for every event, making tau inert.

4. **Both issues share a root cause:** subscription-space density exceeding the system's discrimination resources (LLM attention for matching, embedding geometry for routing).

This note develops the full theoretical framework, including the architectural consequences and, critically, the connections to the tractability results from the trilogy and TEAC papers.

---

## 2. The Two-Constraint Model

### 2.1 Current Model (One Constraint)

The paper's cost model has one resource constraint (context window):

    W_used = t_inst + |S'_c| * t_s + b * t_e + t_resp  <=  W

When W is large (modern cloud LLMs, W >= 128K), this constraint is slack. A0 (all subscriptions in one prompt) is cost-optimal because it minimizes invocations.

### 2.2 Missing Constraint (Discrimination Capacity)

The D2 results reveal a second constraint:

    |S'_c|  <=  D(model)

where D(model) is the maximum number of subscriptions the LLM can evaluate simultaneously with acceptable accuracy. Empirically: D(Haiku) ~ 15-20, D(Sonnet) ~ 40-50.

### 2.3 Joint Optimization

The partitioning problem becomes:

    min_k  sum_c ceil(m_c / b_max(c))              (minimize invocations)
    
    subject to:
      (C1)  |S'_c| * t_s + b * t_e  <=  W - overhead       (context)
      (C2)  |S'_c|  <=  D(model)                             (discrimination)
      (C3)  coverage(e) >= alpha   for all events e           (recall)

For modern LLMs, (C1) is slack and (C2) is binding. This inverts the motivation for clustering: partitioning is needed for **accuracy**, not cost.

### 2.4 Two Crossover Points

The paper derives W_cross (context crossover). The full theory needs:

- **W_cross**: Below which compression saves invocations. Above, A0 is cost-optimal. (Existing result, correct.)
- **|S|_cross**: Below which A0's accuracy is acceptable. Above, partitioning is needed even though A0 is cost-optimal.

For most deployments: W >> W_cross (context abundant) but |S| may exceed |S|_cross (discrimination scarce). The binding constraint flips from the paper's assumption.

---

## 3. The Partitioning Problem

### 3.1 Formal Statement

Given:
- Bipartite matching graph G = (E union S, M) where (e, s) in M iff event e matches subscription s
- Discrimination capacity D(model)
- Recall floor alpha

Find a partition P = {c_1, ..., c_k} of S such that:
- |c_i| <= D(model) for all i                    (discrimination constraint)
- For each event e, Pr(true matches in visited clusters) >= alpha    (recall constraint)
- k is minimized                                   (cost: fewer clusters = fewer invocations)

### 3.2 Hardness in General

This is a constrained balanced graph partitioning problem. In general:

- **NP-hard** (by reduction from minimum bisection on general graphs)
- The recall constraint (C3) makes it harder: it requires that ground-truth matchings are concentrated within clusters, which constrains which subscriptions can be co-located
- Even approximating the minimum number of clusters subject to coverage guarantees is hard on arbitrary bipartite graphs

### 3.3 The Key Question

**Under what structural conditions on the subscription space does this become tractable?**

This is precisely the pattern from the trilogy and TEAC papers: a general NP-hard problem that becomes polynomial-time under graph-structural restrictions.

---

## 4. Connections to the Trilogy and TEAC Tractability Results

### 4.1 The Common Pattern

| Paper | General Problem | NP-hard Because | Tractable When | Structural Property |
|---|---|---|---|---|
| **Trilogy 1** | Allocation on service-dependency DAGs | Complementarities in general DAGs | Tree or series-parallel DAGs | Polymatroid (submodular rank function) |
| **Trilogy 1, Prop. 3** | Allocation on arbitrary DAGs | Same | Quotient graph after encapsulation is tree/SP | Encapsulation restores polymatroidal structure |
| **TEAC** | Multi-level governance optimization | Budget-constrained compliance on general DAGs | Trees (DP) or chains (O(n)) | Supermodularity + Topkis monotonicity |
| **Neural Router** | Subscription partitioning for discrimination | Min-cut balanced partitioning on arbitrary bipartite graphs | **???** | **???** |

The question marks are what this analysis fills in.

### 4.2 Tree-Structured Subscription Taxonomies

Many real-world subscription spaces have hierarchical (tree) structure:

- **EUR-Lex (D2):** EUROVOC is a 3-level hierarchy (domains -> micro-thesauri -> descriptors). The 201 level-2 "subject matters" sit within a tree of 21 level-1 domains.
- **News categorization:** Topic taxonomies are typically hierarchical (e.g., IPTC NewsCodes).
- **IoT device registries:** Sensor type hierarchies (location -> device type -> sensor ID).

**Claim:** When the subscription taxonomy T = (S, E_T) is a rooted tree, the discrimination-constrained partitioning problem becomes polynomial-time solvable.

**Argument sketch:**

Let T be a rooted tree on |S| subscriptions. We seek a partition into clusters of size <= D(model) that minimizes cross-cluster matchings (i.e., events whose true matches span multiple clusters).

This is equivalent to **tree partitioning with capacity constraints**: cut edges in the tree to produce connected subtrees of size <= D(model), minimizing the number of cut edges (each cut edge represents a pair of related subscriptions placed in different clusters, increasing recall loss).

Tree partitioning with capacity constraints is solvable in polynomial time:

1. **Bottom-up greedy:** Process the tree bottom-up. When a subtree exceeds D(model) leaves, cut it at the lowest ancestor that restores feasibility. This produces O(|S|/D) clusters with connected subtrees.

2. **Dynamic programming:** For optimal solutions, DP on the tree with state = (current subtree root, accumulated size). Complexity: O(|S| * D(model)), polynomial.

The polynomial tractability comes from the same structural property as Trilogy 1: **tree structure induces a laminar family** on the subscription subsets, which constrains the partitioning to respect the hierarchy.

### 4.3 Encapsulation: From Arbitrary Graphs to Trees (Trilogy 1, Proposition 3)

Trilogy Paper 1's Proposition 3 shows that arbitrary DAGs can be made tractable via **encapsulation**: integrators partition non-leaf nodes into disjoint clusters, each managing an internal sub-DAG and exposing a single composite capacity. If the **quotient graph** (after contracting each cluster) has tree or SP structure, the allocation problem becomes polymatroidal and polynomial.

The Neural Router can use the **exact same trick**:

1. **Identify natural sub-taxonomies** within the subscription space (e.g., EUROVOC domains, news categories, IoT device groups).
2. **Encapsulate** each sub-taxonomy into a meta-subscription with a summary description.
3. **Build a quotient subscription graph** where each meta-subscription is a node.
4. If the quotient graph is a tree or SP graph, the partitioning is tractable.

**Concretely for D2 (EUR-Lex):**
- 201 EUROVOC descriptors organized under 21 domains
- Encapsulate each domain's descriptors into one meta-cluster
- Quotient graph: 21 domain nodes (a flat set, trivially a forest)
- Each domain has ~10 descriptors, which is <= D(Haiku) ~ 15-20
- **Result:** The discrimination constraint is satisfied without NP-hard optimization, because the taxonomy provides the tree structure for free

This is architecturally significant: many real-world subscription spaces come with taxonomies that provide exactly the structural restriction needed for tractability.

### 4.4 The Polymatroid Connection

In Trilogy 1, the feasibility region under tree/SP DAGs forms a polymatroid (Proposition 1). This enables:
- Greedy algorithms for allocation (polynomial time)
- Walrasian equilibria (competitive pricing)
- DSIC mechanisms (truthful bidding)

For the Neural Router, an analogous result may hold:

**Conjecture (Polymatroidal Subscription Assignment):** When the subscription taxonomy is a tree and the discrimination capacity constraint |S'_c| <= D is uniform, the set of feasible partitions (satisfying discrimination + coverage constraints) forms a polymatroid on the subscription set, parameterized by D and the tree structure.

If this conjecture holds, the optimal partitioning is computable greedily (rank function evaluation = max-flow on the tree, as in Trilogy 1). The proof would follow the same pattern: tree structure -> laminar family -> submodularity of the capacity function.

**Status:** This is a conjecture, not a theorem. Verifying it requires formalizing the coverage constraint (C3) as a matroid rank condition and checking submodularity. The analogy to Trilogy 1 is strong but the mapping is not immediate because the Neural Router's problem is on a bipartite graph (events x subscriptions) rather than a resource DAG.

### 4.5 Supermodularity of Joint Optimization (TEAC Connection)

The TEAC paper shows that in multi-level governance, resolving one level increases the marginal value of resolving others (strict complementarity, interaction term I > 0). This creates supermodular structure: the optimal resolution strategy can be computed via Topkis monotonicity on trees and DP.

For the Neural Router, an analogous supermodularity exists between **routing quality** (how well events are assigned to clusters) and **discrimination quality** (how well the LLM matches within each cluster):

    Value(routing, discrimination) > Value(routing, 0) + Value(0, discrimination) - Value(0, 0)

Improving routing (events reach the right clusters) increases the value of better LLM discrimination (because the LLM sees more relevant subscriptions). Improving discrimination (smaller |S'| per prompt) increases the value of better routing (because each routing error discards a larger fraction of the relevant subscription space).

This supermodularity has a practical consequence: **joint optimization of routing and partitioning outperforms sequential optimization.** The current architecture optimizes them independently (k-means clustering, then cosine routing, then LLM matching). A joint approach would:

1. Partition subscriptions to minimize the expected number of clusters each event must visit (routing-aware partitioning)
2. Route events to clusters based on the partitioning structure (partition-aware routing)
3. Match within clusters with |S'_c| <= D(model) guaranteed

On tree-structured taxonomies, this joint optimization is tractable via DP, matching the TEAC result for multi-level hierarchies.

### 4.6 Phase Transition Analogy (Econometrica Connection)

The Phase Transition paper shows a sharp transition at the C^{0,1}/C^1 differentiability boundary: step-function oversight achieves first-best for every scoring rule, while smooth oversight incurs welfare loss for every non-Brier rule. The transition is discontinuous.

The Neural Router exhibits an analogous phase transition at |S|_cross:

| Below |S|_cross | Above |S|_cross |
|---|---|
| LLM dominates all baselines | Pairwise methods dominate LLM |
| A0 (single prompt) is optimal | Partitioned architecture is necessary |
| Clustering hurts (adds overhead) | Clustering helps (manages discrimination load) |
| Cost model is sufficient | Accuracy model is required |

The transition is sharp, not gradual (D1/D3 at |S|=19 show strong LLM advantage; D2 at |S|=201 shows LLM failure). This mirrors the phase transition's discontinuity: a continuous parameter (|S| or smoothness) creates a discrete change in optimal strategy.

The connection is more than an analogy. In both cases, the mechanism is the same: a **resource limitation** (LLM attention / scoring rule curvature) creates a **regime boundary** beyond which the qualitatively best approach changes. Below the boundary, a "generous" strategy works (single prompt / smooth oversight). Above, a "structured" strategy is required (partitioned matching / step-function oversight).

The Econometrica paper's insight that Brier score is the unique exception (constant G'', allowing smooth oversight) may have an analogue: there may exist specific LLM architectures or attention mechanisms where discrimination capacity does not degrade with |S| (e.g., models with explicit per-subscription attention heads). Identifying such architectures would be the Neural Router analogue of identifying the Brier score as the unique smooth-oversight-compatible scoring rule.

---

## 5. Architectural Implications

### 5.1 Cascade Architecture (Filter -> Focus -> Verify)

The discrimination capacity framework motivates a three-stage cascade:

| Stage | Method | |S| seen | Strength | Cost |
|---|---|---|---|---|
| **Filter** | SBERT cosine (pairwise, independent) | All |S| | No |S|-degradation | O(|S|) embeddings |
| **Focus** | LLM with <= D(model) candidates | Top-r from filter | Deep semantic reasoning | O(r) LLM tokens |
| **Verify** (optional) | LLM cross-cluster reconciliation | Matches from all clusters | Resolves boundary cases | O(k) LLM calls |

This exploits complementary strengths:
- SBERT is a pairwise method: accuracy is independent of |S| (each comparison is independent)
- LLM is a holistic reasoner: powerful but attention-limited

The cascade is an **LLM-as-verifier** architecture: SBERT proposes candidates, LLM verifies. The LLM's per-pair decision load drops from O(|S|) to O(kappa), completely bypassing the discrimination bottleneck.

### 5.2 Hierarchy-Aware Partitioning

When the subscription taxonomy is a tree (as with EUROVOC, IPTC, IoT device registries):

1. **Use the taxonomy directly** as the partitioning structure
2. Cut at the level where subtree size <= D(model)
3. For EUROVOC: cut at level-1 domains (21 groups of ~10 descriptors each)
4. Each LLM prompt sees <= D(model) semantically related subscriptions

This is zero-cost in the computational sense (no optimization needed; the taxonomy provides the partition) and is guaranteed to produce semantically coherent clusters (because the taxonomy reflects the domain's semantic structure).

### 5.3 Adaptive Discrimination Budget

Not all subscriptions are equally hard to discriminate:
- "Sports" vs "Politics" is easy (high inter-subscription distance)
- "Monetary economics" vs "Economic policy" is hard (low inter-subscription distance)

An adaptive system would:
1. Estimate pairwise discriminability from embedding distances or calibration runs
2. Place hard-to-distinguish subscriptions in the same cluster (so the LLM sees them side-by-side)
3. Allow easy-to-distinguish subscriptions to be in separate clusters (routing can handle them)

This transforms D(model) from a hard ceiling into an **allocatable budget**, analogous to how Trilogy 1's Proposition 3 transforms an intractable allocation into a tractable one by choosing the right encapsulation.

### 5.4 Encapsulation for Non-Hierarchical Subscriptions

When no natural taxonomy exists (e.g., free-form subscription descriptions), the encapsulation trick from Trilogy 1 still applies:

1. **Discover latent structure** via embedding clustering (existing k-means step)
2. **Encapsulate** each cluster into a meta-subscription (existing cover/merge step)
3. **Check if quotient graph is tree/SP** (new step: verify the inter-cluster dependency structure)
4. If yes: polynomial-time partitioning via tree DP
5. If no: use heuristics (spectral partitioning, label-tree decomposition) with approximation guarantees

The encapsulation step is exactly what cover/merge already does. The missing piece is using the resulting structure for discrimination-aware partitioning rather than just token compression.

---

## 6. What This Means for the Paper

### 6.1 Scope B: Reformulate the Cost Model

The discrimination capacity framework with the two-constraint model and two crossover points adds approximately 5 pages of theoretical content. It transforms the paper from "here is an architecture with a cost model" to "here is an architecture with a joint cost-accuracy model that characterizes when LLM matching is viable."

The key additions:
1. Discrimination capacity as a formal concept (Section 3)
2. The two-constraint optimization problem (Section 3)
3. |S|_cross crossover alongside W_cross (Section 3)
4. Tractability under tree-structured taxonomies (Section 3 or 4)
5. Connection to the encapsulation pattern from Trilogy 1 (Discussion)
6. Cascade architecture as a consequence (Discussion/Future Work)

### 6.2 What Belongs in This Paper vs Follow-Up

**In this paper (Scope B):**
- The two-constraint model and the discrimination capacity concept
- The two crossover points (W_cross, |S|_cross) with empirical calibration from D1-D3
- The observation that tree-structured taxonomies enable tractable partitioning
- The cascade architecture as a design consequence (described, not fully evaluated)
- Honest D2 results with the discrimination complexity explanation

**In a follow-up paper:**
- Full evaluation of the cascade architecture across |S| scales
- The polymatroid conjecture (Section 4.4): formal proof or disproof
- Discrimination-optimal partitioning algorithms with approximation guarantees
- Controlled |S|-sweep experiments to calibrate the accuracy scaling law
- The phase transition formalization (is |S|_cross truly discontinuous?)
- Joint optimization of routing + discrimination (Section 4.5 supermodularity)

### 6.3 Connection to the Programme

This analysis reveals that the Neural Router is not an isolated system paper but connects deeply to the trilogy's theoretical programme:

- **Trilogy 1** provides the encapsulation pattern: transform arbitrary subscription spaces into tractable structures by choosing the right abstraction level
- **TEAC** provides the supermodularity framework: routing and discrimination are complements, not independent subsystems
- **Phase Transition** provides the regime-boundary concept: there exists a sharp |S|_cross beyond which the optimal architecture changes qualitatively

These connections strengthen both the Neural Router paper (by grounding its design in the programme's theoretical framework) and the trilogy (by providing a concrete systems instantiation of the abstract tractability results).

---

## 7. Integration Plan

| Step | File | Section | Action | Net pages |
|------|------|---------|--------|-----------|
| 1 | Design.tex | 3.6 (Cost Model) | Add discrimination capacity constraint C2 alongside context constraint C1. Present the two-constraint optimization. Define D(model). | +0.75 |
| 2 | Design.tex | 3.6 (Cost Model) | Add |S|_cross derivation alongside W_cross. Present both crossover points. Worked example with D1 vs D2. | +0.5 |
| 3 | Design.tex | 3.6 or new 3.7 | Note that tree-structured taxonomies enable polynomial-time discrimination-optimal partitioning. State the tree-DP result. Reference Trilogy 1's encapsulation (Prop. 3) for the general case. | +0.5 |
| 4 | Results.tex | Ablation results | Report D2 results honestly. Present the |S|-dependent performance pattern across D1/D2/D3. | +0.25 |
| 5 | Discussion.tex | Analysis | New subsection "Discrimination Complexity". Present the framework, the two crossover points, the connection to the tau plateau, and the cascade architecture as a consequence. | +1.5 |
| 6 | Discussion.tex | Limitations | Acknowledge: cost model does not capture accuracy scaling; D2 reveals the gap; the polymatroid conjecture is open. | +0.25 |
| 7 | Discussion.tex | Future Work | Cascade architecture evaluation, controlled |S|-sweep, discrimination-optimal partitioning algorithms, phase transition formalization. | +0.25 |
| 8 | Discussion.tex | Related Work | Connect to XML classification (label trees), "lost in the middle" (attention saturation), concentration of measure. | +0.25 |
| 9 | bibtex/ | References | Add: Liu et al. 2024 (lost in the middle), Radovanovic et al. 2010 (hubness), Vershynin 2018 (high-dim probability), EUR-Lex XML references. Cite Trilogy 1 for encapsulation. | +0.0 |
| | | | **Total** | **+4.25 pages** |

### Priority Order

1. Steps 1-2: **Core theory** (the two-constraint model is the main contribution)
2. Step 4: **Mandatory** (honest reporting)
3. Step 5: **High value** (converts weakness to insight)
4. Step 3: **Novel contribution** (tractability under taxonomy structure)
5. Steps 6-9: **Supporting** (scope, context, references)

---

## 8. Open Questions

1. **Is the polymatroid conjecture true?** Does the set of feasible partitions under tree-structured taxonomies and uniform discrimination capacity form a polymatroid? If yes, greedy algorithms suffice.

2. **Is |S|_cross a true phase transition or a smooth crossover?** The three data points (D1, D2, D3) suggest a sharp boundary, but document length and taxonomy granularity are confounded with |S|. A controlled experiment varying |S| while holding other factors constant would resolve this.

3. **Does the supermodularity of routing + discrimination hold formally?** The TEAC analogy is suggestive but the proof requires formalizing the Neural Router's two-level structure as a governance problem.

4. **What is D(model) for specific LLMs?** The empirical estimates (D(Haiku) ~ 15-20, D(Sonnet) ~ 40-50) are crude. A systematic measurement across model families and sizes would calibrate the framework.

5. **Can the cascade architecture recover D2 performance?** If SBERT proposes top-20 candidates per event and the LLM verifies each, does F1 on D2 approach the D1 level? This is the key empirical validation of the framework.
