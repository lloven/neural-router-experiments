---
title: "Publication Strategy: Neural Router and Discrimination Complexity"
date: 2026-04-06
source: research notes d2-eurlex-underperformance.md, tau-sensitivity-plateau.md, discrimination-complexity-framework.md, polymatroid-applicability-and-integrator-connection.md
status: draft
tags: [publication-strategy]
---

# Publication Strategy: Neural Router and Discrimination Complexity

## Guiding Principle

Each paper has one identity and one central question. No paper partially answers another paper's question. A reader of Paper N should never think "this would have been better in Paper N-1" or "I need to read Paper N+1 to understand this." Cross-references are forward pointers, not load-bearing dependencies.

---

## Paper 1: Neural Router (FGCS Revision)

**Identity:** A systems paper that introduces LLM-driven semantic matching for pub/sub and derives a cost model that tells practitioners when to use it.

**Central question:** How do you build an LLM-based content matching engine, and when does it outperform traditional methods?

**Status:** Under revision at Future Generation Computer Systems (Elsevier).

### Contributions (revised)

| # | Contribution | What changes in revision |
|---|---|---|
| C1 | Neural Router architecture (three algorithms, composable pipeline) | No change |
| C2 | Cost model with analytically derived crossover W_cross | **Extend:** add discrimination constraint. The cost model becomes a cost-accuracy model with two crossover points (W_cross for cost, \|S\|_cross for accuracy). The worked example in Table 3 gets a companion showing the accuracy crossover. |
| C3 | QoE-based heterogeneous backend assignment | No change |
| C4 | Empirical evaluation (3 datasets, 3 backends, 7 configs, 7 baselines) | **Correct:** report D2 honestly. Present the \|S\|-dependent performance pattern. The current abstract's "macro-F1 of 0.69 on CardiffNLP" remains valid. |
| C5 | Cross-domain validation (social media, legal, IoT) | **Reframe:** D2 is now a predicted boundary case (above \|S\|_cross), not an anomaly. D3 (IoT abductive reasoning) remains the qualitative highlight. |

### What goes IN the revision

- **Two-constraint model** (Section 3.6): context capacity W and discrimination capacity D as distinct resources. Both crossover points derived. This is the paper's theoretical upgrade: from "when is compression cost-effective?" to "when is LLM matching viable at all?"
- **Honest D2 results** (Section 5): the LLM underperforms TF-IDF/SBERT at |S|=201. The 2.3x claim is corrected. The discrimination capacity concept explains this cleanly.
- **τ plateau explanation** (Section 5.3): concentration of measure in d=384 embedding space. Default recommendation updated from τ=0.3 to τ~0.1. One paragraph, not a full treatment.
- **Discussion** (~1.5 pages): discrimination capacity as a concept, the two crossover points, the observation that tree-structured taxonomies enable efficient partitioning. Forward pointers to cascade architecture and regime classification as future work.

### What stays OUT

- Full discrimination complexity theory (scaling laws, formal characterization) → Paper 2
- Regime classification (I/II/III) with polymatroid boundaries → Paper 2
- Cascade architecture design and evaluation → Paper 2
- Integrator/encapsulation connection to the trilogy → Paper 2 or Neural Pub-Sub
- Controlled |S|-sweep experiments → Paper 2

### Novelty claim

> We introduce LLM-driven semantic matching for content-based pub/sub and derive a joint cost-accuracy model that identifies two deployment crossover points: a context-window crossover (below which compression reduces cost) and a discrimination-capacity crossover (above which LLM matching degrades below embedding baselines). The cost model gives practitioners a principled decision tool for architecture selection.

### Estimated revision scope

~4 pages of changes within existing structure. Net growth: ~2-3 pages (some existing content is rewritten, not just added). No new sections; the two-constraint model extends Section 3.6, D2 results are corrected in Section 5, Discussion gains a subsection.

---

## Paper 2: Discrimination Complexity in LLM-Based Content Matching

**Identity:** A theory-and-systems paper that characterizes the fundamental limits of LLM-based classification as a function of label-space structure, and derives the optimal matching architecture for each regime.

**Central question:** When can an LLM replace pairwise matching methods, when can't it, and what architecture should you use in each case?

**Status:** New submission. Venue TBD.

### Contributions

| # | Contribution | Novelty |
|---|---|---|
| C1 | **Discrimination complexity framework.** Formal characterization of LLM accuracy degradation as a function of |S|, semantic density H(S), and document length. Introduces D(model) as a measurable system parameter alongside W. | New concept. No prior work models LLM discrimination capacity as a resource constraint separate from context window. Connects to "lost in the middle" (Liu et al. 2024) and extreme multi-label classification (XML) literature, but provides a unified framework rather than ad-hoc observations. |
| C2 | **Regime classification** with polymatroid boundaries. Three regimes (tree+coherent, tree+cross-branch, unstructured) with precise structural conditions for tractability. Revised polymatroid proposition with four explicit conditions. | New result. Maps the NP-hard/polynomial boundary from the combinatorial optimization literature onto the pub/sub matching problem. The four conditions (tree taxonomy, label coherence, uniform capacity, unit weight) are stated precisely and each failure mode is characterized. |
| C3 | **Cascade architecture** (Filter → Focus → Verify). SBERT proposes candidates, LLM verifies. Transforms the LLM's role from classifier (O(\|S\|) simultaneous decisions) to verifier (O(κ) independent binary decisions). | Architectural contribution grounded in the theory: the cascade is the optimal response to discrimination capacity limits, not an ad-hoc engineering choice. The regime classification determines which cascade variant to use. |
| C4 | **Comprehensive evaluation** across |S| scales. Controlled |S|-sweep on D1 (varying |S| from 5 to 200 while holding other factors constant). Cascade evaluation on D2. Cross-regime comparison. | Empirical validation of the framework. The |S|-sweep isolates the discrimination capacity effect from confounds (document length, taxonomy granularity). |

### What goes IN

- Full discrimination complexity theory: formal definitions, scaling law hypothesis, D(model) calibration methodology
- Regime classification with all boundary conditions and failure mode analysis
- Polymatroid proposition (Regime I) with proof sketch (tree DP argument)
- Cascade architecture: design, implementation, complexity analysis
- Controlled experiments: |S|-sweep, cascade vs single-prompt, cross-regime evaluation
- Connections to XML literature, "lost in the middle," concentration of measure (as related work, not reproduced)

### What stays OUT

- Neural Router architecture details (cites Paper 1)
- Cost model derivation (cites Paper 1; uses the W_cross result but does not re-derive it)
- Federation / distributed architecture (cites Neural Pub-Sub paper)
- Trilogy proofs (cites Trilogy 1 for the encapsulation pattern and polymatroid conditions; does not reproduce)
- Mechanism design / incentive compatibility (trilogy's domain, not this paper's)

### Dependency on Paper 1

Paper 2 cites Paper 1 for:
- The Neural Router architecture (algorithms, pipeline)
- The cost model and W_cross derivation
- The empirical finding that D2 underperforms (Paper 1 reports it; Paper 2 explains it)

Paper 2 does NOT depend on Paper 1 for its core theory: the discrimination complexity framework stands alone as a characterization of LLM-based classification limits. A reader who has never seen the Neural Router can still understand Paper 2 if they accept that "LLMs are used for multi-label content matching."

### Novelty claim

> We introduce the discrimination complexity framework, which characterizes LLM-based content matching accuracy as a function of label-space structure. We identify three regimes based on whether the subscription taxonomy is tree-structured and whether matching is label-coherent, prove tractability in Regime I via a polymatroid argument, and design a cascade architecture (embedding filter → LLM verifier) that bypasses the discrimination bottleneck in Regimes II-III. Controlled experiments across |S| from 5 to 200 validate the framework's predictions and demonstrate that the cascade recovers LLM performance in the high-|S| regime where single-prompt matching fails.

### Venue considerations

The paper combines theory (complexity characterization, regime boundaries, polymatroid result) with systems (cascade architecture, comprehensive evaluation). Candidate venues:

| Venue | Fit | Concern |
|---|---|---|
| **IEEE TPDS** | Systems + theory, distributed systems audience | May need stronger distributed-systems angle |
| **ACM TIST** | Intelligent systems, AI+systems crossover | Good fit for the regime classification + cascade |
| **VLDB Journal** | Data management, query optimization | The subscription matching is analogous to query routing |
| **IEEE TKDE** | Knowledge + data engineering | Multi-label classification + system design |
| **FGCS** (same as Paper 1) | Computing continuum, same audience | Risk: two papers in same journal may be seen as salami-slicing. Mitigated by completely different contributions (architecture vs theory). |

---

## Companion: Neural Pub-Sub (IEEE TNSE, in preparation)

**Identity:** A distributed systems paper that federates Neural Router instances across administrative domains using market-based allocation.

**Central question:** How do you coordinate semantic matching across a multi-domain pub/sub federation?

### What it gains from this work

One paragraph in Discussion (~0.3 pages):

> The two-level polymatroid argument. At the federation level (inter-broker), polymatroid structure is guaranteed by designing the broker topology as tree or SP (the architect's choice). At the subscription level (intra-broker), polymatroid structure depends on the subscription space's regime (I/II/III per [Paper 2]). The integrator encapsulation of [Trilogy 1, Proposition 3] bridges the two levels: each broker hides its internal matching complexity and exposes only a scalar capacity summary to the federation, regardless of the internal regime.

This is a forward reference to Paper 2, not a reproduction. The Neural Pub-Sub paper's own contributions (federation protocol, market-based allocation, O-RAN evaluation) are unaffected.

---

## Separation Matrix

To verify no pollution between papers:

| Topic | Paper 1 (Router, FGCS) | Paper 2 (Discrim. Complexity) | Neural Pub-Sub (TNSE) |
|---|---|---|---|
| Neural Router architecture | **Defines** | Cites | Cites |
| Cost model (W, invocations) | **Derives** | Cites | Cites |
| W_cross (context crossover) | **Derives** | Cites | -- |
| \|S\|_cross (discrimination crossover) | **Introduces** (empirical, ~1 page) | **Characterizes** (full theory) | -- |
| D2 underperformance | **Reports honestly** | **Explains** (theory + controlled expt) | -- |
| τ plateau | **Reports + explains briefly** | Cites Paper 1 | -- |
| Discrimination capacity D(model) | **Introduces** as concept | **Formalizes** (scaling law, calibration) | -- |
| Regime classification (I/II/III) | -- (future work pointer) | **Defines and proves** | References |
| Polymatroid boundaries | -- | **Proves** (Regime I) | Uses (Level 2) |
| Cascade architecture | -- (future work pointer) | **Designs and evaluates** | -- |
| Controlled \|S\|-sweep | -- | **Conducts** | -- |
| Integrator connection | -- | **Establishes** (Section 2) | **Uses** (Section 4) |
| Federation protocol | -- | -- | **Defines** |
| Market-based allocation | -- | -- | **Defines** |
| Two-level polymatroid | -- | States (Section 6) | **Applies** (Discussion) |

**Verification:** Each cell is either "Defines/Derives/Proves" (owns it), "Cites/References" (uses it), or "--" (not mentioned). No topic appears as "Defines" in two papers. No paper partially answers another paper's central question.

---

## Timeline

| Paper | Status | Next Step | Target |
|---|---|---|---|
| **Paper 1** (Router, FGCS) | Under revision | Implement Scope B revision (~4 pages of changes) | Submit revision within 4-6 weeks |
| **Paper 2** (Discrim. Complexity) | Research notes complete | Run controlled |S|-sweep experiments; implement cascade; write paper | Submit 3-4 months after Paper 1 revision |
| **Neural Pub-Sub** (TNSE) | In preparation | Add two-level polymatroid paragraph when Paper 2 is drafted | Independent timeline |

### Dependency chain

```
Paper 1 revision  ──→  Paper 2 experiments  ──→  Paper 2 submission
                                                      │
Neural Pub-Sub  ←── two-level paragraph from Paper 2 ─┘
```

Paper 1 must be submitted first (Paper 2 cites it). Paper 2's experiments can begin in parallel with Paper 1's revision (they require no manuscript changes, only code). Neural Pub-Sub is independent but gains a paragraph from Paper 2's results.

---

## Risk Assessment

| Risk | Mitigation |
|---|---|
| **Reviewer sees Paper 1 as incomplete** ("you identified the problem but didn't solve it") | Paper 1's contribution is the two-constraint model and the empirical finding, not the solution. The Discussion clearly scopes: "characterizing the architectural response is future work." The cost model with two crossover points is a complete, self-contained contribution. |
| **Reviewer sees Paper 2 as incremental over Paper 1** ("this is just the follow-up experiment") | Paper 2's central contribution is the discrimination complexity theory and regime classification, not "more experiments for the Neural Router." The theory applies to ANY LLM-based classification system, not just pub/sub. The cascade architecture is a general design pattern. |
| **Salami-slicing accusation** (two papers from one study) | The papers answer different questions: Paper 1 asks "how to build it and when does it work?"; Paper 2 asks "why does it fail and what are the fundamental limits?" The contributions are non-overlapping (see separation matrix). Different venues further mitigate this. |
| **D2 results weaken Paper 1** ("your system doesn't work on 1 of 3 datasets") | The two-constraint model PREDICTS that D2 is above |S|_cross. It's not a failure; it's a validated boundary. The abstract already says "the cost model gives practitioners a principled tool for choosing." The revision extends this to accuracy, not just cost. |
| **Polymatroid result too narrow for Paper 2** ("Regime I is the easy case") | The regime classification IS the contribution, not just Regime I. The paper characterizes all three regimes and provides architectures for each. The polymatroid result is one piece (the tractable case), but the cascade (the intractable case) is equally important. |
