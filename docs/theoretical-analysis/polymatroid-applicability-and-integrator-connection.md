---
title: "Theoretical Analysis: Polymatroid Applicability, Integrator Connection, and Practical Boundaries"
date: 2026-04-06
source: trilogy Papers 1-3, Governance Duality (TEAC), Neural Pub-Sub (TNSE), experimental results
manuscript: Manuscripts/Neural Router (Elsevier FGCS)/
related:
  - docs/theoretical-analysis/discrimination-complexity-framework.md
  - Manuscripts/Neural Pub-Sub (IEEE TNSE)/
  - Manuscripts/Trilogy Paper 1 (IEEE TSC)/
  - Manuscripts/Governance Duality (TEAC)/
status: draft
tags: [theoretical-analysis, polymatroid, integrator, applicability, pub-sub]
---

# Polymatroid Applicability, Integrator Connection, and Practical Boundaries

## 1. The Integrator Connection

### 1.1 Neural Pub-Sub Broker as Trilogy Integrator

The Neural Pub-Sub paper (IEEE TNSE) explicitly positions each domain broker as a trilogy integrator (Section 4.2.4):

> "Each domain's broker acts as an integrator: it encapsulates the domain's internal service-dependency structure into a composite service with scalar capacity."

The architecture maps precisely:

| Trilogy Concept | Neural Pub-Sub Implementation |
|---|---|
| Integrator | Domain broker |
| Internal sub-DAG | Neural Router (matching pipeline: clustering, C&M, LLM matching) |
| Scalar composite capacity | Subscription summary (centroid embedding, cluster radius, available capacity) |
| Quotient graph | Federation overlay (broker-to-broker topology) |
| Walrasian price signal | Per-stage clearing prices exchanged between brokers |
| Agent | Publisher or subscriber |

### 1.2 Neural Router as Internal Matching Component

The Neural Router is the matching engine WITHIN a single broker (integrator). It handles the complexity that the integrator encapsulates:

- Subscription clustering (k-means partitioning)
- Cover/merge compression (subscription encapsulation within clusters)
- LLM-based semantic matching (the actual content evaluation)

The broker exposes only aggregate summaries to federation peers, hiding the internal matching complexity. This is Proposition 3's encapsulation in action.

### 1.3 What Carries Over Directly from the Trilogy

Because the Neural Pub-Sub broker IS an integrator, several trilogy results apply immediately:

**At the federation level (inter-broker):**
- If the broker federation topology is tree or SP, the agent-facing allocation is polymatroidal (Proposition 1)
- Walrasian price convergence holds (Proposition 2 of Trilogy 1)
- VCG mechanisms are DSIC under domain separation (Proposition 4 of Trilogy 2)

**At the broker level (intra-broker):**
- Cover/merge IS subscription encapsulation (compressing multiple subscriptions into a broader description)
- The quotient subscription graph (after C&M) ideally has simpler structure than the original

The open question is whether polymatroidal structure also applies to the **subscription partitioning problem** within a single broker. This is the polymatroid conjecture from the previous note.

---

## 2. The Polymatroid Conjecture, Stated Precisely

### 2.1 What Polymatroid Structure Requires (from Trilogy 1)

Proposition 1 of Trilogy 1 establishes three necessary conditions:

1. **Tree or series-parallel graph structure.** The constraint sets {L_v} (sets of leaves reachable from each internal node) must form a **laminar family**: for any two sets L_v, L_w, either L_v subset L_w, or L_w subset L_v, or L_v intersection L_w = empty. This holds automatically for trees and by inductive construction for SP graphs.

2. **Submodular capacity function.** The rank function f(S) = min_A sum_{v in A} C_v (minimized over antichains A in the laminar family covering S) must be submodular, monotone, and normalized. This follows from the laminar family property.

3. **Gross substitutes valuations.** For the mechanism design results (Walrasian equilibrium, DSIC), agent valuations must satisfy gross substitutes. This requires: unit demand per task, additive separability across tasks, and fixed slice attributes within each epoch.

### 2.2 Mapping to Subscription Partitioning

For the Neural Router's discrimination-constrained partitioning:

| Trilogy 1 Element | Neural Router Analogue |
|---|---|
| Resource DAG nodes | Subscription taxonomy nodes |
| Leaf services (allocated to agents) | Individual subscriptions (assigned to clusters) |
| Node capacity C_v | Discrimination capacity D(model) per cluster |
| Throughput tokens | Subscription-to-cluster assignments |
| Allocation feasibility | Partition feasibility (capacity + coverage) |

The conjecture is: when the subscription taxonomy is a tree and D(model) is uniform, the set of feasible partitions forms a polymatroid.

### 2.3 The Coverage Constraint Is the Problem

The capacity constraint alone ({|c_i| <= D} for all clusters) trivially defines a partition matroid. The difficulty is the **coverage constraint**: for each event e, its true-match subscriptions should be concentrated in few clusters (to bound recall loss from routing).

This coverage constraint creates **dependencies** between which subscriptions can be co-located. Whether these dependencies preserve or break polymatroid structure depends on the structure of the ground-truth matching.

---

## 3. When Polymatroid Structure Holds

### 3.1 Sufficient Condition: Label-Coherent Matching on Tree Taxonomies

**Definition (Label-coherent matching):** A matching instance is label-coherent with respect to a taxonomy tree T if, for every event e, the set of true-match subscriptions M(e) = {s : mu(e,s) = 1} forms a **connected subtree** of T (or a union of at most r small subtrees, for some fixed r).

**Claim:** If the subscription taxonomy is a rooted tree T, matching is label-coherent (r = 1), and the capacity constraint is |c_i| <= D(model), then:
- Any partition into connected subtrees of size <= D automatically satisfies coverage
- The feasible partitions are characterized by the tree structure
- The capacity function inherits submodularity from the tree's laminar family
- Polynomial-time tree DP solves the optimal partition

**Why:** Label-coherent matching means that an event's true matches are all in the same branch of the taxonomy. A partition that respects branches (connected subtrees) automatically co-locates all true matches. The coverage constraint is redundant, and only the capacity constraint remains, which has polymatroid structure on trees.

### 3.2 Practical Use Cases Where This Holds

**Single-label classification with tree taxonomy:**

| Use Case | Taxonomy | Label Coherence | Polymatroid? |
|---|---|---|---|
| **E-commerce product matching** | Product category tree (e.g., Amazon Browse Node hierarchy) | High: queries match items within a category branch | Yes |
| **Simple IoT sensor routing** | Location hierarchy (building -> floor -> room -> sensor) | High: sensor events are location-specific | Yes |
| **DNS-like topic routing** | Hierarchical topic namespace (e.g., MQTT topic tree: home/kitchen/temperature) | By design: topics form a tree, matching is prefix-based | Yes |
| **Organizational routing** | Organizational hierarchy (division -> department -> team) | High: internal communications follow reporting structure | Yes |
| **Single-category news** | Flat or shallow taxonomy (e.g., "Sports", "Politics", "Tech") with single-label assignment | High when categories are disjoint | Yes |

In these cases, the subscription space has natural tree structure, events match within a single branch, and the polymatroid result enables polynomial-time optimal partitioning. The Neural Router's clustering should exploit the taxonomy directly rather than using embedding-based k-means.

---

## 4. When Polymatroid Structure Fails

### 4.1 Failure Mode 1: Cross-Branch Matching (Laminar Family Breaks)

**When events match subscriptions from multiple branches of the taxonomy, the coverage constraint creates cross-cutting dependencies that break laminarity.**

This is the Neural Router analogue of Trilogy 1's failure on general DAGs: "cross-cutting dependencies typically destroy the laminar property required by Proposition 1."

| Use Case | Taxonomy | Cross-Branch Matching? | Polymatroid? |
|---|---|---|---|
| **EUR-Lex / EUROVOC (D2)** | 3-level hierarchy (domains -> micro-thesauri -> descriptors) | Yes: legal documents are labeled with descriptors from multiple domains ("environmental tax regulation" -> taxation + environment + legislation) | **No** |
| **Multi-label news** | Topic hierarchy | Yes: "sports economics" -> sports + economics | **No** |
| **Healthcare / ICD coding** | ICD-10 tree | Yes: comorbidities span branches (diabetes + cardiovascular + renal) | **No** |
| **Academic paper categorization** | Field hierarchy (CS -> AI -> NLP, CS -> Systems -> Distributed) | Yes: interdisciplinary papers span branches | **No** |
| **Social media topic detection** | Topic hierarchy | Yes: tweets commonly span multiple topics | **No** |

**This is precisely the D2 (EUR-Lex) scenario.** EUROVOC descriptors have tree structure, but the ground-truth matching is NOT label-coherent: legal documents routinely receive labels from multiple EUROVOC domains. A partition that respects the tree structure will split an event's true matches across clusters, violating the coverage constraint.

The implication: **polymatroid structure does not help with EUR-Lex and similar multi-label taxonomic use cases.** For these, the partitioning problem remains NP-hard, and the cascade architecture (SBERT filter -> LLM verifier) is the appropriate solution.

### 4.2 Failure Mode 2: No Natural Taxonomy (No Tree to Exploit)

When subscriptions are free-form natural language descriptions with no pre-existing taxonomy, there is no tree structure to exploit.

| Use Case | Taxonomy | Polymatroid? |
|---|---|---|
| **Ad-hoc pub/sub** (arbitrary subscriber interests) | None | **No** |
| **Email filtering** (user-defined rules in natural language) | None | **No** |
| **Content recommendation** (free-text user profiles) | None | **No** |
| **Alert systems** (natural language alert criteria) | None | **No** |

For these cases, the embedding-based k-means clustering is the right approach (it discovers latent structure), but there is no guarantee of polymatroid structure.

### 4.3 Failure Mode 3: Complementarities in Matching (GS Breaks)

In Trilogy 1, gross substitutes requires additive separability: the value of matching event e to subscription s_i is independent of whether e also matches s_j.

For the Neural Router, matching decisions are typically independent (whether e matches "sports" doesn't affect whether e matches "politics"). However, there are cases where matching has complementarities:

| Case | Complementarity | GS Holds? |
|---|---|---|
| **Exclusive categories** ("fiction" vs "non-fiction") | Anti-complementarity: matching one excludes the other | No (but this is easily handled by constraints) |
| **Conditional subscriptions** ("notify if sports AND finance") | Boolean conjunction creates complementarity | No |
| **Priority-based routing** (only route to highest-priority match) | Assignment depends on other matches | No |

In practice, most pub/sub matching is independent (each subscription is evaluated separately), so GS holds in the common case. But conditional or compound subscriptions would break it.

### 4.4 Failure Mode 4: Non-Scalar Capacity (Proposition 3 Condition ii Breaks)

Proposition 3's encapsulation requires each integrator to export a **single, homogeneous slice type** with scalar capacity. In the Neural Router context, this means the discrimination capacity D(model) must be a single number, independent of which subscriptions are in the cluster.

This assumption is approximately correct (the LLM can handle D subscriptions regardless of their content) but may fail when:
- Subscriptions vary dramatically in length (short vs long descriptions consume different attention)
- Some subscription pairs are harder to discriminate than others (the effective D depends on semantic density within the cluster)

In these cases, D(model) is not scalar but **content-dependent**, and the polymatroid argument does not directly apply.

---

## 5. The Boundary Map

### 5.1 Three Regimes

The polymatroid applicability analysis reveals three regimes for the Neural Router's subscription partitioning:

| Regime | Subscription Structure | Matching Pattern | Polymatroid? | Optimal Architecture |
|---|---|---|---|---|
| **I: Tractable** | Tree taxonomy | Label-coherent (single-branch) | **Yes** | Taxonomy-aware tree partitioning (polynomial DP) |
| **II: Partially structured** | Tree taxonomy | Multi-label (cross-branch) | **No** | Encapsulation + cascade (SBERT filter -> LLM verifier) |
| **III: Unstructured** | No taxonomy (free-form) | Arbitrary | **No** | Embedding clustering + cascade |

### 5.2 D1, D2, D3 Regime Classification

| Dataset | |S| | Taxonomy | Matching Pattern | Regime | Partitioning Strategy |
|---|---|---|---|---|---|
| **D1 (CardiffNLP)** | 19 | Flat (coarse topics) | Mostly single-label | I (trivially: |S| < D) | No partitioning needed (A0 works) |
| **D2 (EUR-Lex)** | 201 | Tree (EUROVOC) | Multi-label, cross-branch | **II** | Cascade needed; taxonomy partitioning alone insufficient |
| **D3 (CASAS IoT)** | 19 | Flat (activity types) | Single-label | I (trivially: |S| < D) | No partitioning needed |

D2 falls in Regime II: the taxonomy exists but matching is cross-branch. This is precisely why the polymatroid conjecture is WRONG for D2, and why a more nuanced approach is needed.

### 5.3 Implications for the Broader Pub/Sub Programme

The three regimes map onto three tiers of the computing continuum's pub/sub workloads:

**Regime I (Tractable, tree-structured):**
- IoT device registries, MQTT topic trees, organizational routing
- Subscription spaces designed with hierarchical structure
- The Neural Router can exploit the taxonomy directly; the trilogy's polynomial-time results apply
- Typical deployment: edge/fog tier with engineered topic namespaces

**Regime II (Partially structured, multi-label):**
- Regulatory/compliance monitoring (EUROVOC, ICD), academic paper matching, multi-topic news
- Subscription spaces have natural hierarchies but events span multiple branches
- Cascade architecture (SBERT -> LLM) is necessary; taxonomy provides useful but insufficient structure
- Typical deployment: governance/analytics tier where accuracy matters more than cost

**Regime III (Unstructured, arbitrary):**
- Ad-hoc content matching, email filtering, recommendation systems
- No pre-existing taxonomy; embedding space is the only structure available
- Full cascade or LLM-as-verifier architecture required
- Typical deployment: application tier with diverse, user-generated subscriptions

---

## 6. Two-Level Polymatroid Structure

### 6.1 The Multi-Level Argument

The Neural Pub-Sub architecture has two levels where polymatroid structure can apply independently:

**Level 1 (Within broker): Subscription partitioning**
- Holds only in Regime I (tree taxonomy, label-coherent matching)
- Enables polynomial-time optimal cluster assignment

**Level 2 (Federation): Broker-to-broker allocation**
- Controlled by the system architect (federation topology is a design choice)
- The Pub-Sub paper designs the federation as a tree (hierarchical federation for 50+ domains) or star (small deployments)
- Polymatroid structure holds by construction at this level

The key insight: **even when Level 1 (subscription partitioning) lacks polymatroid structure, Level 2 (federation) can be designed to have it.** The integrator encapsulation of Proposition 3 applies: each broker hides its internal matching complexity (which may be NP-hard) and exposes only a scalar capacity summary to the federation.

### 6.2 What This Means Architecturally

The trilogy's encapsulation theorem (Proposition 3) tells us:
- The federation-level allocation is tractable if the broker topology is tree/SP
- The within-broker matching can be arbitrarily complex
- The broker's scalar capacity summary (subscription summary in the Pub-Sub paper) is the encapsulation mechanism

For the Neural Router specifically:
- At the federation level: the system architect ensures polymatroid structure by choosing tree/SP broker topology
- At the subscription level: the architecture must adapt to the regime (I, II, or III)

This two-level separation is the correct way to think about the problem. The polymatroid conjecture should NOT be stated as a universal claim about subscription partitioning. Instead:

> The Neural Pub-Sub federation inherits polymatroid structure from the trilogy's encapsulation theorem when the broker topology is tree or SP (by design). Within each broker, subscription partitioning is polymatroidal only when the subscription taxonomy is a tree AND matching is label-coherent (Regime I). For multi-label or unstructured subscription spaces (Regimes II-III), the internal matching problem does not have polymatroid structure, and alternative approaches (cascade architecture, approximate partitioning) are required.

---

## 7. Revised Conjecture

The original polymatroid conjecture from the previous note was too broad. The correct statement:

### Original (too strong):
> When the subscription taxonomy is a tree and D is uniform, the set of feasible partitions forms a polymatroid.

### Revised (precise):
> **Proposition (Polymatroidal Subscription Partitioning).** Let the subscription taxonomy T be a rooted tree. Let the matching instance be label-coherent with parameter r = 1 (every event's true matches form a connected subtree of T). Let the discrimination capacity D be uniform across clusters.
>
> Then the set of feasible partitions (satisfying both capacity and coverage) has polymatroid structure: the capacity function induced by the tree's laminar family is submodular, monotone, and normalized, and optimal partitioning is solvable in polynomial time via tree DP.

### Conditions (explicit):
1. **Taxonomy is a tree** (not DAG, not flat)
2. **Matching is label-coherent** (r = 1: events match within a single branch)
3. **Capacity is uniform** (D is the same for all clusters)
4. **Subscriptions have unit weight** (each subscription consumes one unit of cluster capacity)

### What fails without each condition:
1. Without tree structure: no laminar family, submodularity not guaranteed
2. Without label coherence: coverage constraint creates cross-cutting dependencies, breaks laminarity
3. Without uniform capacity: the capacity function becomes asymmetric, may lose submodularity
4. Without unit weight: subscriptions of different sizes create a bin-packing subproblem (NP-hard)

---

## 8. Practical Implications for the Neural Router Paper

### 8.1 What to Include in the Paper

1. **The integrator positioning.** State explicitly that the Neural Router is the matching component within a Neural Pub-Sub broker, which acts as a trilogy integrator. This connects the paper to the broader programme.

2. **The three-regime classification.** Present the boundary map (Regime I/II/III) as a practical guide for when taxonomy-aware partitioning is applicable vs when cascade architectures are needed.

3. **The precise polymatroid conditions.** State the revised conjecture with all four conditions. Be explicit about what fails for D2 (condition 2: label coherence) and for free-form subscriptions (condition 1: no tree).

4. **The two-level argument.** Note that polymatroid structure at the federation level (Level 2) is guaranteed by broker topology design, independent of the subscription-level (Level 1) structure.

### 8.2 What to Defer to Follow-Up

1. Full proof of the revised polymatroid proposition (Regime I)
2. Approximation guarantees for Regime II partitioning
3. Empirical validation of the cascade architecture across regimes
4. The interaction between taxonomy granularity and discrimination capacity

### 8.3 D2 Reframing

The D2 results should now be understood as:

> EUR-Lex falls in Regime II: the EUROVOC taxonomy provides useful hierarchical structure, but multi-label matching (documents labeled with descriptors from multiple domains) violates the label-coherence condition required for polymatroidal partitioning. The Neural Router's structure-blind k-means clustering cannot exploit the taxonomy, and the LLM's discrimination capacity is exceeded by |S| = 201. The appropriate architecture for Regime II workloads is a cascade (embedding filter -> focused LLM verification), with the taxonomy providing coarse pre-grouping but not the fine-grained discrimination-optimal partition.

This is honest, precise, and connects the empirical finding to the theoretical framework without over-claiming.

---

## 9. Integration Plan

| Step | File | Section | Action | Net pages |
|------|------|---------|--------|-----------|
| 1 | Design.tex or Discussion.tex | System Model or Discussion | Add 1-2 paragraphs positioning Neural Router as matching component within Neural Pub-Sub broker (integrator). Reference Trilogy 1 Proposition 3. | +0.3 |
| 2 | Discussion.tex | Discrimination Complexity subsection | Add regime classification (I/II/III) with boundary conditions. Present the revised polymatroid proposition with all four conditions. Note which D1/D2/D3 datasets fall in which regime. | +0.5 |
| 3 | Discussion.tex | Discrimination Complexity subsection | Add the two-level argument: Level 1 (subscription) may lack polymatroid structure, but Level 2 (federation) has it by design. The encapsulation theorem bridges the levels. | +0.3 |
| 4 | Discussion.tex | Future Work | Note: formal proof of revised proposition, Regime II approximation algorithms, cascade architecture evaluation. | +0.2 |
| | | | **Total** | **+1.3 pages** |

---

## 10. Summary

The polymatroid conjecture from the previous note was too broad. Careful analysis, grounded in the precise conditions from Trilogy 1, reveals:

1. **Polymatroid structure holds for Regime I** (tree taxonomy + label-coherent matching + uniform capacity + unit weights). This covers IoT registries, MQTT topics, organizational routing, and single-label classification with tree taxonomies.

2. **Polymatroid structure fails for Regime II** (tree taxonomy but cross-branch matching). This covers EUR-Lex, multi-label news, healthcare coding, and any multi-label classification where ground truth spans multiple taxonomy branches. D2 is squarely in this regime.

3. **Polymatroid structure fails for Regime III** (no taxonomy). This covers ad-hoc pub/sub, email filtering, recommendation, and any free-form subscription space.

4. **At the federation level** (inter-broker), polymatroid structure is guaranteed by design (the architect chooses tree/SP broker topology). This is independent of the subscription-level regime.

5. **The Neural Router paper should present this regime map** rather than a universal polymatroid claim, and should connect it to the trilogy's encapsulation framework. The D2 failure is explained by Regime II membership (cross-branch matching breaks label coherence), and the appropriate architectural response is the cascade.
