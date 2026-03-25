"""Neural Router: core matching engine.

Implements the three algorithms from the paper (Section 3):

  Algorithm 1 -- OptimizeSubscriptions (offline phase, Section 3.2):
      Embed subscriptions, cluster via k-means, and compress each cluster
      with CoverAndMerge. Produces an optimised subscription index.

  Algorithm 2 -- MatchEvents (online phase, Section 3.3):
      Embed incoming events, assign them to clusters via cosine pre-filter
      (threshold tau), and invoke the LLM for semantic matching within
      each cluster.

  Algorithm 3 -- CoverAndMerge (Section 3.2):
      LLM-assisted iterative compression that identifies subsumption
      (covering) pairs and mergeable subscriptions to reduce prompt size.

Key classes:
  RouterConfig  -- full configuration (ablation toggles + hyperparameters)
  NeuralRouter  -- stateful engine that runs the offline/online pipeline
  Cluster       -- a group of subscriptions with shared centroid
  MatchResult   -- per-event list of matched subscription IDs
  RouterStats   -- timing, token, and invocation counters
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from .data import Dataset, Event, Subscription
from .llm import DryRunLLMClient, LLMClient, LLMResponse
from .embeddings import EmbeddingModel, OllamaEmbeddings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Cluster:
    """A subscription cluster produced by k-means during the offline phase.

    Each cluster holds its original subscriptions, an optional compressed
    version (after CoverAndMerge), and a centroid embedding used for the
    cosine pre-filter during event assignment.
    """
    id: int
    subscriptions: list[Subscription]          # original subscriptions in this cluster
    compressed: Optional[list[Subscription]] = None  # after CoverAndMerge (Algorithm 3)
    centroid: Optional[np.ndarray] = None       # mean embedding of member subscriptions
    llm: Optional[str] = None                   # LLM model assigned to this cluster
    queue: list[Event] = field(default_factory=list)  # events routed here during matching

    @property
    def active_subscriptions(self) -> list[Subscription]:
        """Return compressed subscriptions if available, else originals."""
        return self.compressed if self.compressed is not None else self.subscriptions

    @property
    def compression_ratio(self) -> float:
        """Per-cluster compression ratio: rho_c = |S'_c| / |S_c| (Eq. 3)."""
        if self.compressed is None:
            return 1.0
        return len(self.compressed) / max(len(self.subscriptions), 1)


@dataclass
class MatchResult:
    """Result of matching a single event against the subscription set.

    Contains the ordered list of subscription IDs the LLM deemed relevant,
    optionally accompanied by relevance scores.
    """
    event_id: str
    matched_subscription_ids: list[str]
    scores: Optional[list[float]] = None  # relevance scores if available


@dataclass
class RouterStats:
    """Timing, token, and invocation statistics from a routing run.

    Tracks both the offline phase (embedding, clustering, CoverAndMerge)
    and the online phase (cosine filter, LLM inference, post-processing).
    """
    num_events: int = 0
    num_subscriptions: int = 0
    num_clusters: int = 0
    compression_ratio: float = 1.0
    llm_invocations: int = 0
    total_prompt_tokens: int = 0
    total_response_tokens: int = 0
    embedding_time_s: float = 0.0
    clustering_time_s: float = 0.0
    cover_merge_time_s: float = 0.0
    cosine_filter_time_s: float = 0.0
    llm_inference_time_s: float = 0.0
    post_processing_time_s: float = 0.0
    total_time_s: float = 0.0
    num_subscriptions_effective: int = 0  # after context-window truncation


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RouterConfig:
    """Configuration for a Neural Router run.

    Each ablation variant (A0-A6, Table 2 in the paper) is a specific
    combination of the boolean component toggles below. Parameter defaults
    correspond to the full pipeline (A3).

    Attributes:
        use_clustering: Enable k-means subscription clustering (Algorithm 1).
        use_cover_merge: Enable CoverAndMerge compression (Algorithm 3).
        use_reunite: After C&M, merge all clusters back into one (A4 variant).
        use_event_clustering: Cluster events before matching (A5 variant).
        use_cosine_filter: Pre-filter events by cosine threshold tau.
    """
    # Components (ablation toggles)
    use_clustering: bool = True       # cluster subscriptions
    use_cover_merge: bool = True      # compress via C&M
    use_reunite: bool = False         # merge clusters back after C&M
    use_event_clustering: bool = False  # cluster events before matching
    use_cosine_filter: bool = True    # pre-filter events by cosine threshold

    # Core hyperparameters (Section 4.3)
    k: int = 19                       # number of subscription clusters (k in the paper)
    tau: float = 0.3                  # cosine similarity threshold (tau in the paper)
    kappa: int = 3                    # top-kappa matches returned per event
    adaptive_kappa: bool = False      # let LLM decide match count dynamically

    # Event clustering
    k_events: int = 5                 # number of event clusters (if enabled)

    # Cover/Merge
    cm_max_iterations: int = 10

    # LLM settings (used in the cost model, Eq. 4)
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 16384       # max output tokens (Haiku: 4096)
    context_window: int = 128000      # W: context window size in tokens
    t_inst: int = 200                 # t_inst: instruction overhead in tokens
    t_resp: int = 500                 # t_resp: reserved response budget in tokens

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"

    # Parallelism
    max_parallel: int = 10
    use_async: bool = True             # use async LLM calls for parallel matching
    max_batch_events: int = 200        # hard cap on events per LLM call (avoids JSON serialization issues)
    llm_timeout: int = 600             # per-request timeout in seconds (increase for local models)

    # Prompt size control
    max_event_words: int | None = None  # truncate event text in LLM prompts (None = no truncation)
    max_sub_words: int | None = None    # truncate subscription text in LLM prompts (None = no truncation)
    max_context_tokens: int | None = None  # hard token budget for crossover experiment (None = use context_window)

    # Random seed (for k-means)
    seed: int = 42

    @property
    def config_name(self) -> str:
        """Return a short descriptive name matching the ablation table (A0-A6)."""
        if not self.use_clustering and not self.use_cover_merge:
            return "A0_raw_llm"
        elif self.use_clustering and not self.use_cover_merge:
            return "A1_cluster_only"
        elif not self.use_clustering and self.use_cover_merge:
            return "A2_cm_only"
        elif self.use_reunite:
            return "A4_reunite"
        elif self.use_event_clustering:
            return "A5_event_clust"
        elif not self.use_cosine_filter:
            return "A6_no_cosine"
        else:
            return "A3_clust_cm"


# Predefined ablation configurations (Table 2 in the paper):
#   A0: Raw LLM baseline (no clustering, no C&M, no cosine filter)
#   A1: Clustering only (+ cosine filter, no C&M)
#   A2: CoverAndMerge only (no clustering, no cosine filter)
#   A3: Full pipeline (clustering + C&M + cosine filter)
#   A4: Full + reunite (merge clusters back after C&M)
#   A5: Full + event clustering
#   A6: Full but no cosine filter (tau=0 equivalent)
ABLATION_CONFIGS = {
    "A0": RouterConfig(
        use_clustering=False, use_cover_merge=False,
        use_cosine_filter=False,
    ),
    "A1": RouterConfig(
        use_clustering=True, use_cover_merge=False,
        use_cosine_filter=True,
    ),
    "A2": RouterConfig(
        use_clustering=False, use_cover_merge=True,
        use_cosine_filter=False,
    ),
    "A3": RouterConfig(
        use_clustering=True, use_cover_merge=True,
        use_cosine_filter=True,
    ),
    "A4": RouterConfig(
        use_clustering=True, use_cover_merge=True,
        use_reunite=True, use_cosine_filter=False,
    ),
    "A5": RouterConfig(
        use_clustering=True, use_cover_merge=True,
        use_event_clustering=True, use_cosine_filter=True,
    ),
    "A6": RouterConfig(
        use_clustering=True, use_cover_merge=True,
        use_cosine_filter=False,  # tau=0 equivalent
    ),
}


# ---------------------------------------------------------------------------
# Context-window truncation (crossover experiment support)
# ---------------------------------------------------------------------------

def _estimate_tokens_simple(text: str) -> int:
    """Fast token estimate: ~4 chars per token (GPT-family heuristic)."""
    return max(1, len(text) // 4)


def truncate_subscriptions_to_budget(
    subscriptions: list,
    budget_tokens: int | None,
    t_inst: int = 200,
    t_resp: int = 500,
) -> list:
    """Return the longest prefix of *subscriptions* that fits within *budget_tokens*.

    The budget accounts for instruction overhead (t_inst) and response budget
    (t_resp).  Subscriptions are added greedily in order until the next one
    would exceed the remaining capacity.

    Args:
        subscriptions: Ordered subscription list.
        budget_tokens: Hard token ceiling.  ``None`` means unlimited.
        t_inst: Instruction template overhead in tokens.
        t_resp: Reserved response budget in tokens.

    Returns:
        A (possibly shorter) prefix of *subscriptions*.
    """
    if budget_tokens is None:
        return list(subscriptions)

    available = budget_tokens - t_inst - t_resp
    if available <= 0:
        return []

    result = []
    running = 0
    for sub in subscriptions:
        line = f"- [{sub.id}] {sub.description}"
        tokens = _estimate_tokens_simple(line)
        if running + tokens > available:
            break
        running += tokens
        result.append(sub)

    return result


# ---------------------------------------------------------------------------
# Neural Router
# ---------------------------------------------------------------------------

class NeuralRouter:
    """Neural Router matching engine (Algorithms 1-3).

    Orchestrates the offline phase (optimize_subscriptions) and online phase
    (match_events). Maintains stateful cluster assignments and cumulative
    statistics across calls.

    Args:
        config: Full router configuration (ablation toggles + hyperparameters).
        llm_client: LLM client for semantic matching and CoverAndMerge.
        embedding_model: Embedding backend (OllamaEmbeddings or EmbeddingModel).
    """

    def __init__(
        self,
        config: RouterConfig,
        llm_client: LLMClient,
        embedding_model: Optional[EmbeddingModel | OllamaEmbeddings] = None,
    ):
        self.config = config
        self.llm = llm_client
        self.embedder = embedding_model
        self.clusters: list[Cluster] = []
        self.stats = RouterStats()

    # -------------------------------------------------------------------
    # Algorithm 1: OptimizeSubscriptions (offline)
    # -------------------------------------------------------------------

    def optimize_subscriptions(self, subscriptions: list[Subscription]) -> list[Cluster]:
        """Cluster and compress subscriptions (offline phase, Algorithm 1).

        Steps (Section 3.2):
          1. Embed all subscription descriptions with the sentence-transformer.
          2. K-means clustering into k groups (if use_clustering is True).
          3. CoverAndMerge per cluster to compress subscriptions (if use_cover_merge).
          4. Optionally reunite all clusters into one (A4 variant).

        Args:
            subscriptions: The full subscription set S.

        Returns:
            List of Cluster objects with centroid, original, and (optionally)
            compressed subscriptions.
        """
        self.stats.num_subscriptions = len(subscriptions)
        t0 = time.time()

        # Step 1: Embed subscriptions
        t_embed = time.time()
        sub_texts = [s.description for s in subscriptions]
        sub_embeddings = self.embedder.encode(sub_texts)
        self.stats.embedding_time_s = time.time() - t_embed
        logger.info(f"Embedded {len(subscriptions)} subscriptions in {self.stats.embedding_time_s:.2f}s")

        # Step 2: Cluster (or single cluster)
        t_clust = time.time()
        if self.config.use_clustering and self.config.k > 1:
            k = min(self.config.k, len(subscriptions))
            kmeans = KMeans(n_clusters=k, random_state=self.config.seed, n_init=10)
            labels = kmeans.fit_predict(sub_embeddings)

            clusters = []
            for i in range(k):
                mask = labels == i
                cluster_subs = [subscriptions[j] for j in range(len(subscriptions)) if mask[j]]
                cluster_embeds = sub_embeddings[mask]
                centroid = cluster_embeds.mean(axis=0)
                clusters.append(Cluster(
                    id=i,
                    subscriptions=cluster_subs,
                    centroid=centroid,
                    llm=self.config.llm_model,
                ))
        else:
            # Single cluster (A0 or A2)
            centroid = sub_embeddings.mean(axis=0)
            clusters = [Cluster(
                id=0,
                subscriptions=list(subscriptions),
                centroid=centroid,
                llm=self.config.llm_model,
            )]
        self.stats.clustering_time_s = time.time() - t_clust
        self.stats.num_clusters = len(clusters)
        logger.info(f"Created {len(clusters)} clusters in {self.stats.clustering_time_s:.2f}s")

        # Step 3: Cover/Merge per cluster
        t_cm = time.time()
        if self.config.use_cover_merge:
            for cluster in clusters:
                cluster.compressed = self._cover_and_merge(cluster.subscriptions)
                logger.info(
                    f"Cluster {cluster.id}: {len(cluster.subscriptions)} → "
                    f"{len(cluster.compressed)} subscriptions (ρ={cluster.compression_ratio:.2f})"
                )
        self.stats.cover_merge_time_s = time.time() - t_cm

        # Step 4: Reunite (if enabled)
        if self.config.use_reunite and len(clusters) > 1:
            all_compressed = []
            for c in clusters:
                all_compressed.extend(c.active_subscriptions)
            merged_centroid = sub_embeddings.mean(axis=0)
            clusters = [Cluster(
                id=0,
                subscriptions=[s for c in clusters for s in c.subscriptions],
                compressed=all_compressed,
                centroid=merged_centroid,
                llm=self.config.llm_model,
            )]
            logger.info(f"Reunited into single cluster with {len(all_compressed)} subscriptions")

        self.clusters = clusters

        # Compute overall compression ratio
        total_original = sum(len(c.subscriptions) for c in clusters)
        total_compressed = sum(len(c.active_subscriptions) for c in clusters)
        self.stats.compression_ratio = total_compressed / max(total_original, 1)

        self.stats.total_time_s = time.time() - t0
        return clusters

    # -------------------------------------------------------------------
    # Algorithm 2: MatchEvents (online)
    # -------------------------------------------------------------------

    def match_events(self, events: list[Event]) -> list[MatchResult]:
        """Match events to subscriptions (online phase, Algorithm 2).

        Steps (Section 3.3):
          1. Embed all event texts.
          2. Assign each event to clusters whose centroid similarity >= tau
             (cosine pre-filter). Falls back to nearest cluster if no threshold
             is met.
          3. For each cluster, batch its queued events and invoke the LLM.
          4. Deduplicate and trim results to top-kappa per event.

        Args:
            events: The event stream M to be matched.

        Returns:
            One MatchResult per event, in the same order as the input list.

        Raises:
            RuntimeError: If optimize_subscriptions() has not been called.
        """
        self.stats.num_events = len(events)
        t0 = time.time()

        if not self.clusters:
            raise RuntimeError("Call optimize_subscriptions() first")

        # Step 1: Embed events
        t_embed = time.time()
        event_texts = [e.text for e in events]
        event_embeddings = self.embedder.encode(event_texts)
        self.stats.embedding_time_s += time.time() - t_embed

        # Step 2: Assign events to clusters
        t_filter = time.time()
        # Clear previous queues
        for cluster in self.clusters:
            cluster.queue = []

        if self.config.use_cosine_filter and len(self.clusters) > 1:
            # Cosine pre-filter: assign event to clusters where similarity >= tau
            centroids = np.array([c.centroid for c in self.clusters])
            similarities = cosine_similarity(event_embeddings, centroids)

            for i, event in enumerate(events):
                assigned = False
                for j, cluster in enumerate(self.clusters):
                    if similarities[i, j] >= self.config.tau:
                        cluster.queue.append(event)
                        assigned = True
                if not assigned:
                    # Assign to nearest cluster as fallback
                    nearest = int(np.argmax(similarities[i]))
                    self.clusters[nearest].queue.append(event)
        else:
            # No filter: all events go to all clusters
            for cluster in self.clusters:
                cluster.queue = list(events)

        self.stats.cosine_filter_time_s = time.time() - t_filter

        # Log assignment distribution
        for c in self.clusters:
            logger.debug(f"Cluster {c.id}: {len(c.queue)} events assigned")

        # Step 3: LLM matching per cluster
        t_llm = time.time()

        if self.config.use_async and not isinstance(self.llm, DryRunLLMClient):
            all_results = self._run_async_matching()
        else:
            all_results = self._run_sync_matching()

        self.stats.llm_inference_time_s = time.time() - t_llm

        # Post-process: deduplicate and trim to top-kappa
        t_post = time.time()
        final_results = []
        for event in events:
            if event.id in all_results:
                result = all_results[event.id]
                # Deduplicate subscription matches
                seen = set()
                unique_matches = []
                for sid in result.matched_subscription_ids:
                    if sid not in seen:
                        seen.add(sid)
                        unique_matches.append(sid)
                if not self.config.adaptive_kappa:
                    unique_matches = unique_matches[:self.config.kappa]
                result.matched_subscription_ids = unique_matches
                final_results.append(result)
            else:
                final_results.append(MatchResult(event_id=event.id, matched_subscription_ids=[]))

        self.stats.post_processing_time_s = time.time() - t_post
        self.stats.total_time_s = time.time() - t0

        return final_results

    # -------------------------------------------------------------------
    # Algorithm 3: CoverAndMerge
    # -------------------------------------------------------------------

    def _cover_and_merge(self, subscriptions: list[Subscription]) -> list[Subscription]:
        """LLM-assisted subscription compression (Algorithm 3, Section 3.2).

        Iteratively asks the LLM to identify:
          - Covering pairs: s_i semantically subsumes s_j, so s_j is removed.
          - Merge candidates: s_i and s_j are combined into a new, broader
            subscription with a concatenated identifier (e.g., "id1+id2").

        Converges when the LLM returns no further operations or cm_max_iterations
        is reached.

        Args:
            subscriptions: Subscriptions within a single cluster to compress.

        Returns:
            Compressed subscription list (may be shorter than input).
        """
        current = list(subscriptions)

        for iteration in range(self.config.cm_max_iterations):
            if len(current) <= 1:
                break

            prompt = self._build_cover_merge_prompt(current)
            response = self.llm.invoke(prompt)
            self.stats.llm_invocations += 1
            self.stats.total_prompt_tokens += response.prompt_tokens
            self.stats.total_response_tokens += response.response_tokens

            try:
                operations = self._parse_cover_merge_response(response.text, current)
            except Exception as e:
                logger.warning(f"Failed to parse C&M response (iter {iteration}): {e}")
                break

            covers = operations.get("covers", [])
            merges = operations.get("merges", [])

            if not covers and not merges:
                logger.info(f"C&M converged after {iteration + 1} iterations")
                break

            # Apply covers: remove subsumed subscriptions
            to_remove = set()
            for parent_id, child_id in covers:
                to_remove.add(child_id)

            current = [s for s in current if s.id not in to_remove]

            # Apply merges: replace pairs with merged subscription
            merge_remove = set()
            new_subs = []
            for s1_id, s2_id, merged_desc in merges:
                merge_remove.add(s1_id)
                merge_remove.add(s2_id)
                # Create merged subscription preserving both IDs
                new_subs.append(Subscription(
                    id=f"{s1_id}+{s2_id}",
                    name=f"Merged({s1_id}, {s2_id})",
                    description=merged_desc,
                ))

            current = [s for s in current if s.id not in merge_remove]
            current.extend(new_subs)

            logger.debug(
                f"C&M iter {iteration}: {len(covers)} covers, {len(merges)} merges "
                f"→ {len(current)} subscriptions"
            )

        return current

    # -------------------------------------------------------------------
    # LLM prompt construction
    # -------------------------------------------------------------------

    def _match_cluster(self, cluster: Cluster) -> list[MatchResult]:
        """Match all events in a cluster's queue against its subscriptions.

        Automatically batches events to fit within the LLM's context window,
        using the cost model formula (Eq. 4) to determine batch size.

        Args:
            cluster: Cluster with a non-empty event queue.

        Returns:
            One MatchResult per queued event.
        """
        subs = cluster.active_subscriptions
        events = cluster.queue
        results = []

        # Context-window truncation (crossover experiment): drop subscriptions
        # that don't fit within the hard token budget.
        if self.config.max_context_tokens is not None:
            subs = truncate_subscriptions_to_budget(
                subs,
                budget_tokens=self.config.max_context_tokens,
                t_inst=self.config.t_inst,
                t_resp=self.config.t_resp,
            )
            self.stats.num_subscriptions_effective += len(subs)
            if len(subs) < len(cluster.active_subscriptions):
                logger.info(
                    f"  Context truncation: {len(cluster.active_subscriptions)} → "
                    f"{len(subs)} subscriptions (budget={self.config.max_context_tokens})"
                )

        # Compute batch size from the cost model (Eq. 4):
        #   b_max = floor((W - t_inst - |S_c| * t_s - t_resp) / t_e)
        sub_text = "\n".join(f"- [{s.id}] {s.description}" for s in subs)
        avg_sub_tokens = self._estimate_tokens(sub_text)
        # Estimate per-event tokens from actual event texts
        if events:
            sample = events[:min(20, len(events))]
            avg_event_tokens = max(10, int(
                sum(self._estimate_tokens(f"- [{e.id}] {e.text}") for e in sample)
                / len(sample)
            ))
        else:
            avg_event_tokens = 50

        available = self.config.context_window - self.config.t_inst - avg_sub_tokens - self.config.t_resp
        if available <= 0:
            batch_size = 1
        else:
            batch_size = max(1, int(available / max(avg_event_tokens, 1)))
            batch_size = max(1, int(batch_size * 0.8))  # 20% safety margin
            batch_size = min(batch_size, self.config.max_batch_events)  # hard cap

        # Process in batches
        for i in range(0, len(events), batch_size):
            batch = events[i:i + batch_size]
            prompt = self._build_match_prompt(subs, batch)

            # Validate prompt fits; halve batch if too large
            prompt_tokens = self._estimate_tokens(prompt)
            while prompt_tokens > self.config.context_window * 0.95 and len(batch) > 1:
                batch = batch[:len(batch) // 2]
                prompt = self._build_match_prompt(subs, batch)
                prompt_tokens = self._estimate_tokens(prompt)
            response = self.llm.invoke(prompt)
            self.stats.llm_invocations += 1
            self.stats.total_prompt_tokens += response.prompt_tokens
            self.stats.total_response_tokens += response.response_tokens

            try:
                batch_results = self._parse_match_response(response.text, batch)
                results.extend(batch_results)
            except Exception as e:
                logger.warning(f"Failed to parse match response for batch {i}: {e}")
                # Return empty matches for failed batch
                for event in batch:
                    results.append(MatchResult(event_id=event.id, matched_subscription_ids=[]))

        return results

    def _run_sync_matching(self) -> dict[str, MatchResult]:
        """Run LLM matching sequentially (original path)."""
        all_results: dict[str, MatchResult] = {}
        for cluster in self.clusters:
            if not cluster.queue:
                continue
            cluster_results = self._match_cluster(cluster)
            for result in cluster_results:
                if result.event_id in all_results:
                    existing = all_results[result.event_id]
                    existing.matched_subscription_ids.extend(
                        result.matched_subscription_ids
                    )
                else:
                    all_results[result.event_id] = result
        return all_results

    def _run_async_matching(self) -> dict[str, MatchResult]:
        """Run LLM matching with async parallelism via asyncio.

        Builds all prompts upfront across all clusters, fires them
        concurrently (bounded by max_parallel semaphore), then parses
        and merges results. Falls back to a thread-pool if an event loop
        is already running (e.g., inside Jupyter).

        Returns:
            Dict mapping event_id to its aggregated MatchResult.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self._async_match_all_clusters())
                    return future.result()
            else:
                return loop.run_until_complete(self._async_match_all_clusters())
        except RuntimeError:
            return asyncio.run(self._async_match_all_clusters())

    async def _async_match_all_clusters(self) -> dict[str, MatchResult]:
        """Async implementation: build all prompts, fire concurrently, merge."""
        import litellm

        semaphore = asyncio.Semaphore(self.config.max_parallel)

        # Phase 1: Build all (prompt, cluster, batch) tuples upfront
        prompt_tasks: list[tuple[str, Cluster, list[Event]]] = []

        for cluster in self.clusters:
            if not cluster.queue:
                continue
            subs = cluster.active_subscriptions
            events = cluster.queue

            # Context-window truncation (crossover experiment)
            if self.config.max_context_tokens is not None:
                subs = truncate_subscriptions_to_budget(
                    subs,
                    budget_tokens=self.config.max_context_tokens,
                    t_inst=self.config.t_inst,
                    t_resp=self.config.t_resp,
                )
                self.stats.num_subscriptions_effective += len(subs)

            # Compute batch size (same logic as _match_cluster)
            sub_text = "\n".join(f"- [{s.id}] {s.description}" for s in subs)
            avg_sub_tokens = self._estimate_tokens(sub_text)
            if events:
                sample = events[:min(20, len(events))]
                avg_event_tokens = max(10, int(
                    sum(self._estimate_tokens(f"- [{e.id}] {e.text}") for e in sample)
                    / len(sample)
                ))
            else:
                avg_event_tokens = 50

            available = (
                self.config.context_window
                - self.config.t_inst
                - avg_sub_tokens
                - self.config.t_resp
            )
            batch_size = max(1, int(available / max(avg_event_tokens, 1))) if available > 0 else 1

            # Safety margin: reduce by 20% to account for prompt template overhead
            batch_size = max(1, int(batch_size * 0.8))
            batch_size = min(batch_size, self.config.max_batch_events)  # hard cap

            logger.info(
                f"  Cluster {cluster.id}: {len(events)} events, "
                f"batch_size={batch_size}, avg_event_tokens={avg_event_tokens}"
            )

            for i in range(0, len(events), batch_size):
                batch = events[i:i + batch_size]
                prompt = self._build_match_prompt(subs, batch)

                # Validate prompt fits context window; halve batch if too large
                prompt_tokens = self._estimate_tokens(prompt)
                while prompt_tokens > self.config.context_window * 0.95 and len(batch) > 1:
                    batch = batch[:len(batch) // 2]
                    prompt = self._build_match_prompt(subs, batch)
                    prompt_tokens = self._estimate_tokens(prompt)
                    logger.warning(
                        f"  Prompt too large ({prompt_tokens} est. tokens), "
                        f"reduced batch to {len(batch)} events"
                    )

                prompt_tasks.append((prompt, cluster, batch))

        logger.info(f"Async matching: {len(prompt_tasks)} prompts across {len(self.clusters)} clusters")

        # Phase 2: Fire all prompts concurrently (with retry on rate limits)
        def _is_fatal_error(e: Exception) -> bool:
            """Detect non-transient API errors that should abort the run."""
            err_str = str(e).lower()
            fatal_patterns = [
                "usage limit",       # billing/budget exhausted
                "invalid_api_key",   # bad credentials
                "authentication",    # auth failure
                "permission",        # permission denied
                "not_found_error",   # model deprecated/removed
            ]
            return any(p in err_str for p in fatal_patterns)

        async def _invoke_one(prompt: str) -> LLMResponse:
            max_retries = 5
            for attempt in range(max_retries):
                async with semaphore:
                    t0 = time.time()
                    try:
                        response = await asyncio.wait_for(
                            litellm.acompletion(
                                model=self.config.llm_model,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=self.config.llm_temperature,
                                max_tokens=self.config.llm_max_tokens,
                                timeout=self.config.llm_timeout,
                            ),
                            timeout=self.config.llm_timeout + 60,
                        )
                        text = response.choices[0].message.content or ""
                        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                        response_tokens = response.usage.completion_tokens if response.usage else 0
                        latency = time.time() - t0
                        return LLMResponse(
                            text=text,
                            prompt_tokens=prompt_tokens,
                            response_tokens=response_tokens,
                            latency_s=latency,
                            model=self.config.llm_model,
                        )
                    except Exception as e:
                        latency = time.time() - t0
                        # Fatal errors (billing, auth) should not be retried
                        if _is_fatal_error(e):
                            logger.error(f"Fatal API error (aborting run): {e}")
                            raise
                        is_rate_limit = "rate_limit" in str(e).lower() or "429" in str(e)
                        if is_rate_limit and attempt < max_retries - 1:
                            wait = 2 ** attempt * 5  # 5s, 10s, 20s, 40s
                            logger.warning(f"Rate limited, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                            await asyncio.sleep(wait)
                            continue
                        logger.error(f"Async LLM call failed after {latency:.2f}s: {e}")
                        raise

        coros = [_invoke_one(prompt) for prompt, _, _ in prompt_tasks]
        responses = await asyncio.gather(*coros, return_exceptions=True)

        # Phase 3: Parse responses and merge results
        all_results: dict[str, MatchResult] = {}

        # Check for fatal errors first: if ANY response is a fatal error,
        # abort immediately rather than producing silent all-empty results
        for resp in responses:
            if isinstance(resp, BaseException) and _is_fatal_error(resp):
                logger.error(f"Fatal error detected in batch, aborting: {resp}")
                raise resp

        # Count failures for early-abort on mass transient failures
        failure_count = sum(1 for r in responses if isinstance(r, BaseException))
        if failure_count == len(responses) and len(responses) > 0:
            first_error = next(r for r in responses if isinstance(r, BaseException))
            logger.error(f"All {len(responses)} async calls failed. First error: {first_error}")
            raise RuntimeError(
                f"All {len(responses)} LLM calls failed. First error: {first_error}"
            )

        for (prompt, cluster, batch), resp in zip(prompt_tasks, responses):
            if isinstance(resp, BaseException):
                logger.warning(f"Async call failed for cluster {cluster.id}: {resp}")
                for event in batch:
                    if event.id not in all_results:
                        all_results[event.id] = MatchResult(
                            event_id=event.id, matched_subscription_ids=[]
                        )
                continue

            # Track stats
            self.stats.llm_invocations += 1
            self.stats.total_prompt_tokens += resp.prompt_tokens
            self.stats.total_response_tokens += resp.response_tokens

            try:
                batch_results = self._parse_match_response(resp.text, batch)
            except Exception as e:
                logger.warning(f"Failed to parse async response for cluster {cluster.id}: {e}")
                batch_results = [
                    MatchResult(event_id=ev.id, matched_subscription_ids=[])
                    for ev in batch
                ]

            for result in batch_results:
                if result.event_id in all_results:
                    existing = all_results[result.event_id]
                    existing.matched_subscription_ids.extend(
                        result.matched_subscription_ids
                    )
                else:
                    all_results[result.event_id] = result

        return all_results

    def _build_match_prompt(
        self,
        subscriptions: list[Subscription],
        events: list[Event],
    ) -> str:
        """Build the event matching prompt (Listing 2 from the paper).

        The prompt instructs the LLM to evaluate every (event, subscription)
        pair semantically, rank matches by relevance, and return a JSON dict
        mapping event IDs to lists of matched subscription IDs.

        Args:
            subscriptions: Active (possibly compressed) subscriptions for this cluster.
            events: Batch of events to match in a single LLM call.

        Returns:
            Complete prompt string ready for LLM invocation.
        """
        def _truncate(text: str, max_words: int | None) -> str:
            if max_words is None:
                return text
            words = text.split()
            if len(words) <= max_words:
                return text
            return " ".join(words[:max_words]) + " [...]"

        sub_text = "\n".join(
            f"- [{s.id}] {_truncate(s.description, self.config.max_sub_words)}"
            for s in subscriptions
        )

        event_text = "\n\n".join(
            f"Event [{e.id}]:\n{_truncate(e.text, self.config.max_event_words)}" for e in events
        )

        kappa_instruction = (
            "Return all subscriptions that match."
            if self.config.adaptive_kappa
            else f"Return the top {self.config.kappa} matches."
        )

        return f"""# Role
Semantic matching engine for a content-based publish/subscribe system.

# Task
For each event below, identify which subscriptions it satisfies. A **match** means the event content is relevant to the subscriber's stated interest. Consider meaning, paraphrases, and related concepts, not just keyword overlap.

# Subscriptions
{sub_text}

# Events
{event_text}

# Steps
For each event:
1. Evaluate it against every subscription using semantic similarity.
2. Rank matching subscriptions by relevance.
3. {kappa_instruction} If fewer match, return only those that do. If none match, return an empty list.

# Constraints
- Evaluate every (event, subscription) pair; do not skip subscriptions.
- When uncertain, prefer inclusion (favour recall over precision).

# Output
Return a JSON object mapping event IDs to lists of matching subscription IDs.
Example: {{"event_1": ["sub_a", "sub_b"], "event_2": ["sub_c"]}}
Only return the JSON, no other text."""

    def _build_cover_merge_prompt(self, subscriptions: list[Subscription]) -> str:
        """Build the CoverAndMerge prompt (Listing 1 from the paper).

        Instructs the LLM to identify subsumption pairs and merge candidates,
        returning structured JSON with "covers" and "merges" keys.

        Args:
            subscriptions: Current subscription set within one cluster.

        Returns:
            Complete prompt string ready for LLM invocation.
        """
        sub_text = "\n".join(
            f"- [{s.id}] {s.description}" for s in subscriptions
        )

        return f"""# Role
Subscription optimiser for a content-based publish/subscribe system.

# Task
Compress the subscription set below while preserving matching accuracy. Every original identifier **must** be retained.

# Subscriptions
{sub_text}

# Steps
1. **Subsumption test** -- For every pair, check whether one subscription semantically subsumes another (i.e., every event that matches the narrower one also matches the broader one).
2. **Combine subsumed** -- Merge each subsumed subscription into its parent. Concatenate identifier lists.
3. **Merge similar** -- Where subsumption does not apply, merge subscriptions that can be combined into a single, more general description without reducing accuracy. Each identifier must appear exactly once across all outputs.
4. **Compress text** -- Shorten every remaining subscription to the most concise phrasing that preserves its semantic scope.

# Constraints
- Never drop or duplicate an identifier.
- Prefer recall: when uncertain whether two subscriptions overlap, keep them separate.

# Output
Return a JSON object with two keys:
- "covers": list of [parent_id, child_id] pairs where parent subsumes child
- "merges": list of [id1, id2, "merged description"] triples
Example: {{"covers": [["sports", "football"]], "merges": [["arts_&_culture", "music", "Arts, culture, and music entertainment"]]}}
Only return the JSON, no other text."""

    def _parse_match_response(
        self, response_text: str, events: list[Event],
    ) -> list[MatchResult]:
        """Parse the LLM's JSON response for event matching.

        Handles markdown code fences, nested lists, and missing event keys.

        Args:
            response_text: Raw LLM output (expected: JSON dict).
            events: The event batch that was sent in the prompt.

        Returns:
            One MatchResult per event in the batch.

        Raises:
            json.JSONDecodeError: If the response is not valid JSON.
        """
        # Try to extract JSON from response
        text = response_text.strip()
        if text.startswith("```"):
            # Strip markdown code fences
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        data = json.loads(text)

        results = []
        for event in events:
            matches = data.get(event.id, [])
            if isinstance(matches, list):
                # Flatten any nested lists and ensure string IDs
                flat = []
                for m in matches:
                    if isinstance(m, str):
                        flat.append(m)
                    elif isinstance(m, list) and m:
                        flat.append(str(m[0]))
                results.append(MatchResult(
                    event_id=event.id,
                    matched_subscription_ids=flat,
                ))
            else:
                results.append(MatchResult(event_id=event.id, matched_subscription_ids=[]))

        return results

    def _parse_cover_merge_response(
        self, response_text: str, subscriptions: list[Subscription],
    ) -> dict[str, list]:
        """Parse the LLM's JSON response for CoverAndMerge operations.

        Args:
            response_text: Raw LLM output (expected: JSON with "covers" and "merges").
            subscriptions: Current subscriptions (for validation context).

        Returns:
            Dict with "covers" (list of (parent, child) tuples) and
            "merges" (list of (id1, id2, description) triples).
        """
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        data = json.loads(text)

        covers = []
        for pair in data.get("covers", []):
            if len(pair) >= 2:
                covers.append((str(pair[0]), str(pair[1])))

        merges = []
        for triple in data.get("merges", []):
            if len(triple) >= 3:
                merges.append((str(triple[0]), str(triple[1]), str(triple[2])))

        return {"covers": covers, "merges": merges}

    def _estimate_tokens(self, text: str) -> int:
        """Conservative token count estimate (~3.2 chars per token).

        GPT tokenizers average 3-3.5 characters per token for English text.
        We use 3.2 to avoid under-estimating and exceeding context windows.
        """
        return int(len(text) / 3.2)

    # -------------------------------------------------------------------
    # Cost model
    # -------------------------------------------------------------------

    def predict_invocations(self) -> int:
        """Predict total LLM invocations using the cost model (Eq. 4).

        I = sum_c ceil(|M_c| / b_max_c), where b_max_c is the max batch
        size for cluster c given the context window budget.

        Returns:
            Estimated total number of LLM calls for the current cluster state.
        """
        if not self.clusters:
            return 0

        total = 0
        for cluster in self.clusters:
            subs = cluster.active_subscriptions
            avg_t_s = 80  # tokens per subscription (estimate)
            avg_t_e = 50  # tokens per event (estimate)

            # b_max = floor((W - t_inst - |S'_c| * t_s - t_resp) / t_e)
            b_max = max(1, (
                self.config.context_window
                - self.config.t_inst
                - len(subs) * avg_t_s
                - self.config.t_resp
            ) // avg_t_e)

            m_c = len(cluster.queue)
            i_c = (m_c + b_max - 1) // b_max  # ceil(|M_c| / b_max)
            total += i_c

        return total

    def predict_latency(self, t_llm: float = 0.2) -> float:
        """Predict end-to-end latency using the cost model (Eq. 5).

        L = ceil(I / P) * t_llm, where I is total invocations and P is the
        parallelism degree.

        Args:
            t_llm: Mean wall-clock time per LLM invocation in seconds.

        Returns:
            Estimated wall-clock latency in seconds.
        """
        I = self.predict_invocations()
        P = self.config.max_parallel
        R = (I + P - 1) // P  # ceil(I / P): number of sequential rounds
        return R * t_llm
