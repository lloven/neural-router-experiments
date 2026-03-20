"""Dataset loading and preprocessing for Neural Router experiments.

Provides a unified interface over three benchmark datasets used to evaluate
the Neural Router's content-based publish/subscribe matching (Table 1 in the
paper).  The datasets form a progression of semantic difficulty:

  D1: CardiffNLP Tweet Topic -- multi-label, 19 topics, short tweets (~25 words).
                                Low semantic gap: keywords overlap between events
                                and subscription descriptions.
  D2: MultiEURLEX           -- multi-label, 127 EUROVOC level-2 labels, long EU
                                legal documents (~700 words, truncated to 2000).
                                High semantic gap: domain-specific legal jargon,
                                structured taxonomy.
  D3: CASAS Aruba IoT       -- single-label, ~11 ADL activities, smart home
                                sensor event sequences (~20-50 readings/event).
                                Very high semantic gap: raw sensor data vs.
                                natural-language activity descriptions.  Requires
                                abductive reasoning to bridge modality mismatch.

Key abstractions:
  Subscription  -- a topic (filter) that events can match against
  Event         -- a document (notification) to be routed to matching subscriptions
  Dataset       -- a loaded corpus with events, subscriptions, and metadata

Usage:
    dataset = load_dataset_by_name("D1", cache_dir="data", max_events=500)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Subscription:
    """A subscription (topic/filter) in the pub/sub model.

    Maps to a single node in the subscription set S (Definition 1 in the paper).
    The ``description`` field is the natural-language text passed to the LLM
    and the embedding model for semantic matching.
    """
    id: str
    name: str
    description: str  # natural-language description for LLM matching


@dataclass
class Event:
    """An event (document/notification) to be routed to matching subscriptions.

    Maps to an element of the event stream M (Definition 2 in the paper).
    ``ground_truth`` holds the oracle matching subscription IDs for evaluation.
    """
    id: str
    text: str
    ground_truth: list[str]  # list of matching subscription IDs


@dataclass
class Dataset:
    """A loaded dataset ready for Neural Router evaluation.

    Bundles events and subscriptions together with descriptive metadata.
    Provides summary statistics used in Table 1 of the paper.
    """
    name: str
    short_name: str  # D1, D2, D3
    events: list[Event]
    subscriptions: list[Subscription]
    metadata: dict = field(default_factory=dict)

    @property
    def num_events(self) -> int:
        return len(self.events)

    @property
    def num_subscriptions(self) -> int:
        return len(self.subscriptions)

    @property
    def mean_labels_per_event(self) -> float:
        if not self.events:
            return 0.0
        return np.mean([len(e.ground_truth) for e in self.events])

    @property
    def mean_words_per_event(self) -> float:
        if not self.events:
            return 0.0
        return np.mean([len(e.text.split()) for e in self.events])

    def summary(self) -> str:
        return (
            f"{self.short_name}: {self.name}\n"
            f"  Events: {self.num_events}\n"
            f"  Subscriptions: {self.num_subscriptions}\n"
            f"  Mean labels/event: {self.mean_labels_per_event:.1f}\n"
            f"  Mean words/event: {self.mean_words_per_event:.0f}"
        )


def load_dataset_by_name(
    name: str,
    cache_dir: Optional[str] = None,
    max_events: Optional[int] = None,
) -> Dataset:
    """Load a dataset by short name or alias.

    Args:
        name: Dataset identifier. Accepts short names (D1, D2, D3) or
            aliases (cardiffnlp, eurlex, casas).
        cache_dir: Directory for HuggingFace / Zenodo caches.
        max_events: If set, randomly subsample to this many events
            (deterministic, seed=42).

    Returns:
        A fully populated Dataset instance.

    Raises:
        ValueError: If `name` does not match any known dataset.
    """
    name_lower = name.lower().strip()
    if name_lower in ("d1", "cardiffnlp", "tweet_topic"):
        return load_cardiffnlp(cache_dir=cache_dir, max_events=max_events)
    elif name_lower in ("d2", "eurlex", "multi_eurlex", "multieurlex"):
        return load_eurlex(cache_dir=cache_dir, max_events=max_events)
    elif name_lower in ("d3", "casas", "casas_iot", "aruba"):
        return load_casas(cache_dir=cache_dir, max_events=max_events)
    else:
        raise ValueError(f"Unknown dataset: {name}. Use D1/D2/D3.")


# ---------------------------------------------------------------------------
# D1: CardiffNLP Tweet Topic (multi-label)
# ---------------------------------------------------------------------------

# Human-readable descriptions for the 19 CardiffNLP topics
CARDIFFNLP_DESCRIPTIONS = {
    "arts_&_culture": "Arts and culture, including visual arts, literature, theater, and cultural events",
    "business_&_entrepreneurs": "Business, entrepreneurship, startups, and corporate news",
    "celebrity_&_pop_culture": "Celebrity news, pop culture, entertainment industry gossip",
    "diaries_&_daily_life": "Personal diary entries, daily routines, and everyday life updates",
    "family": "Family relationships, parenting, and family activities",
    "fashion_&_style": "Fashion trends, clothing, style advice, and beauty",
    "film_tv_&_video": "Film, television shows, streaming content, and video media",
    "fitness_&_health": "Fitness routines, health tips, wellness, and medical topics",
    "food_&_dining": "Food, cooking, restaurants, recipes, and dining experiences",
    "gaming": "Video games, esports, gaming culture, and game reviews",
    "learning_&_educational": "Education, learning resources, academic topics, and teaching",
    "music": "Music artists, albums, concerts, genres, and music industry",
    "news_&_social_concern": "Current news, social issues, politics, and civic concerns",
    "other_hobbies": "Hobbies and interests not covered by other categories",
    "relationships": "Romantic relationships, dating, and interpersonal connections",
    "science_&_technology": "Science discoveries, technology news, and innovation",
    "sports": "Sports events, athletes, teams, and athletic competitions",
    "travel_&_adventure": "Travel destinations, adventure activities, and tourism",
    "youth_&_student_life": "Student life, youth culture, campus activities, and academic experiences",
}


def load_cardiffnlp(
    cache_dir: Optional[str] = None,
    max_events: Optional[int] = None,
) -> Dataset:
    """Load D1: CardiffNLP Tweet Topic multi-label dataset from HuggingFace.

    Args:
        cache_dir: HuggingFace cache directory.
        max_events: Cap on number of events (deterministic subsample).

    Returns:
        Dataset with 19 subscriptions and multi-label events.
    """
    from datasets import load_dataset as hf_load

    logger.info("Loading CardiffNLP Tweet Topic dataset...")
    ds = hf_load("cardiffnlp/tweet_topic_multi", cache_dir=cache_dir)

    # Combine train and test splits (the dataset uses _coling2022 suffixes
    # for split names, but some versions drop the suffix)
    all_examples = []
    for split_name in ["train_coling2022", "test_coling2022"]:
        if split_name in ds:
            all_examples.extend(ds[split_name])
        elif split_name.replace("_coling2022", "") in ds:
            all_examples.extend(ds[split_name.replace("_coling2022", "")])

    # Fallback: iterate all available splits if named splits were not found
    if not all_examples:
        for split_name in ds.keys():
            all_examples.extend(ds[split_name])

    logger.info(f"Loaded {len(all_examples)} examples from CardiffNLP")

    # Build subscription list from the 19 topic labels
    label_names = list(CARDIFFNLP_DESCRIPTIONS.keys())
    subscriptions = []
    for label_name in label_names:
        subscriptions.append(Subscription(
            id=label_name,
            name=label_name.replace("_", " ").replace("&", "and"),
            description=CARDIFFNLP_DESCRIPTIONS[label_name],
        ))

    # Build event list
    events = []
    for i, example in enumerate(all_examples):
        # label is a binary vector of length 19
        label_vec = example["label"]
        active_labels = [label_names[j] for j, v in enumerate(label_vec) if v == 1]

        if not active_labels:
            continue  # skip events with no labels

        events.append(Event(
            id=f"cardiffnlp_{i}",
            text=example["text"],
            ground_truth=active_labels,
        ))

    if max_events and len(events) > max_events:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(events), max_events, replace=False)
        events = [events[i] for i in sorted(indices)]

    return Dataset(
        name="CardiffNLP Tweet Topic",
        short_name="D1",
        events=events,
        subscriptions=subscriptions,
        metadata={"source": "cardiffnlp/tweet_topic_multi", "multi_label": True},
    )


# ---------------------------------------------------------------------------
# D2: MultiEURLEX (multi-label, EUROVOC level-2)
# ---------------------------------------------------------------------------

def load_eurlex(
    cache_dir: Optional[str] = None,
    max_events: Optional[int] = None,
    label_level: str = "level_2",
) -> Dataset:
    """Load D2: MultiEURLEX dataset from HuggingFace with EUROVOC level-2 labels.

    The EUROVOC concept IDs are numeric; human-readable descriptions come from
    ``data/eurovoc_labels.csv`` (generated by ``scripts/fetch_eurovoc.py``).
    Documents longer than 2000 words are truncated to control LLM token costs.

    Args:
        cache_dir: HuggingFace cache directory.
        max_events: Cap on number of events (deterministic subsample).
        label_level: EUROVOC hierarchy level (default "level_2", 127 labels).

    Returns:
        Dataset with ~127 subscriptions and multi-label events.

    Note:
        The nlpaueb/multi_eurlex dataset requires ``trust_remote_code=True``
        because it uses a custom loading script.
    """
    from datasets import load_dataset as hf_load

    logger.info(f"Loading MultiEURLEX dataset (label_level={label_level})...")
    ds = hf_load(
        "nlpaueb/multi_eurlex", "en",
        label_level=label_level,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    # Get label names from the dataset features
    label_feature = ds["train"].features["labels"].feature
    label_names = label_feature.names  # EUROVOC concept IDs as strings
    logger.info(f"Found {len(label_names)} labels at {label_level}")

    # Build subscriptions from EUROVOC IDs
    # Try to load textual descriptions from a mapping file
    eurovoc_descriptions = _load_eurovoc_descriptions(label_names, cache_dir)

    subscriptions = []
    for label_id in label_names:
        desc = eurovoc_descriptions.get(label_id, label_id)
        subscriptions.append(Subscription(
            id=label_id,
            name=desc,
            description=desc,
        ))

    # Build events from all splits
    events = []
    event_idx = 0
    for split_name in ["train", "validation", "test"]:
        if split_name not in ds:
            continue
        for example in ds[split_name]:
            label_indices = example["labels"]
            active_labels = [label_names[j] for j in label_indices]

            if not active_labels:
                continue

            # Truncate very long documents to first 2000 words to manage costs
            text = example["text"]
            words = text.split()
            if len(words) > 2000:
                text = " ".join(words[:2000])

            events.append(Event(
                id=f"eurlex_{event_idx}",
                text=text,
                ground_truth=active_labels,
            ))
            event_idx += 1

    if max_events and len(events) > max_events:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(events), max_events, replace=False)
        events = [events[i] for i in sorted(indices)]

    return Dataset(
        name=f"MultiEURLEX ({label_level})",
        short_name="D2",
        events=events,
        subscriptions=subscriptions,
        metadata={
            "source": "nlpaueb/multi_eurlex",
            "label_level": label_level,
            "multi_label": True,
            "num_labels": len(label_names),
        },
    )


def _load_eurovoc_descriptions(
    label_ids: list[str],
    cache_dir: Optional[str] = None,
) -> dict[str, str]:
    """Load EUROVOC concept ID to textual description mapping.

    Looks for ``<cache_dir>/eurovoc_labels.csv`` (two columns: id, description).
    If the file is missing, falls back to raw numeric IDs and logs a warning.

    Args:
        label_ids: List of EUROVOC concept ID strings (e.g., "100163").
        cache_dir: Directory to search for the CSV mapping file.

    Returns:
        Dict mapping concept ID strings to human-readable descriptions.
    """
    # Check for local mapping file
    mapping_path = Path(cache_dir or "data") / "eurovoc_labels.csv"
    if mapping_path.exists():
        df = pd.read_csv(mapping_path)
        return dict(zip(df["id"].astype(str), df["description"]))

    logger.warning(
        f"EUROVOC description file not found at {mapping_path}. "
        "Using concept IDs as labels. Run scripts/fetch_eurovoc.py to create mapping."
    )
    return {lid: lid for lid in label_ids}


# ---------------------------------------------------------------------------
# D3: CASAS Aruba IoT (smart home ADL activities, free-living)
# ---------------------------------------------------------------------------

def load_casas(
    cache_dir: Optional[str] = None,
    max_events: Optional[int] = None,
) -> Dataset:
    """Load D3: CASAS Aruba smart home IoT dataset from pre-fetched CSV.

    The CASAS Aruba dataset must first be downloaded and converted using
    ``scripts/fetch_casas.py``, which produces ``data/casas_iot.csv``
    (events) and ``data/casas_iot_subscriptions.csv`` (subscriptions).

    The Aruba dataset contains naturalistic free-living sensor data from a
    single-resident apartment over approximately 8 months (Nov 2010 - Jun 2011).
    Each event is a sequence of motion/door sensor readings from one annotated
    activity segment.  Ground truth is single-label (one activity per segment).

    This dataset has the highest semantic gap in the evaluation: raw sensor
    readings (e.g., "sensor M014 (motion) reported ON") share no vocabulary
    with activity descriptions (e.g., "resident is preparing food").  Accurate
    matching requires abductive reasoning about what sensor patterns imply.

    Args:
        cache_dir: Directory containing the pre-fetched CSV files.
        max_events: Cap on number of events (deterministic subsample).

    Returns:
        Dataset with ~11 subscriptions and single-label events.
    """
    data_dir = Path(cache_dir or "data")
    events_path = data_dir / "casas_iot.csv"
    subs_path = data_dir / "casas_iot_subscriptions.csv"

    if not events_path.exists():
        raise FileNotFoundError(
            f"CASAS events CSV not found at {events_path}. "
            "Run scripts/fetch_casas.py first to download and convert the dataset."
        )

    logger.info("Loading CASAS Aruba IoT dataset...")
    events_df = pd.read_csv(events_path)
    subs_df = pd.read_csv(subs_path)
    logger.info(f"Loaded {len(events_df)} events, {len(subs_df)} subscriptions from CASAS Aruba")

    # Build subscriptions
    subscriptions = []
    for _, row in subs_df.iterrows():
        subscriptions.append(Subscription(
            id=row["id"],
            name=row["name"],
            description=row["description"],
        ))

    # Build events
    events = []
    for _, row in events_df.iterrows():
        events.append(Event(
            id=row["id"],
            text=row["text"],
            ground_truth=[row["ground_truth"]],  # single label
        ))

    if max_events and len(events) > max_events:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(events), max_events, replace=False)
        events = [events[i] for i in sorted(indices)]

    return Dataset(
        name="CASAS Aruba IoT (Smart Home ADL)",
        short_name="D3",
        events=events,
        subscriptions=subscriptions,
        metadata={
            "source": "zenodo:15708568 (Aruba subset)",
            "multi_label": False,
            "note": "Each event is a sensor sequence from one annotated activity segment",
        },
    )
