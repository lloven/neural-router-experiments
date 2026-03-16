#!/usr/bin/env python3
"""Download and verify all three benchmark datasets (D1, D2, D3).

Downloads datasets via HuggingFace (D1, D2) and Zenodo (D3) into the cache
directory, then prints a summary of each (event count, subscription count,
label statistics). Use this script to pre-populate the data cache before
running experiments.

Usage:
    python scripts/fetch_datasets.py [--cache-dir data]
    python scripts/fetch_datasets.py --max-events 100  # quick verification
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_dataset_by_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, default="data")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Limit events per dataset (for quick verification)")
    args = parser.parse_args()

    datasets_to_load = ["D1", "D2", "D3"]

    for ds_name in datasets_to_load:
        logger.info(f"\n{'='*50}")
        logger.info(f"Loading {ds_name}...")
        logger.info(f"{'='*50}")

        try:
            dataset = load_dataset_by_name(
                ds_name,
                cache_dir=args.cache_dir,
                max_events=args.max_events,
            )
            print(dataset.summary())
            print(f"  Subscriptions: {[s.id for s in dataset.subscriptions[:5]]}...")
            print(f"  First event: {dataset.events[0].text[:100]}...")
            print(f"  First event labels: {dataset.events[0].ground_truth}")
            print()
        except Exception as e:
            logger.error(f"Failed to load {ds_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
