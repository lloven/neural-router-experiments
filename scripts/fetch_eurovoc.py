#!/usr/bin/env python3
"""Fetch EUROVOC level-2 concept descriptions for the D2 dataset.

The MultiEURLEX dataset uses numeric EUROVOC concept IDs as labels (e.g.,
"100163"). This script maps those IDs to human-readable English descriptions
(e.g., "political framework") needed for LLM-based semantic matching.

Strategy:
  1. Load the MultiEURLEX dataset to enumerate the 127 level-2 label IDs.
  2. Attempt to fetch English prefLabels from the EUROVOC RDF endpoint
     (each concept at https://eurovoc.europa.eu/<id> returns RDF/XML
     with skos:prefLabel elements).
  3. If the API fails for too many early requests (>3 of the first 10),
     fall back to the hardcoded FALLBACK_MAPPING dictionary below.
  4. Write the final mapping to data/eurovoc_labels.csv (columns: id, description).

Output: data/eurovoc_labels.csv

Usage:
    python scripts/fetch_eurovoc.py
"""

from __future__ import annotations

import csv
import logging
import re
import time
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Project root (one level up from scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "eurovoc_labels.csv"

# ---------------------------------------------------------------------------
# Hardcoded fallback: all 127 EUROVOC level-2 concept descriptions.
# Extracted from https://eurovoc.europa.eu/<id> RDF skos:prefLabel (en),
# with the numeric code prefix stripped (e.g., "0406 political framework"
# becomes "political framework").
# ---------------------------------------------------------------------------
FALLBACK_MAPPING: dict[str, str] = {
    "100163": "political framework",
    "100164": "political party",
    "100165": "electoral procedure and voting",
    "100166": "parliament",
    "100167": "parliamentary proceedings",
    "100168": "politics and public safety",
    "100169": "executive power and public service",
    "100170": "international affairs",
    "100171": "cooperation policy",
    "100172": "international security",
    "100173": "defence",
    "100174": "EU institutions and European civil service",
    "100175": "European Union law",
    "100176": "European construction",
    "100177": "EU finance",
    "100178": "sources and branches of the law",
    "100179": "civil law",
    "100180": "criminal law",
    "100181": "justice",
    "100182": "organisation of the legal system",
    "100183": "international law",
    "100184": "rights and freedoms",
    "100185": "economic policy",
    "100186": "economic conditions",
    "100187": "regions and regional policy",
    "100188": "economic structure",
    "100189": "national accounts",
    "100190": "economic analysis",
    "100191": "trade policy",
    "100192": "tariff policy",
    "100193": "trade",
    "100194": "international trade",
    "100195": "consumption",
    "100196": "marketing",
    "100197": "distributive trades",
    "100198": "monetary relations",
    "100199": "monetary economics",
    "100200": "financial institutions and credit",
    "100201": "free movement of capital",
    "100202": "financing and investment",
    "100203": "insurance",
    "100204": "public finance and budget policy",
    "100205": "budget",
    "100206": "taxation",
    "100207": "prices",
    "100208": "family",
    "100209": "migration",
    "100210": "demography and population",
    "100211": "social framework",
    "100212": "social affairs",
    "100213": "culture and religion",
    "100214": "social protection",
    "100215": "health",
    "100216": "construction and town planning",
    "100217": "education",
    "100218": "teaching",
    "100219": "organisation of teaching",
    "100220": "documentation",
    "100221": "communications",
    "100222": "information and information processing",
    "100223": "information technology and data processing",
    "100224": "natural and applied sciences",
    "100225": "humanities",
    "100226": "business organisation",
    "100227": "business classification",
    "100228": "legal form of organisations",
    "100229": "management",
    "100230": "accounting",
    "100231": "competition",
    "100232": "employment",
    "100233": "labour market",
    "100234": "organisation of work and working conditions",
    "100235": "personnel management and staff remuneration",
    "100236": "labour law and labour relations",
    "100237": "transport policy",
    "100238": "organisation of transport",
    "100239": "land transport",
    "100240": "maritime and inland waterway transport",
    "100241": "air and space transport",
    "100242": "environmental policy",
    "100243": "natural environment",
    "100244": "deterioration of the environment",
    "100245": "agricultural policy",
    "100246": "agricultural structures and production",
    "100247": "farming systems",
    "100248": "cultivation of agricultural land",
    "100249": "means of agricultural production",
    "100250": "agricultural activity",
    "100251": "forestry",
    "100252": "fisheries",
    "100253": "plant product",
    "100254": "animal product",
    "100255": "processed agricultural produce",
    "100256": "beverages and sugar",
    "100257": "foodstuff",
    "100258": "agri-foodstuffs",
    "100259": "food technology",
    "100260": "production",
    "100261": "technology and technical regulations",
    "100262": "research and intellectual property",
    "100263": "energy policy",
    "100264": "coal and mining industries",
    "100265": "oil industry",
    "100266": "electrical and nuclear industries",
    "100267": "soft energy",
    "100268": "industrial structures and policy",
    "100269": "chemistry",
    "100270": "iron, steel and other metal industries",
    "100271": "mechanical engineering",
    "100272": "electronics and electrical engineering",
    "100273": "building and public works",
    "100274": "wood industry",
    "100275": "leather and textile industries",
    "100276": "miscellaneous industries",
    "100277": "Europe",
    "100278": "regions of EU Member States",
    "100279": "America",
    "100280": "Africa",
    "100281": "Asia and Oceania",
    "100282": "economic geography",
    "100283": "political geography",
    "100284": "overseas countries and territories",
    "100285": "United Nations",
    "100286": "European organisations",
    "100287": "extra-European organisations",
    "100288": "world organisations",
    "100289": "non-governmental organisations",
}


def get_label_ids_from_dataset() -> list[str]:
    """Load the MultiEURLEX dataset and extract the level-2 label IDs.

    Returns:
        List of EUROVOC concept ID strings (e.g., ["100163", "100164", ...]).
    """
    from datasets import load_dataset

    logger.info("Loading MultiEURLEX to get level-2 label IDs...")
    ds = load_dataset(
        "nlpaueb/multi_eurlex",
        "en",
        label_level="level_2",
        cache_dir=str(DATA_DIR),
        trust_remote_code=True,
    )
    label_feature = ds["train"].features["labels"].feature
    label_ids = label_feature.names
    logger.info(f"Found {len(label_ids)} level-2 labels")
    return label_ids


def fetch_label_from_rdf(concept_id: str, timeout: int = 15) -> str | None:
    """Fetch the English prefLabel for a EUROVOC concept from its RDF page.

    The EUROVOC site returns RDF/XML with skos:prefLabel elements. The
    English label typically has a numeric code prefix (e.g., "0406 political
    framework") which is stripped.

    Args:
        concept_id: Numeric EUROVOC concept ID (e.g., "100163").
        timeout: HTTP request timeout in seconds.

    Returns:
        Cleaned English label string, or None on failure.
    """
    url = f"https://eurovoc.europa.eu/{concept_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8")
        matches = re.findall(
            r'<skos:prefLabel xml:lang="en">(.*?)</skos:prefLabel>', data
        )
        if matches:
            # Strip the numeric code prefix (e.g., "0406 " -> "")
            return re.sub(r"^\d+\s*", "", matches[0]).strip()
    except Exception as e:
        logger.debug(f"RDF fetch failed for {concept_id}: {e}")
    return None


def fetch_all_labels(label_ids: list[str]) -> tuple[dict[str, str], int]:
    """Fetch descriptions for all label IDs, with fallback to hardcoded mapping.

    Strategy:
      1. Try the EUROVOC RDF endpoint for each concept ID.
      2. If the endpoint fails for more than 3 of the first 10 requests,
         abandon the API and use the hardcoded fallback for all remaining.
      3. Any individual IDs missed by the API are filled from FALLBACK_MAPPING.

    Args:
        label_ids: List of EUROVOC concept ID strings.

    Returns:
        Tuple of (descriptions dict mapping ID to text, count fetched from API).
    """
    results: dict[str, str] = {}
    api_available = True
    failed_count = 0

    logger.info("Fetching labels from EUROVOC RDF endpoint...")
    for i, cid in enumerate(label_ids):
        if not api_available:
            break

        label = fetch_label_from_rdf(cid)
        if label:
            results[cid] = label
            logger.info(f"  [{i + 1}/{len(label_ids)}] {cid} -> {label}")
        else:
            failed_count += 1
            logger.warning(f"  [{i + 1}/{len(label_ids)}] {cid} -> FAILED")
            if i < 10 and failed_count > 3:
                logger.warning("Too many API failures, switching to fallback mapping")
                api_available = False

        # Rate-limit: short pause every 20 requests
        if i % 20 == 19:
            time.sleep(0.5)

    from_api = len(results)

    # Fill in any missing IDs from the fallback mapping
    missing = [cid for cid in label_ids if cid not in results]
    if missing:
        logger.info(f"Using fallback mapping for {len(missing)} missing IDs")
        for cid in missing:
            if cid in FALLBACK_MAPPING:
                results[cid] = FALLBACK_MAPPING[cid]
            else:
                results[cid] = f"EUROVOC concept {cid}"
                logger.warning(f"No description available for {cid}")

    return results, from_api


def save_csv(label_ids: list[str], descriptions: dict[str, str]) -> None:
    """Save the id-to-description mapping to CSV, preserving dataset label order.

    Args:
        label_ids: Ordered list of concept IDs (determines CSV row order).
        descriptions: Mapping from concept ID to textual description.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "description"])
        for cid in label_ids:
            writer.writerow([cid, descriptions.get(cid, cid)])
    logger.info(f"Saved {len(label_ids)} labels to {OUTPUT_PATH}")


def main() -> None:
    label_ids = get_label_ids_from_dataset()
    descriptions, from_api = fetch_all_labels(label_ids)
    save_csv(label_ids, descriptions)

    from_fallback = len(label_ids) - from_api
    logger.info(f"Done: {from_api} from API, {from_fallback} from fallback. "
                f"Total: {len(descriptions)} labels.")


if __name__ == "__main__":
    main()
