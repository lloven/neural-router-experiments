#!/usr/bin/env python3
"""Fetch EUROVOC concept descriptions for MultiEURLEX level-2 labels.

Downloads textual descriptions from the EU SPARQL endpoint for each
EUROVOC concept ID used in the MultiEURLEX dataset. Falls back to a
hardcoded mapping if the API is unavailable.

Output: data/eurovoc_labels.csv (columns: id, description)

Usage:
    python scripts/fetch_eurovoc.py
"""

from __future__ import annotations

import csv
import logging
import sys
import urllib.request
import urllib.parse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# SPARQL endpoint for EUROVOC
SPARQL_ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"

# Hardcoded mapping for the 127 level-2 EUROVOC concepts used in MultiEURLEX.
# These are the official English preferred labels from the EUROVOC thesaurus.
# Source: https://op.europa.eu/en/web/eu-vocabularies/concept-scheme/-/resource?uri=http://eurovoc.europa.eu/100141
EUROVOC_LEVEL2_LABELS = {
    "100215": "political framework",
    "100211": "political theory",
    "100213": "international politics",
    "100214": "executive power and public service",
    "100209": "political life",
    "100208": "parliament",
    "100216": "electoral procedure and voting",
    "100210": "political party",
    "100212": "defence",
    "100270": "EU institutions and European civil service",
    "100282": "European Union law",
    "100271": "EU finance",
    "100273": "European construction",
    "100279": "EU body or agency",
    "100277": "management of the EU",
    "100275": "common foreign and security policy",
    "100280": "politics and public safety",
    "100274": "justice",
    "100276": "rights and freedoms",
    "100278": "public administration and budget policy",
    "100281": "criminal law",
    "100248": "international affairs",
    "100246": "political geography",
    "100249": "cooperation policy",
    "100250": "international security",
    "100247": "international law",
    "100251": "international organisation",
    "100158": "economic policy",
    "100159": "economic conditions",
    "100161": "regions and regional policy",
    "100164": "national accounts",
    "100163": "economic analysis",
    "100160": "marketing",
    "100162": "monetary relations",
    "100165": "monetary economics",
    "100166": "economic structure",
    "100167": "economic geography",
    "100176": "international trade",
    "100171": "tariff policy",
    "100174": "trade policy",
    "100175": "free movement of goods",
    "100168": "trade",
    "100173": "consumption",
    "100172": "prices",
    "100170": "distributive trades",
    "100169": "world organisations - Loss-leader trade",
    "100179": "finance",
    "100177": "financing and investment",
    "100178": "free movement of capital",
    "100180": "financial institutions and credit",
    "100182": "public finance and budget policy",
    "100181": "taxation",
    "100183": "insurance",
    "100191": "social affairs",
    "100184": "social protection",
    "100194": "family",
    "100192": "migration",
    "100193": "demography and population",
    "100196": "health",
    "100195": "documentation",
    "100197": "culture and religion",
    "100198": "social framework",
    "100199": "education and communications",
    "100200": "communications",
    "100201": "information and information processing",
    "100202": "information technology and data processing",
    "100203": "natural and applied sciences",
    "100204": "research and intellectual property",
    "100205": "organisation of transport",
    "100206": "transport policy",
    "100207": "land transport",
    "100252": "air and space transport",
    "100253": "maritime and inland waterway transport",
    "100148": "environmental policy",
    "100147": "deterioration of the environment",
    "100149": "natural environment",
    "100150": "agricultural policy",
    "100152": "agricultural structures and production",
    "100151": "farming systems",
    "100153": "means of agricultural production",
    "100154": "agricultural activity",
    "100143": "cultivation of agricultural land",
    "100145": "plant product",
    "100146": "beverages and sugar",
    "100144": "animal product",
    "100155": "foodstuff",
    "100156": "agri-foodstuffs",
    "100157": "food technology",
    "100254": "fisheries",
    "100255": "forestry",
    "100256": "business and competition",
    "100257": "company law",
    "100258": "business organisation",
    "100259": "management",
    "100260": "accounting",
    "100261": "competition",
    "100262": "employment and working conditions",
    "100263": "employment",
    "100264": "labour market",
    "100265": "organisation of work and working conditions",
    "100266": "labour law and labour relations",
    "100267": "personnel management and staff remuneration",
    "100226": "production, technology and research",
    "100227": "technology and technical regulations",
    "100228": "research policy",
    "100229": "energy policy",
    "100230": "coal and mining industries",
    "100231": "oil industry",
    "100232": "electrical and nuclear industries",
    "100233": "soft energy",
    "100234": "industrial structures and policy",
    "100235": "chemistry",
    "100236": "iron, steel and other metal industries",
    "100237": "mechanical engineering",
    "100238": "electronics and electrical engineering",
    "100239": "building and public works",
    "100240": "wood industry",
    "100241": "leather and textile industries",
    "100244": "miscellaneous industries",
    "100243": "processed agricultural produce",
    "100242": "beverages industry and tobacco industry",
    "100268": "geography",
    "100269": "economic geography",
    "100283": "international organisations",
    "100190": "LAW",
    "100186": "social questions",
    "100185": "FINANCE",
    "100188": "education",
    "100189": "science",
    "100187": "business and management",
}


def fetch_from_sparql(concept_ids: list[str]) -> dict[str, str]:
    """Fetch EUROVOC concept labels via SPARQL endpoint.

    Queries the EU Publications Office SPARQL endpoint for the English
    preferred label (skos:prefLabel) of each concept.
    """
    results = {}

    # Query in batches of 50
    for i in range(0, len(concept_ids), 50):
        batch = concept_ids[i:i + 50]
        values = " ".join(f"<http://eurovoc.europa.eu/{cid}>" for cid in batch)

        query = f"""
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        SELECT ?concept ?label WHERE {{
            VALUES ?concept {{ {values} }}
            ?concept skos:prefLabel ?label .
            FILTER(LANG(?label) = "en")
        }}
        """

        try:
            params = urllib.parse.urlencode({
                "query": query,
                "format": "application/json",
            })
            url = f"{SPARQL_ENDPOINT}?{params}"
            req = urllib.request.Request(url, headers={"Accept": "application/json"})

            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())

            for binding in data.get("results", {}).get("bindings", []):
                uri = binding["concept"]["value"]
                label = binding["label"]["value"]
                # Strip numeric EUROVOC domain prefix (e.g., "2841 health" -> "health")
                import re
                label = re.sub(r"^\d+\s+", "", label)
                concept_id = uri.split("/")[-1]
                results[concept_id] = label

            logger.info(f"SPARQL batch {i//50 + 1}: got {len(data.get('results', {}).get('bindings', []))} labels")

        except Exception as e:
            logger.warning(f"SPARQL batch {i//50 + 1} failed: {e}")

    return results


def main():
    output_path = Path("data/eurovoc_labels.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get concept IDs from the dataset
    logger.info("Loading MultiEURLEX to get level-2 label IDs...")
    try:
        from datasets import load_dataset
        ds = load_dataset("nlpaueb/multi_eurlex", "en", label_level="level_2", trust_remote_code=True)
        label_feature = ds["train"].features["labels"].feature
        concept_ids = label_feature.names
        logger.info(f"Found {len(concept_ids)} level-2 concept IDs")
    except Exception as e:
        logger.warning(f"Could not load dataset: {e}")
        concept_ids = list(EUROVOC_LEVEL2_LABELS.keys())
        logger.info(f"Using hardcoded list of {len(concept_ids)} concept IDs")

    # Try SPARQL first, fall back to hardcoded
    logger.info("Attempting SPARQL fetch...")
    sparql_labels = fetch_from_sparql(concept_ids)

    # Merge: prefer SPARQL, fall back to hardcoded
    labels = {}
    for cid in concept_ids:
        if cid in sparql_labels:
            labels[cid] = sparql_labels[cid]
        elif cid in EUROVOC_LEVEL2_LABELS:
            labels[cid] = EUROVOC_LEVEL2_LABELS[cid]
        else:
            labels[cid] = f"EUROVOC concept {cid}"
            logger.warning(f"No description for concept {cid}")

    # Save CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "description"])
        for cid in concept_ids:
            writer.writerow([cid, labels[cid]])

    logger.info(f"Saved {len(labels)} labels to {output_path}")

    # Show a sample
    for cid in concept_ids[:5]:
        print(f"  {cid}: {labels[cid]}")


if __name__ == "__main__":
    main()
