#!/usr/bin/env python3
"""Download and convert the CASAS smart home dataset for Neural Router.

Dataset:  CASAS Smart Home (Zenodo 15708568), home hh113
          Naturalistic free-living sensor data from a single-resident apartment.
          Ambient sensors recorded motion and door events; activities were
          annotated by human observers using begin/end markers in the raw stream.

          Home hh113 is one of the largest annotated homes in the CASAS collection,
          with ~2.3M sensor events and ~23K annotated activity segments across 34
          activity types over an extended period.

Design decisions (documented here for reviewer transparency):
------------------------------------------------------------------------
1. SUBSCRIPTION DESCRIPTIONS are GENERIC:  Each subscription describes the
   activity in natural language based solely on the activity name (e.g.,
   "Meal Preparation", "Sleeping").  The descriptions do NOT reference
   specific sensor IDs or the apartment layout; they would be equally valid
   in any smart-home deployment.

2. EVENT DESCRIPTIONS are TEMPLATED:  Every sensor reading is rendered with
   the same template:
       "At {time}, sensor in {location} reported {value}."
   Location names come from the CASAS sensor identifiers (e.g., "Bedroom",
   "Kitchen"), NOT from the activity label.  This prevents information
   leakage from the ground truth into the event text.

3. EVENT GROUPING:  Activity segments are delimited by "begin" and "end"
   markers in the CASAS annotation.  Each segment forms one Event whose
   ground truth is the annotated activity label.

4. ACTIVITY MERGING:  Fine-grained activity variants (e.g., Cook_Breakfast,
   Cook_Lunch, Cook_Dinner) are merged into coarser categories (e.g., Cook)
   to produce a manageable subscription set.  This mirrors realistic pub/sub
   where subscribers express interest at a functional level, not at the
   granularity of meal-specific cooking.

5. GROUND TRUTH is the CASAS dataset's own human annotation.

6. SAMPLING:  If the total number of activity segments exceeds --max-events,
   a stratified random sample is drawn (preserving the activity distribution)
   with seed=42 for reproducibility.

Source:   https://zenodo.org/records/15708568
License:  CC-BY-4.0
Citation: D.J. Cook, "Learning setting-generalized activity models for smart
          spaces," IEEE Intelligent Systems, 2012.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import zipfile
from collections import Counter, OrderedDict
from pathlib import Path

import urllib.request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ZENODO_RECORD = "15708568"
LABELED_DATA_URL = (
    f"https://zenodo.org/api/records/{ZENODO_RECORD}"
    "/files/labeled_data.zip/content"
)

# Default home to extract (hh113 is one of the richest annotated homes:
# ~2.3M sensor events, ~23K activity segments, 34 activity types)
DEFAULT_HOME = "hh113"

# Minimum activity instances to include (exclude very rare activities)
MIN_ACTIVITY_COUNT = 20

# Activity merging: map fine-grained activities to coarser categories.
# Activities not listed here are used as-is.
ACTIVITY_MERGE = {
    "Cook_Breakfast": "Cook",
    "Cook_Lunch": "Cook",
    "Cook_Dinner": "Cook",
    "Eat_Breakfast": "Eat",
    "Eat_Lunch": "Eat",
    "Eat_Dinner": "Eat",
    "Wash_Breakfast_Dishes": "Wash_Dishes",
    "Wash_Lunch_Dishes": "Wash_Dishes",
    "Wash_Dinner_Dishes": "Wash_Dishes",
    "Morning_Meds": "Take_Medicine",
    "Evening_Meds": "Take_Medicine",
    "Sleep_Out_Of_Bed": "Sleep",
    "Bed_Toilet_Transition": "Toilet",
    "Work_On_Computer": "Work",
    "Work_At_Table": "Work",
    "Step_Out": "Leave_Home",
}

# Generic subscription descriptions per merged activity.
ACTIVITY_SUBSCRIPTIONS = {
    "Sleep": (
        "Notify when the resident is sleeping or resting in bed, including "
        "going to bed, sleeping through the night, and waking up"
    ),
    "Toilet": (
        "Alert when the resident is using the toilet or bathroom for "
        "personal hygiene needs"
    ),
    "Personal_Hygiene": (
        "Report when the resident is performing personal hygiene activities "
        "such as brushing teeth, washing face, or similar grooming routines"
    ),
    "Bathe": (
        "Notify when the resident is bathing or showering, involving "
        "extended water use in the bathroom"
    ),
    "Groom": (
        "Alert when the resident is grooming, such as combing hair, "
        "applying cosmetics, or similar personal care"
    ),
    "Dress": (
        "Report when the resident is getting dressed or changing clothes, "
        "typically involving activity near the closet or bedroom"
    ),
    "Cook": (
        "Report when the resident is preparing food or cooking a meal, "
        "involving kitchen activity such as using a stove, opening cabinets, "
        "or handling ingredients"
    ),
    "Eat": (
        "Notify when the resident is consuming a meal or snack, typically "
        "involving activity at a dining area or kitchen"
    ),
    "Drink": (
        "Alert when the resident is having a drink or beverage, involving "
        "brief kitchen activity"
    ),
    "Wash_Dishes": (
        "Alert when the resident is washing dishes, involving water use "
        "and activity near the kitchen sink"
    ),
    "Work": (
        "Report when the resident is engaged in work activity, such as using "
        "a computer, reading documents, or performing tasks at a desk or table"
    ),
    "Relax": (
        "Notify when the resident is relaxing or engaged in leisure activity, "
        "such as watching television, reading, or sitting quietly"
    ),
    "Watch_TV": (
        "Alert when the resident is watching television, indicated by "
        "sustained presence in the living area"
    ),
    "Read": (
        "Report when the resident is reading a book, newspaper, or similar "
        "material, involving quiet sustained activity"
    ),
    "Phone": (
        "Notify when the resident is making or receiving a telephone call, "
        "including picking up, talking on, or hanging up the phone"
    ),
    "Leave_Home": (
        "Report when the resident leaves the home, indicated by exit "
        "through the front door and absence of indoor activity"
    ),
    "Enter_Home": (
        "Report when the resident enters the home, indicated by the front "
        "door opening followed by indoor activity"
    ),
    "Entertain_Guests": (
        "Notify when the resident is entertaining guests or visitors, "
        "indicated by social activity and multiple presences"
    ),
    "Take_Medicine": (
        "Alert when the resident is taking medication, a brief routine "
        "activity typically in the kitchen or bathroom"
    ),
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_file(url: str, cache_path: Path) -> None:
    """Download a file from URL to cache_path if not already cached."""
    if cache_path.exists():
        logger.info(f"Using cached {cache_path}")
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, cache_path)
    logger.info(
        f"Saved to {cache_path} ({cache_path.stat().st_size / 1e6:.1f} MB)"
    )


def _extract_home_from_zip(zip_path: Path, home_id: str) -> str | None:
    """Extract a specific home's data file from the labeled_data.zip archive.

    Args:
        zip_path: Path to the labeled_data.zip archive.
        home_id: Home identifier (e.g., "hh113").

    Returns:
        The raw text content of the data file, or None if not found.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Look for the home's CSV file
        target_csv = f"labeled/{home_id}.csv"
        target_txt = f"labeled/{home_id}.txt"

        for target in [target_csv, target_txt]:
            if target in zf.namelist():
                info = zf.getinfo(target)
                logger.info(
                    f"Extracting {target} "
                    f"({info.file_size / 1e6:.1f} MB)"
                )
                with zf.open(target) as f:
                    return f.read().decode("utf-8", errors="replace")

        logger.error(f"Home {home_id} not found in {zip_path}")
        # Show available homes
        homes = sorted(
            set(
                n.split("/")[1].split(".")[0]
                for n in zf.namelist()
                if "/" in n and not n.endswith("/")
            )
        )
        logger.info(f"Available homes: {homes[:20]}...")
        return None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_casas_csv(raw_text: str) -> list[dict]:
    """Parse CASAS CSV sensor data with activity annotations.

    Expected format (comma-separated):
        2011-06-15,01:31:23.155386,Bedroom,ON,Sleep="begin"
        2011-06-15,01:31:25.011146,Bedroom,OFF
        ...

    Activity annotations appear as an optional 5th field:
        Activity="begin" or Activity="end"

    Returns:
        List of dicts: date, time, sensor_location, sensor_value,
        activity (optional), marker (optional: "begin" or "end")
    """
    records = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        fields = line.split(",")
        if len(fields) < 4:
            continue

        record = {
            "date": fields[0],
            "time": fields[1],
            "sensor_location": fields[2],
            "sensor_value": fields[3],
        }

        # Activity annotation (5th field, format: Activity="begin")
        if len(fields) >= 5:
            annotation = fields[4].strip()
            # Parse Activity="begin" or Activity="end"
            match = re.match(r'(\w+)="(begin|end)"', annotation)
            if match:
                record["activity"] = match.group(1)
                record["marker"] = match.group(2)

        records.append(record)

    return records


def _segment_activities(
    records: list[dict],
    merge_map: dict[str, str] | None = None,
) -> list[dict]:
    """Segment the continuous sensor stream into activity instances.

    Uses begin/end markers to delimit activity segments. Each segment
    contains all sensor events between the "begin" and "end" markers.

    Args:
        records: Parsed CASAS sensor records.
        merge_map: Optional dict mapping fine-grained activities to coarser
            categories (e.g., Cook_Breakfast -> Cook).

    Returns:
        List of segment dicts: activity, readings, start_time, end_time,
        segment_id, original_activity
    """
    merge_map = merge_map or {}
    segments = []
    active_segments: dict[str, dict] = {}  # raw_activity -> segment
    segment_counter = 0

    for record in records:
        marker = record.get("marker")
        raw_activity = record.get("activity")

        if marker == "begin" and raw_activity:
            # Close any previous unclosed segment for this activity
            if raw_activity in active_segments:
                prev = active_segments[raw_activity]
                prev["end_time"] = record["date"] + " " + record["time"]
                segments.append(prev)

            segment_counter += 1
            merged = merge_map.get(raw_activity, raw_activity)
            active_segments[raw_activity] = {
                "activity": merged,
                "original_activity": raw_activity,
                "readings": [record],
                "start_time": record["date"] + " " + record["time"],
                "end_time": None,
                "segment_id": f"seg_{segment_counter:05d}",
            }

        elif marker == "end" and raw_activity:
            if raw_activity in active_segments:
                seg = active_segments.pop(raw_activity)
                seg["readings"].append(record)
                seg["end_time"] = record["date"] + " " + record["time"]
                segments.append(seg)

        else:
            # Regular sensor event: add to all currently active segments
            for seg in active_segments.values():
                seg["readings"].append(record)

    # Close remaining open segments
    for raw_activity, seg in active_segments.items():
        if seg["readings"]:
            seg["end_time"] = (
                seg["readings"][-1]["date"] + " " + seg["readings"][-1]["time"]
            )
            segments.append(seg)

    return segments


# ---------------------------------------------------------------------------
# Neural Router format conversion
# ---------------------------------------------------------------------------

def _render_event_text(readings: list[dict], max_readings: int = 50) -> str:
    """Convert sensor readings to natural-language event text.

    Uses a fixed template for ALL events (no activity-specific wording).
    Location names come from CASAS sensor identifiers, not from the label.
    """
    if len(readings) > max_readings:
        readings = readings[:max_readings]

    lines = []
    for r in readings:
        location = r["sensor_location"]
        lines.append(
            f"At {r['time']}, sensor in {location} reported {r['sensor_value']}."
        )
    return " ".join(lines)


def convert_to_neural_router_format(
    segments: list[dict],
    max_readings_per_event: int = 50,
    max_events: int | None = None,
    min_readings: int = 3,
) -> tuple[list[dict], list[dict]]:
    """Convert activity segments to Neural-Router events and subscriptions.

    Args:
        segments: Activity segments from _segment_activities().
        max_readings_per_event: Cap on sensor readings per event text.
        max_events: If set, stratified random sample to this many events.
        min_readings: Minimum sensor readings for a valid segment.

    Returns:
        (events_list, subscriptions_list)
    """
    # Filter short segments
    valid = [s for s in segments if len(s["readings"]) >= min_readings]
    logger.info(
        f"Valid segments (>={min_readings} readings): "
        f"{len(valid)} of {len(segments)}"
    )

    # Count activities and filter rare ones
    activity_counts = Counter(s["activity"] for s in valid)
    logger.info("Activity counts before filtering:")
    for act, count in activity_counts.most_common():
        logger.info(f"  {act}: {count}")

    valid_activities = {
        act
        for act, count in activity_counts.items()
        if count >= MIN_ACTIVITY_COUNT
    }
    valid = [s for s in valid if s["activity"] in valid_activities]

    excluded = set(activity_counts.keys()) - valid_activities
    if excluded:
        logger.info(
            f"Excluded rare activities (<{MIN_ACTIVITY_COUNT}): {excluded}"
        )

    # Build subscriptions
    subscriptions = []
    for activity in sorted(valid_activities):
        description = ACTIVITY_SUBSCRIPTIONS.get(activity)
        if description is None:
            readable = activity.replace("_", " ").lower()
            description = f"Notify when the resident is {readable}"
            logger.warning(
                f"No handcrafted subscription for {activity}, using fallback"
            )
        subscriptions.append({
            "id": activity,
            "name": activity.replace("_", " "),
            "description": description,
        })

    # Stratified sampling
    if max_events and len(valid) > max_events:
        import numpy as np

        rng = np.random.RandomState(42)
        by_activity: dict[str, list[int]] = {}
        for idx, seg in enumerate(valid):
            by_activity.setdefault(seg["activity"], []).append(idx)

        selected_indices: list[int] = []
        for act, indices in by_activity.items():
            n_select = max(
                1, int(max_events * len(indices) / len(valid))
            )
            n_select = min(n_select, len(indices))
            selected = rng.choice(indices, n_select, replace=False).tolist()
            selected_indices.extend(selected)

        # Fill up if short
        remaining = set(range(len(valid))) - set(selected_indices)
        while len(selected_indices) < max_events and remaining:
            extra = rng.choice(list(remaining))
            selected_indices.append(extra)
            remaining.remove(extra)

        selected_indices.sort()
        valid = [valid[i] for i in selected_indices[:max_events]]
        logger.info(f"Stratified sample: {len(valid)} events")

    # Convert to events
    events = []
    for idx, seg in enumerate(valid):
        event_text = _render_event_text(
            seg["readings"], max_readings_per_event
        )
        events.append({
            "id": f"casas_{idx:05d}",
            "text": event_text,
            "ground_truth": seg["activity"],
            "segment_id": seg["segment_id"],
            "start_time": seg["start_time"],
            "duration_readings": len(seg["readings"]),
        })

    return events, subscriptions


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_neural_router_csv(
    events: list[dict],
    subscriptions: list[dict],
    output_path: Path,
) -> None:
    """Save in the format expected by src/data.py."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    event_fields = [
        "id", "text", "ground_truth", "segment_id",
        "start_time", "duration_readings",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=event_fields)
        writer.writeheader()
        writer.writerows(events)
    logger.info(f"Saved {len(events)} events to {output_path}")

    subs_path = output_path.parent / (output_path.stem + "_subscriptions.csv")
    with open(subs_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "description"])
        writer.writeheader()
        writer.writerows(subscriptions)
    logger.info(f"Saved {len(subscriptions)} subscriptions to {subs_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download CASAS smart home dataset and convert to "
            "Neural Router format"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/casas",
        help="Directory for raw downloads (default: data/casas)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/casas_iot.csv",
        help="Output CSV path (default: data/casas_iot.csv)",
    )
    parser.add_argument(
        "--home",
        type=str,
        default=DEFAULT_HOME,
        help=f"CASAS home ID to extract (default: {DEFAULT_HOME})",
    )
    parser.add_argument(
        "--max-readings",
        type=int,
        default=50,
        help="Max sensor readings per event sequence (default: 50)",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Max total events via stratified sampling (default: all)",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output_path = Path(args.output)

    # Download
    labeled_zip = cache_dir / "labeled_data.zip"
    _download_file(LABELED_DATA_URL, labeled_zip)

    # Extract home data
    logger.info(f"Extracting home {args.home} from archive...")
    raw_text = _extract_home_from_zip(labeled_zip, args.home)
    if raw_text is None:
        return

    # Parse
    logger.info("Parsing sensor data...")
    records = _parse_casas_csv(raw_text)
    logger.info(f"Parsed {len(records)} sensor readings")

    # Segment
    logger.info("Segmenting into activity instances...")
    segments = _segment_activities(records, merge_map=ACTIVITY_MERGE)
    logger.info(f"Found {len(segments)} activity segments")

    if not segments:
        logger.error("No activity segments found.")
        lines = raw_text.splitlines()[:10]
        logger.info("First 10 lines:")
        for line in lines:
            logger.info(f"  {line}")
        return

    # Convert
    events, subscriptions = convert_to_neural_router_format(
        segments,
        max_readings_per_event=args.max_readings,
        max_events=args.max_events,
    )

    # Summary
    activity_dist = Counter(e["ground_truth"] for e in events)
    logger.info(
        f"Final dataset: {len(events)} events, "
        f"{len(subscriptions)} subscriptions"
    )
    logger.info("Activity distribution:")
    for act, count in activity_dist.most_common():
        logger.info(f"  {act}: {count}")

    # Save
    save_neural_router_csv(events, subscriptions, output_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
