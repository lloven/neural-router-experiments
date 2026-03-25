#!/usr/bin/env python3
"""analyze_blackbox.py — Post-incident analyzer for blackbox recorder data.

Reads JSONL output from blackbox.sh and produces a diagnostic report:
- Timeline of load, NFS ops, Python process count, GPU memory
- Correlation analysis: did NFS spike before load, or load before NFS?
- Python ppid analysis: who spawned the Python processes?
- Annotated events: load thresholds crossed, first Python process, etc.

Usage:
    python3 analyze_blackbox.py /var/nvme-cache/lloven/blackbox.jsonl
    python3 analyze_blackbox.py --threshold 30 blackbox.jsonl
"""
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


LOAD_THRESHOLD = 30.0  # default, matches watchdog


def parse_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file into a list of sample dicts."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def find_events(samples: list[dict], threshold: float) -> list[dict]:
    """Identify notable events in the timeline."""
    events = []
    first_python = None
    load_crossed = None
    prev_load = 0.0

    for i, s in enumerate(samples):
        ts = s["ts"]
        load = s.get("load_1m", 0)

        # First Python process appearance
        if first_python is None and s.get("python_count", 0) > 0:
            first_python = i
            delta = _seconds_since_start(samples, i)
            events.append({
                "ts": ts,
                "type": "first_python",
                "detail": f"First Python process appeared at T+{delta}s (count={s['python_count']})",
            })

        # Load crossed threshold
        if load_crossed is None and load >= threshold and prev_load < threshold:
            load_crossed = i
            delta = _seconds_since_start(samples, i)
            events.append({
                "ts": ts,
                "type": "load_spike",
                "detail": f"Load crossed {threshold} at T+{delta}s (load={load})",
            })

        # Load exceeded specific thresholds
        if prev_load < 10 and load >= 10:
            delta = _seconds_since_start(samples, i)
            events.append({
                "ts": ts,
                "type": "load_exceeded_10",
                "detail": f"Load exceeded 10 at T+{delta}s (load={load})",
            })

        prev_load = load

    # Peak load
    if samples:
        peak_idx = max(range(len(samples)), key=lambda i: samples[i].get("load_1m", 0))
        peak = samples[peak_idx]
        delta = _seconds_since_start(samples, peak_idx)
        events.append({
            "ts": peak["ts"],
            "type": "peak_load",
            "detail": f"Peak load={peak['load_1m']} at T+{delta}s",
        })

    return events


def _seconds_since_start(samples: list[dict], idx: int) -> float:
    """Compute seconds between first sample and sample at idx."""
    if idx == 0 or not samples:
        return 0.0
    try:
        t0 = datetime.fromisoformat(samples[0]["ts"])
        ti = datetime.fromisoformat(samples[idx]["ts"])
        return (ti - t0).total_seconds()
    except (ValueError, KeyError):
        return idx * 2.0  # fallback: assume 2s intervals


def analyze_correlation(samples: list[dict]) -> dict:
    """Analyze whether NFS ops spike before or after load spike."""
    if len(samples) < 3:
        return {"result": "insufficient data"}

    # Find first significant NFS increase (>50% jump)
    nfs_spike_idx = None
    for i in range(1, len(samples)):
        prev_nfs = samples[i - 1].get("nfs_ops", 0)
        curr_nfs = samples[i].get("nfs_ops", 0)
        if prev_nfs > 0 and curr_nfs > prev_nfs * 1.5:
            nfs_spike_idx = i
            break

    # Find first significant load increase (>5x or >5.0)
    load_spike_idx = None
    for i in range(1, len(samples)):
        prev_load = samples[i - 1].get("load_1m", 0)
        curr_load = samples[i].get("load_1m", 0)
        if curr_load > 5.0 and (prev_load == 0 or curr_load > prev_load * 2):
            load_spike_idx = i
            break

    if nfs_spike_idx is not None and load_spike_idx is not None:
        if nfs_spike_idx < load_spike_idx:
            return {"result": "NFS spike preceded load spike", "nfs_first": True,
                    "nfs_idx": nfs_spike_idx, "load_idx": load_spike_idx}
        elif load_spike_idx < nfs_spike_idx:
            return {"result": "Load spike preceded NFS spike", "nfs_first": False,
                    "nfs_idx": nfs_spike_idx, "load_idx": load_spike_idx}
        else:
            return {"result": "NFS and load spiked simultaneously",
                    "nfs_idx": nfs_spike_idx, "load_idx": load_spike_idx}

    return {"result": "Could not determine correlation",
            "nfs_spike_idx": nfs_spike_idx, "load_spike_idx": load_spike_idx}


def analyze_ppids(samples: list[dict]) -> dict:
    """Analyze parent PIDs of Python processes across all samples."""
    all_ppids: list[int] = []
    for s in samples:
        ppids = s.get("python_ppids", [])
        if isinstance(ppids, list):
            all_ppids.extend(ppids)

    if not all_ppids:
        return {"unique_parents": 0, "counts": {}}

    counts = Counter(all_ppids)
    return {
        "unique_parents": len(counts),
        "counts": dict(counts.most_common()),
    }


def format_report(samples: list[dict], threshold: float) -> str:
    """Generate a human-readable diagnostic report."""
    lines = []
    lines.append("=" * 60)
    lines.append("BLACK BOX RECORDER — INCIDENT ANALYSIS")
    lines.append("=" * 60)

    if not samples:
        lines.append("No samples found.")
        return "\n".join(lines)

    # Summary
    lines.append(f"\nSamples: {len(samples)}")
    lines.append(f"Time range: {samples[0]['ts']} to {samples[-1]['ts']}")
    duration = _seconds_since_start(samples, len(samples) - 1)
    lines.append(f"Duration: {duration:.0f}s")

    # Load timeline
    loads = [s.get("load_1m", 0) for s in samples]
    lines.append(f"\n--- LOAD ---")
    lines.append(f"Min: {min(loads):.2f}  Max: {max(loads):.2f}  Final: {loads[-1]:.2f}")

    # Python processes
    py_counts = [s.get("python_count", 0) for s in samples]
    lines.append(f"\n--- PYTHON PROCESSES ---")
    lines.append(f"Max count: {max(py_counts)}")
    ppid_info = analyze_ppids(samples)
    if ppid_info["unique_parents"] > 0:
        lines.append(f"Unique parent PIDs (ppid): {ppid_info['unique_parents']}")
        for ppid, count in ppid_info["counts"].items():
            lines.append(f"  ppid {ppid}: spawned {count} Python process samples")
    else:
        lines.append("No Python processes observed.")

    # NFS ops
    nfs_ops = [s.get("nfs_ops", 0) for s in samples]
    lines.append(f"\n--- NFS OPS ---")
    lines.append(f"Min: {min(nfs_ops)}  Max: {max(nfs_ops)}  Final: {nfs_ops[-1]}")

    # Correlation
    corr = analyze_correlation(samples)
    lines.append(f"\n--- NFS / LOAD CORRELATION ---")
    lines.append(f"Result: {corr['result']}")

    # GPU
    gpu_mems = [s.get("gpu_mem_mb", 0) for s in samples]
    gpu_utils = [s.get("gpu_util_pct", 0) for s in samples]
    lines.append(f"\n--- GPU ---")
    lines.append(f"Memory: min={min(gpu_mems)} MB, max={max(gpu_mems)} MB")
    lines.append(f"Utilization: min={min(gpu_utils)}%, max={max(gpu_utils)}%")

    # Events
    events = find_events(samples, threshold)
    lines.append(f"\n--- EVENTS ---")
    for e in events:
        lines.append(f"[{e['ts']}] {e['detail']}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze blackbox recorder JSONL data")
    parser.add_argument("jsonl_file", help="Path to blackbox.jsonl")
    parser.add_argument("--threshold", type=float, default=LOAD_THRESHOLD,
                        help=f"Load threshold for spike detection (default: {LOAD_THRESHOLD})")
    args = parser.parse_args()

    path = Path(args.jsonl_file)
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)

    samples = parse_jsonl(path)
    report = format_report(samples, args.threshold)
    print(report)


if __name__ == "__main__":
    main()
