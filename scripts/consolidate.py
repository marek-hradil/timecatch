"""Consolidate outputs/ timestamped dirs into flat results/{model}/ CSV files.

For each (experiment, dataset, model) triple, picks the latest outputs/ run
and copies it to results/{model}/{experiment}_{dataset}.csv.

Usage:
    python scripts/consolidate.py [--outputs-dir outputs] [--results-dir results]
"""
import argparse
import re
import shutil
from pathlib import Path

# Known model slugs — used to split the dir name into (experiment, model)
MODELS = [
    "qwen2-5-vl-7b",
    "qwen3-vl-8b",
    "qwen3-vl-8b-video",
    "intern-vl-3",
    "intern-vl",
    "molmo-7b",
]

# Pattern: {experiment}_{model}_{timestamp}
# The timestamp is always YYYY-MM-DD_HH-MM-SS at the end
TS_RE = re.compile(r"_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$")


def parse_dir_name(name: str):
    m = TS_RE.search(name)
    if not m:
        return None, None, None
    timestamp = m.group(1)
    prefix = name[: m.start()]  # everything before the timestamp
    # Try to match a known model suffix
    model = None
    experiment = prefix
    for slug in MODELS:
        if prefix.endswith("_" + slug):
            model = slug
            experiment = prefix[: -(len(slug) + 1)]
            break
    return experiment, model, timestamp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    outputs = Path(args.outputs_dir)
    results_root = Path(args.results_dir)

    # Group dirs by (experiment, model) → [(timestamp, path), ...]
    groups: dict[tuple, list] = {}
    for d in sorted(outputs.iterdir()):
        if not d.is_dir():
            continue
        csv = d / "results.csv"
        if not csv.exists():
            continue
        experiment, model, timestamp = parse_dir_name(d.name)
        if model is None:
            continue
        key = (experiment, model)
        groups.setdefault(key, []).append((timestamp, csv))

    copied = 0
    for (experiment, model), runs in sorted(groups.items()):
        # Pick the latest run
        runs.sort(key=lambda x: x[0])
        latest_ts, latest_csv = runs[-1]

        # Skip tiny/failed runs (≤2 rows means only header or 1 row)
        lines = sum(1 for _ in latest_csv.open())
        if lines <= 2:
            continue

        dest_dir = results_root / model
        dest_dir.mkdir(exist_ok=True)
        dest = dest_dir / f"{experiment}.csv"

        # Only overwrite if the source is newer or destination doesn't exist
        if dest.exists():
            src_lines = lines
            dst_lines = sum(1 for _ in dest.open())
            if dst_lines >= src_lines:
                continue  # already up to date or better

        shutil.copy2(latest_csv, dest)
        print(f"  {model}/{experiment}.csv  ({lines-1} rows, ts={latest_ts})")
        copied += 1

    print(f"\n{copied} files updated.")


if __name__ == "__main__":
    main()
