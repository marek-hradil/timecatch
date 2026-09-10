#!/usr/bin/env python3
"""
Rank the gathered papers by review priority.

The `relevance` column in papers.csv is a raw 0-6 keyword score. This script
combines it with recency and provenance into a single priority score:

  priority = keyword_relevance        (0-6, as in papers.csv)
           + recency bonus            (up to +4: 2026 -> +4, 2025 -> +3,
                                        2024 -> +1, older -> 0)
           + seed-provenance bonus    (+3 if the paper cites one of our seed
                                        papers -- it lives in the same
                                        literature neighborhood)
           + log-scaled citations     (log10(1+c), max ~+4 for 10k cites)
           - classic penalty          (-2 if pre-2024 AND highly cited --
                                        foundational background, not new
                                        literature to review)

Usage:
  python prioritize.py            # top 30, reads papers.csv
  python prioritize.py -n 100     # top 100
  python prioritize.py --csv      # write full ranked table to papers_ranked.csv
"""

import argparse
import csv
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def priority(r: dict) -> float:
    score = float(r.get("relevance") or 0)

    # recency: the review targets *new* literature
    year = int(r.get("year") or 0)
    recency = {2026: 4, 2025: 3, 2024: 1}.get(year, 0)
    score += recency

    # provenance: pulled in via citation tracing from a seed
    if r.get("cites_seed"):
        score += 3

    # citation-weighted attention signal (log scale)
    cites = int(r.get("citation_count") or 0)
    score += math.log10(1 + cites)

    # old + famous = background classic, deprioritize for a *new literature* review
    if year < 2024 and cites >= 100:
        score -= 2

    return score


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=30, help="how many to print")
    ap.add_argument("--csv", action="store_true",
                    help="also write papers_ranked.csv (all papers)")
    ap.add_argument("--min-score", type=float, default=None,
                    help="only show papers at or above this score")
    args = ap.parse_args()

    with open(HERE / "papers.csv") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("title")]

    for r in rows:
        r["priority"] = f"{priority(r):.1f}"
    rows.sort(key=lambda r: (-float(r["priority"]), r["title"]))

    if args.csv:
        out = HERE / "papers_ranked.csv"
        cols = list(rows[0].keys())
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out} ({len(rows)} papers)")

    shown = 0
    print(f"{'prio':>5}  {'rel':>3} {'year':>4} {'cites':>5}  title  [seed provenance]")
    for r in rows:
        if args.min_score is not None and float(r["priority"]) < args.min_score:
            break
        if shown >= args.n:
            break
        shown += 1
        seeds = f" <- {r['cites_seed'][:30]}" if r.get("cites_seed") else ""
        print(f"{r['priority']:>5}  {r['relevance']:>3} {r['year'] or '?':>4} "
              f"{r['citation_count'] or 0:>5}  {r['title'][:78]}{seeds}")


if __name__ == "__main__":
    main()
