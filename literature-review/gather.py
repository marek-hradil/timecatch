#!/usr/bin/env python3
"""
Gather literature for the TimeCatch follow-up / systematic review.

Pipeline (PRISMA-style snowballing):
  1. Resolve seed papers (arXiv IDs in config.json) on OpenAlex.
  2. Forward citations  : papers citing the seeds  (new literature).
  3. Backward references: papers cited by the seeds  (older foundations).
  4. arXiv keyword search with configurable boolean queries.
  5. Optional: same citation tracing on Semantic Scholar (--with-s2),
     best with a free API key (see README) since the unauthenticated
     pool is heavily rate-limited.
  6. Dedupe (DOI > arXiv ID > normalized title), relevance-score, export.

Outputs (in this folder):
  cache/          raw API responses (re-runs don't re-hit the APIs)
  papers.csv      all unique papers + provenance + relevance + screening columns
  papers.bib      BibTeX for everything in the CSV
  report.md       run summary (counts per source, top papers by score)

Usage:
  python gather.py                 # full run (OpenAlex + arXiv)
  python gather.py --no-cache      # ignore cached responses
  python gather.py --seeds-only    # just resolve seeds (debugging)
  python gather.py --with-s2       # also pull Semantic Scholar citations
                                  (export S2_API_KEY=... for real rate limits)

Dependencies: Python 3.10+ stdlib only.
"""

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "cache"
CONFIG_PATH = HERE / "config.json"

OA_BASE = "https://api.openalex.org/works"
S2_BASE = "https://api.semanticscholar.org/graph/v1"
ARXIV_BASE = "https://export.arxiv.org/api/query"

S2_FIELDS = "title,year,abstract,venue,citationCount,externalIds,authors"
S2_SLEEP = 2.0
S2_RETRIES = 3  # low on purpose: unauthenticated S2 is often unusable
OA_SLEEP = 0.3
ARXIV_SLEEP = 3.5  # arXiv asks for 3s politeness delay


# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------

def http_get(url: str) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": "temporal-misalignment-literature-review/0.2 (research use)"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def cached_get(url: str, use_cache: bool = True, retries: int = 5,
               retry_status: set[int] | None = None) -> tuple[bytes, bool]:
    """GET with disk cache and retry/backoff (handles 429 rate limits).

    Returns (data, from_cache). No sleeping when served from cache.
    """
    key = hashlib.sha256(url.encode()).hexdigest()[:24]
    cache_file = CACHE_DIR / f"{key}.json"
    if use_cache and cache_file.exists():
        return cache_file.read_bytes(), True

    last_err = None
    for attempt in range(retries):
        try:
            data = http_get(url)
            CACHE_DIR.mkdir(exist_ok=True)
            cache_file.write_bytes(data)
            return data, False
        except urllib.error.HTTPError as e:
            if retry_status and e.code in retry_status:
                wait = min(120, 2 ** attempt * 5)
                print(f"  HTTP {e.code}, backing off {wait}s ({url[:80]}...)",
                      file=sys.stderr)
                time.sleep(wait)
                last_err = e
                continue
            raise
    raise last_err


# --------------------------------------------------------------------------
# OpenAlex
# --------------------------------------------------------------------------

def oa_params(extra: dict, mailto: str) -> dict:
    p = dict(extra)
    if mailto:
        p["mailto"] = mailto
    return p


def oa_request(params: dict, mailto: str, use_cache: bool) -> dict:
    url = f"{OA_BASE}?{urllib.parse.urlencode(oa_params(params, mailto))}"
    raw, from_cache = cached_get(url, use_cache=use_cache,
                                  retry_status={403, 429, 500, 502, 503, 504})
    if not from_cache:
        time.sleep(OA_SLEEP)  # only rate-limit real requests
    return json.loads(raw)


def abstract_from_inverted(inv: dict | None) -> str:
    """Reconstruct an abstract from OpenAlex's abstract_inverted_index."""
    if not inv:
        return ""
    positions = []
    for word, idxs in inv.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def work_to_paper(work: dict) -> dict:
    doi = (work.get("doi") or "").replace("https://doi.org/", "")
    arxiv_id = ""
    m = re.match(r"10\.48550/arxiv\.(.+)", doi, re.I)
    if m:
        arxiv_id = m.group(1)
        doi = ""
    location = (work.get("primary_location") or {}).get("source") or {}
    return {
        "title": work.get("display_name") or "",
        "abstract": abstract_from_inverted(work.get("abstract_inverted_index")),
        "year": work.get("publication_year"),
        "venue": location.get("display_name") or "",
        "citation_count": work.get("cited_by_count", 0),
        "authors": ", ".join(
            (a.get("author") or {}).get("display_name", "")
            for a in (work.get("authorships") or [])[:10]
        ),
        "doi": doi,
        "arxiv_id": arxiv_id,
        "openalex_id": (work.get("id") or "").rsplit("/", 1)[-1],
    }


def oa_find_seed(seed: dict, mailto: str, use_cache: bool) -> list[dict]:
    """Resolve a seed to OpenAlex works (may be several duplicate records).

    OpenAlex often has separate records for the arXiv preprint and the
    published venue version, so we collect both the DOI lookup and any
    strong title-search matches and trace citations through all of them.
    """
    works: dict[str, dict] = {}

    arxiv_id = seed.get("arxiv_id")
    if arxiv_id:
        try:
            url = f"{OA_BASE}/https://doi.org/10.48550/arxiv.{arxiv_id}"
            if mailto:
                url += f"?mailto={urllib.parse.quote(mailto)}"
            raw, _ = cached_get(url, use_cache=use_cache,
                                retry_status={403, 429, 500, 502, 503, 504})
            time.sleep(OA_SLEEP)
            work = json.loads(raw)
            works[work["id"]] = work
        except urllib.error.HTTPError as e:
            if e.code != 404:
                print(f"  ! OpenAlex lookup {e.code} for arXiv:{arxiv_id}",
                      file=sys.stderr)

    # title search (finds published versions + duplicates)
    # strip wildcard chars (* ?) which break OpenAlex filter parsing
    search_title = re.sub(r"[*?]", "", seed["title"]).strip()
    try:
        data = oa_request({"filter": f"title.search:{search_title}",
                           "per_page": 10, "select": "id,display_name"},
                          mailto, use_cache)
    except urllib.error.HTTPError as e:
        print(f"  ! title search failed ({e.code}), trying shorter query",
              file=sys.stderr)
        short = " ".join(search_title.split()[:6])
        data = oa_request({"filter": f"title.search:{short}",
                           "per_page": 10, "select": "id,display_name"},
                          mailto, use_cache)
    for w in data.get("results", []):
        if title_overlap(seed["title"], w.get("display_name", "")) >= 0.5:
            works.setdefault(w["id"], w)

    # re-fetch everything in full (title search returned id/display_name only,
    # and we need referenced_works + counts for the citation tracing)
    if works:
        ids = [w["id"].rsplit("/", 1)[-1] for w in works.values()]
        full = oa_batch_works(ids, mailto, use_cache)
        return full or list(works.values())
    return list(works.values())


def oa_forward_citations(work_id: str, mailto: str, use_cache: bool,
                         max_papers: int = 3000) -> list[dict]:
    """All works citing a given OpenAlex work, via cursor pagination."""
    out, cursor = [], "*"
    while True:
        data = oa_request({"filter": f"cites:https://openalex.org/{work_id}",
                           "per_page": 200, "cursor": cursor}, mailto, use_cache)
        out.extend(data.get("results", []))
        cursor = (data.get("meta") or {}).get("next_cursor")
        if not cursor or len(out) >= max_papers:
            break
    return out


def oa_batch_works(ids: list[str], mailto: str, use_cache: bool) -> list[dict]:
    """Fetch works by OpenAlex ID in chunks (pipe-separated filter)."""
    out = []
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        data = oa_request({"filter": "openalex:" + "|".join(chunk),
                           "per_page": 100}, mailto, use_cache)
        out.extend(data.get("results", []))
    return out


# --------------------------------------------------------------------------
# Semantic Scholar (optional enrichment)
# --------------------------------------------------------------------------

def s2_request(path: str, params: dict, use_cache: bool) -> dict:
    import os
    query = urllib.parse.urlencode(params)
    url = f"{S2_BASE}{path}?{query}"
    headers = {}
    if os.environ.get("S2_API_KEY"):
        headers = {"x-api-key": os.environ["S2_API_KEY"]}
    raw, _ = cached_get(url, use_cache=use_cache,
                        retries=S2_RETRIES, retry_status={429, 502, 503})
    time.sleep(S2_SLEEP)
    return json.loads(raw)


def s2_paginate(path: str, params: dict, use_cache: bool, key: str,
                max_papers: int = 2000) -> list[dict]:
    out, offset = [], 0
    while True:
        data = s2_request(path, dict(params, limit=1000, offset=offset), use_cache)
        batch = data.get("data", [])
        if not batch:
            break
        out.extend(batch)
        nxt = data.get("next")
        if nxt is None or len(out) >= max_papers:
            break
        offset = nxt
    return [item[key] for item in out if item.get(key)]


def paper_from_s2(p: dict) -> dict:
    ext = p.get("externalIds") or {}
    return {
        "title": p.get("title") or "",
        "abstract": p.get("abstract") or "",
        "year": p.get("year"),
        "venue": p.get("venue") or "",
        "citation_count": p.get("citationCount", 0),
        "authors": ", ".join(a.get("name", "") for a in (p.get("authors") or [])[:10]),
        "doi": ext.get("DOI", ""),
        "arxiv_id": ext.get("ArXiv", ""),
        "s2_id": p.get("paperId", ""),
    }


# --------------------------------------------------------------------------
# arXiv
# --------------------------------------------------------------------------

ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}


def _arxiv_entries(raw: bytes) -> list[tuple[str, str]]:
    out = []
    root = ET.fromstring(raw)
    for entry in root.findall("a:entry", ARXIV_NS):
        arxiv_id = entry.findtext("a:id", "", ARXIV_NS).rsplit("/", 1)[-1]
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
        title = " ".join(entry.findtext("a:title", "", ARXIV_NS).split())
        out.append((arxiv_id, title))
    return out


def arxiv_search(query: str, max_results: int, use_cache: bool) -> list[dict]:
    params = urllib.parse.urlencode({
        "search_query": query, "start": 0, "max_results": max_results,
        "sortBy": "submittedDate", "sortOrder": "descending",
    })
    raw, from_cache = cached_get(f"{ARXIV_BASE}?{params}", use_cache=use_cache,
                                  retry_status={503})
    if not from_cache:
        time.sleep(ARXIV_SLEEP)

    papers = []
    root = ET.fromstring(raw)
    for entry in root.findall("a:entry", ARXIV_NS):
        arxiv_id = entry.findtext("a:id", "", ARXIV_NS).rsplit("/", 1)[-1]
        papers.append({
            "title": " ".join(entry.findtext("a:title", "", ARXIV_NS).split()),
            "abstract": " ".join(entry.findtext("a:summary", "", ARXIV_NS).split()),
            "year": int(entry.findtext("a:published", "", ARXIV_NS)[:4]),
            "authors": ", ".join(a.findtext("a:name", "", ARXIV_NS)
                                 for a in entry.findall("a:author", ARXIV_NS)[:10]),
            "arxiv_id": re.sub(r"v\d+$", "", arxiv_id),
            "doi": (entry.findtext("a:doi", "", ARXIV_NS) or "").strip(),
            "venue": "arXiv preprint",
        })
    return papers


def arxiv_find_by_title(title: str, use_cache: bool) -> tuple[str, float] | None:
    """Find a paper on arXiv by (fuzzy) title match. Returns (arxiv_id, overlap)."""
    params = urllib.parse.urlencode({
        "search_query": f'all:"{title}"', "start": 0, "max_results": 20,
    })
    try:
        raw, _ = cached_get(f"{ARXIV_BASE}?{params}", use_cache=use_cache,
                            retry_status={503})
    except Exception as e:
        print(f"  ! arXiv title search failed: {e}", file=sys.stderr)
        return None
    time.sleep(ARXIV_SLEEP)

    best_id, best_score = None, 0.0
    for arxiv_id, entry_title in _arxiv_entries(raw):
        score = title_overlap(title, entry_title)
        if score > best_score:
            best_id, best_score = arxiv_id, score
    if best_id and best_score >= 0.6:
        return best_id, best_score
    return None


# --------------------------------------------------------------------------
# Dedupe / scoring / export
# --------------------------------------------------------------------------

def norm_title(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^a-z0-9 ]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def title_overlap(a: str, b: str) -> float:
    """Token overlap ratio between two titles (for fuzzy matching)."""
    ta = {t for t in norm_title(a).split() if len(t) > 2}
    tb = {t for t in norm_title(b).split() if len(t) > 2}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def dedupe_key(paper: dict) -> tuple:
    if paper.get("doi"):
        return ("doi", paper["doi"].lower())
    if paper.get("arxiv_id"):
        return ("arxiv", paper["arxiv_id"])
    return ("title", norm_title(paper.get("title", "")))


def relevance_score(paper: dict, keywords: dict) -> int:
    text = (paper.get("title", "") + " " + (paper.get("abstract") or "")).lower()
    score = 0
    for group, weight in (("temporal", 3), ("vision", 2), ("eval", 1)):
        if any(kw in text for kw in keywords.get(group, [])):
            score += weight
    return score


KNOWN_CONFS = {"ACL", "EMNLP", "NAACL", "CVPR", "ICCV", "ECCV",
               "NeurIPS", "ICML", "ICLR", "AAAI", "COLM"}


def to_bibtex(paper: dict) -> str:
    year = paper.get("year") or ""
    first_author = (paper.get("authors") or "Unknown").split(",")[0].split()[-1]
    cite_key = f"{first_author}{year}"
    if paper.get("arxiv_id"):
        cite_key += f"_{paper['arxiv_id'].replace('/', '_').replace('.', '_')}"

    fields = [
        ("title", paper.get("title", "")),
        ("author", paper.get("authors", "")),
        ("year", str(year)),
    ]
    venue = paper.get("venue") or ""
    if venue:
        kind = "booktitle" if any(c in venue for c in KNOWN_CONFS) else "journal"
        fields.append((kind, venue))
    if paper.get("arxiv_id"):
        fields += [("eprint", paper["arxiv_id"]), ("archivePrefix", "arXiv")]
    if paper.get("doi"):
        fields.append(("doi", paper["doi"]))

    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields)
    return f"@misc{{{cite_key},\n{body}\n}}"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--seeds-only", action="store_true")
    ap.add_argument("--with-s2", action="store_true",
                    help="also pull Semantic Scholar citations (use S2_API_KEY)")
    args = ap.parse_args()
    use_cache = not args.no_cache

    config = json.loads(CONFIG_PATH.read_text())
    min_year = config.get("min_year", 2023)
    mailto = config.get("mailto", "")

    records: dict[tuple, dict] = {}
    # secondary index: normalized title -> canonical key, so records that only
    # differ in identifier coverage (e.g. OpenAlex preprint-with-DOI vs.
    # published-without-DOI duplicates) still merge
    title_index: dict[str, tuple] = {}

    def add(paper: dict, source: str, seed_hit: str = "") -> None:
        if not paper.get("title"):
            return
        key = dedupe_key(paper)
        nt = norm_title(paper["title"])
        # resolve through the title alias when possible
        canonical = key if key in records else title_index.get(nt)
        if canonical is not None and canonical in records:
            key = canonical

        if key in records:
            rec = records[key]
            rec["sources"].add(source)
            if seed_hit and seed_hit not in rec.get("seed_hits", set()):
                rec.setdefault("seed_hits", set()).add(seed_hit)
            for k, v in paper.items():
                if v and not rec.get(k):
                    rec[k] = v
        else:
            paper = dict(paper, sources={source})
            if seed_hit:
                paper["seed_hits"] = {seed_hit}
            records[key] = paper
        title_index.setdefault(nt, key)

    # -- 1. seeds + OpenAlex citation tracing --------------------------------
    print("== Resolving seeds on OpenAlex ==")
    seeds = []
    for seed in config["seeds"]:
        print(f"  seed: {seed['title']}")
        seed_works = oa_find_seed(seed, mailto, use_cache)
        if not seed_works:
            print("    NOT FOUND on OpenAlex — check arxiv_id/title in config.json",
                  file=sys.stderr)
            continue
        seed_name = seed.get("note", seed["title"]).split(",")[0]
        for w in seed_works:
            print(f"    -> {w.get('display_name')} "
                  f"({w.get('publication_year')}, "
                  f"{w.get('cited_by_count')} citations)")
        seeds.append((seed, seed_works, seed_name))

    if args.seeds_only:
        return

    # seeds themselves belong in the screening table too
    for seed, seed_works, name in seeds:
        for w in seed_works:
            p = work_to_paper(w)
            if p["title"]:
                add(p, "seed", name)

    print("\n== Forward citations (papers citing the seeds) ==")
    for seed, seed_works, name in seeds:
        seen_ids: set[str] = set()
        total = 0
        for work in seed_works:
            wid = work["id"].rsplit("/", 1)[-1]
            try:
                for w in oa_forward_citations(wid, mailto, use_cache):
                    if w["id"] in seen_ids:
                        continue
                    seen_ids.add(w["id"])
                    total += 1
                    if (w.get("publication_year") or 0) >= min_year:
                        add(work_to_paper(w), "forward-cite", name)
            except Exception as e:
                print(f"  ! forward-cite failed for {name} ({e}) — skipping",
                      file=sys.stderr)
        print(f"  {name}: {total} unique citing works")

    print("\n== Backward references (papers cited by the seeds) ==")
    for seed, seed_works, name in seeds:
        ref_ids: list[str] = []
        for work in seed_works:
            ref_ids += [r.rsplit("/", 1)[-1]
                        for r in work.get("referenced_works", [])]
        ref_ids = list(dict.fromkeys(ref_ids))  # dedupe, keep order
        print(f"  {name}: {len(ref_ids)} references")
        try:
            for w in oa_batch_works(ref_ids, mailto, use_cache):
                add(work_to_paper(w), "backward-ref")
        except Exception as e:
            print(f"  ! backward-refs failed for {name} ({e}) — skipping",
                  file=sys.stderr)

    # -- 2. optional Semantic Scholar enrichment ------------------------------
    if args.with_s2:
        print("\n== Semantic Scholar citation tracing (optional) ==")
        import os
        if not os.environ.get("S2_API_KEY"):
            print("  note: no S2_API_KEY set — unauthenticated requests are often "
                  "rate-limited; skipping failures gracefully", file=sys.stderr)
        s2_ok = True
        for seed, _, name in seeds:
            if not seed.get("arxiv_id") or not s2_ok:
                continue
            pid = f"arXiv:{seed['arxiv_id']}"
            try:
                citing = s2_paginate(f"/paper/{pid}/citations",
                                     {"fields": S2_FIELDS}, use_cache, "citingPaper")
                fresh = [p for p in citing if (p.get("year") or 0) >= min_year]
                print(f"  {name}: {len(citing)} citing on S2, "
                      f"{len(fresh)} since {min_year}")
                for p in fresh:
                    add(paper_from_s2(p), "s2-forward-cite", name)
                cited = s2_paginate(f"/paper/{pid}/references",
                                    {"fields": S2_FIELDS}, use_cache, "citedPaper")
                for p in cited:
                    add(paper_from_s2(p), "s2-backward-ref")
            except urllib.error.HTTPError as e:
                print(f"  ! S2 failed ({e.code}) for {pid} — giving up on S2 "
                      f"for this run (get a free API key: "
                      f"https://www.semanticscholar.org/product/api)",
                      file=sys.stderr)
                s2_ok = False

    # -- 3. arXiv search ------------------------------------------------------
    print("\n== arXiv keyword search ==")
    for query in config.get("arxiv_queries", []):
        n = config.get("arxiv_max_results_per_query", 150)
        try:
            papers = arxiv_search(query, n, use_cache)
        except Exception as e:
            print(f"  ! arXiv query failed: {e}", file=sys.stderr)
            continue
        kept = [p for p in papers if p["year"] >= min_year]
        print(f"  [{len(papers):3d} results, {len(kept)} >= {min_year}] "
              f"{query[:70]}...")
        for p in kept:
            add(p, "arxiv-search")

    # -- 4. score + export -----------------------------------------------------
    print(f"\n== Exporting {len(records)} unique papers ==")
    rows = []
    for p in records.values():
        p["source"] = "; ".join(sorted(p["sources"]))
        p["cites_seed"] = "; ".join(sorted(p.pop("seed_hits", set()) or set()))
        p.pop("sources", None)
        p["relevance"] = relevance_score(p, config.get("relevance_keywords", {}))
        rows.append(p)
    rows.sort(key=lambda r: (-r["relevance"], -(r.get("citation_count") or 0),
                             r.get("title", "")))

    csv_path = HERE / "papers.csv"
    cols = ["relevance", "title", "year", "authors", "venue", "citation_count",
            "cites_seed", "arxiv_id", "doi", "source", "decision", "notes",
            "abstract"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(dict(r, decision="", notes=""))

    with open(HERE / "papers.bib", "w") as f:
        for r in rows:
            f.write(to_bibtex(r) + "\n\n")

    report = [
        "# Literature review — gather run",
        "",
        f"Unique papers: **{len(rows)}**  (min year: {min_year})",
        "",
        "Sources:",
        f"- Seeds resolved: {len(seeds)}/{len(config['seeds'])}",
        f"- From forward citations: "
        f"{sum(1 for r in rows if 'forward-cite' in r['source'])}",
        f"- From backward references: "
        f"{sum(1 for r in rows if 'backward-ref' in r['source'])}"
        f" (+ S2: {sum(1 for r in rows if 's2-' in r['source'])})",
        f"- From arXiv search: {sum(1 for r in rows if 'arxiv-search' in r['source'])}",
        "",
        "Note: brand-new preprints (2026) often have no citation or reference "
        "data in OpenAlex yet — run with `--with-s2` (and a free S2_API_KEY) "
        "to pull their references, and re-run periodically as citations "
        "accumulate.",
        "",
        f"Papers with relevance score >= 5 (strong candidates): "
        f"**{sum(1 for r in rows if r['relevance'] >= 5)}**",
        "",
        "## Top 40 by relevance",
        "",
        "| score | year | cites | title |",
        "|---|---|---|---|",
    ]
    for r in rows[:40]:
        title = r["title"][:90].replace("|", "/")
        report.append(f"| {r['relevance']} | {r.get('year', '')} | "
                      f"{r.get('citation_count', 0)} | {title} |")
    (HERE / "report.md").write_text("\n".join(report) + "\n")

    print(f"  wrote {csv_path}")
    print(f"  wrote {HERE / 'papers.bib'}")
    print(f"  wrote {HERE / 'report.md'}")
    print("\nNext: open papers.csv, fill the 'decision' column "
          "(include/exclude/maybe) while screening titles+abstracts.")


if __name__ == "__main__":
    main()
