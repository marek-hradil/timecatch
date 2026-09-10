# Literature Review — TimeCatch follow-up

Systematic literature gathering for the TimeCatch paper ("What's the Catch?
Evaluating Temporal Consistency in Vision-Language Models") and its follow-up
work on temporal grounding / temporal reasoning in VLMs.

## What's here

| File | Purpose |
|---|---|
| `config.json` | Edit this: seed papers, arXiv queries, relevance keywords, min year |
| `gather.py` | Gathers papers from Semantic Scholar (citation tracing) + arXiv (keyword search) |
| `prioritize.py` | Ranks `papers.csv` by review priority (relevance + recency + provenance); use this to decide what to read first |
| `papers.csv` | Output: all unique papers + relevance score + `decision`/`notes` screening columns |
| `papers.bib` | Output: BibTeX for all gathered papers |
| `report.md` | Output: run summary + top papers by relevance |
| `cache/` | Raw API responses (re-runs are fast and polite; delete to force refresh) |
| `screening.md` | PRISMA-style screening log template |

`gather.py` uses only the Python standard library — run it with any Python 3.10+:

```bash
python gather.py               # full run
python gather.py --seeds-only  # just check that seed titles resolve
python gather.py --no-cache    # force fresh API calls
```

Optional (higher Semantic Scholar rate limits):

```bash
export S2_API_KEY=...   # free key: https://www.semanticscholar.org/product/api
```

## Method

Snowballing + keyword search, PRISMA-style:

1. **Seed resolution** — each seed title in `config.json` is resolved to a
   Semantic Scholar paper. Run `--seeds-only` after editing seeds to confirm
   they resolve to the right papers (fix titles if the match is wrong).
2. **Forward citations** — every paper that *cites* a seed, filtered to
   `min_year` and later. This is the main "new literature" channel.
3. **Backward references** — everything the seeds cite (foundations, older
   benchmarks).
4. **arXiv search** — boolean queries over titles/abstracts, newest first.
5. **Dedupe** by DOI → arXiv ID → normalized title.
6. **Relevance score** (0–6): +3 if temporal keywords hit, +2 if VLM/video
   keywords hit, +1 if evaluation keywords hit. Scoring only *sorts* — nothing
   is dropped.
7. **Export** to `papers.csv` / `papers.bib` / `report.md`.

Rate limits: the script sleeps between requests and retries with backoff on
429/503. A full run takes ~10–20 minutes unauthenticated (S2 caps ~100
requests / 5 min); cached responses make re-runs instant.

## Screening workflow

1. Run `gather.py`.
2. Run `python prioritize.py -n 50` to get a reading order (or `--csv` for a
   full ranked table in `papers_ranked.csv`). See "Priority scoring" below.
3. Open `papers.csv` (Excel/Numbers/Google Sheets works fine).
4. Screen in priority order, checking title + abstract, and fill:
   - `decision`: `include` / `exclude` / `maybe`
   - `notes`: one-line reason (e.g. "temporal benchmark, must cite",
     "single-frame only, exclude")
5. Log counts in `screening.md` (PRISMA flow).
6. When you find a highly relevant paper that is *not* a seed yet → add it to
   `config.json` seeds and re-run. 2–3 rounds of this converge quickly.

## Priority scoring

Two layers, on purpose:

- **`relevance` (0–6, in papers.csv)** — pure keyword score from
  `config.json`: temporal keywords (+3), vision/VLM keywords (+2),
  evaluation keywords (+1). Deliberately coarse; it only sorts, never drops.
- **`priority` (from prioritize.py)** — composite for *review* priority:
  relevance + recency bonus (2026: +4, 2025: +3, 2024: +1) + seed-provenance
  bonus (+3 if the paper cites one of our seeds — same literature
  neighborhood) + log-scaled citations − classic penalty (pre-2024 papers with
  ≥100 citations get −2; they're background, not new literature).

Tune the weights in `prioritize.py` if the ranking doesn't match your
  intuition — it's ~20 lines.

## Keeping it fresh

- Re-run `gather.py --no-cache` every few weeks (forward citations accumulate
  as the field cites the seeds).
- **Known gap:** OpenAlex has *no citation/reference data yet* for the newest
  2026 preprints (they show 0 citations, 0 references). Semantic Scholar has
  it — get a free API key at <https://www.semanticscholar.org/product/api>,
  then run `S2_API_KEY=... python gather.py --with-s2` to pull the new papers'
  reference lists. Without a key, S2's unauthenticated pool is usually
  rate-limited to unusability.
- Alternatively set up Google Scholar alerts for "temporal reasoning
  vision-language" and citation alerts on the seed papers — those are free and
  arrive by email.
- Once TimeCatch is on arXiv, add its own arXiv ID to the seeds.
