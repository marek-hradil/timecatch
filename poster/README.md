# TimeCatch poster

A0 portrait poster for "What's the Catch? Evaluating Temporal Consistency in
Vision-Language Models", built on the `a0poster` class (University of
Copenhagen template lineage: KU crimson title accents). Minimal layout:
white page, hairline beat separators, color only in data elements.

Fonts match the paper's Figure 1: Montserrat (bold) for headings, Inter for
body text — both from TeX Live's `montserrat`/`inter` packages, so plain
pdflatex works (no XeLaTeX needed).

## Build

```sh
pdflatex poster.tex
```

Requires a TeX Live with `a0poster` and `pgfplots` (any full TeX Live works;
tested with TeX Live 2025).

## Assets (`img/`)

- `dive1..4.jpg` — MTL-AQA scene 17_49 frames in their *correct* order
  (`datasets/MTL-AQA/17_49/00065096|121|146|171.jpg`). The poster shows them
  as 1,2,4,3 — the ground-truth swap for this scene is positions 3–4
  (`human_subset/swap_localize_mtl-aqa.json`).
- `noise.jpg` — generated Gaussian-noise frame (unused since the beat-3
  panel now embeds Figure 1; kept for the recreated-strip variant).
- `thumb_*.jpg` — one representative frame per dataset, center-cropped 16:9.
- `visual_abstract.pdf` — Figure 1 from the paper, embedded as-is (vector)
  in beat 3.

## Data sources (paper ground truth)

- Beat 3 is the paper's Figure 1 verbatim (strips, task questions,
  human-vs-VLM bars).
- Mini-charts: LPIPS bins (Fig. 6, Qwen3-VL-8B real-world), scaling (Fig. 8,
  frame/temporal task averages), prompting deltas (Fig. 9), sequence length
  (Fig. 10, model averages, 4–14 frames; the n=5 bucket at 16 is omitted).

## Print note

The beat-1 dive frames are 640×360 source JPGs (~105 DPI at print size) —
fine from viewing distance; swap in higher-res frames if available. Beat 1
deliberately prints no answer — the swap (frames 3 & 4) is revealed only in
conversation, though a sharp eye will spot the same dive highlighted in the
Figure 1 panel below it.
