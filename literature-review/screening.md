# Screening log (PRISMA-style)

Update these numbers after each screening pass. Date format: YYYY-MM-DD.

## Run 1 — DATE

- Records gathered (papers.csv rows): ___
- Seeds resolved: ___ / ___
- From forward citations: ___
- From backward references: ___
- From arXiv search: ___

### Screening decisions

- Include: ___
- Maybe: ___
- Exclude: ___ (most common exclusion reasons: ___)

### Notes

- New papers promoted to seeds for the next run: ___

## Inclusion criteria (draft — refine as you screen)

1. **Population**: (M)LLMs / VLMs processing video or image sequences.
2. **Construct**: temporal understanding — order, duration, motion continuity,
   causal/temporal consistency across frames (not text-only temporal reasoning,
   not purely spatial compositionality).
3. **Study type**: benchmarks/evaluations, mechanistic analyses, training/data
   interventions targeting temporal capability.
4. **Time window**: ≥ 2023 for benchmark papers; older items allowed from
   backward references if foundational (e.g., TGIF-QA).

### Exclusion rules of thumb

- Text-only temporal reasoning (temporal commonsense in LLMs) → exclude
- Purely spatial compositionality benchmarks (e.g. ARO-style) → exclude
- Video generation/editing with no evaluation angle → exclude
- Application papers (surveillance, action recognition CNNs) → exclude
