# TimeCatch

### What's the Catch? Evaluating Temporal Consistency in Vision-Language Models

**Marek Hradil**<sup>1</sup> &nbsp; **Danae Sánchez Villegas**<sup>1,2</sup>

<sup>1</sup>University of Copenhagen &nbsp;&nbsp; <sup>2</sup>The National Center for AI in Society (CAISA)

`rsc784@alumni.ku.dk` &nbsp;&nbsp; `davi@samf.ku.dk`

---

Vision-language models (VLMs) achieve strong performance on video and image-sequence
benchmarks, yet it remains unclear whether they capture temporal structure. **TimeCatch**
formulates temporal grounding as an *anomaly detection* problem, giving a simple and
controlled test of sensitivity to temporal consistency.

Two kinds of anomaly are injected into image sequences:

- **Temporal anomalies** — a randomly chosen consecutive frame pair `(I_i, I_i+1)` is
  swapped. Detecting one requires reasoning across frames.
- **Frame-level anomalies** — a uniformly sampled frame is replaced with Gaussian noise.
  This is a *control condition*: it is identifiable without any temporal context, so
  strong performance here shows the model does attend to the sequence.

Because only the *order* of frames changes, the underlying visual content is held fixed
and temporal consistency can be evaluated in isolation.

**Core finding.** In the main evaluation setting, VLMs reliably detect and localize
frame-level anomalies (up to 99.6% / 99.4%) but generally perform near chance on temporal
anomaly detection (best: 57.4%) with limited localization (best: 43.2%). Humans reach
75.0–91.7% and 83.3–88.9% respectively on the same subset. Analyses across model scale,
prompting, sequence length, and visual similarity show performance can improve under some
conditions — notably larger scale and removing scene descriptions — while substantial
gaps remain.

## Tasks

| Paper term | Repo script | Output space | Random chance |
|---|---|---|---|
| Temporal Anomaly Detection ("Temporal Detect") | `scripts/swap_detect.py` | `{yes, no}` | 50% |
| Temporal Anomaly Localization ("Temporal Localize") | `scripts/swap_localize.py` | index `1..n-1` | 1/(n−1) |
| Frame-Level Anomaly Detection ("Frame Detect") | `scripts/corrupt_detect.py` | `{yes, no}` | 50% |
| Frame-Level Anomaly Localization ("Frame Localize") | `scripts/corrupt_localize.py` | index `1..n` | 1/n |

The paper's *frame-level anomaly* is the *corruption* (Gaussian-noise) condition in the
code — same thing, different name. Keep this mapping in mind when reading `results/`
CSVs or `config/experiments/` filenames.

Random-chance localization differs between the two families because a swap cannot occur
at the last position: temporal-localize chance is 4.3 / 5.5 / 8.7 / 13.2% and
frame-localize chance is 13.5 / 14.9 / 18.1 / 22.4% for CLEVRER / CRAFT / DriveLM /
MTL-AQA.

## Datasets

Four dataset test splits spanning synthetic and real-world domains, selected so that
(i) events are interpretable without domain expertise, (ii) each sequence follows a single
temporal trajectory, (iii) consecutive frames show distinguishable change, and (iv) the
viewpoint is consistent.

| Dataset | Domain | Content | Sequences | Avg. length | Used for |
|---|---|---|---|---|---|
| CLEVRER | Synthetic | Abstract 3D objects, collisions | 4,997 | 7.5 | main results |
| CRAFT | Synthetic | Abstract 2D objects, collisions | 858 | 6.9 | main results |
| DriveLM | Real-world | Egocentric driving | 696 | 5.9 | main results |
| MTL-AQA | Real-world | Olympic competitive diving | 338 | 4.5 | main results |
| CRAFT-long | Synthetic | Abstract 2D objects, collisions | 1,101 | 10.6 | sequence-length analysis only |

**Sampling and filtering.** Videos are subsampled to image sequences: every 1.5 s for
CRAFT and CLEVRER, every 1 s for MTL-AQA. DriveLM is not sampled from video — it uses
nuScenes' native discrete image sequences. Consecutive pairs whose
[LPIPS](https://arxiv.org/abs/1801.03924) distance falls below **0.05** are dropped, since
near-identical pairs make a swap imperceptible. Sequences left with fewer than four frames
are discarded.

**CRAFT is split in two.** After filtering there are 1,959 CRAFT sequences (avg 9.0
frames). Only the 4–8-frame subset (858 sequences) enters the main results table; the
remaining 9–16-frame sequences become **CRAFT-long**, used *only* for the sequence-length
analysis. Don't conflate the two when reading `results/`.

Mean LPIPS after filtering: CRAFT 0.093, CLEVRER 0.131, DriveLM 0.428, MTL-AQA 0.584 —
the real-world datasets have much larger consecutive-frame differences than the synthetic
ones, which have largely static backgrounds.

**Scene descriptions** are appended as plain text after the image tokens. CLEVRER and
CRAFT use the datasets' own event annotations serialized chronologically; DriveLM uses
nuScenes scene-level metadata; MTL-AQA descriptions are generated from the structured dive
annotations (rotation type, body position, somersault and twist count, armstand flag).

### Licensing

CLEVRER is released under CC0. CRAFT is released under CC BY 4.0. DriveLM's annotations
are released under CC BY-NC-SA 4.0. MTL-AQA is used consistent with standard academic
reuse of an established public benchmark — the original release does not specify an
explicit license.

Datasets are **not** distributed with this repo. Obtain each from its original source,
then run the preparation scripts in `scripts/<dataset>/` (see below).

## Models

Five open-weight VLMs with native multi-image support, all evaluated zero-shot:

- Qwen2.5-VL-7B
- Qwen3-VL-8B — best overall performer, and the model used for the analyses
- Gemma-4-E4B
- InternVL3-8B
- InternVL3.5-8B

The scale analysis additionally evaluates Qwen3-VL at 2B / 4B / 32B.

All models are served with [vLLM](https://github.com/vllm-project/vllm) using **constrained
decoding** — detection outputs restricted to `{yes, no}`, localization outputs to valid
frame indices — on NVIDIA A100 GPUs.

## Setup

Requires Python ≥ 3.13. Dependencies are managed with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

### Data preparation

Each dataset has a preparation pipeline under `scripts/<dataset>/`:

```bash
python3 scripts/clevrer/restructure_data.py    # raw download -> scenes/ layout
python3 scripts/clevrer/prepare_frames.py      # video -> subsampled frames
python3 scripts/clevrer/describe_dataset.py    # build scene descriptions
python3 scripts/add_lpips.py                   # per-pair LPIPS
python3 scripts/add_seq_lpips.py               # sequence-level LPIPS + filtering
```

DriveLM uses `scripts/prepare_drivelm.py` instead of a video-sampling step. CRAFT and
MTL-AQA additionally have `*_full.py` variants that keep long sequences, which is how
**CRAFT-long** is produced. The result is a `scenes_filtered.json` manifest per dataset,
which is what the dataset YAMLs point at.

## Running experiments

Experiments are fully YAML-driven. One file per (task × dataset × model) combination:

```yaml
# config/experiments/swap-detect-clevrer-qwen3-vl-8b.yaml
model: qwen3-vl-8b
dataset: clevrer
seed: 42
script: swap_detect
batch_size: 64

slurm:
  qos: acc_ehpc
  time: "01:00:00"
  gpus: 1
  cpus_per_gpu: 20

system_prompt: >
  You are given a sequence of images showing a scene unfolding over time. ...
```

referencing a model YAML (`config/models/`) and a dataset YAML (`config/datasets/`).

Locally:

```bash
python3 scripts/swap_detect.py config/experiments/swap-detect-clevrer-qwen3-vl-8b.yaml
```

On a SLURM cluster (the paper's runs used MareNostrum 5), copy `cluster/env.sh.example` to
`cluster/env.sh` and fill in your values, then:

```bash
./run.sh config/experiments/swap-detect-clevrer-qwen3-vl-8b.yaml
```

`run.sh` builds the offline bundle (Linux wheels + HF model cache), rsyncs it, and submits
the job. `cluster/run_matrix.sh` submits a whole model family, polls until done, and
consolidates the CSVs. See `cluster/` for the full set of entry points.

Each run writes a timestamped directory under `outputs/`; consolidated per-experiment CSVs
land in `results/<model>/`. **`results/` is the source of truth for every number in the
paper.**

## Reproducing figures and tables

All figures are regenerated from the CSVs in `results/`:

| Paper item (label) | Generator |
|---|---|
| `fig:visual-abstract` — overview | `figures/visual_abstract.html` (hand-authored) |
| `fig:dataset-examples` — dataset examples | `plotting/plot_dataset_examples.py` |
| `fig:dataset-histograms` — sequence length distributions | `plotting/plot_dataset_lengths.py` |
| `tab:dataset_stats_1` — dataset statistics | `scripts/print_dataset_stats_table.py` |
| `tab:detailed_results` — main results | `plotting/print_main_table_detailed.py` |
| `fig:model-failure` — MTL-AQA localization failure | `scripts/render_scene_figure.py` |
| `fig:human_results` — human vs. model gap | `plotting/plot_human_model_gap.py`, `plotting/print_human_comparison.py` |
| `fig:visual-similarity` — LPIPS bins, Qwen3-VL-8B | `plotting/plot_lpips_bins.py` |
| `fig:localize-heatmap-qwen` — confusion matrix, Qwen3-VL-8B | `plotting/plot_localize_heatmap.py` |
| `fig:scaling` — model scale | `plotting/plot_qwen3_scaling.py` |
| `fig:textual-grounding-deltas` — prompting deltas | `plotting/plot_ablation_deltas.py` |
| `fig:by-length` — sequence length | `plotting/plot_craft_by_length.py` |
| `tab:dataset_stats` — stats before/after filtering (App. B) | `scripts/print_dataset_stats_table.py` |
| `fig:lpips-filtering` — LPIPS distributions (App. B) | *generator not currently in repo* |
| `fig:annotation-interface` / `-viewer` — UI screenshots (App. C) | manual screenshots of `annotation-tool/` |
| `tab:finetuning` — fine-tuning results (App. D) | *not currently reproducible from `results/` — see below* |
| `tab:no_scene_desc_delta` — no-scene-desc deltas (App. D) | `plotting/print_no_scene_desc_delta_table.py` |
| `fig:lpips-bins` — LPIPS bins, all models (App. D) | `plotting/plot_lpips_bins.py` |
| `fig:localize-heatmap` — confusion matrices, all models (App. D) | `plotting/plot_localize_heatmap.py` |
| `fig:quadrant-examples` — qualitative examples (App. D) | `scripts/render_quadrant_figures.py` |

Referenced by LaTeX label rather than number, since float placement shifts the numbering.

Figures are written to `figures/`; the copies used by the LaTeX build live in
`paper/img/`.

## Prompting

The four main task prompts are in Appendix A of the paper and are set per experiment via
the `system_prompt` field of the experiment YAML.

The prompting analysis (`fig:textual-grounding-deltas`) covers three axes, all on temporal detection:

- **Prompt phrasing** — base Prompt A plus four variants B–E, ranging from a minimal
  phrasing to a step-through chain-of-thought style. Configs in
  `config/experiments/` (prompt-ablation), results in `results/prompt-ablation/`.
- **No scene descriptions** — configs in `config/experiments/no-scene-desc/`, driven by
  `cluster/run_matrix_no_scene_desc.sh`, results in `results/no-scene-desc/`.
- **Reasoning ("thinking") mode** — `config/models/qwen3-vl-8b-thinking.yaml`, results in
  `results/thinking-ablation/`.

Alternative phrasings have little effect. Removing scene descriptions often yields modest
gains on temporal tasks, but the *largest* improvements are concentrated on frame-level
localization (e.g. InternVL3.5-8B: +39.6 pp on CLEVRER, +66.0 pp on MTL-AQA) — the text
acts more as a distractor on the control task than as a temporal shortcut. Enabling
reasoning gives mixed, dataset-dependent results with no consistent benefit.

## Human study

`annotation-tool/` is a custom Prolific-integrated annotation platform (FastAPI backend +
React frontend). It presents an image sequence with its scene description, walks
annotators through guidelines with visual examples and a video walkthrough, and offers an
interactive frame-by-frame viewer navigable with arrow keys — which annotators found
meaningfully helpful for spotting swaps.

- 24 participants (3 per dataset × task configuration), all with at least an undergraduate
  degree, 16 countries, mean age 35.2, 16M / 8F.
- Effective mean pay > £8.00/hour, above Prolific's £6.00/hour minimum. Mean completion
  time 9.8 minutes for 15 sequences.
- **Quality control:** 3 manually chosen maximally-easy attention-check sequences per
  participant, excluded from results; participants failing ≥ 2 of 3 are rejected.
- The human-vs-model comparison (`fig:human_results`) uses a fixed subset of **72 samples per dataset**,
  exported to `human_subset/` by `annotation-tool/backend/export_human_subset.py` and
  evaluated on models via `config/experiments/human-subset/` +
  `scripts/human_swap_detect.py` / `scripts/human_swap_localize.py`.

Human scores are **per-dataset ranges, not single numbers**: temporal detection
75.0–91.7% (best CRAFT, worst DriveLM), temporal localization 83.3–88.9%, and ~100% on
the frame-level control tasks. On the shared subset no evaluated VLM exceeds 70% temporal
detection or 50% temporal localization on any dataset — the gap holds dataset by dataset,
not just on average.

## Fine-tuning experiment (Appendix D)

A preliminary experiment asks whether the limitation can be mitigated with targeted
supervision. Qwen3-VL-2B is LoRA fine-tuned on temporal anomaly detection using an 80%
split of CLEVRER and evaluated on the held-out 20% plus the three unseen datasets.

Training uses [ms-swift](https://github.com/modelscope/ms-swift): 3 epochs, learning rate
1e-4, effective batch size 16 (2 × 8 gradient accumulation), max sequence length 4096,
LoRA rank 8 / alpha 32, seed 42, single A100. The reported checkpoint is step 714.

```bash
python3 scripts/split_finetune_clevrer.py          # stratified 80/20 scene split
python3 scripts/export_finetune_swap_detect.py     # -> ms-swift SFT JSONL
./finetune.sh datasets/CLEVRER                     # build wheels, rsync, sbatch
python3 scripts/full_matrix_eval.py <lora_path> clevrer
```

The cluster job is `cluster/finetune_swap_detect.sbatch`; evaluation configs are the
`*-finetune-heldout-*.yaml` files in `config/experiments/` and `config/datasets/`.

Fine-tuning improves held-out CLEVRER detection to 96.5% and transfers to the unseen
datasets (CRAFT 63.6%, DriveLM 52.7%, MTL-AQA 68.1%), with the fine-tuned 2B model
matching or exceeding zero-shot Qwen3-VL-8B. These results are preliminary and limited to
one model family; a controlled study across scales and families is left to future work.

## Repository structure

```
scripts/            One script per task: swap_detect.py, swap_localize.py,
                    corrupt_detect.py, corrupt_localize.py, plus dataset
                    preparation (clevrer/, craft/, drivelm/, mtl-aqa/),
                    LPIPS tooling, and figure rendering
config/
  experiments/      One YAML per (task x dataset x model); subdirs for the
                    no-scene-desc and human-subset conditions
  models/           One YAML per model (HF path, tensor parallelism)
  datasets/         One YAML per dataset (manifest, sequence length bounds)
inference.py        Shared vLLM wrapper (Model class, constrained decoding)
dataset.py          Dataset loading, swap/corruption injection
results.py          Results aggregation (Results class used by task scripts)
results/            Output CSVs, one per experiment - source of truth
plotting/           Reproduces the paper's figures and tables from results/
outputs/            Raw per-run output dirs, <task>_<dataset>_<timestamp>/
cluster/            SLURM infrastructure: sbatch jobs, matrix submitters,
                    offline wheel/HF-cache bundling
annotation-tool/    Prolific-compatible human annotation platform
human_subset/       The fixed 72-samples-per-dataset human comparison subset
paper/              LaTeX source and figures
```

## Citation

```bibtex
@inproceedings{hradil2026timecatch,
  title     = {What's the Catch? Evaluating Temporal Consistency in Vision-Language Models},
  author    = {Hradil, Marek and S\'{a}nchez Villegas, Danae},
  year      = {2026}
}
```

## Acknowledgments

This work was supported by a research grant (VIL53122) from VILLUM FONDEN. We acknowledge
the EuroHPC Joint Undertaking for awarding this project access to the EuroHPC
supercomputer MareNostrum 5, hosted by the Barcelona Supercomputing Center (BSC), Spain,
under project ID EHPC-DEV-2025D11-030.
