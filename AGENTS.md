# Temporal Misalignment — Project Guide

## What This Project Is

**Paper:** "What's the Catch? Evaluating Temporal Consistency in Vision-Language Models" — Marek Hradil (University of Copenhagen) and Danae Sánchez Villegas (University of Copenhagen / CAISA). Camera-ready, de-anonymized. Contact: `rsc784@alumni.ku.dk`, `davi@samf.ku.dk`.

> ⚠️ **`paper/latex/acl_latex.tex` on disk is still the older anonymous submission** (4 tables, no fine-tuning appendix). The camera-ready adds Appendix D's fine-tuning experiment and softens several claims. Until the tex is updated, treat this file and `README.md` as the description of the camera-ready, not the on-disk tex.

Funding to credit: VILLUM FONDEN grant VIL53122; EuroHPC / MareNostrum 5 (BSC) under project ID EHPC-DEV-2025D11-030. BSC is a compute acknowledgment, not an affiliation.

**Benchmark name: TimeCatch.** The paper's formal task names differ from the repo's script names — keep this mapping in mind when reading the paper vs. the code:

| Paper term | Repo script |
|---|---|
| Temporal Anomaly Detection ("Temporal Detect") | `scripts/swap_detect.py` |
| Temporal Anomaly Localization ("Temporal Localize") | `scripts/swap_localize.py` |
| Frame-Level Anomaly Detection ("Frame Detect") | `scripts/corrupt_detect.py` |
| Frame-Level Anomaly Localization ("Frame Localize") | `scripts/corrupt_localize.py` |

("Frame-level anomaly" in the paper = the corruption/Gaussian-noise baseline in the repo — same thing, different name.)

**Core claim:** Current open-weight VLMs detect individual-frame anomalies (corrupted frames) well above chance, but *in the main evaluation setting* generally score near random chance on detecting a swap of two consecutive frames, with limited localization. Humans are near-ceiling on both.

Mind the hedging — the camera-ready deliberately weakened the earlier absolute framing. Some conditions **do** improve temporal performance (larger scale; removing scene descriptions on some datasets; fine-tuning). The claim is a persistent *gap* between frame-level and temporal anomaly performance across the conditions examined, **not** that temporal performance is immovable. Do not write "models are at chance, full stop" or "capacity is ruled out."

**Two tasks:**
- **Swap detection** — binary: are two consecutive frames swapped? (random chance = 50%)
- **Swap localization** — which pair is swapped? (random chance = 1/(n-1))

**Two baselines used in evaluation:**
- **Corruption baseline** — one frame replaced with Gaussian noise; models perform near-ceiling here, confirming they attend to the sequence
- **Human baseline** — per-dataset ranges, not single numbers: 75.0–91.7% temporal detection, 83.3–88.9% temporal localization (see the Human Study section below)

**Key finding on prompting:** Removing scene descriptions has *very different* magnitudes by task — it produces only modest changes on temporal tasks, but massive gains on frame-level localization specifically (e.g. InternVL3.5-8B: +39.6pp on CLEVRER, +66.0pp on MTL-AQA frame-localize; see `tab:no_scene_desc_delta`). Scene descriptions appear to act as a textual distractor/shortcut more than a temporal-attention shortcut — the effect is concentrated on the *frame-level* control task, not the temporal one. Enabling "thinking"/reasoning mode gives mixed, dataset-dependent results on temporal tasks with no consistent overall benefit.

**Key finding on localization errors:** Confusion matrices (`fig:localize-heatmap-qwen`, `fig:localize-heatmap`) show wrong predictions are *not* near-misses clustered around the true swap location. Qwen3-VL-8B and InternVL3.5-8B exhibit a strong, dataset-independent bias toward predicting position 2 regardless of ground truth; Gemma-4, Qwen2.5-VL, and InternVL3 instead produce near-uniform (i.e. near-random) predictions across positions. Two distinct failure modes, both consistent with "not actually localizing," but useful to distinguish when writing analysis code over `results/`.

**Other robustness checks (mostly negative, but not uniformly — read the scale bullet carefully):**
- Model scale (Qwen3-VL 2B/4B/8B/32B): frame-level tasks scale to near-ceiling. **Temporal detection also improves with scale, particularly from 4B to 32B**, though by less than the frame-level tasks; temporal localization improves less consistently and stays comparatively low. ⚠️ An earlier revision of this file said temporal tasks "improve only marginally even at 32B" — that is *not* the camera-ready claim. Capacity contributes; it just does not close the gap.
- Sequence length (CRAFT-long, up to 16 frames): temporal detect/localize accuracy is flat across lengths; frame-level *localization* actually degrades as sequences get longer (more candidate positions). The n=16 bucket has only 5 samples — treat as noisy if reproducing `fig:by-length`.
- Visual similarity (LPIPS-binned, 0.1-wide bins from 0–0.9): temporal detection accuracy ticks up slightly for perceptually distant swapped pairs but stays far below human level in every bin.

---

## Datasets

Four datasets used in the main evaluation, all filtered to 4–8 frames per sequence. **CRAFT is split in two**: after LPIPS filtering there are 1,959 CRAFT sequences total (avg 9.0 frames), but only the 4–8-frame subset (858 sequences, avg 6.9 frames) is used in the main results table. The remaining longer sequences are pulled out into a separate **CRAFT-long** subset (1,101 sequences, avg 10.6 frames, spanning 9–16 frames) used *only* for the sequence-length analysis (paper §6.5, `sec:sec_length`) — don't conflate the two when reading `results/` CSVs or plotting scripts.

| Dataset     | Content                          | Sequences | Avg frames | Used for |
|-------------|-----------------------------------|-----------|------------|----------|
| CLEVRER     | Abstract 3D objects, collisions   | 4,997     | 7.5        | main results |
| CRAFT       | Abstract 2D objects, collisions   | 858       | 6.9        | main results |
| DriveLM     | Egocentric driving (not video-derived; native nuScenes image sequences) | 696 | 5.9 | main results |
| MTL-AQA     | Olympic competitive diving        | 338       | 4.5        | main results |
| CRAFT-long  | Abstract 2D objects, collisions   | 1,101     | 10.6       | sequence-length ablation only |

Mean LPIPS (post-filter) by dataset: CRAFT 0.093, CLEVRER 0.131, DriveLM 0.428, MTL-AQA 0.584 — real-world datasets have much larger consecutive-frame differences than the synthetic ones (mostly static backgrounds).

**Frame sampling:** 1 frame/1.5 s for CRAFT/CLEVRER; 1 frame/s for MTL-AQA; DriveLM is not sampled from video (uses nuScenes' native discrete image sequences). Consecutive pairs below LPIPS threshold T=0.05 are dropped. Sequences falling below 4 frames after filtering are discarded.

**Random-chance baselines** (for sanity-checking new results against `results.py` aggregation): detection is always 50%. Localization is 1/(n−1) for temporal tasks (swap can't be at the last position) vs. 1/n for frame tasks (corruption can land anywhere) — e.g. temporal-localize random ≈ 4.3% (CLEVRER) / 5.5% (CRAFT) / 8.7% (DriveLM) / 13.2% (MTL-AQA), while frame-localize random ≈ 13.5% / 14.9% / 18.1% / 22.4% for the same datasets, respectively.

Scene descriptions are appended as plain text after image tokens. For CRAFT/CLEVRER: collision/event annotations serialized chronologically. For DriveLM: nuScenes scene metadata. For MTL-AQA: dive type (rotation, twist count, body position, armstand flag).

---

## Models Evaluated

All ~7–9B parameter open-weight VLMs, served with vLLM + constrained decoding (yes/no for detection; integer index for localization):

- **Qwen3-VL-8B** — best overall performer
- **Qwen2.5-VL-7B**
- **InternVL3-8B**
- **InternVL3.5-8B**
- **Gemma-4-E4B**

Model scale ablation done on Qwen3-VL at 2B / 4B / 8B / 32B. Corruption tasks scale well with size; swap tasks do not.

---

## Repo Structure

```
scripts/          # One script per task: swap_detect.py, swap_localize.py,
                  #   corrupt_detect.py, corrupt_localize.py
config/
  experiments/    # One YAML per (task × dataset × model) combination
  models/         # One YAML per model (model_id, path, etc.)
inference.py      # Shared vLLM inference wrapper (Model class, constrained decoding)
dataset.py        # Dataset loading utilities
results.py        # Results aggregation
results/          # Output CSVs (one per experiment), rsync'd back from cluster
plotting/         # Reproduces all paper figures from results/ CSVs
outputs/          # Raw per-run output dirs (named <task>_<dataset>_<timestamp>/)
cluster/          # All cluster infrastructure (see below)
annotation-tool/  # Custom human-study annotation platform (Prolific-compatible)
main.py           # Early prototype / local testing harness (uses nobodywho, not vLLM)
```

---

## Prompts

### Main task prompts (used in all experiments)

**Swap detection:**
> You are given a sequence of images showing a scene unfolding over time. The frames should appear in a natural temporal order, but two consecutive frames may have been swapped. Reply with only 'yes' if you detect a swap, or 'no' if the order looks correct.

**Swap localization:**
> You are given a sequence of images showing a scene unfolding over time. Exactly two consecutive frames have been swapped. Reply with only the two frame numbers that are out of order, separated by a comma. For example: 3,4 means frames 3 and 4 were swapped. Use 1-based indexing (1 for the first frame).

**Corrupt detection:**
> You are given a sequence of images showing a scene unfolding over time. One frame may have been replaced by random noise. Reply with only 'yes' if you see a corrupted frame, or 'no' if all frames look normal.

**Corrupt localization:**
> You are given a sequence of images showing a scene unfolding over time. Exactly one frame has been replaced with random noise. Reply with only the number of the corrupted frame. Use 1-based indexing (1 for the first frame).

### Prompt variants (ablation, swap detection only)
- **A** (base): above
- **B**: ask if sequence is in correct temporal order with no frames swapped
- **C**: examine each consecutive pair — does transition make physical sense?
- **D**: minimal — "Look at these images in order. Have any two neighboring images been swapped?"
- **E**: step-through chain-of-thought style, asking about each transition

---

## Cluster Infrastructure (BSC MareNostrum 5)

### Setup

Copy and fill in `cluster/env.sh` (gitignored; see `cluster/env.sh.example`):
```bash
CLUSTER_USER, CLUSTER_TRANSFER_HOST, CLUSTER_INTERNET_HOST, CLUSTER_LOGIN_HOST
PROJECT, ACCOUNT, REMOTE_DIR, HF_CACHE, CLUSTER_PYTHON_VERSION
```

### Key scripts

| Script | What it does |
|--------|-------------|
| `cluster/prepare.sh <exp.yaml>` | On login node: creates venv, installs from local wheels, checks HF model is present in cache, submits job |
| `cluster/submit.sh <exp.yaml>` | Lightweight sbatch wrapper (assumes venv already prepared) |
| `cluster/run.sbatch` | Actual compute job — loads modules, activates venv, runs `scripts/$SCRIPT.py $EXP` |
| `cluster/run_matrix.sh` | Submits all experiments for a model family, polls until done, consolidates CSVs |
| `cluster/run_matrix_model.sh` | Variant targeting a specific model |
| `cluster/run_matrix_no_scene_desc.sh` | Variant for the no-scene-description ablation |
| `cluster/stream_model.sh` | Interactive model streaming for debugging |

### Module stack on compute nodes
```
oneapi  hdf5  python  sqlite3/3.45.2-gcc  nvidia-hpc-sdk/25.3
```
Environment notes:
- `PYTHONHOME`/`PYTHONPATH` unset to isolate venv from system site-packages
- `LD_LIBRARY_PATH` extended with nvidia-hpc-sdk CUDA 12.8 compat libs (cluster native driver = CUDA 12.2; forward-compat allows torch+cu128 / vLLM 0.11)
- `CC=gcc`, `CXX=g++` — overrides `nvc` placed first on PATH by nvhpc module
- `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` — for Molmo's 4096 position embedding limit
- `VLLM_USE_V1=0` — V1 engine breaks Gemma-4's dummy-input generation in vLLM 0.19.1
- HF models loaded offline: `TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1`

### GPU: single NVIDIA A100 per job

### Rsync pattern (laptop → cluster)
```bash
rsync -avz ./ $CLUSTER_USER@$CLUSTER_TRANSFER_HOST:$REMOTE_DIR/
```
Results back:
```bash
rsync -avz $CLUSTER_TRANSFER_HOST:$REMOTE_DIR/results-<model>/ ./results/<model>/
```

---

## Human Study

Custom annotation platform (`annotation-tool/`) integrating with Prolific. **Study is complete** (paper reports final numbers, not a pilot):
- Shows image sequence + scene description; user clicks detect or localize
- Walks annotators through guidelines with visual examples and a video walkthrough before annotation
- Interactive frame-by-frame viewer (arrow-key navigation) available during annotation — found to meaningfully help annotators spot swaps
- 24 Prolific participants total (3 per dataset × task configuration), all with ≥ undergraduate degree, 16 countries, mean age 35.2, 16M/8F. Mean pay > £8/hour (Prolific minimum £6/hour). Mean completion time 9.8 min for 15 sequences.
- **Quality control:** 3 attention-check sequences (manually chosen to be maximally easy) inserted per participant's task, excluded from results; participants failing ≥ 2 of the 3 are rejected.
- Human-vs-model comparison in the paper (`fig:human_results`) is run on a fixed subset of **72 samples per dataset**, not the full annotated set.

Human scores are **per-dataset ranges, not single numbers** — do not quote "91.7%/97.5%" as overall figures:
- Temporal detection: 75.0%–91.7% across datasets (best: CRAFT; worst: DriveLM)
- Temporal localization: 83.3%–88.9% across datasets
- Frame-level tasks: ~100% for humans (trivial control condition)
- No evaluated VLM exceeds 70% temporal-detect accuracy or 50% temporal-localize accuracy on any dataset — the human/VLM gap holds dataset-by-dataset, not just on average.

---

## Results Summary

In the **main (zero-shot) evaluation setting**, temporal detection is generally near random chance (50%) and temporal localization is limited: across the four datasets the highest detection accuracy by any model is 57.4% and the highest localization accuracy is 43.2% (both Qwen3-VL-8B on MTL-AQA). Frame-level tasks reach up to 99.6% (detect) and 99.4% (localize). Best model overall: **Qwen3-VL-8B**.

Scope that claim to the main setting — it does not hold across scale or after fine-tuning (see below).

Full per-dataset, per-model numbers in `tab:detailed_results` (`results/` CSVs are the source of truth — the table is generated from them). Figures reproduced by running scripts in `plotting/` against CSVs in `results/`.

---

## Fine-Tuning Experiment (Appendix D)

Added in the camera-ready; **absent from the anonymous submission and from `paper/latex/acl_latex.tex` on disk.** If you grep the on-disk tex for "fine-tun" you get zero hits and will wrongly conclude this track is dead code. It is not.

A preliminary experiment: Qwen3-VL-2B LoRA fine-tuned on temporal anomaly detection using an 80% split of CLEVRER, evaluated on the held-out 20% plus the three unseen datasets (transfer — the other datasets are **not** trained on).

Hyperparameters (all from `cluster/finetune_swap_detect.sbatch`, verified to match the paper): ms-swift, LoRA rank 8 / alpha 32, 3 epochs, lr 1e-4, effective batch size 16 (per-device 2 × grad-accum 8), max_length 4096, seed 42, single A100. Reported checkpoint: step 714.

Results (temporal detection accuracy, %):

| | CLEVRER | CRAFT | DriveLM | MTL-AQA |
|---|---|---|---|---|
| Qwen3-VL-2B zero-shot | 49.6 | 50.5 | 50.0 | 44.1 |
| Qwen3-VL-2B fine-tuned | 96.5 | 63.6 | 52.7 | 68.1 |
| Qwen3-VL-8B zero-shot (reference) | 51.9 | 53.3 | 51.9 | 57.4 |

The fine-tuned 2B matches or exceeds zero-shot 8B. Framing in the paper is deliberately cautious: preliminary, one model family, suggests *part* of the limitation is attributable to available supervision rather than scale alone.

### Which finetune files are load-bearing

Only the CLEVRER-trained path is in the paper. Keep:
- `scripts/split_finetune_clevrer.py` → stratified 80/20 scene split
- `scripts/export_finetune_swap_detect.py` → ms-swift SFT JSONL
- `scripts/full_matrix_eval.py` → held-out + transfer eval (produces the table)
- `cluster/finetune_swap_detect.sbatch`, `cluster/full_matrix_eval.sbatch`
- `finetune.sh`, `cluster/prepare_finetune.sh`, `cluster/requirements-finetune.txt`
- `config/models/qwen3-vl-2b-lora-swap-detect-clevrer.yaml`, `config/datasets/clevrer-finetune-heldout.yaml`, the `swap-*-clevrer-finetune-heldout-*.yaml` experiment configs

Everything else in the finetune track is exploratory and unused by the paper: the per-dataset training pipelines for CRAFT / MTL-AQA / DriveLM (`split_finetune_craft*`, `expand_finetune_*`, `export_finetune_swap_detect_{craft,mtlaqa,drivelm,no_scenedesc}`), all `*_combined*` variants, `cap_finetune_train.py`, `subsample_clevrer_train.py`, `check_overfit.py`, `verify_checkpoint.py`, `combined_eval.py`, `eval_clevrer_no_scenedesc.py`, and `datasets/combined/`. (`checkpoint_sweep.py` is the only file mentioning step 714, so it documents how that checkpoint was chosen even though it is not needed to reproduce.)

> ⚠️ **`eval_clevrer_no_scenedesc.py` is finetune-side, not the paper's no-scene-description ablation.** That ablation lives in `config/experiments/no-scene-desc/` + `cluster/run_matrix_no_scene_desc.sh` → `results/no-scene-desc/`.

### Known gap

**The fine-tuning numbers are not reproducible from this repo.** There are no finetune CSVs under `results/`, no matching run dirs in `outputs/`, and no `checkpoints/`. Unlike every other table in the paper, `tab:finetuning` does not trace back to a committed artifact — the values live only on the cluster or in SLURM logs. Recover them before release.
