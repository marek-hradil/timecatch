# Annotation Tool

Web-based human annotation platform for the temporal-misalignment study.
Participants are shown video frame sequences and asked to detect or localise a swap of two consecutive frames.

## Live Deployment

**URL:** `https://annotation-tool-649785408950.europe-west1.run.app`

Study links follow the pattern:
```
https://annotation-tool-649785408950.europe-west1.run.app/?PROLIFIC_PID={{%PROLIFIC_PID%}}&dataset=CRAFT&task=detect
```

Valid values: `dataset` ∈ {CRAFT, CLEVRER, MTL-AQA, drive_lm} · `task` ∈ {detect, localize}

## Tasks

- **Detect** — given a sequence of 4–8 frames, decide whether two consecutive frames were swapped (binary yes/no). ~50% of sequences are actually swapped.
- **Localize** — given a sequence where a swap is always present, identify which adjacent pair was swapped. Buttons are labelled "Frame i ↔ i+1".

## Architecture

| Layer | Technology |
|-------|-----------|
| Frontend | React 19 + TypeScript + Vite + Zustand |
| Backend | FastAPI + uvicorn |
| Database | Firestore (project: `lamp-visual-misalignment`) |
| Frame storage | GCS bucket `lamp-annotation-datasets` (public read) |
| Hosting | Google Cloud Run, region `europe-west1` |
| Container registry | `europe-west1-docker.pkg.dev/lamp-visual-misalignment/annotation-tool/app` |

Frame images and instruction media are served directly from GCS (bypassing Cloud Run) at:
`https://storage.googleapis.com/lamp-annotation-datasets/`

## Running Locally

```bash
# Backend (port 8000)
uv run uvicorn backend.main:app --reload

# Frontend (port 5173, proxies /api and /frames to backend)
cd frontend && npm run dev
```

Open: `http://localhost:5173/?PROLIFIC_PID=test123&dataset=CRAFT&task=detect`

Note: same `PROLIFIC_PID` always yields the same sequences (deterministic seed via SHA-256). Use different PIDs to see different sequences.

## Redeploying

```bash
# From repo root (temporal-misalignment/)
gcloud builds submit --config annotation-tool/cloudbuild.yaml .

gcloud run deploy annotation-tool \
  --image europe-west1-docker.pkg.dev/lamp-visual-misalignment/annotation-tool/app:latest \
  --region europe-west1 --platform managed --allow-unauthenticated \
  --port 8080 --memory 1Gi --cpu 1 --min-instances 1 --max-instances 4 \
  --add-volume name=datasets,type=cloud-storage,bucket=lamp-annotation-datasets \
  --add-volume-mount volume=datasets,mount-path=/datasets \
  --set-env-vars DATASETS_ROOT=/datasets/datasets,GCS_PUBLIC_BASE=https://storage.googleapis.com/lamp-annotation-datasets
```

## Session & Sampling

- Sessions are keyed by `(prolific_pid, dataset, task)` — same triple always resumes the same session.
- Seed = `SHA-256(prolific_pid) % 2^31`. Deterministic across restarts.
- 15 samples per participant. Detect: ~50/50 swapped/unswapped. Localize: always swapped.
- Swap position drawn uniformly from the n−1 adjacent pairs.
- Annotations are saved to Firestore immediately on each response.
- To reset a test session, delete the Firestore documents with the matching `(prolific_pid, dataset, task)` key.

## Firestore Collections

- `participants` — doc ID: `{pid}__{dataset}__{task}`, fields: prolific_pid, dataset, task, seed, created_at, completed_at
- `annotations` — doc ID: `{pid}__{dataset}__{task}__{sample_index}`, fields: prolific_pid, dataset, task, sample_index, sample_name, n_frames, ground_truth, human_answer, answered_at

## Instruction Files

Per-condition markdown files live in `backend/instructions/{DATASET}_{TASK}.md`.
Images and videos referenced in instructions are stored in `introductions/` locally and at `gs://lamp-annotation-datasets/introductions/` in GCS.

## Human Study Results (local annotations)

2 participants per condition · 15 sequences each · 30 annotations per condition · 240 total.

| Task | CRAFT | CLEVRER | DriveLM | MTL-AQA | Avg |
|------|-------|---------|---------|---------|-----|
| Detect | 100.0% | 90.0% | 93.3% | 83.3% | 91.7% |
| Localize | 100.0% | 96.7% | 96.7% | 96.7% | 97.5% |
| **Overall** | | | | | **94.6%** |
