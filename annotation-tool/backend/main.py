"""
FastAPI application — endpoints + static file mounts only.
All sampling/ground-truth logic lives in study.py and sampling.py.
"""
import re
from pathlib import Path
from typing import Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import (
    DATASETS_ROOT,
    VALID_DATASETS,
    VALID_TASKS,
    prolific_completion_url,
    SAMPLES_PER_PARTICIPANT,
)
from .db import (
    count_annotations,
    init_db,
    mark_completed,
    update_instruction_metrics,
    upsert_annotation,
    upsert_participant,
)
from .study import build_plan, ground_truth_for, pid_to_seed

# ── Startup validation ────────────────────────────────────────────────────────

_INSTRUCTIONS_DIR = Path(__file__).parent / "instructions"

def _load_instructions() -> dict[str, str]:
    """Load all 8 instruction files at startup; fail loudly on missing."""
    texts: dict[str, str] = {}
    for dataset in VALID_DATASETS:
        for task in VALID_TASKS:
            key = f"{dataset}_{task}"
            path = _INSTRUCTIONS_DIR / f"{key}.md"
            if not path.exists():
                raise RuntimeError(f"Missing instruction file: {path}")
            texts[key] = path.read_text(encoding="utf-8")
    return texts


_INSTRUCTIONS = _load_instructions()
init_db()

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Temporal Annotation Tool", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve raw frame images: /frames/{dataset}/scenes/{scene}/{N}.jpg
app.mount("/frames", StaticFiles(directory=str(DATASETS_ROOT)), name="frames")

# Serve instruction media (screenshots, videos)
_INTRODUCTIONS_DIR = Path(__file__).parent.parent / "introductions"
if _INTRODUCTIONS_DIR.exists():
    app.mount("/introductions", StaticFiles(directory=str(_INTRODUCTIONS_DIR)), name="introductions")

# In production, serve the built React app.
# Use an explicit SPA fallback route rather than StaticFiles at "/" — mounting
# StaticFiles at "/" is a catch-all that intercepts POST /api/* and returns 405.
_FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(_FRONTEND_DIST / "assets")), name="frontend_assets")


# ── Input validation helpers ──────────────────────────────────────────────────

_PID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")

def _validate_pid(pid: str) -> None:
    if not pid or not _PID_RE.match(pid):
        raise HTTPException(status_code=400, detail="Invalid or missing PROLIFIC_PID")

def _validate_condition(dataset: str, task: str) -> None:
    if dataset not in VALID_DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset {dataset!r}")
    if task not in VALID_TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task {task!r}")


# ── Schemas ───────────────────────────────────────────────────────────────────

class SessionRequest(BaseModel):
    prolific_pid: str
    dataset: str
    task: str
    scene_id: str = "main"
    study_id: str | None = None
    session_id: str | None = None


class SessionMetricsRequest(BaseModel):
    prolific_pid: str
    dataset: str
    task: str
    scene_id: str = "main"
    instructions_duration_s: float | None = None
    instructions_lightbox_opens: int | None = None
    video_played: bool | None = None


class SampleOut(BaseModel):
    sample_index: int
    sample_name: str
    n_frames: int
    frame_urls: list[str]
    scene_description: str | None


class SessionResponse(BaseModel):
    prolific_pid: str
    dataset: str
    task: str
    samples: list[SampleOut]
    instructions_md: str
    prolific_completion_url: str
    total: int


class AnnotateRequest(BaseModel):
    prolific_pid: str
    dataset: str
    task: str
    sample_index: int
    human_answer: Union[bool, list[int]]  # bool for detect, [i, j] for localize
    scene_id: str = "main"
    view_duration_s: float | None = None
    viewer_opens: int | None = None


class AnnotateResponse(BaseModel):
    ok: bool
    annotated: int
    total: int
    completed: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/api/session", response_model=SessionResponse)
def create_session(req: SessionRequest) -> SessionResponse:
    _validate_pid(req.prolific_pid)
    _validate_condition(req.dataset, req.task)

    seed = pid_to_seed(req.prolific_pid)
    upsert_participant(
        req.prolific_pid, req.dataset, req.task, seed,
        scene_id=req.scene_id, study_id=req.study_id, session_id=req.session_id,
    )

    plan = build_plan(req.prolific_pid, req.dataset, req.task)
    instructions_key = f"{req.dataset}_{req.task}"

    return SessionResponse(
        prolific_pid=req.prolific_pid,
        dataset=req.dataset,
        task=req.task,
        samples=[
            SampleOut(
                sample_index=s.sample_index,
                sample_name=s.sample_name,
                n_frames=s.n_frames,
                frame_urls=s.frame_urls,
                scene_description=s.scene_description,
            )
            for s in plan
        ],
        instructions_md=_INSTRUCTIONS[instructions_key],
        prolific_completion_url=prolific_completion_url(req.dataset, req.task),
        total=len(plan),
    )


@app.post("/api/annotate", response_model=AnnotateResponse)
def annotate(req: AnnotateRequest) -> AnnotateResponse:
    _validate_pid(req.prolific_pid)
    _validate_condition(req.dataset, req.task)

    n = SAMPLES_PER_PARTICIPANT
    if not (0 <= req.sample_index < n):
        raise HTTPException(status_code=400, detail=f"sample_index must be 0..{n-1}")

    # Resolve ground truth server-side
    gt_list = ground_truth_for(req.prolific_pid, req.dataset, req.task, n)
    gt = gt_list[req.sample_index].ground_truth

    # Normalise human_answer
    if req.task == "detect":
        if not isinstance(req.human_answer, bool):
            raise HTTPException(status_code=400, detail="detect answer must be a boolean")
        ha: Union[bool, tuple[int, int]] = req.human_answer
    else:
        if not isinstance(req.human_answer, list) or len(req.human_answer) != 2:
            raise HTTPException(status_code=400, detail="localize answer must be [i, j]")
        ha = (int(req.human_answer[0]), int(req.human_answer[1]))

    # Fetch sample metadata from the plan
    plan = build_plan(req.prolific_pid, req.dataset, req.task, n)
    sample = plan[req.sample_index]

    upsert_annotation(
        req.prolific_pid,
        req.dataset,
        req.task,
        req.sample_index,
        sample.sample_name,
        sample.n_frames,
        gt,
        ha,
        scene_id=req.scene_id,
        is_attention_check=sample.is_attention_check,
        view_duration_s=req.view_duration_s,
        viewer_opens=req.viewer_opens,
    )

    annotated = count_annotations(req.prolific_pid, req.dataset, req.task, req.scene_id)
    completed = annotated >= n

    if completed:
        mark_completed(req.prolific_pid, req.dataset, req.task, req.scene_id)

    return AnnotateResponse(ok=True, annotated=annotated, total=n, completed=completed)


@app.post("/api/session/metrics", response_model=dict)
def session_metrics(req: SessionMetricsRequest) -> dict:
    _validate_pid(req.prolific_pid)
    _validate_condition(req.dataset, req.task)
    update_instruction_metrics(
        req.prolific_pid, req.dataset, req.task, req.scene_id,
        req.instructions_duration_s, req.instructions_lightbox_opens, req.video_played,
    )
    return {"ok": True}


# ── SPA fallback (production only) ───────────────────────────────────────────

if _FRONTEND_DIST.exists():
    _INDEX_HTML = _FRONTEND_DIST / "index.html"

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str) -> FileResponse:
        candidate = _FRONTEND_DIST / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_INDEX_HTML)
