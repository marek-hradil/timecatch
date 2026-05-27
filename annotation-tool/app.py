"""
Swap annotation tool — run with: uv run python app.py
"""
import sys
import csv
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset import Dataset, SwapDataset  # noqa: E402

ROOT = Path(__file__).parent.parent
ANNOTATIONS_DIR = Path(__file__).parent / "annotations"
MAX_PAIRS = 7  # max adjacent pairs for an 8-frame sequence

DATASETS = {
    "CRAFT":    {"path": ROOT / "datasets/CRAFT",    "manifest": "scenes_filtered.json", "seq_len_min": 4, "seq_len_max": 8},
    "CLEVRER":  {"path": ROOT / "datasets/CLEVRER",  "manifest": "scenes_filtered.json", "seq_len_min": 4, "seq_len_max": 8},
    "MTL-AQA":  {"path": ROOT / "datasets/MTL-AQA",  "manifest": "scenes_filtered.json", "seq_len_min": 4, "seq_len_max": 8},
    "drive_lm": {"path": ROOT / "datasets/drive_lm", "manifest": "scenes_filtered.json", "seq_len_min": 4, "seq_len_max": 8},
}


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load(dataset_name, task, n, seed):
    cfg = DATASETS[dataset_name]
    base = Dataset(str(cfg["path"]), cfg["seq_len_min"], cfg["seq_len_max"], seed=seed, manifest=cfg["manifest"])
    ds = SwapDataset(base)
    return ds.sample_binary(n) if task == "detect" else ds.sample_position(n)


def _out_path(dataset_name, task):
    ANNOTATIONS_DIR.mkdir(exist_ok=True)
    return ANNOTATIONS_DIR / f"{task}_{dataset_name}.csv"


def _done_names(path):
    if not path.exists():
        return set()
    with open(path) as f:
        return {row["sample_name"] for row in csv.DictReader(f)}


def _write(path, name, n_frames, gt, human):
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_name", "n_frames", "ground_truth", "human_answer"])
        if is_new:
            w.writeheader()
        w.writerow({"sample_name": name, "n_frames": n_frames, "ground_truth": str(gt), "human_answer": str(human)})


# ── UI update helpers ─────────────────────────────────────────────────────────

def _render(sample, task, cursor, total):
    """Build the full list of component updates for a single sample."""
    frames = [(img, str(i)) for i, img in enumerate(sample.frames)]
    n = len(sample.frames)
    scene = f"*{sample.scene_description}*" if sample.scene_description else ""
    progress = f"**Sample {cursor + 1} / {total}** — `{sample.name}` ({n} frames)"

    if task == "detect":
        pairs = [gr.update(visible=False)] * MAX_PAIRS
        return progress, frames, scene, gr.update(visible=True), gr.update(visible=True), *pairs

    pairs = [
        gr.update(value=f"Swapped: {i} ↔ {i+1}", visible=(i < n - 1))
        for i in range(MAX_PAIRS)
    ]
    return progress, frames, scene, gr.update(visible=False), gr.update(visible=False), *pairs


def _all_done(total):
    blanks = [gr.update(visible=False)] * MAX_PAIRS
    return (
        f"✅ All {total} samples annotated. Results saved to `annotations/`.",
        None, "",
        gr.update(visible=False), gr.update(visible=False),
        *blanks,
    )


# ── Event handlers ────────────────────────────────────────────────────────────

def start(dataset_name, task, n, seed, _st):
    samples = _load(dataset_name, task, int(n), int(seed))
    path = _out_path(dataset_name, task)
    pending = [s for s in samples if s.name not in _done_names(path)]
    st = {"pending": pending, "cursor": 0, "path": str(path), "task": task}
    if not pending:
        return st, *_all_done(len(samples))
    return st, *_render(pending[0], task, 0, len(pending))


def annotate(answer, st):
    pending = st.get("pending", [])
    cursor = st.get("cursor", 0)
    if not pending or cursor >= len(pending):
        blanks = [gr.update(visible=False)] * MAX_PAIRS
        return st, "Nothing to annotate.", None, "", gr.update(visible=False), gr.update(visible=False), *blanks  # noqa: E501

    sample = pending[cursor]
    path = Path(st["path"])
    gt = sample.anomaly if st["task"] == "detect" else sample.position
    _write(path, sample.name, len(sample.frames), gt, answer)

    cursor += 1
    st = {**st, "cursor": cursor}
    if cursor >= len(pending):
        return st, *_all_done(len(pending))
    return st, *_render(pending[cursor], st["task"], cursor, len(pending))


# ── Layout ────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Swap Annotation Tool", theme=gr.themes.Base()) as demo:
    st = gr.State({})

    gr.Markdown(
        "## Swap Annotation Tool\n"
        "Frames are shown left→right in temporal order. "
        "**detect**: did any adjacent pair get swapped? "
        "**localize**: click the swapped pair."
    )

    with gr.Row():
        dataset_dd = gr.Dropdown(list(DATASETS), value="CRAFT", label="Dataset")
        task_dd    = gr.Dropdown(["detect", "localize"], value="detect", label="Task")
        n_sl       = gr.Slider(10, 60, value=30, step=5, label="N samples")
        seed_nb    = gr.Number(42, label="Seed", precision=0)
        start_btn  = gr.Button("▶  Start / Resume", variant="primary")

    progress_md = gr.Markdown("Select a dataset and task, then click **Start / Resume**.")
    gallery     = gr.Gallery(label="Frames", columns=8, height=260, object_fit="contain", show_label=True)
    scene_md    = gr.Markdown("")

    with gr.Row():
        yes_btn = gr.Button("✓  YES — out of order", variant="primary",  visible=False)
        no_btn  = gr.Button("✗  NO — correct order", variant="secondary", visible=False)

    with gr.Row():
        pair_btns = [gr.Button(f"Swapped: {i} ↔ {i+1}", visible=False) for i in range(MAX_PAIRS)]

    outputs = [st, progress_md, gallery, scene_md, yes_btn, no_btn, *pair_btns]

    start_btn.click(start, [dataset_dd, task_dd, n_sl, seed_nb, st], outputs)
    yes_btn.click(lambda s: annotate(True,  s), [st], outputs)
    no_btn.click( lambda s: annotate(False, s), [st], outputs)
    for i, btn in enumerate(pair_btns):
        btn.click(lambda s, idx=i: annotate((idx, idx + 1), s), [st], outputs)


if __name__ == "__main__":
    demo.launch()
