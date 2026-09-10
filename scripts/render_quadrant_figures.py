"""
Generate quadrant figures for specific scenes:
  both_correct:  CRAFT scene_5_002227, MTL-AQA scene_26_117
  llm_correct:   Drive-LM scene_429  (human wrong, LLMs right)
  both_wrong:    MTL-AQA localize scene_02_77

Saves PDFs to figures/ alongside the existing human_failure_* files.
"""

import csv
import json
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent.parent

DATASETS = ["clevrer", "craft", "drive-lm", "mtl-aqa"]
TASKS    = ["detect", "localize"]
MODELS   = ["qwen2-5-vl-7b", "qwen3-vl-8b", "gemma-4-e4b", "intern-vl-3", "intern-vl-3-5"]

DS_PATH = {
    "clevrer":  ROOT / "datasets/CLEVRER",
    "craft":    ROOT / "datasets/CRAFT",
    "drive-lm": ROOT / "datasets/drive_lm",
    "mtl-aqa":  ROOT / "datasets/MTL-AQA",
}

THUMB_W  = 220
THUMB_H  = 160
PAD      = 10
LABEL_H  = 22
DESC_H   = 20
BORDER   = 4

# Dark background palette
BG       = (18, 22, 30)
FG_TITLE = (235, 240, 255)
DIM      = (130, 140, 155)
RED      = (210, 65, 65)
GREEN    = (65, 195, 110)
ORANGE   = (210, 140, 50)   # for category banner

MODEL_SHORT = {
    "qwen2-5-vl-7b": "Qwen2.5-7B",
    "qwen3-vl-8b":   "Qwen3-8B",
    "gemma-4-e4b":   "Gemma4-E4B",
    "intern-vl-3":   "IVL3-8B",
    "intern-vl-3-5": "IVL3.5-8B",
}

CATEGORY_COLORS = {
    "both_correct": (50, 160, 90),
    "llm_correct":  (60, 120, 200),
    "both_wrong":   (170, 55, 55),
}

CATEGORY_LABELS = {
    "both_correct": "BOTH CORRECT",
    "llm_correct":  "LLM CORRECT · HUMAN WRONG",
    "both_wrong":   "BOTH WRONG",
}


def load_font(size):
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def scene_dir(ds, scene_name):
    base = DS_PATH[ds]
    candidate = base / "scenes" / scene_name
    if candidate.exists():
        return candidate
    for p in base.rglob(scene_name):
        if p.is_dir():
            return p
    return None


def load_frames(ds, scene_name, shown_order):
    d = scene_dir(ds, scene_name)
    if d is None:
        raise FileNotFoundError(f"Scene dir not found for {ds}/{scene_name}")
    stem_to_path = {}
    for p in d.iterdir():
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            try:
                stem_to_path[int(p.stem)] = p
            except ValueError:
                pass
    return [Image.open(stem_to_path[i]).convert("RGB") for i in shown_order]


def load_description(ds, scene_name):
    manifest = DS_PATH[ds] / "scenes_filtered.json"
    if not manifest.exists():
        return ""
    with open(manifest) as f:
        scenes = json.load(f)
    for s in scenes:
        paths = s.get("scene_paths", [])
        if paths and Path(paths[0]).parent.name == scene_name:
            return s.get("scene_description", "")
    return ""


_PAIR_GT = re.compile(r"^\(\s*(\d+)\s*,\s*(\d+)\s*\)$")
_PAIR_AN = re.compile(r"^(\d+)\s*,\s*(\d+)$")


def parse_pair(s):
    m = _PAIR_GT.match(str(s).strip()) or _PAIR_AN.match(str(s).strip())
    return (int(m.group(1)), int(m.group(2))) if m else None


def model_correct(task, row):
    if task == "detect":
        ans = row["answer"].strip().lower()
        if ans not in ("yes", "no"):
            return None
        return (str(row["ground_truth"]).strip() == "True") == (ans == "yes")
    gt = parse_pair(row["ground_truth"])
    ans = parse_pair(row["answer"])
    if gt is None or ans is None:
        return None
    return ans == (gt[0] + 1, gt[1] + 1)


def format_model_answer(task, row):
    if task == "detect":
        return row["answer"].strip().lower()
    ans = parse_pair(row["answer"])
    return f"pos {ans[0]-1},{ans[1]-1}" if ans else row["answer"].strip()


def format_human_answer(task, ha):
    if task == "detect":
        return "yes" if ha is True else "no"
    if isinstance(ha, list) and len(ha) == 2:
        return f"pos {ha[0]},{ha[1]}"
    return str(ha)


def human_correct_fn(task, stimulus, ann):
    gt_raw = stimulus["ground_truth"]
    ha = ann["human_answer"]
    if task == "detect":
        return bool(gt_raw) == bool(ha)
    if isinstance(gt_raw, list) and isinstance(ha, list):
        return tuple(int(x) for x in gt_raw) == tuple(int(x) for x in ha)
    return None


def swap_highlight_indices(task, shown_order, ground_truth):
    n = len(shown_order)
    if task == "detect" and ground_truth is True:
        sorted_order = sorted(shown_order)
        for i in range(n - 1):
            if shown_order[i] != sorted_order[i]:
                return {i, i + 1}
    elif task == "localize" and isinstance(ground_truth, list):
        return {ground_truth[0], ground_truth[1]}
    return set()


def render_strip(ds, task, scene_name, shown_order, ground_truth,
                 thumb_w=320, thumb_h=220, pad=6, border=5):
    """Render just the frame strip with red borders on swapped frames."""
    frames = load_frames(ds, scene_name, shown_order)
    n = len(frames)
    hi = swap_highlight_indices(task, shown_order, ground_truth)

    W = pad + n * (thumb_w + pad)
    H = pad + thumb_h + pad

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for i, (frame, _) in enumerate(zip(frames, shown_order)):
        x = pad + i * (thumb_w + pad)
        y = pad
        is_hi = i in hi
        bc = RED if is_hi else (200, 200, 200)
        draw.rectangle(
            [x - border, y - border, x + thumb_w + border, y + thumb_h + border],
            outline=bc, width=border,
        )
        canvas.paste(frame.resize((thumb_w, thumb_h)), (x, y))

    return canvas


def render(ds, task, scene_name, shown_order, ground_truth, description,
           model_answers, human_answers, category):
    frames = load_frames(ds, scene_name, shown_order)
    n = len(frames)
    swap_indices = swap_highlight_indices(task, shown_order, ground_truth)

    font_sm    = load_font(11)
    font_md    = load_font(12)
    font_title = load_font(13)
    font_cat   = load_font(13)

    MODEL_ROW_H = 19
    n_rows = len(human_answers) + 1 + len(model_answers)
    models_block_h = PAD + n_rows * MODEL_ROW_H + PAD

    BANNER_H = 26
    total_w = PAD + n * (THUMB_W + PAD)
    total_h = (BANNER_H
               + PAD + 18 + PAD
               + THUMB_H + BORDER*2 + LABEL_H + PAD
               + DESC_H + PAD
               + models_block_h)

    canvas = Image.new("RGB", (total_w, total_h), BG)
    draw   = ImageDraw.Draw(canvas)

    cat_color = CATEGORY_COLORS[category]
    draw.rectangle([0, 0, total_w, BANNER_H], fill=cat_color)
    draw.text((PAD, 5), CATEGORY_LABELS[category], font=font_cat, fill=(255, 255, 255))

    title = f"{task.upper()} · {ds} · {scene_name} · gt={ground_truth}"
    draw.text((PAD, BANNER_H + PAD), title, font=font_title, fill=FG_TITLE)

    for i, (frame, orig_idx) in enumerate(zip(frames, shown_order)):
        x = PAD + i * (THUMB_W + PAD)
        y = BANNER_H + PAD + 18 + PAD
        is_swapped = i in swap_indices
        border_color = RED if is_swapped else (45, 48, 58)
        draw.rectangle(
            [x - BORDER, y - BORDER, x + THUMB_W + BORDER, y + THUMB_H + BORDER],
            outline=border_color, width=BORDER,
        )
        canvas.paste(frame.resize((THUMB_W, THUMB_H)), (x, y))
        label_color = RED if is_swapped else DIM
        draw.text((x + 3, y + THUMB_H + 4),
                  f"pos {i}  (orig {orig_idx})", font=font_sm, fill=label_color)

    desc_y = BANNER_H + PAD + 18 + PAD + THUMB_H + BORDER*2 + LABEL_H + PAD
    if description:
        max_chars = (total_w - 2*PAD) // 7
        desc_text = description[:max_chars] + ("…" if len(description) > max_chars else "")
        draw.text((PAD, desc_y), desc_text, font=font_md, fill=DIM)

    sep_y = desc_y + DESC_H
    draw.line([(PAD, sep_y), (total_w - PAD, sep_y)], fill=(50, 54, 65), width=1)
    my = sep_y + PAD

    for correct, answer_str in human_answers:
        color = GREEN if correct else RED
        draw.text((PAD, my), f"{'✓' if correct else '✗'}  Human           → {answer_str}",
                  font=font_md, fill=color)
        my += MODEL_ROW_H

    draw.line([(PAD, my - 2), (total_w // 4, my - 2)], fill=(50, 54, 65), width=1)

    for model, (correct, answer_str) in model_answers.items():
        color = GREEN if correct else RED
        short = MODEL_SHORT.get(model, model)
        draw.text((PAD, my), f"{'✓' if correct else '✗'}  {short:<14} → {answer_str}",
                  font=font_md, fill=color)
        my += MODEL_ROW_H

    return canvas


def load_model_results(task, ds):
    results = {}
    for model in MODELS:
        path = ROOT / "results" / model / f"human_swap_{task}_{ds}.csv"
        if path.exists():
            with open(path) as f:
                results[model] = {r["scenario"]: r for r in csv.DictReader(f)}
    return results


def build_case(task, ds, scene_name, category):
    # Load manifest
    manifest_path = ROOT / "human_subset" / f"swap_{task}_{ds}.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    stimulus = next(s for s in manifest["stimuli"] if s["scene_name"] == scene_name)
    model_results = load_model_results(task, ds)

    model_answers = {}
    for model in MODELS:
        if model not in model_results:
            continue
        row = model_results[model].get(scene_name)
        if row is None:
            continue
        model_answers[model] = (model_correct(task, row), format_model_answer(task, row))

    human_answers = [
        (human_correct_fn(task, stimulus, ann), format_human_answer(task, ann["human_answer"]))
        for ann in stimulus["annotations"]
    ]

    description = load_description(ds, scene_name)
    return stimulus, model_answers, human_answers, description


TARGETS = [
    # (category,       task,       ds,         scene_name,      out_stem)
    ("both_correct",  "detect",  "craft",    "scene_5_002227",  "both_correct_craft_scene_5_002227"),
    ("both_correct",  "detect",  "mtl-aqa",  "scene_26_117",    "both_correct_mtl-aqa_scene_26_117"),
    ("llm_correct",   "detect",  "drive-lm", "scene_429",       "llm_correct_drive-lm_scene_429"),
    ("both_wrong",    "localize","mtl-aqa",  "scene_02_77",     "both_wrong_mtl-aqa_localize_scene_02_77"),
]


def main():
    out_dir = ROOT / "figures"
    out_dir.mkdir(exist_ok=True)

    for category, task, ds, scene_name, out_stem in TARGETS:
        print(f"Rendering {category} / {task} / {ds} / {scene_name} …")
        try:
            stimulus, model_answers, human_answers, description = build_case(
                task, ds, scene_name, category
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        img = render(
            ds, task, scene_name,
            stimulus["shown_order"],
            stimulus["ground_truth"],
            description,
            model_answers,
            human_answers,
            category,
        )
        out_path = out_dir / f"{out_stem}.pdf"
        img.save(str(out_path), "PDF", resolution=150)
        print(f"  → saved {out_path.relative_to(ROOT)}")

        strip = render_strip(
            ds, task, scene_name,
            stimulus["shown_order"],
            stimulus["ground_truth"],
        )
        strip_path = out_dir / f"{out_stem}_strip.pdf"
        strip.save(str(strip_path), "PDF", resolution=150)
        print(f"  → saved {strip_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
