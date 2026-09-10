"""
Browse scenes where all models failed but humans succeeded.

Shows frames in the exact order the human participant saw (shown_order),
with the swapped pair highlighted in red. Press Enter to advance.

Usage:
    python3 scripts/show_human_failures.py
    python3 scripts/show_human_failures.py --task localize --dataset drive-lm
"""

import argparse
import csv
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent.parent

DATASETS = ["clevrer", "craft", "drive-lm", "mtl-aqa"]
TASKS = ["detect", "localize"]
MODELS = ["qwen2-5-vl-7b", "qwen3-vl-8b", "gemma-4-e4b", "intern-vl-3", "intern-vl-3-5"]

DS_PATH = {
    "clevrer":  ROOT / "datasets/CLEVRER",
    "craft":    ROOT / "datasets/CRAFT",
    "drive-lm": ROOT / "datasets/drive_lm",
    "mtl-aqa":  ROOT / "datasets/MTL-AQA",
}

THUMB_W = 220
THUMB_H = 160
PAD = 8
LABEL_H = 22
DESC_H = 20
BG = (20, 20, 20)
FG = (220, 220, 100)
DIM = (140, 140, 140)
RED = (220, 60, 60)
GREEN = (60, 200, 60)
BORDER = 4


def load_font(size):
    for path in ["/System/Library/Fonts/Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
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
        raise FileNotFoundError(f"Scene dir not found: {scene_name}")
    stem_to_path = {}
    for p in d.iterdir():
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            try:
                stem_to_path[int(p.stem)] = p
            except ValueError:
                pass
    return [Image.open(stem_to_path[i]).convert("RGB") for i in shown_order]


MODEL_SHORT = {
    "qwen2-5-vl-7b":  "Qwen2.5-7B",
    "qwen3-vl-8b":    "Qwen3-8B",
    "gemma-4-e4b":    "Gemma4-E4B",
    "intern-vl-3":    "IVL3-8B",
    "intern-vl-3-5":  "IVL3.5-8B",
}


def render(ds, task, scene_name, shown_order, ground_truth, scene_description, model_answers, human_answers):
    frames = load_frames(ds, scene_name, shown_order)
    n = len(frames)

    swap_indices = set()
    if isinstance(ground_truth, list) and len(ground_truth) == 2:
        swap_indices = {ground_truth[0], ground_truth[1]}

    font_label  = load_font(11)
    font_desc   = load_font(12)
    font_title  = load_font(15)
    font_model  = load_font(12)

    MODEL_ROW_H = 18
    n_rows = len(human_answers) + 1 + len(model_answers)  # humans + divider + models
    models_block_h = PAD + n_rows * MODEL_ROW_H + PAD

    total_w = PAD + n * (THUMB_W + PAD)
    total_h = (PAD + 20 + PAD                          # title
               + THUMB_H + BORDER * 2 + LABEL_H + PAD  # frames
               + DESC_H + PAD                           # description
               + models_block_h)                        # model answers

    canvas = Image.new("RGB", (total_w, total_h), BG)
    draw = ImageDraw.Draw(canvas)

    title = f"{task.upper()} / {ds}  ·  {scene_name}  ·  gt={ground_truth}"
    draw.text((PAD, PAD), title, font=font_title, fill=FG)

    for i, (frame, orig_idx) in enumerate(zip(frames, shown_order)):
        x = PAD + i * (THUMB_W + PAD)
        y = PAD + 20 + PAD

        is_swapped = i in swap_indices
        border_color = RED if is_swapped else (50, 50, 50)

        draw.rectangle([x - BORDER, y - BORDER,
                        x + THUMB_W + BORDER, y + THUMB_H + BORDER],
                       outline=border_color, width=BORDER)
        canvas.paste(frame.resize((THUMB_W, THUMB_H)), (x, y))

        label_color = RED if is_swapped else DIM
        draw.text((x + 3, y + THUMB_H + 4),
                  f"pos {i}  (orig {orig_idx})", font=font_label, fill=label_color)

    desc_y = PAD + 20 + PAD + THUMB_H + BORDER * 2 + LABEL_H + PAD
    draw.text((PAD, desc_y), scene_description or "", font=font_desc, fill=DIM)

    # answers panel
    sep_y = desc_y + DESC_H
    draw.line([(PAD, sep_y), (total_w - PAD, sep_y)], fill=(60, 60, 60), width=1)
    my = sep_y + PAD

    for i, (correct, answer_str) in enumerate(human_answers):
        color = GREEN if correct else RED
        draw.text((PAD, my), f"{'✓' if correct else '✗'}  {'Human ' + str(i+1):<14} → {answer_str}",
                  font=font_model, fill=color)
        my += MODEL_ROW_H

    draw.line([(PAD, my), (total_w // 3, my)], fill=(60, 60, 60), width=1)
    my += MODEL_ROW_H // 2

    for model, (correct, answer_str) in model_answers.items():
        color = GREEN if correct else RED
        short = MODEL_SHORT.get(model, model)
        draw.text((PAD, my), f"{'✓' if correct else '✗'}  {short:<14} → {answer_str}",
                  font=font_model, fill=color)
        my += MODEL_ROW_H

    return canvas


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
    shifted = (gt[0] + 1, gt[1] + 1)
    return ans == shifted or ans == (shifted[1], shifted[0])


def human_correct(task, stimulus, ann):
    gt_raw, ha = stimulus["ground_truth"], ann["human_answer"]
    if task == "detect":
        gt = bool(gt_raw)
        ans = bool(ha) if isinstance(ha, bool) else str(ha).lower() == "true"
        return gt == ans
    if not isinstance(gt_raw, list) or not isinstance(ha, list):
        return None
    gt_t = (int(gt_raw[0]), int(gt_raw[1]))
    ans_t = (int(ha[0]), int(ha[1]))
    return gt_t == ans_t or gt_t == (ans_t[1], ans_t[0])


def format_model_answer(task, row):
    if task == "detect":
        return row["answer"].strip().lower()
    ans = parse_pair(row["answer"])
    # model answers are 1-indexed; subtract 1 to match 0-indexed gt display
    return f"pos {ans[0]-1},{ans[1]-1}" if ans else row["answer"].strip()


def format_human_answer(task, ha):
    if task == "detect":
        return str(ha).lower()
    if isinstance(ha, list) and len(ha) == 2:
        return f"pos {ha[0]},{ha[1]}"
    return str(ha)


def collect_failures(filter_task=None, filter_ds=None):
    model_results = {}
    for model in MODELS:
        model_results[model] = {}
        for task in TASKS:
            for ds in DATASETS:
                path = ROOT / "results" / model / f"human_swap_{task}_{ds}.csv"
                if path.exists():
                    with open(path) as f:
                        model_results[model][(task, ds)] = {
                            r["scenario"]: r for r in csv.DictReader(f)
                        }

    cases = []
    tasks = [filter_task] if filter_task else TASKS
    datasets = [filter_ds] if filter_ds else DATASETS

    for task in tasks:
        for ds in datasets:
            manifest_path = ROOT / "human_subset" / f"swap_{task}_{ds}.json"
            if not manifest_path.exists():
                continue
            with open(manifest_path) as f:
                manifest = json.load(f)

            for s in manifest["stimuli"]:
                anns = s["annotations"]
                if not anns:
                    continue
                h_results = [human_correct(task, s, a) for a in anns]
                if None in h_results or not all(h_results):
                    continue

                mc = {}
                for model in MODELS:
                    key = (task, ds)
                    if key not in model_results.get(model, {}):
                        continue
                    rows = model_results[model][key]
                    if s["scene_name"] in rows:
                        row = rows[s["scene_name"]]
                        mc[model] = (model_correct(task, row), format_model_answer(task, row))

                if len(mc) < 5:
                    continue
                if not all(correct is False for correct, _ in mc.values()):
                    continue

                human_ans = [
                    (human_correct(task, s, a), format_human_answer(task, a["human_answer"]))
                    for a in anns
                ]

                cases.append({
                    "task":          task,
                    "ds":            ds,
                    "scene_name":    s["scene_name"],
                    "shown_order":   s["shown_order"],
                    "ground_truth":  s["ground_truth"],
                    "n_humans":      len(anns),
                    "model_answers": mc,
                    "human_answers": human_ans,
                })

    return cases


def load_description(ds, scene_name):
    manifest = DS_PATH[ds] / "scenes_filtered.json"
    if not manifest.exists():
        return ""
    with open(manifest) as f:
        scenes = json.load(f)
    for s in scenes:
        paths = s.get("scene_paths", [])
        if paths:
            name = Path(paths[0]).parent.name
            if name == scene_name:
                return s.get("scene_description", "")
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, default=None)
    parser.add_argument("--dataset", choices=DATASETS, default=None)
    args = parser.parse_args()

    cases = collect_failures(args.task, args.dataset)
    if not cases:
        print("No matching cases found.")
        return

    print(f"Found {len(cases)} cases. Press Enter to advance, q to quit.\n")

    for i, case in enumerate(cases):
        desc = load_description(case["ds"], case["scene_name"])
        case["description"] = desc

        print(f"[{i+1}/{len(cases)}]  {case['task']}/{case['ds']}  "
              f"{case['scene_name']}  gt={case['ground_truth']}  "
              f"n_humans={case['n_humans']}")

        img = render(
            case["ds"], case["task"], case["scene_name"],
            case["shown_order"], case["ground_truth"], desc,
            case["model_answers"], case["human_answers"],
        )
        img.show()

        cmd = input("  Enter=next  q=quit  > ").strip().lower()
        if cmd == "q":
            break


if __name__ == "__main__":
    main()
