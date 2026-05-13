"""
Visualize a CRAFT scene as a strip of frames with its description.

Usage:
    python3 scripts/craft/show_scene.py [--json scenes_filtered.json] [--scene <video_index>]

If --scene is omitted a random scene is chosen.
"""

import argparse
import json
import os
from PIL import Image, ImageDraw, ImageFont

CRAFT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../..", "datasets", "CRAFT"))

THUMB_W = 200
THUMB_H = 150
PAD = 6
LABEL_H = 20
DESC_H = 50
BG = (30, 30, 30)
FG = (220, 220, 100)
DIM = (160, 160, 160)


def load_font(size: int):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        return ImageFont.load_default()


def wrap_text(text: str, font, max_width: int) -> list[str]:
    words = text.split()
    lines, current = [], ""
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    for word in words:
        candidate = f"{current} {word}".strip()
        w = draw.textlength(candidate, font=font)
        if w <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def render_scene(scene: dict) -> Image.Image:
    paths = scene["scene_paths"]
    n = len(paths)
    desc = scene.get("scene_description", "")

    font_label = load_font(11)
    font_desc = load_font(13)

    desc_lines = wrap_text(desc, font_desc, THUMB_W * n + PAD * (n - 1))
    desc_block_h = DESC_H + (len(desc_lines) - 1) * 16 if desc_lines else DESC_H

    total_w = n * THUMB_W + (n + 1) * PAD
    total_h = PAD + THUMB_H + LABEL_H + PAD + desc_block_h + PAD

    canvas = Image.new("RGB", (total_w, total_h), BG)
    draw = ImageDraw.Draw(canvas)

    for i, rel_path in enumerate(paths):
        abs_path = os.path.join(CRAFT_DIR, rel_path)
        img = Image.open(abs_path).convert("RGB").resize((THUMB_W, THUMB_H))
        x = PAD + i * (THUMB_W + PAD)
        y = PAD
        canvas.paste(img, (x, y))
        label = os.path.basename(rel_path)
        draw.text((x + 3, y + THUMB_H + 3), label, font=font_label, fill=DIM)

    desc_y = PAD + THUMB_H + LABEL_H + PAD
    for line in desc_lines:
        draw.text((PAD, desc_y), line, font=font_desc, fill=FG)
        desc_y += 16

    return canvas


PAGE_SIZE = 20


def pick_scene(scenes: list[dict]) -> dict | None:
    page = 0
    total_pages = (len(scenes) - 1) // PAGE_SIZE + 1

    while True:
        os.system("clear")
        start = page * PAGE_SIZE
        chunk = scenes[start: start + PAGE_SIZE]

        print(f"  {'#':>4}  {'ID':>6}  {'F':>3}  Description")
        print("  " + "-" * 72)
        for i, s in enumerate(chunk):
            idx = start + i + 1
            desc = s.get("scene_description", "")[:52]
            frames = len(s["scene_paths"])
            print(f"  {idx:>4}  {s['video_index']:>6}  {frames:>3}  {desc}")

        print(f"\n  Page {page + 1}/{total_pages}   [n]ext  [p]rev  [q]uit")
        print("  Enter number or scene ID: ", end="", flush=True)

        raw = input().strip().lower()

        if raw == "q":
            return None
        if raw == "n":
            page = min(page + 1, total_pages - 1)
            continue
        if raw == "p":
            page = max(page - 1, 0)
            continue

        if raw.isdigit():
            val = int(raw)
            # treat as list number first, then as video_index
            if 1 <= val <= len(scenes):
                return scenes[val - 1]
            matches = [s for s in scenes if s["video_index"] == val]
            if matches:
                return matches[0]
            print(f"  Not found: {val}")
            input("  Press enter to continue...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="scenes_filtered.json")
    parser.add_argument("--scene", type=int, default=None)
    args = parser.parse_args()

    json_path = os.path.join(CRAFT_DIR, args.json)
    with open(json_path) as f:
        scenes = json.load(f)

    if args.scene is not None:
        matches = [s for s in scenes if s["video_index"] == args.scene]
        if not matches:
            print(f"Scene {args.scene} not found in {args.json}.")
            return
        scene = matches[0]
    else:
        scene = pick_scene(scenes)
        if scene is None:
            return

    os.system("clear")
    print(f"Scene {scene['video_index']}  |  {len(scene['scene_paths'])} frames")
    print(scene["scene_description"])

    img = render_scene(scene)
    img.show()


if __name__ == "__main__":
    main()
