"""Generate a synthetic number-in-image dataset.

Produces 50 series (5 per each length in 5..14), saved under
datasets/single_image_number/seq_XXX/ as 0.png, 1.png, ...

Each image shows only the frame index as a large centered number.
"""

import os
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = "datasets/single_image_number"
SERIES_PER_LENGTH = 5
LENGTHS = range(5, 15)
IMG_SIZE = 400
FONT_SIZE = 160

def render_frame(index):
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), "white")
    draw = ImageDraw.Draw(img)
    text = str(index)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (IMG_SIZE - w) // 2 - bbox[0]
    y = (IMG_SIZE - h) // 2 - bbox[1]
    draw.text((x, y), text, fill="black", font=font)
    return img


def generate_series(series_id, n_snapshots):
    out_dir = os.path.join(OUT_DIR, f"seq_{series_id:03d}")
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n_snapshots):
        img = render_frame(i)
        img.save(os.path.join(out_dir, f"{i}.png"))


def main():
    total = len(LENGTHS) * SERIES_PER_LENGTH
    print(f"Generating {total} series → {OUT_DIR}")
    sid = 0
    for length in LENGTHS:
        for _ in range(SERIES_PER_LENGTH):
            generate_series(sid, length)
            print(f"  seq_{sid:03d}  {length} snapshots")
            sid += 1
    print("Done.")


if __name__ == "__main__":
    main()
