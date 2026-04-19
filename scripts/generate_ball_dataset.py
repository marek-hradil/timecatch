"""Generate a synthetic dot-on-curve image dataset.

Produces 50 series (5 per each length in 5..14), saved under
datasets/single_image_abstract_ball/seq_XXX/ as 0.png, 1.png, ...
"""

import os
import numpy as np
from PIL import Image, ImageDraw

# ── Config ────────────────────────────────────────────────────────────────────

OUT_DIR = "datasets/single_image_abstract_ball"
SERIES_PER_LENGTH = 5
LENGTHS = range(5, 15)   # 5, 6, ..., 14  →  10 × 5 = 50 series
IMG_SIZE = 400
DOT_RADIUS = 10
LINE_WIDTH = 3
CURVE_POINTS = 1000  # resolution when sampling the Bezier path
MARGIN = 60          # keep curves away from image edges
SEED = 42

# ── Bezier curve ──────────────────────────────────────────────────────────────

def random_control_points(rng, n=4):
    """Return n control points within the drawable area."""
    lo, hi = MARGIN, IMG_SIZE - MARGIN
    xs = rng.uniform(lo, hi, n)
    ys = rng.uniform(lo, hi, n)
    return np.stack([xs, ys], axis=1)


def cubic_bezier(p0, p1, p2, p3, t):
    """Evaluate a cubic Bezier at scalar or array t ∈ [0, 1]."""
    t = np.asarray(t)[:, None]
    return (
        (1 - t) ** 3 * p0
        + 3 * (1 - t) ** 2 * t * p1
        + 3 * (1 - t) * t ** 2 * p2
        + t ** 3 * p3
    )


def make_curve(rng):
    """Return (N, 2) array of evenly-sampled points along a random Bezier."""
    pts = random_control_points(rng)
    ts = np.linspace(0, 1, CURVE_POINTS)
    return cubic_bezier(pts[0], pts[1], pts[2], pts[3], ts)


# ── Drawing ───────────────────────────────────────────────────────────────────

def random_color(rng):
    """Return a vivid RGB color (avoid near-white and near-black)."""
    while True:
        r, g, b = rng.integers(0, 256, 3)
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        if 60 < brightness < 210:
            return (int(r), int(g), int(b))


def render_frame(curve, t, dot_color):
    """Draw the full curve with the dot at position t ∈ [0, 1]."""
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), "white")
    draw = ImageDraw.Draw(img)

    # Draw curve as polyline
    polyline = [tuple(p) for p in curve]
    draw.line(polyline, fill="black", width=LINE_WIDTH)

    # Draw dot
    idx = int(t * (len(curve) - 1))
    x, y = curve[idx]
    bbox = [x - DOT_RADIUS, y - DOT_RADIUS, x + DOT_RADIUS, y + DOT_RADIUS]
    draw.ellipse(bbox, fill=dot_color)

    return img


# ── Dataset generation ────────────────────────────────────────────────────────

def generate_series(series_id, n_snapshots, rng):
    curve = make_curve(rng)
    dot_color = random_color(rng)

    out_dir = os.path.join(OUT_DIR, f"seq_{series_id:03d}")
    os.makedirs(out_dir, exist_ok=True)

    ts = np.linspace(0, 1, n_snapshots)
    for i, t in enumerate(ts):
        img = render_frame(curve, t, dot_color)
        img.save(os.path.join(out_dir, f"{i}.png"))


def main():
    rng = np.random.default_rng(SEED)
    total = len(LENGTHS) * SERIES_PER_LENGTH
    print(f"Generating {total} series → {OUT_DIR}")
    sid = 0
    for length in LENGTHS:
        for _ in range(SERIES_PER_LENGTH):
            generate_series(sid, length, rng)
            print(f"  seq_{sid:03d}  {length} snapshots")
            sid += 1
    print("Done.")


if __name__ == "__main__":
    main()
