import json
import os
import sys

from PIL import Image

DATASETS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "datasets"))


def dataset_dir(dataset_name: str) -> str:
    return os.path.join(DATASETS_DIR, dataset_name)


def load_metadata(dataset_name: str) -> dict:
    path = os.path.join(dataset_dir(dataset_name), "metadata.json")
    with open(path) as f:
        return json.load(f)


def make_composite(image_paths: list[str], gap: int = 4) -> Image.Image:
    imgs = [Image.open(p) for p in image_paths]
    w = sum(im.width for im in imgs) + gap * (len(imgs) - 1)
    h = max(im.height for im in imgs)
    composite = Image.new("RGB", (w, h), "white")
    x = 0
    for im in imgs:
        composite.paste(im, (x, 0))
        x += im.width + gap
    return composite


def show_scene(dataset_name: str, scene_id: str):
    data = load_metadata(dataset_name)

    if scene_id not in data:
        print(f"Scene '{scene_id}' not found in dataset '{dataset_name}'.")
        sys.exit(1)

    scene = data[scene_id]
    print(f"Dataset: {dataset_name}")
    print(f"Scene: {scene_id}")
    print(f"Description: {scene['scene_description']}")
    print(f"Frames: {len(scene['frames'])}")

    base = dataset_dir(dataset_name)
    paths = [os.path.join(base, f) for f in scene["frames"]]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"Missing images: {missing}")
        sys.exit(1)

    composite = make_composite(paths)
    composite.show()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <dataset_name> <scene_id>")
        sys.exit(1)
    show_scene(sys.argv[1], sys.argv[2])
