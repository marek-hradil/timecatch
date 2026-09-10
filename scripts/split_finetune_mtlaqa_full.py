"""Fresh 80/20 split of MTL-AQA's FULL filtered pool (1,372 scenes, train+test
combined — see scripts/expand_finetune_mtlaqa_train.py for how that pool was
recovered) for the LoRA fine-tuning pilot.

Supersedes the earlier "expand train, keep the old 68-scene held-out fixed"
approach for this dataset: the original held-out set was only 68 scenes
(20% of the original 338-scene test-only pool), too small for a precise
standalone read of MTL-AQA fine-tuning. This draws a new 80/20 split from
the full 1,372-scene pool instead, matching split_finetune_mtlaqa.py's
approach exactly.

Reads:  datasets/MTL-AQA/scenes_filtered_full.json (1,372 scenes)
Writes: datasets/MTL-AQA/scenes_finetune_train_full80.json
        datasets/MTL-AQA/scenes_finetune_heldout_full80.json

Usage:
    python3 scripts/split_finetune_mtlaqa_full.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "datasets" / "MTL-AQA" / "scenes_filtered_full.json"
TRAIN_OUT = ROOT / "datasets" / "MTL-AQA" / "scenes_finetune_train_full80.json"
HELDOUT_OUT = ROOT / "datasets" / "MTL-AQA" / "scenes_finetune_heldout_full80.json"

SEED = 42
TRAIN_FRACTION = 0.8


def main():
    scenes = json.loads(SRC.read_text())

    buckets = defaultdict(list)
    for scene in scenes:
        buckets[len(scene["scene_paths"])].append(scene)

    rng = random.Random(SEED)
    train, heldout = [], []
    for length in sorted(buckets):
        group = list(buckets[length])
        rng.shuffle(group)
        n_train = round(len(group) * TRAIN_FRACTION)
        train.extend(group[:n_train])
        heldout.extend(group[n_train:])
        print(f"  len={length}: {len(group)} scenes -> {n_train} train / {len(group) - n_train} held-out")

    rng.shuffle(train)
    rng.shuffle(heldout)

    train_ids = {s["video_index"] for s in train}
    heldout_ids = {s["video_index"] for s in heldout}
    assert not (train_ids & heldout_ids), "leakage between splits"

    TRAIN_OUT.write_text(json.dumps(train, indent=2))
    HELDOUT_OUT.write_text(json.dumps(heldout, indent=2))

    print(f"\nTotal: {len(scenes)} scenes -> {len(train)} train / {len(heldout)} held-out")
    print(f"Written to {TRAIN_OUT.name} and {HELDOUT_OUT.name}")


if __name__ == "__main__":
    main()
