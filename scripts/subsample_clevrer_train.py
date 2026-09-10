"""Subsample CLEVRER's fine-tuning train set down to ~1000 scenes, stratified
by frame-length bucket, to test whether CLEVRER's train-set size (3998
scenes, 4-15x more than the other three datasets) explains why fine-tuning
found a shortcut there but not on CRAFT/DriveLM/MTL-AQA — rather than
something specific to CLEVRER's data regardless of amount.

The held-out set is UNCHANGED (still the original 999-scene split used for
the full-CLEVRER run) so results are directly comparable.

Reads:  datasets/CLEVRER/scenes_finetune_train.json (3998 scenes)
Writes: datasets/CLEVRER/scenes_finetune_train_<N>.json

Usage:
    python3 scripts/subsample_clevrer_train.py [target_n]
    # defaults to 1000
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "datasets" / "CLEVRER" / "scenes_finetune_train.json"

SEED = 42
TARGET_N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
OUT = ROOT / "datasets" / "CLEVRER" / f"scenes_finetune_train_{TARGET_N}.json"


def main():
    scenes = json.loads(SRC.read_text())

    buckets = defaultdict(list)
    for scene in scenes:
        buckets[len(scene["scene_paths"])].append(scene)

    rng = random.Random(SEED)
    fraction = TARGET_N / len(scenes)
    selected = []
    for length in sorted(buckets):
        group = list(buckets[length])
        rng.shuffle(group)
        k = round(len(group) * fraction)
        selected.extend(group[:k])
        print(f"  len={length}: {len(group)} scenes -> {k} selected")

    rng.shuffle(selected)

    OUT.write_text(json.dumps(selected, indent=2))
    print(f"\n{len(scenes)} -> {len(selected)} scenes (target {TARGET_N})")
    print(f"Written to {OUT.name}")


if __name__ == "__main__":
    main()
