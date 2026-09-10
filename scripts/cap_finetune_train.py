"""Cap a fine-tuning train manifest down to a target size, stratified by
frame-length bucket, so datasets of very different native sizes (e.g. CRAFT's
7,855-scene full80 train set vs DriveLM's 557) can be combined into one joint
training run without the largest dataset dominating gradients.

Mirrors scripts/subsample_clevrer_train.py's bucketing logic, generalized to
any dataset directory/manifest.

Usage:
    python3 scripts/cap_finetune_train.py <dataset_dir> <src_manifest> <target_n> [out_manifest]
    # e.g.
    python3 scripts/cap_finetune_train.py datasets/CRAFT scenes_finetune_train_full80.json 557
    # -> datasets/CRAFT/scenes_finetune_train_full80_cap557.json
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

SEED = 42


def main():
    dataset_dir = Path(sys.argv[1])
    src_manifest = sys.argv[2]
    target_n = int(sys.argv[3])
    src = dataset_dir / src_manifest
    out = dataset_dir / (sys.argv[4] if len(sys.argv) > 4 else f"{src.stem}_cap{target_n}.json")

    scenes = json.loads(src.read_text())
    if target_n >= len(scenes):
        print(f"{src}: already {len(scenes)} <= target {target_n}, copying as-is")
        out.write_text(json.dumps(scenes, indent=2))
        return

    buckets = defaultdict(list)
    for scene in scenes:
        buckets[len(scene["scene_paths"])].append(scene)

    rng = random.Random(SEED)
    fraction = target_n / len(scenes)
    selected = []
    for length in sorted(buckets):
        group = list(buckets[length])
        rng.shuffle(group)
        k = round(len(group) * fraction)
        selected.extend(group[:k])
        print(f"  len={length}: {len(group)} scenes -> {k} selected")

    rng.shuffle(selected)

    out.write_text(json.dumps(selected, indent=2))
    print(f"\n{src.name}: {len(scenes)} -> {len(selected)} scenes (target {target_n})")
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
