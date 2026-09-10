"""Expand MTL-AQA's fine-tuning train set from the official 353-clip TEST
split (338 after LPIPS filtering) to the full train+test pool (1,372 after
filtering), recovered by re-downloading the MUSDL-hosted MTL-AQA frame
archive and pairing it with train_split_0.pkl (the split scripts/mtl-aqa's
original prepare_frames.py had deleted, keeping only test-split clips).

The held-out set is UNCHANGED (still the original 68-scene split used for
the original MTL-AQA fine-tune) so results are directly comparable to the
270-train-scene result. Train = every filtered scene (train+test combined)
NOT in that held-out set = 1,372 - 68 = 1,304.

Reads:  datasets/MTL-AQA/scenes_filtered_full.json (1372 scenes, train+test)
        datasets/MTL-AQA/scenes_finetune_heldout.json (68 scenes, fixed)
Writes: datasets/MTL-AQA/scenes_finetune_train_full.json

Usage:
    python3 scripts/expand_finetune_mtlaqa_train.py
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "datasets" / "MTL-AQA" / "scenes_filtered_full.json"
HELDOUT = ROOT / "datasets" / "MTL-AQA" / "scenes_finetune_heldout.json"
OUT = ROOT / "datasets" / "MTL-AQA" / "scenes_finetune_train_full.json"

SEED = 42


def main():
    all_scenes = json.loads(SRC.read_text())
    heldout = json.loads(HELDOUT.read_text())
    heldout_ids = {s["video_index"] for s in heldout}

    train = [s for s in all_scenes if s["video_index"] not in heldout_ids]

    train_ids = {s["video_index"] for s in train}
    assert not (train_ids & heldout_ids), "leakage between splits"
    assert len(train) + len(heldout) == len(all_scenes)

    rng = random.Random(SEED)
    rng.shuffle(train)

    OUT.write_text(json.dumps(train, indent=2))

    from collections import Counter
    lens = Counter(len(s["scene_paths"]) for s in train)
    print(f"{len(all_scenes)} filtered scenes, {len(heldout)} held out (unchanged)")
    print(f"Train: {len(train)} scenes, length distribution: {sorted(lens.items())}")
    print(f"Written to {OUT.name}")


if __name__ == "__main__":
    main()
