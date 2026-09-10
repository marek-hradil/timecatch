"""Expand CRAFT's fine-tuning train set to include CRAFT-long (9-16 frame
sequences), which split_finetune_craft.py originally excluded solely to keep
CRAFT-long clean for the paper's separate sequence-length ablation (measured
on the zero-shot base model — unaffected by reusing these frames here).

The held-out set is UNCHANGED (still the original 172-scene, 4-8-frame-only
split used for the original CRAFT fine-tune) so results are directly
comparable to the 686-train-scene result. Train = every filtered CRAFT scene
NOT in that held-out set, i.e. the original 686 (4-8 frames) plus all 1,101
CRAFT-long scenes (9-16 frames) = 1,787.

Reads:  datasets/CRAFT/scenes_filtered.json (1959 scenes, 4-16 frames)
        datasets/CRAFT/scenes_finetune_heldout.json (172 scenes, fixed)
Writes: datasets/CRAFT/scenes_finetune_train_full.json

Usage:
    python3 scripts/expand_finetune_craft_train.py
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "datasets" / "CRAFT" / "scenes_filtered.json"
HELDOUT = ROOT / "datasets" / "CRAFT" / "scenes_finetune_heldout.json"
OUT = ROOT / "datasets" / "CRAFT" / "scenes_finetune_train_full.json"

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

    lengths_in = sum(1 for s in train if 4 <= len(s["scene_paths"]) <= 8)
    lengths_long = sum(1 for s in train if 9 <= len(s["scene_paths"]) <= 16)
    print(f"{len(all_scenes)} filtered scenes, {len(heldout)} held out (unchanged, 4-8 frames only)")
    print(f"Train: {len(train)} scenes ({lengths_in} at 4-8 frames, {lengths_long} at 9-16 frames)")
    print(f"Written to {OUT.name}")


if __name__ == "__main__":
    main()
