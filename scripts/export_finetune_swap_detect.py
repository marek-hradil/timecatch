"""Export CLEVRER train-half swap_detect examples as an ms-swift SFT JSONL.

Reads the fine-tuning train manifest (datasets/CLEVRER/scenes_finetune_train.json,
written by split_finetune_clevrer.py) and reuses SwapDataset's swap logic so the
50/50 swap/no-swap label balance matches eval. System prompt, scene-description
placement, and yes/no target text match scripts/swap_detect.py and
config/experiments/swap-detect-clevrer-qwen3-vl-2b.yaml exactly, so the training
distribution matches what's evaluated afterward.

Image paths are written relative to the repo root (e.g.
"datasets/CLEVRER/scenes/scene_10000/0.jpg") — run this script, and ms-swift
training, with the repo root as the working directory.

Usage (from repo root):
    python3 scripts/export_finetune_swap_detect.py [manifest] [output]
    # e.g. for the 1000-scene subsample:
    python3 scripts/export_finetune_swap_detect.py scenes_finetune_train_1000.json sft_swap_detect_train_1000.jsonl
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import Dataset, SwapDataset

MANIFEST = sys.argv[1] if len(sys.argv) > 1 else "scenes_finetune_train.json"
OUT_PATH = Path("datasets/CLEVRER") / (sys.argv[2] if len(sys.argv) > 2 else "sft_swap_detect_train.jsonl")

# Matches config/experiments/swap-detect-clevrer-qwen3-vl-2b.yaml exactly.
SYSTEM_PROMPT = (
    "You are given a sequence of images showing a scene unfolding over time. "
    "The frames should appear in a natural temporal order, but two consecutive "
    "frames may have been swapped. Reply with only 'yes' if you detect a swap, "
    "or 'no' if the order looks correct."
)


def main():
    dataset = Dataset("datasets/CLEVRER", manifest=MANIFEST)
    scenarios = SwapDataset(dataset).sample_binary(n=None)

    n_pos = 0
    with open(OUT_PATH, "w") as f:
        for s in scenarios:
            assert s.paths is not None, f"scenario {s.name!r} has no frame paths"
            # Mirrors inference.py's _messages(): image tokens first, scene
            # description text (if any) appended after.
            user_content = "<image>" * len(s.paths) + (s.scene_description or "")
            label = "yes" if s.anomaly else "no"
            n_pos += s.anomaly
            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": label},
                ],
                "images": s.paths,
            }
            f.write(json.dumps(example) + "\n")

    print(f"{len(scenarios)} examples written to {OUT_PATH} ({n_pos} swapped / {len(scenarios) - n_pos} unswapped)")


if __name__ == "__main__":
    main()
