"""Export DriveLM train-half swap_detect examples as an ms-swift SFT JSONL.

Mirrors scripts/export_finetune_swap_detect.py — see that file for rationale.
Reads the fine-tuning train manifest (datasets/drive_lm/scenes_finetune_train.json,
written by split_finetune_drivelm.py). System prompt is identical across all
four datasets (see scripts/swap_detect.py / config/experiments), so no
dataset-specific prompt needed here.

Image paths are written relative to the repo root — run this script, and
ms-swift training, with the repo root as the working directory.

Usage (from repo root):
    python3 scripts/export_finetune_swap_detect_drivelm.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import Dataset, SwapDataset

OUT_PATH = Path("datasets/drive_lm/sft_swap_detect_train.jsonl")

SYSTEM_PROMPT = (
    "You are given a sequence of images showing a scene unfolding over time. "
    "The frames should appear in a natural temporal order, but two consecutive "
    "frames may have been swapped. Reply with only 'yes' if you detect a swap, "
    "or 'no' if the order looks correct."
)


def main():
    dataset = Dataset("datasets/drive_lm", manifest="scenes_finetune_train.json")
    scenarios = SwapDataset(dataset).sample_binary(n=None)

    n_pos = 0
    with open(OUT_PATH, "w") as f:
        for s in scenarios:
            assert s.paths is not None, f"scenario {s.name!r} has no frame paths"
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
