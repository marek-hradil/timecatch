"""Export CRAFT train-half swap_detect examples as an ms-swift SFT JSONL.

Mirrors scripts/export_finetune_swap_detect.py — see that file for rationale.
Reads the fine-tuning train manifest (datasets/CRAFT/scenes_finetune_train.json,
written by split_finetune_craft.py).

Usage (from repo root):
    python3 scripts/export_finetune_swap_detect_craft.py [manifest] [output]
    # e.g. for the CRAFT-long-expanded train set:
    python3 scripts/export_finetune_swap_detect_craft.py scenes_finetune_train_full.json sft_swap_detect_train_full.jsonl
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import Dataset, SwapDataset

MANIFEST = sys.argv[1] if len(sys.argv) > 1 else "scenes_finetune_train.json"
OUT_PATH = Path("datasets/CRAFT") / (sys.argv[2] if len(sys.argv) > 2 else "sft_swap_detect_train.jsonl")

SYSTEM_PROMPT = (
    "You are given a sequence of images showing a scene unfolding over time. "
    "The frames should appear in a natural temporal order, but two consecutive "
    "frames may have been swapped. Reply with only 'yes' if you detect a swap, "
    "or 'no' if the order looks correct."
)


def main():
    # seq_len_max=16 so CRAFT-long scenes (up to 16 frames) in expanded
    # manifests aren't silently dropped by Dataset's default 4-8 bound.
    dataset = Dataset("datasets/CRAFT", manifest=MANIFEST, seq_len_max=16)
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
