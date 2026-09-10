"""Direct overfit sanity check.

Queries the model on the exact 24 examples used to train the overfit LoRA
adapter (datasets/CLEVRER/sft_swap_detect_overfit.jsonl) and compares
predictions to the labels baked into that file.

Deliberately bypasses the manifest/experiment-config eval pipeline
(scripts/swap_detect.py) — that re-samples swap positions from a dataset
manifest via SwapDataset, which wouldn't reproduce the literal examples the
model was trained on. This checks the actual training examples directly.

Usage (on the cluster, in the vLLM eval venv, not .venv-finetune):
    python3 scripts/check_overfit.py checkpoints/qwen3-vl-2b-lora-swap-detect-overfit/<run>/checkpoint-N
"""

import json
import re
import sys
from pathlib import Path

from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).parent.parent))
from inference import Model

SYSTEM_PROMPT = (
    "You are given a sequence of images showing a scene unfolding over time. "
    "The frames should appear in a natural temporal order, but two consecutive "
    "frames may have been swapped. Reply with only 'yes' if you detect a swap, "
    "or 'no' if the order looks correct."
)


def main():
    if len(sys.argv) != 2:
        print("Usage: check_overfit.py <lora_checkpoint_dir>", file=sys.stderr)
        sys.exit(1)
    lora_path = sys.argv[1]

    with open("datasets/CLEVRER/sft_swap_detect_overfit.jsonl") as f:
        examples = [json.loads(line) for line in f]

    model = Model("Qwen/Qwen3-VL-2B-Instruct", lora_path=lora_path, lora_rank=8)

    requests = []
    labels = []
    for ex in examples:
        user_content = ex["messages"][1]["content"]
        scene_desc = re.sub(r"^(<image>)+", "", user_content) or None
        images = [PILImage.open(p) for p in ex["images"]]
        requests.append((images, SYSTEM_PROMPT, scene_desc))
        labels.append(ex["messages"][2]["content"])

    predictions = model.ask_batch(requests, pattern=r"yes|no")

    correct = sum(p == l for p, l in zip(predictions, labels))
    print(f"{correct}/{len(labels)} correct ({100 * correct / len(labels):.1f}%)")
    for ex, pred, label in zip(examples, predictions, labels):
        marker = "OK" if pred == label else "WRONG"
        print(f"  [{marker}] {ex['images'][0]}: predicted={pred!r} expected={label!r}")


if __name__ == "__main__":
    main()
