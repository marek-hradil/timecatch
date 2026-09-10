"""Cross-check a single checkpoint via the original, single-adapter Model
code path (lora_path fixed at construction, no per-call override) — used to
verify whether scripts/checkpoint_sweep.py's new multi-adapter-swap code
(lora_path passed per ask_batch call) produces the same result as the
already-validated path, on the same held-out examples.

Usage (on the cluster, vLLM eval venv):
    python3 scripts/verify_checkpoint.py <lora_checkpoint_dir> [n_examples]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset import Dataset, SwapDataset
from inference import Model

SWAP_PROMPT = (
    "You are given a sequence of images showing a scene unfolding over time. "
    "The frames should appear in a natural temporal order, but two consecutive "
    "frames may have been swapped. Reply with only 'yes' if you detect a swap, "
    "or 'no' if the order looks correct."
)


def main():
    if len(sys.argv) < 2:
        print("Usage: verify_checkpoint.py <lora_checkpoint_dir> [n_examples]", file=sys.stderr)
        sys.exit(1)
    lora_path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    dataset = Dataset("datasets/CLEVRER", manifest="scenes_finetune_heldout.json")
    scenarios = SwapDataset(dataset).sample_binary(n=None)[:n]

    model = Model("Qwen/Qwen3-VL-2B-Instruct", lora_path=lora_path, lora_rank=8)

    correct = 0
    n_yes = 0
    for i in range(0, len(scenarios), 64):
        batch = scenarios[i:i + 64]
        requests = [(s.frames, SWAP_PROMPT, s.scene_description) for s in batch]
        answers = model.ask_batch(requests, pattern=r"yes|no")
        for s, answer in zip(batch, answers):
            expected = "yes" if s.anomaly else "no"
            answer = answer.strip().lower()
            if answer == expected:
                correct += 1
            if answer == "yes":
                n_yes += 1

    print(f"single-adapter path: {correct}/{len(scenarios)} = {100 * correct / len(scenarios):.1f}%  (yes rate {n_yes / len(scenarios):.2f})")


if __name__ == "__main__":
    main()
