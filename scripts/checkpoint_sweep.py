"""Checkpoint sweep: evaluate every retained LoRA checkpoint from a training
run against the real held-out CLEVRER set, for both swap-detect (the trained
task) and corrupt-detect (the untrained regression control).

The first training run (checkpoint-650/714 only, selected by ms-swift's
internal 5%-of-train-pool eval split) looked healthy on paper but had
actually collapsed into a "yes"-biased near-chance policy on real held-out
data — including on corrupt-detect, which it was never trained on. That
internal split doesn't reveal this because it's drawn from the same narrow
training distribution. This script checks the real held-out set at every
checkpoint to show whether/when the collapse happens, alongside a step-0
(no-adapter, base model) reference point.

Loads the base model once (multi_lora=True) and swaps LoRA adapters between
checkpoints instead of reloading, which is much faster across ~15 checkpoints.

Usage (on the cluster, vLLM eval venv):
    python3 scripts/checkpoint_sweep.py checkpoints/qwen3-vl-2b-lora-swap-detect-clevrer/v4-<timestamp>
    python3 scripts/checkpoint_sweep.py <run_dir> <dataset_root> <heldout_manifest>
    # e.g. python3 scripts/checkpoint_sweep.py checkpoints/.../v0-<ts> datasets/drive_lm scenes_finetune_heldout.json
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset import CorruptDataset, Dataset, SwapDataset
from inference import Model

SWAP_PROMPT = (
    "You are given a sequence of images showing a scene unfolding over time. "
    "The frames should appear in a natural temporal order, but two consecutive "
    "frames may have been swapped. Reply with only 'yes' if you detect a swap, "
    "or 'no' if the order looks correct."
)
CORRUPT_PROMPT = (
    "You are given a sequence of images showing a scene unfolding over time. "
    "One frame may have been replaced by random noise. Reply with only 'yes' "
    "if you see a corrupted frame, or 'no' if all frames look normal."
)

BATCH_SIZE = 64
LORA_RANK = 8


def find_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    checkpoints = [(int(d.name.split("-")[1]), d) for d in run_dir.glob("checkpoint-*")]
    return sorted(checkpoints)


def evaluate(model: Model, scenarios, prompt: str, lora_path: str | None) -> tuple[float, float]:
    n_correct = 0
    n_yes = 0
    for i in range(0, len(scenarios), BATCH_SIZE):
        batch = scenarios[i:i + BATCH_SIZE]
        requests = [(s.frames, prompt, s.scene_description) for s in batch]
        answers = model.ask_batch(requests, pattern=r"yes|no", lora_path=lora_path)
        for s, answer in zip(batch, answers):
            expected = "yes" if s.anomaly else "no"
            answer = answer.strip().lower()
            if answer == expected:
                n_correct += 1
            if answer == "yes":
                n_yes += 1
    n = len(scenarios)
    return n_correct / n, n_yes / n


def main():
    if len(sys.argv) not in (2, 4):
        print("Usage: checkpoint_sweep.py <run_dir> [dataset_root heldout_manifest]", file=sys.stderr)
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    dataset_root = sys.argv[2] if len(sys.argv) > 2 else "datasets/CLEVRER"
    heldout_manifest = sys.argv[3] if len(sys.argv) > 2 else "scenes_finetune_heldout.json"
    checkpoints = find_checkpoints(run_dir)
    print(f"Found {len(checkpoints)} checkpoints: {[s for s, _ in checkpoints]}")

    dataset = Dataset(dataset_root, manifest=heldout_manifest)
    swap_scenarios = SwapDataset(dataset).sample_binary(n=None)
    corrupt_scenarios = CorruptDataset(dataset).sample_binary(n=None)
    print(f"Held-out set: {len(swap_scenarios)} swap-detect / {len(corrupt_scenarios)} corrupt-detect scenarios")

    model = Model("Qwen/Qwen3-VL-2B-Instruct", multi_lora=True, lora_rank=LORA_RANK)

    # step 0 = base model, no adapter — reference point before any LoRA drift.
    steps = [(0, None)] + [(step, str(path)) for step, path in checkpoints]

    out_path = run_dir / "checkpoint_sweep.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "swap_acc", "swap_yes_rate", "corrupt_acc", "corrupt_yes_rate"])

        for step, lora_path in steps:
            swap_acc, swap_yes = evaluate(model, swap_scenarios, SWAP_PROMPT, lora_path)
            corrupt_acc, corrupt_yes = evaluate(model, corrupt_scenarios, CORRUPT_PROMPT, lora_path)
            print(
                f"step={step:>4}  swap_acc={swap_acc:.3f} (yes={swap_yes:.2f})  "
                f"corrupt_acc={corrupt_acc:.3f} (yes={corrupt_yes:.2f})"
            )
            writer.writerow([step, f"{swap_acc:.4f}", f"{swap_yes:.4f}", f"{corrupt_acc:.4f}", f"{corrupt_yes:.4f}"])
            f.flush()

    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
