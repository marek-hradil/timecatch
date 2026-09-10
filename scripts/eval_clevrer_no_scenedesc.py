"""Evaluate a checkpoint (or the zero-shot base model) on CLEVRER's own
held-out set, WITHOUT scene_description text in the prompt -- the eval-side
counterpart of scripts/export_finetune_swap_detect_no_scenedesc.py's
training-side ablation.

Usage (on the cluster, vLLM eval venv):
    python3 scripts/eval_clevrer_no_scenedesc.py [lora_path]
    # omit lora_path to evaluate the zero-shot base model
"""

import csv
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
LOCALIZE_PROMPT = (
    "You are given a sequence of images showing a scene unfolding over time. "
    "Exactly two consecutive frames have been swapped. Reply with only the two "
    "frame numbers that are out of order, separated by a comma. For example: "
    "3,4 means frames 3 and 4 were swapped. Use 1-based indexing (1 for the "
    "first frame)."
)

BATCH_SIZE = 64
LORA_RANK = 8
SEQ_LEN_MIN = 4
SEQ_LEN_MAX = 8
HELDOUT_MANIFEST = "scenes_finetune_heldout.json"


def evaluate_detect(model: Model, scenarios) -> tuple[float, float, int]:
    n_correct = 0
    n_yes = 0
    for i in range(0, len(scenarios), BATCH_SIZE):
        batch = scenarios[i:i + BATCH_SIZE]
        requests = [(s.frames, SWAP_PROMPT, None) for s in batch]
        answers = model.ask_batch(requests, pattern=r"yes|no")
        for s, answer in zip(batch, answers):
            expected = "yes" if s.anomaly else "no"
            answer = answer.strip().lower()
            if answer == expected:
                n_correct += 1
            if answer == "yes":
                n_yes += 1
    n = len(scenarios)
    return n_correct / n, n_yes / n, n


def evaluate_localize(model: Model, scenarios) -> tuple[float, int]:
    n_correct = 0
    for i in range(0, len(scenarios), BATCH_SIZE):
        batch = scenarios[i:i + BATCH_SIZE]
        requests = [(s.frames, LOCALIZE_PROMPT, None) for s in batch]
        answers = model.ask_batch(requests, pattern=r"\d+,\d+")
        for s, answer in zip(batch, answers):
            i0, j0 = s.position
            expected = f"{i0 + 1},{j0 + 1}"
            if answer.strip() == expected:
                n_correct += 1
    n = len(scenarios)
    return n_correct / n, n


def main():
    lora_path = sys.argv[1] if len(sys.argv) > 1 else None
    label = lora_path if lora_path else "zero-shot"

    model = Model("Qwen/Qwen3-VL-2B-Instruct", lora_path=lora_path, lora_rank=LORA_RANK)

    heldout_dataset = Dataset("datasets/CLEVRER", manifest=HELDOUT_MANIFEST, seq_len_min=SEQ_LEN_MIN, seq_len_max=SEQ_LEN_MAX)
    swap_data = SwapDataset(heldout_dataset)

    detect_scenarios = swap_data.sample_binary(n=None)
    acc, yes_rate, n = evaluate_detect(model, detect_scenarios)
    print(f"[{label}] no-scenedesc held-out swap-detect: {acc:.3f} (yes={yes_rate:.2f}, n={n})")

    localize_scenarios = swap_data.sample_position(n=None)
    loc_acc, n_loc = evaluate_localize(model, localize_scenarios)
    print(f"[{label}] no-scenedesc held-out swap-localize: {loc_acc:.3f} (n={n_loc})")

    out_path = Path(lora_path).parent / "eval_no_scenedesc.csv" if lora_path else Path("results") / "clevrer_no_scenedesc_zeroshot.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["eval", "accuracy", "yes_rate", "n"])
        writer.writerow(["detect", f"{acc:.4f}", f"{yes_rate:.4f}", n])
        writer.writerow(["localize", f"{loc_acc:.4f}", "", n_loc])
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
