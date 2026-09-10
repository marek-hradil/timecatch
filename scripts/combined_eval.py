"""Evaluate the jointly-trained (all four datasets combined) swap-detect
checkpoint against each dataset's own held-out set.

Unlike scripts/full_matrix_eval.py, there's no single "home" dataset here --
this checkpoint was trained on a capped, standardized 557-scene slice of
all four datasets at once (see cluster/finetune_swap_detect_combined.sbatch),
so every dataset is "home". Skips the transfer-to-other-datasets step
entirely: since all four were already trained on, transfer no longer means
anything, and evaluating against each dataset's *full* manifest (rather than
its held-out slice) would leak training examples back in.

Uses each dataset's own correct held-out manifest -- the original 80/20 split
for CLEVRER/DriveLM, the full80 fresh-pool 80/20 split for CRAFT/MTL-AQA
(these are held-out sets, so they were never capped/subsampled themselves).

Usage (on the cluster, vLLM eval venv):
    python3 scripts/combined_eval.py <lora_path>
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
SEQ_LEN_MAX = 16  # CRAFT-full80's held-out set includes scenes up to 16 frames

DATASETS = {
    "clevrer": {"root": "datasets/CLEVRER", "heldout_manifest": "scenes_finetune_heldout.json"},
    "craft": {"root": "datasets/CRAFT", "heldout_manifest": "scenes_finetune_heldout_full80.json"},
    "drive-lm": {"root": "datasets/drive_lm", "heldout_manifest": "scenes_finetune_heldout.json"},
    "mtl-aqa": {"root": "datasets/MTL-AQA", "heldout_manifest": "scenes_finetune_heldout_full80.json"},
}


def evaluate_detect(model: Model, scenarios) -> tuple[float, float, int]:
    n_correct = 0
    n_yes = 0
    for i in range(0, len(scenarios), BATCH_SIZE):
        batch = scenarios[i:i + BATCH_SIZE]
        requests = [(s.frames, SWAP_PROMPT, s.scene_description) for s in batch]
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
        requests = [(s.frames, LOCALIZE_PROMPT, s.scene_description) for s in batch]
        answers = model.ask_batch(requests, pattern=r"\d+,\d+")
        for s, answer in zip(batch, answers):
            i0, j0 = s.position
            expected = f"{i0 + 1},{j0 + 1}"
            if answer.strip() == expected:
                n_correct += 1
    n = len(scenarios)
    return n_correct / n, n


def main():
    if len(sys.argv) != 2:
        print("Usage: combined_eval.py <lora_path>", file=sys.stderr)
        sys.exit(1)
    lora_path = sys.argv[1]

    model = Model("Qwen/Qwen3-VL-2B-Instruct", lora_path=lora_path, lora_rank=LORA_RANK)

    results = []
    for key, spec in DATASETS.items():
        heldout_dataset = Dataset(spec["root"], manifest=spec["heldout_manifest"], seq_len_min=SEQ_LEN_MIN, seq_len_max=SEQ_LEN_MAX)
        swap_data = SwapDataset(heldout_dataset)

        detect_scenarios = swap_data.sample_binary(n=None)
        acc, yes_rate, n = evaluate_detect(model, detect_scenarios)
        print(f"[{key}] held-out swap-detect: {acc:.3f} (yes={yes_rate:.2f}, n={n})")
        results.append(["heldout_detect", key, f"{acc:.4f}", f"{yes_rate:.4f}", n])

        localize_scenarios = swap_data.sample_position(n=None)
        loc_acc, n_loc = evaluate_localize(model, localize_scenarios)
        print(f"[{key}] held-out swap-localize: {loc_acc:.3f} (n={n_loc})")
        results.append(["heldout_localize", key, f"{loc_acc:.4f}", "", n_loc])

    out_path = Path(lora_path).parent / "combined_eval.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["eval", "dataset", "accuracy", "yes_rate", "n"])
        writer.writerows(results)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
