"""Consolidated full-matrix evaluation for one fine-tuned (swap-detect-only)
checkpoint: (1) own held-out swap-detect, (2) own held-out swap-localize
(a task never trained on), (3) swap-detect transfer to the other three full
datasets. Loads the model once with a single fixed adapter — no need for
multi_lora here since we're only evaluating one checkpoint, unlike
checkpoint_sweep.py which sweeps many.

Consolidating into one job avoids ~4 separate cluster submissions per
trained model, each of which reloads the 2B base model from scratch and is
independently subject to SLURM queue waits (seen up to 25-30 min each
during this project).

Usage (on the cluster, vLLM eval venv):
    python3 scripts/full_matrix_eval.py <lora_path> <home_dataset_key> [heldout_manifest]
    # home_dataset_key one of: clevrer, craft, drive-lm, mtl-aqa
    # heldout_manifest defaults to scenes_finetune_heldout.json; override for
    # variant runs (e.g. mtl-aqa-full80's fresh scenes_finetune_heldout_full80.json)
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

# root: dataset directory. full_manifest: complete dataset (for transfer
# eval into datasets this checkpoint was never trained on at all).
# heldout_manifest: the 20% split disjoint from this dataset's own training
# data (only meaningful for the home dataset).
DATASETS = {
    "clevrer": {"root": "datasets/CLEVRER", "full_manifest": "scenes_filtered.json"},
    "craft": {"root": "datasets/CRAFT", "full_manifest": "scenes_filtered.json"},
    "drive-lm": {"root": "datasets/drive_lm", "full_manifest": "scenes_filtered.json"},
    "mtl-aqa": {"root": "datasets/MTL-AQA", "full_manifest": "scenes_filtered.json"},
}
HELDOUT_MANIFEST = "scenes_finetune_heldout.json"


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
    if len(sys.argv) not in (3, 4):
        print("Usage: full_matrix_eval.py <lora_path> <home_dataset_key> [heldout_manifest]", file=sys.stderr)
        sys.exit(1)
    lora_path = sys.argv[1]
    home_key = sys.argv[2]
    heldout_manifest = sys.argv[3] if len(sys.argv) > 3 else HELDOUT_MANIFEST
    if home_key not in DATASETS:
        print(f"Unknown dataset key {home_key!r}, expected one of {list(DATASETS)}", file=sys.stderr)
        sys.exit(1)

    model = Model("Qwen/Qwen3-VL-2B-Instruct", lora_path=lora_path, lora_rank=LORA_RANK)

    results = []

    # (1) own held-out swap-detect + (2) own held-out swap-localize
    home = DATASETS[home_key]
    heldout_dataset = Dataset(home["root"], manifest=heldout_manifest, seq_len_min=SEQ_LEN_MIN, seq_len_max=SEQ_LEN_MAX)
    swap_data = SwapDataset(heldout_dataset)

    detect_scenarios = swap_data.sample_binary(n=None)
    acc, yes_rate, n = evaluate_detect(model, detect_scenarios)
    print(f"[{home_key}] own held-out swap-detect: {acc:.3f} (yes={yes_rate:.2f}, n={n})")
    results.append(["own_heldout_detect", home_key, f"{acc:.4f}", f"{yes_rate:.4f}", n])

    localize_scenarios = swap_data.sample_position(n=None)
    loc_acc, n_loc = evaluate_localize(model, localize_scenarios)
    print(f"[{home_key}] own held-out swap-localize: {loc_acc:.3f} (n={n_loc})")
    results.append(["own_heldout_localize", home_key, f"{loc_acc:.4f}", "", n_loc])

    # (3) swap-detect transfer to the other three full datasets
    for other_key, other in DATASETS.items():
        if other_key == home_key:
            continue
        other_dataset = Dataset(other["root"], manifest=other["full_manifest"], seq_len_min=SEQ_LEN_MIN, seq_len_max=SEQ_LEN_MAX)
        other_scenarios = SwapDataset(other_dataset).sample_binary(n=None)
        acc, yes_rate, n = evaluate_detect(model, other_scenarios)
        print(f"[{home_key}] transfer swap-detect -> {other_key}: {acc:.3f} (yes={yes_rate:.2f}, n={n})")
        results.append(["transfer_detect", other_key, f"{acc:.4f}", f"{yes_rate:.4f}", n])

    out_path = Path(lora_path).parent / f"full_matrix_eval_{home_key}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["eval", "dataset", "accuracy", "yes_rate", "n"])
        writer.writerows(results)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
