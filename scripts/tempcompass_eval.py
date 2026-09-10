"""Evaluate a checkpoint (or the zero-shot base model) on TempCompass's
yes_no task -- 2,453 yes/no questions over 410 real videos spanning 5
temporal dimensions (action, direction, speed, order, attribute_change).
Chosen over TempCompass's other three tasks (multi-choice, caption_matching,
captioning) because its answer format (a single yes/no token) is identical
to what the swap-detect LoRA was trained to produce, so no new answer
parsing/prompt-format adaptation is needed on the model side.

Reads:  datasets/TempCompass/yes_no.json (video_id, question, answer, dim)
        datasets/TempCompass/frames/<video_id>/0.jpg..7.jpg (8 uniformly
        sampled frames per video, written by scripts/tempcompass_prepare_frames.py)

Usage (on the cluster, vLLM eval venv):
    python3 scripts/tempcompass_eval.py [lora_path]
    # omit lora_path to evaluate the zero-shot base model
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).parent.parent))
from inference import Model, Prompt, build_messages

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
ROOT = Path(__file__).parent.parent / "datasets" / "TempCompass"
BATCH_SIZE = 64
LORA_RANK = 8

SYSTEM_PROMPT = (
    "You are given a sequence of frames sampled uniformly from a video. "
    "Answer the following yes/no question about the video. Reply with only "
    "'yes' or 'no'."
)


def main():
    lora_path = sys.argv[1] if len(sys.argv) > 1 else None
    label = lora_path if lora_path else "zero-shot"

    records = json.loads((ROOT / "yes_no.json").read_text())
    records = [r for r in records if (ROOT / "frames" / r["video_id"]).exists()]
    print(f"{len(records)} questions with frames available")

    model = Model(MODEL_ID, lora_path=lora_path, lora_rank=LORA_RANK)

    results = []
    n_correct = 0
    per_dim_correct = defaultdict(int)
    per_dim_total = defaultdict(int)

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        requests = []
        for r in batch:
            frame_dir = ROOT / "frames" / r["video_id"]
            paths = sorted(frame_dir.glob("*.jpg"), key=lambda p: int(p.stem))
            frames = [PILImage.open(p).copy() for p in paths]
            prompt = Prompt(frames, SYSTEM_PROMPT, r["question"])
            requests.append((build_messages(prompt, MODEL_ID), prompt.images))

        answers = model.ask_batch(requests, pattern=r"yes|no")
        for r, answer in zip(batch, answers):
            expected = r["answer"].strip().lower()
            answer = answer.strip().lower()
            correct = answer == expected
            n_correct += correct
            per_dim_total[r["dim"]] += 1
            per_dim_correct[r["dim"]] += correct
            results.append([r["video_id"], r["dim"], expected, answer, correct])

    n = len(records)
    print(f"\n[{label}] overall accuracy: {n_correct / n:.3f} (n={n})")
    for dim in sorted(per_dim_total):
        acc = per_dim_correct[dim] / per_dim_total[dim]
        print(f"  {dim}: {acc:.3f} (n={per_dim_total[dim]})")

    out_name = f"tempcompass_yes_no_{'zeroshot' if not lora_path else Path(lora_path).name}.csv"
    out_path = ROOT / out_name
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "dim", "expected", "answer", "correct"])
        writer.writerows(results)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
